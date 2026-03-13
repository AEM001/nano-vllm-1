"""

"""

import pickle
import torch
import torch.distributed as dist

from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    High-performance model execution engine with tensor parallelism support.
    
    The ModelRunner is responsible for:
    1. **Model Loading**: Loading and distributing model across GPUs
    2. **KV Cache Management**: Allocating and managing memory for attention states
    3. **CUDA Graphs**: Pre-capturing computation graphs for decode phase optimization
    4. **Batch Execution**: Running prefill and decode phases efficiently
    5. **Inter-process Communication**: Coordinating across tensor parallel ranks
    
    Process Architecture:
    - Rank 0: Main process that coordinates with LLMEngine
    - Rank 1..N-1: Worker processes that run in event-driven loop
    - Shared Memory: For fast inter-process communication
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):

        self.config = config
        hf_config = config.hf_config#due to model runner is gonna directly working with the GPU, it is neccessary to knwo the exact structure

        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size#Total number of processes in tensor parallel group

        self.rank = rank#this process's ID
        self.event = event#Synchronization object for coordinating with other ranks,Rank 0 gets list[Event] from all workers

        #Initialize pytorch distributed communication between process
        #set up NCCL for GPU-to-GPU communication
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        # Assign this process to a specific GPU
        torch.cuda.set_device(rank)
        
        
        # Setup model precision and device
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # Load model and initialize components
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # Performance optimizations
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # Reset device settings
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # Multi-process coordination setup
        if self.world_size > 1:
            if rank == 0:
                # Main process: create shared memory
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # Worker process: connect and start event loop
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """
        Gracefully shutdown ModelRunner and clean up resources.
        
        Cleanup Sequence:
        1. Close shared memory for inter-process communication
        2. Destroy CUDA graphs if they were created
        3. Synchronize CUDA operations
        4. Destroy distributed process group
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        Event-driven loop for worker processes (rank > 0).
        
        Worker processes continuously:
        1. Wait for method calls via shared memory
        2. Execute the requested method
        3. Exit when "exit" method is called
        
        This enables coordination without expensive RPC calls.
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        Read method call from shared memory (worker processes only).
        
        Returns:
            tuple: (method_name, args) to execute
            
        Communication Protocol:
        - Event signals data availability
        - First 4 bytes: data length (little endian)
        - Remaining bytes: pickled [method_name, *args]
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        Write method call to shared memory (main process only).
        
        Args:
            method_name: Name of method to execute on workers
            *args: Arguments to pass to the method
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        Execute method call across all tensor parallel ranks.
        
        Coordination Pattern:
        - Rank 0: Broadcasts method call to workers via shared memory
        - All ranks: Execute the method locally
        - Returns result from rank 0 only
        
        Args:
            method_name: Name of method to execute
            *args: Method arguments
            
        Returns:
            Result from rank 0 (None for other ranks)
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        Warmup the model to allocate memory and optimize CUDA kernels.
        
        Warmup Benefits:
        - Allocates initial GPU memory to avoid fragmentation
        - Compiles CUDA kernels for optimal performance
        - Pre-allocates buffers and internal state
        - Reduces first-inference latency
        
        Strategy:
        Create maximum-sized batch to ensure all memory allocations occur.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        Allocate KV cache memory based on available GPU memory.
        
        KV Cache Strategy:
        - Calculate memory needed per block (2 * layers * block_size * heads * head_dim)
        - Determine maximum blocks based on GPU memory utilization
        - Pre-allocate contiguous memory tensor for all blocks
        - Assign cache tensors to each attention layer
        
        Memory Calculation:
        block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
        """
        config = self.config
        hf_config = config.hf_config
        
        # Get GPU memory information
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # Calculate KV cache parameters
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # Calculate maximum number of blocks we can allocate
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        
        # Pre-allocate KV cache tensor
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        # Assign cache tensors to attention layers
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        Prepare block tables for efficient KV cache indexing.
        
        Block Tables map logical sequence positions to physical KV cache blocks:
        - Each sequence has a list of allocated block indices
        - Tables are padded to equal length for GPU efficiency
        - -1 indicates invalid/unused slots
        
        Args:
            seqs: List of sequences to prepare block tables for
            
        Returns:
            torch.Tensor: Padded block tables [num_seqs, max_blocks]
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        Prepare batch data for prefill phase (prompt processing).
        
        Prefill Phase Characteristics:
        - Processes entire prompt tokens at once
        - Writes initial KV cache states
        - Variable sequence lengths in batch
        - May use prefix cache optimization
        
        Args:
            seqs: List of sequences to prepare for prefill
            
        Returns:
            tuple: (input_ids, positions) tensors for model input
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            # Get tokens that need processing (exclude cached tokens)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            # Calculate sequence lengths for cu_seqlens
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # Calculate KV cache slot mapping
            if not seq.block_table:    # warmup phase
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        
        # Handle prefix cache optimization
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        # Convert to tensors and transfer to GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # Set global context for attention layers
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        Prepare batch data for decode phase (token generation).
        
        Decode Phase Characteristics:
        - Processes exactly one token per sequence
        - Reads from existing KV cache states
        - Appends new KV states to cache
        - Fixed batch size (1 token per sequence)
        
        Args:
            seqs: List of sequences to prepare for decode
            
        Returns:
            tuple: (input_ids, positions) tensors for model input
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        
        for seq in seqs:
            # Each sequence contributes exactly one token
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # Calculate KV cache slot for new token
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        
        # Convert to tensors and transfer to GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # Set global context for attention layers
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        Prepare sampling parameters for token generation.
        
        Args:
            seqs: List of sequences needing sampling
            
        Returns:
            torch.Tensor: Temperature parameters for each sequence
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        Execute model forward pass with CUDA graph optimization.
        
        Execution Strategy:
        - Prefill or large batches: Use eager execution
        - Small decode batches: Use pre-captured CUDA graphs
        - CUDA graphs eliminate Python overhead and kernel launch costs
        
        Args:
            input_ids: Token IDs to process
            positions: Position IDs for each token
            is_prefill: Whether this is prefill phase
            
        Returns:
            torch.Tensor: Logits for next token prediction
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Eager execution for prefill, forced eager, or large batches
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA graph execution for optimized decode
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # Update graph input variables
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # Replay captured graph
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        Main execution method for model inference.
        
        Pipeline:
        1. Prepare batch data (prefill or decode)
        2. Prepare sampling parameters
        3. Execute model forward pass
        4. Sample next tokens
        5. Reset global context
        
        Args:
            seqs: List of sequences to process
            is_prefill: Whether this is prefill phase
            
        Returns:
            list[int]: Generated token IDs (rank 0 only)
        """
        # Prepare batch data based on phase
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        
        # Prepare sampling parameters (rank 0 only)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        # Execute model
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # Sample tokens (rank 0 only)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        # Clean up global context
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        Capture CUDA graphs for optimized decode phase execution.
        
        CUDA Graph Benefits:
        - Eliminates Python overhead for each decode step
        - Reduces CUDA kernel launch costs
        - Pre-allocates all memory allocations
        - Dramatically improves decode throughput
        
        Capture Strategy:
        - Pre-capture graphs for common batch sizes (1, 2, 4, 8, 16, 32, ...)
        - Use memory pool to share allocations across graphs
        - Warmup each graph before capture to ensure stable memory
        - Store graph variables for dynamic updates during replay
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # Pre-allocate tensors for graph capture
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # Define batch sizes to capture (powers of 2 + multiples of 16)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # Capture graphs in reverse order (largest first) for memory efficiency
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            
            # Setup context for this batch size
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # Warmup run to ensure stable memory state
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # Capture the graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # Use first graph's memory pool for all subsequent graphs
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # Store references to graph variables for dynamic updates
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
