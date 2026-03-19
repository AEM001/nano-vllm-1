import pickle
import time
import logging
import torch
import torch.distributed as dist

from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.layers.sampler import Sampler

# Get module-specific logger
logger = logging.getLogger(__name__)

class ModelRunner:

    def __init__(self, config: Config, rank: int, events: Event | list[Event]):

        self.config = config
        hf_config = config.hf_config#due to model runner is gonna directly working with the GPU, it is neccessary to knwo the exact structure

        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size#Total number of processes in tensor parallel group

        self.rank = rank#this process's ID
        self.event = events#Synchronization object for coordinating with other ranks,Rank 0 gets list[Event] from all workers

        logger.info(f"Model runner {rank} initialized")

        #Initialize pytorch distributed communication between process
        #set up NCCL for GPU-to-GPU communication
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        # Assign this process to a specific GPU
        torch.cuda.set_device(rank)
        logger.info(f"GPU device set to rank {rank}")
        
        
        # Setup model precision and device
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
      
        self.model = Qwen3ForCausalLM(hf_config)
        t0 = time.time()
        load_model(self.model, config.model)
        t1 = time.time()
        logger.info(f"Model loaded on rank {rank} in {t1 - t0:.4f} seconds")


        self.sampler = Sampler()
        
        # Performance optimizations
        logger.info(f"Warming up model on rank {rank}...")
        self.warmup_model()
        logger.info(f"Allocating KV cache on rank {rank}...")
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
                self.loop()#rank0 doesn't execute

    def prepare_block_tables(self, seqs: list[Sequence]):
        
        max_len = max(len(seq.block_table) for seq in seqs)#find maximum block table length

        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]#pad all tables to equal length with -1

        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)#convert to tensor and move to GPU

        return block_tables


    def exit(self):

        if self.world_size > 1:
            self.shm.close()#Every process (rank 0,1,2,3) closes its connection to shared memory
            dist.barrier()#synchronization, wait until every process finish closing shared memory
            
            if self.rank == 0:
                self.shm.unlink()#Only rank 0 unlink (delete) the shared memory

        if not self.enforce_eager:
            del self.graphs, self.graph_pool#delete cuda graphs and memory pool
        torch.cuda.synchronize()#Blocks CPU until all GPU operations finish

        dist.destroy_process_group()#Destroys the NCCL communication group

    def loop(self):
        
        while True:
            method_name, args = self.read_shm()

            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()

        n = int.from_bytes(self.shm.buf[0:4], "little")

        method_name, *args = pickle.loads(self.shm.buf[4:n+4])

        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
               
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])#seiralize the method call for inter-process communication
        n = len(data)

        self.shm.buf[0:4] = n.to_bytes(4, "little")#write the length of the data to the first 4 bytes
        self.shm.buf[4:n+4] = data#write the data to the shared memory

        for event in self.event:
            event.set()#two conditions:set() or clear()

    def call(self, method_name, *args):
 
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)#broadcast and execute

        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len

        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)

        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        t0 = time.time()
        self.run(seqs, True)
        t1 = time.time()
        logger.info(f"Model warmed up in {t1 - t0:.4f} seconds")
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        
        config = self.config
        hf_config = config.hf_config
        
        # Get GPU memory information
        free, total = torch.cuda.mem_get_info()
        logger.info(f"Free memory: {free / 1024**3:.2f} GB")
     
        used = total - free 
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        logger.info(f"Peak memory: {peak / 1024**3:.2f} GB")
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        logger.info(f"Current memory: {current / 1024**3:.2f} GB")
        
        # Calculate KV cache parameters
        num_kv_heads = hf_config.num_key_value_heads // self.world_size

        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # Calculate maximum number of blocks we can allocate
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        logger.info(f"Number of KV cache blocks: {config.num_kvcache_blocks}")
        # Pre-allocate KV cache tensor
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        logger.info(f"Free memory after KV cache allocation: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")
        # Assign cache tensors to attention layers
        layer_id = 0

        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                """
                Iterates through all model modules (linear layers, attention layers, etc.)
                Filters for attention layers that have k_cache and v_cache attributes
                Only attention layers need KV cache storage
                """
                module.k_cache = self.kv_cache[0, layer_id]#Each layer gets its own memory slice
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        
        max_len = max(len(seq.block_table) for seq in seqs)#find maximum block table length

        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]#pad all tables to equal length with -1

        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)#convert to tensor and move to GPU

        return block_tables


    def prepare_prefill(self, seqs: list[Sequence]):
        # Batch preparation for prefill phase - processes multiple sequences efficiently

        input_ids = []  # Flattened new tokens from all sequences
        positions = []  # REAL positions for each token (critical for embeddings)

        cu_seqlens_q = [0]  # Cumulative query lengths (new tokens only)
        cu_seqlens_k = [0]  # Cumulative key lengths (all tokens for KV cache)

        max_seqlen_q = 0    # Max new tokens in any sequence
        max_seqlen_k = 0    # Max total tokens in any sequence

        slot_mapping = []   # Maps logical positions to physical KV cache slots
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            
            # Extract only NEW tokens (skip cached ones for efficiency)
            input_ids.extend(seq[seq.num_cached_tokens:])
            # Generate REAL positions for new tokens (not 0,1,2!)
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # Example: Seq has tokens [10,20,30,40,50], cached=2
            # New tokens: [30,40,50], positions: [2,3,4] (not [0,1,2]!)

            # Build cumulative lengths for GPU batch processing
            seqlen_q = seqlen - seq.num_cached_tokens  # New tokens count
            seqlen_k = seqlen                           # Total tokens count

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)  # Running total of new tokens
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)  # Running total of all tokens
            # Example batch: Seq1(3 new), Seq2(2 new), Seq3(4 new)
            # cu_seqlens_q: [0, 3, 5, 9] → GPU knows: seq0=[0:3], seq1=[3:5], seq2=[5:9]

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # Map logical sequence blocks to physical KV cache memory (PagedAttention)
            if not seq.block_table:    # Skip warmup phase (no cache yet)
                logger.debug("Skipping warmup phase in prepare_prefill")
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size  # Physical start slot

                if i != seq.num_blocks - 1:
                    end = start + self.block_size            # Full block
                else:
                    end = start + seq.last_block_num_tokens # Partial last block

                slot_mapping.extend(list(range(start, end)))
                # Example: block_table=[7,3], block_size=4, cached_blocks=0
                # Block 0: slots 7*4=28 to 31, Block 1: slots 3*4=12 to 15
                # slot_mapping: [28,29,30,31,12,13,14,15]
        
        # Optimization: only prepare block tables if we have cached data
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        # Convert to GPU tensors (pin_memory = faster CPU->GPU transfer)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # Set global context for attention layers (avoids passing many parameters)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # Batch preparation for decode phase - processes exactly 1 token per sequence
        
        input_ids = []    # Last token of each sequence (only 1 token per seq!)
        positions = []    # Position of that last token
        slot_mapping = [] # Where to store NEW KV state for this token
        context_lens = [] # How many tokens to attend to (sequence length)
        
        for seq in seqs:
            # Each sequence contributes exactly ONE token for decode
            input_ids.append(seq.last_token)
            # Position is always len(seq)-1 (the last position)
            positions.append(len(seq) - 1)
            # Context length = full sequence length for attention mask
            context_lens.append(len(seq))
            # Example: seq has 10 tokens, context_lens=10 means "attend to all 10 previous tokens"
            
            # Calculate KV cache slot for NEW token's KV state
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            # Example: last_block=5, block_size=4, last_block_num_tokens=2
            # slot = 5*4 + 2 - 1 = 21 (where to store KV for new token)
        
        # Convert to GPU tensors (pin_memory = faster CPU->GPU transfer)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        block_tables = self.prepare_block_tables(seqs)
        
        # Set global context for attention layers (decode mode = False)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
     
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Eager execution for prefill, forced eager, or large batches
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA graph execution for optimized decode
            bs = input_ids.size(0)#get batch size
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
        # Pre-capture CUDA graphs for ultra-fast decode execution
        # Eliminates Python overhead and kernel launch costs during inference
        
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # Pre-allocate tensors for graph capture (max possible size)
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # Define batch sizes to capture (powers of 2 + multiples of 16)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # Example: max_bs=32 → graph_bs = [1,2,4,8,16,32]
        # During inference, batch size 12 will use graph size 16
        self.graphs = {}
        self.graph_pool = None

        # Capture graphs in reverse order (largest first) for memory efficiency
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            
            # Setup context for this batch size
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # Warmup run to ensure stable memory state (prevents memory fragmentation)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # Capture the graph (records all GPU operations)
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # Example: For bs=8, this captures forward pass for 8 tokens
            
            # Use first graph's memory pool for all subsequent graphs (memory sharing)
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            
            self.graphs[bs] = graph
            torch.cuda.synchronize()  # Ensure capture completes
            reset_context()

        # Store references to graph variables for dynamic updates during replay
        self.graph_vars = dict(
            input_ids=input_ids,      # Will be updated with real tokens
            positions=positions,      # Will be updated with real positions
            slot_mapping=slot_mapping, # Will be updated with real KV slots
            context_lens=context_lens, # Will be updated with real context lengths
            block_tables=block_tables, # Will be updated with real block tables
            outputs=outputs,          # Will contain model outputs
        )
        # During inference: graph_vars["input_ids"][:bs] = real_input_ids
