import pickle
import time
import logging
import torch
import torch.distributed as dist
from collections import deque

from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
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

        self.block_size = Sequence.block_size
        self.enforce_eager = config.enforce_eager
        self.is_warmup = False  # Flag to suppress logging during warmup
        self.world_size = config.tensor_parallel_size#Total number of processes in tensor parallel group

        self.rank = rank#this process's ID
        self.event = events#Synchronization object for coordinating with other ranks,Rank 0 gets list[Event] from all workers

        logger.info(f"[ModelRunner] Rank {rank} initialized")

        #Initialize pytorch distributed communication between process
        #set up NCCL for GPU-to-GPU communication
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        # Assign this process to a specific GPU
        torch.cuda.set_device(rank)
        logger.info(f"[ModelRunner] GPU device set to rank {rank}")
        
        
        # Setup model precision and device
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
      
        self.model = Qwen3ForCausalLM(hf_config)
        t0 = time.time()
        load_model(self.model, config.model)
        t1 = time.time()
        logger.info(f"[ModelRunner] Model loaded on rank {rank} in {t1 - t0:.2f}s")


        self.sampler = Sampler()
        
        # Performance optimizations
        logger.info(f"[ModelRunner] Allocating KV cache on rank {rank}...")
        # Skip warmup for now - TODO: fix warmup to work with continuous batching
        # self.warmup_model()
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
        
        # Set sequences to fully prefilled for warmup (they act like decode sequences)
        for seq in seqs:
            seq.num_cached_tokens = seq.num_prompt_tokens
        
        # Set warmup flag to suppress verbose logging
        self.is_warmup = True
        
        t0 = time.time()
        # Convert to deque for warmup
        from collections import deque
        self.run(deque(seqs))
        t1 = time.time()
        logger.info(f"[ModelRunner] Warmup completed in {t1 - t0:.2f}s")
        
        # Clear warmup flag
        self.is_warmup = False
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        
        config = self.config
        hf_config = config.hf_config
        
        # Get GPU memory information
        free, total = torch.cuda.mem_get_info()
        logger.info(f"[ModelRunner] GPU memory: {free/1024**3:.2f}GB free / {total/1024**3:.2f}GB total")
     
        used = total - free 
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        logger.info(f"[ModelRunner] Model weights: {peak/1024**3:.2f}GB")
        
        # Calculate KV cache parameters
        num_kv_heads = hf_config.num_key_value_heads // self.world_size

        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # Calculate maximum number of blocks we can allocate
        # Available memory = total * utilization - peak (model weights)
        # Don't subtract 'used' since it already includes peak
        available_memory = int(total * config.gpu_memory_utilization - peak)
        config.num_kvcache_blocks = available_memory // block_bytes

        logger.info(f"[ModelRunner] KV cache memory: {available_memory/1024**3:.2f}GB available, block_size={block_bytes} bytes")
        assert config.num_kvcache_blocks > 0
        logger.info(f"[ModelRunner] KV cache blocks: {config.num_kvcache_blocks}")


        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        logger.info(f"[ModelRunner] KV cache allocated: {self.kv_cache.shape}")

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

    def prepare_block_tables(self, seqs: list[(Sequence,int)]):
        
        max_len = max(len(seq.block_table) for seq,_ in seqs)#find maximum block table length

        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq,_ in seqs]#pad all tables to equal length with -1

        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)#convert to tensor and move to GPU

        return block_tables

    def prepare(self, seqs: list[(Sequence,int)]):
        input_ids = []
        positions = []
        slot_mapping = []
        mask=[]
        seq_mask = []
        context_lens = []
        
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        

        for seq,num in seqs:

            if seq.num_cached_tokens < seq.num_prompt_tokens:
                # Handle prefill sequences: process chunked tokens
                start_idx = seq.num_cached_tokens
                end_idx = start_idx + num
                input_ids.extend(seq[start_idx:end_idx])
                positions.extend(list(range(start_idx, end_idx)))
                
                seqlen_q = end_idx - start_idx
                seqlen_k = end_idx
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
                max_seqlen_q = max(max_seqlen_q, seqlen_q)
                max_seqlen_k = max(max_seqlen_k, seqlen_k)
                mask.append([-1]*seqlen_q)
                seq_mask.append(-1)
                context_lens.append(end_idx)  # Always set context_lens for mixed path
                
                # Build slot mapping for NEW tokens only
                if seq.block_table:
                    for token_pos in range(start_idx, end_idx):
                        block_idx = token_pos // self.block_size
                        block_offset = token_pos % self.block_size
                        if block_idx < len(seq.block_table):
                            physical_block = seq.block_table[block_idx]
                            slot = physical_block * self.block_size + block_offset
                            slot_mapping.append(slot)
                        
            elif seq.num_cached_tokens >= seq.num_prompt_tokens:
                # Handle decode sequences: process exactly 1 token
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                seqlen_q = 1
                seqlen_k = len(seq)
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
                max_seqlen_q = max(max_seqlen_q, seqlen_q)
                max_seqlen_k = max(max_seqlen_k, seqlen_k)
                mask.append([0])
                seq_mask.append(0)
                context_lens.append(len(seq))
                
                # Only add slot mapping if block_table exists
                if seq.block_table:
                    slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        
        # Always prepare block tables for mixed path
        block_tables = self.prepare_block_tables(seqs)
        
        # Convert to GPU tensors
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        query_mask = torch.tensor([value for group in mask for value in group], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        seq_mask = torch.tensor(seq_mask, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Always use mixed batch context format
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, query_mask, seq_mask)
        # logger.info(f"mask: {mask}")
        return input_ids, positions, mask

    
    def prepare_sample(self, seqs: list[Sequence]):
     
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, mask:torch.Tensor):
        context = get_context()

        if context.is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Eager execution for prefill, forced eager, or large batches
            return self.model.compute_logits(self.model(input_ids, positions,mask))
        else:
            # CUDA graph execution for optimized decode
            bs = input_ids.size(0)#get batch size

            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # Update graph input variables
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"].fill_(-1)
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # Replay captured graph
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, scheduled_seqs: deque[(Sequence,int)]) -> list[int]:
        # create a list of token_ids for the whole scheduled sequences
        token_ids = [0] * len(scheduled_seqs) if self.rank == 0 else None
        input_ids, positions, mask = self.prepare(scheduled_seqs)
        
        # Count prefill vs decode tokens from mask
        flat_mask = [m for seq_mask in mask for m in seq_mask]
        num_prefill_tokens = sum(1 for m in flat_mask if m == -1)
        num_decode_tokens = sum(1 for m in flat_mask if m == 0)
        
        # logger.info(f"the number of batched tokens: {input_ids.size(0)}")
        # logger.info(f"prefill tokens: {num_prefill_tokens}, decode tokens: {num_decode_tokens}")
        logits=self.run_model(input_ids, positions, mask)
        if self.rank == 0:
            sample_indices = [
                i for i, (seq, _) in enumerate(scheduled_seqs)
                if seq.num_cached_tokens >= seq.num_prompt_tokens
            ]#filter the seqs that need to be sampled
            if sample_indices:
                temperatures = self.prepare_sample([scheduled_seqs[i][0] for i in sample_indices])
                sampled_token_ids = self.sampler(logits, temperatures).tolist()

                for batch_idx, token_id in zip(sample_indices, sampled_token_ids):
                    token_ids[batch_idx] = token_id
                #since the sampl_indices are only the index, while the sampled_token_ids are the real output, use the index and value to update the original token_ids 
        reset_context()

        return token_ids if token_ids is not None else []

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
