from collections import deque
import logging

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# Configure logger
logger = logging.getLogger(__name__)


class Scheduler:
    
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.chunk_size = config.chunk_size
        
        # KV cache memory management
        logger.debug("[Scheduler] Initializing block manager...")
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # Sequence queues
        self.waiting: deque[Sequence] = deque()  # Sequences waiting to be processed
        self.running: deque[Sequence] = deque()  # Currently executing sequences

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        logger.debug(f"[Scheduler] Adding seq={seq.seq_id} to waiting queue")
        self.waiting.append(seq)

    def schedule(self) -> deque[Sequence]:
    
        logger.debug("[Scheduler] Scheduling sequences")
        
        # THIS BATCH: sequences to execute RIGHT NOW
        scheduled_seqs = deque()
        num_batched_tokens = 0
        
        # Step 1: Process existing running sequences (decode phase)
        for seq in self.running:
            if seq.status == SequenceStatus.DECODE:
                # Check if we can append token to this sequence
                other_running = [s for s in self.running if s != seq and s.status == SequenceStatus.DECODE]
                
                while not self.block_manager.can_append(seq):
                    if other_running:
                        self.preempt(other_running.pop())
                    else:
                        self.preempt(seq)
                        break
                
                if seq.status == SequenceStatus.DECODE:  # Check if not preempted
                    self.block_manager.may_append(seq)
                    num_batched_tokens += 1
                    scheduled_seqs.append(seq)
        
        # Step 2: Process waiting sequences (prefill phase)
        for seq in self.waiting:
            # Calculate how many tokens to prefill this chunk
            remaining_tokens = len(seq) - seq.prefilled_tokens
            len_to_prefill = min(self.chunk_size, remaining_tokens)
            
            if num_batched_tokens + len_to_prefill > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
                
            self.block_manager.allocate(seq, len_to_prefill)
            num_batched_tokens += len_to_prefill
            
            # Move sequence from waiting to running
            self.waiting.remove(seq)
            self.running.append(seq)
            
            if seq.prefilled_tokens < len(seq):
                seq.status = SequenceStatus.PREFILL_ING
            else:
                seq.status = SequenceStatus.DECODE
            
            scheduled_seqs.append(seq)
            
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        # Kick out sequence to free memory: Running -> Waiting (high priority)
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)  # Free KV cache memory
        self.waiting.appendleft(seq)  # Front of waiting queue (priority)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        
        for seq, token_id in zip(seqs, token_ids):
            # Append generated token to sequence
            logger.debug(f"[Scheduler] Appending token {token_id} to seq={seq.seq_id}")
                
            seq.append_token(token_id)
            
            # Update sequence status from PREFILL to DECODE after first token generation
            if seq.status == SequenceStatus.PREFILL:
                seq.status = SequenceStatus.DECODE
            
            #eos matters and current generated token is eos or max tokens reached
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
