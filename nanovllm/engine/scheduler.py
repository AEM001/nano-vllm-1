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

    def schedule(self) -> tuple[list[Sequence], bool]:
        """       
        Scheduling Logic:
        - Prioritize prefill to get new sequences started
        - Fall back to decode if no prefill candidates
        - Preempt sequences if memory constraints require it
        """
        logger.debug("[Scheduler] Scheduling sequences")
        
        # THIS BATCH: sequences to execute RIGHT NOW
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # === PHASE 1: PREFILL ===
        # Process NEW sequences from waiting queue
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # Next new sequence
            
            # Check if we have resources for this sequence
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                logger.debug(f"[Scheduler] Cannot allocate seq={seq.seq_id} (tokens={num_batched_tokens}+{len(seq)})")
                break  # No space, stop prefill
            
            # Move: Waiting -> Running -> THIS BATCH
            num_seqs += 1
            logger.debug(f"[Scheduler] Allocating seq={seq.seq_id}")
                
            self.block_manager.allocate(seq)  # Reserve KV cache
            
            logger.info(f"[Scheduler] Prefill allocated seq={seq.seq_id} blocks={len(seq.block_table)} tokens={len(seq)}")
            
            # Update batched tokens count (NEW tokens only, not cached ones)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            logger.debug(f"[Scheduler] Prefill batch_tokens={num_batched_tokens} (added {len(seq) - seq.num_cached_tokens})")
                
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()     # Remove from waiting queue
            self.running.append(seq)    # Add to running queue
            scheduled_seqs.append(seq)  # Add to THIS BATCH for execution
        
        if scheduled_seqs:# prefill until there is no space for more sequences
            return scheduled_seqs, True  # PREFILL BATCH
      
        # === PHASE 2: DECODE ===
        # Process EXISTING sequences from running queue
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()#get next running sequence
            
            # Handle memory pressure: need to free up space
            while not self.block_manager.can_append(seq):
                if self.running:
                    # Preempt another running sequence to free memory
                    self.preempt(self.running.pop())
                else:
                    # No other sequences to preempt, preempt this one
                    self.preempt(seq)
                    break
            else:
                # Space available, add to THIS BATCH
                num_seqs += 1
                logger.debug(f"[Scheduler] Appending to seq={seq.seq_id}")
                    
                self.block_manager.may_append(seq)
                
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs  # Should always have at least one sequence
        # Restore running queue order (reversed to maintain fairness)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # DECODE BATCH

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
            
            #eos matters and current generated token is eos or max tokens reached
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
