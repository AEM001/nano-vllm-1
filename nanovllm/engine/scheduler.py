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
        self.long_prefill_threshold=config.long_prefill_threshold
        self.max_num_partial_prefills = config.max_num_partial_prefills
        self.max_long_partial_prefills = config.max_long_partial_prefills
        # KV cache memory management
        logger.debug("[Scheduler] Initializing block manager...")
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # Sequence queues
        self.waiting: deque[Sequence] = deque()  # Sequences waiting to be processed
        self.running: deque[Sequence] = deque()  # Currently executing sequences
        self.finished: deque[Sequence] = deque()  # Completed sequences

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        logger.debug(f"[Scheduler] Adding seq={seq.seq_id} to waiting queue")
        self.waiting.append(seq)

    def schedule(self) -> deque[(Sequence,int)]:
    
        logger.debug(f"[Scheduler] Scheduling: waiting={len(self.waiting)}, running={len(self.running)}")
        
        # THIS BATCH: sequences to execute RIGHT NOW
        scheduled_seqs = deque()
        num_batched_tokens = 0
        num_seqs=0
        num_partial_prefills=0
        num_long_partial_prefills=0
        # Step 1: Process existing running sequences (both prefilling and decoding)
        running_list = list(self.running)  # Convert to list to avoid modification during iteration
        for seq in [s for s in self.running if s.num_cached_tokens>=s.num_prompt_tokens]:
            if num_seqs>=self.max_num_seqs:
                break
           

            # Line 52-53: Instead of random pop(), sort by progress:
            other_running = sorted(
                [s for s in running_list if s != seq and s.num_cached_tokens >= s.num_prompt_tokens],
                key=lambda s: s.num_completion_tokens  # Preempt those with least progress
            )    
            
            while not (self.max_num_batched_tokens - num_batched_tokens >= 1):
                if other_running:
                    self.preempt(other_running.pop())
                    logger.warning(f"[Scheduler] Preempted seq={seq.seq_id} due to budget")
                else:
                    self.preempt(seq)
                    break
            
            scheduled_seqs.append((seq, 1))
            num_batched_tokens += 1
            num_seqs += 1
       
        for seq in [s for s in self.running if s.num_cached_tokens<s.num_prompt_tokens]:
            # Prefill complete, ready for first decode
            if num_seqs >= self.max_num_seqs or num_batched_tokens == self.max_num_batched_tokens:
                break
            remaining_tokens = seq.num_prompt_tokens - seq.num_cached_tokens
            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            len_to_prefill = min(self.chunk_size, remaining_tokens, remaining_budget)
            logger.warning(f"PARTIALLY PREFILLING: seq_id{seq.seq_id} is prefilled {len_to_prefill} tokens")
            
            # Allocate blocks for this partial prefill chunk
            if not self.block_manager.can_allocate(seq, len_to_prefill):
                break
            self.block_manager.allocate(seq, len_to_prefill)
            
            # Check num_seqs before adding
            num_batched_tokens += len_to_prefill
            scheduled_seqs.append((seq, len_to_prefill))
            num_seqs += 1  # Increment when actually scheduled
            if len_to_prefill+seq.num_cached_tokens == seq.num_prompt_tokens:
                if seq.num_cached_tokens!=0:
                    num_partial_prefills-=1
                    if seq.num_prompt_tokens >= self.long_prefill_threshold:
                        num_long_partial_prefills-=1
                    
        # Step 2: Process waiting sequences (start prefill phase)
        waiting_list = list(self.waiting)  # Convert to list
        for seq in waiting_list:
            if num_seqs >= self.max_num_seqs:  # Check num_seqs at start of each iteration
                break
            # Calculate how many tokens to prefill this chunk
            remaining_tokens = len(seq) - seq.num_cached_tokens
            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            if remaining_budget < remaining_tokens:
                if num_partial_prefills >= self.max_num_partial_prefills:
                    continue
                if seq.num_prompt_tokens >= self.long_prefill_threshold:
                    if num_long_partial_prefills >= self.max_long_partial_prefills:
                        continue
                    num_long_partial_prefills += 1
                num_partial_prefills += 1
            
            len_to_prefill = min(self.chunk_size, remaining_tokens, remaining_budget)
            logger.warning(f" !!! Prefill !!!: seq_id{seq.seq_id} is prefilling {len_to_prefill} tokens and is gonna be allocated")
            
            if not self.block_manager.can_allocate(seq,len_to_prefill):#!!!!!!!!!!!!!!!fix this
                break
                
            self.block_manager.allocate(seq,len_to_prefill)
            num_batched_tokens += len_to_prefill
            
            # Move sequence from waiting to running
            self.waiting.remove(seq)
            self.running.append(seq)
            
            seq.status = SequenceStatus.RUNNING
            scheduled_seqs.append((seq, len_to_prefill))
            num_seqs += 1  # Increment when actually scheduled
        
        logger.debug(f"[Scheduler] Scheduled {len(scheduled_seqs)} sequences, {num_batched_tokens} tokens")
        logger.info(f"number of running and waiting seqs: {len(self.running)} and {len(self.waiting)}")
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        # Kick out sequence to free memory: Running -> Waiting (high priority)
        logger.info(f"[Scheduler] Preempting seq={seq.seq_id}, cached={seq.num_cached_tokens}/{seq.num_prompt_tokens}")
        seq.status = SequenceStatus.WAITING
        self.running.remove(seq)
        self.waiting.appendleft(seq)
        self.block_manager.deallocate(seq)  # Free KV cache memory
        # Reset prefill progress since we're deallocating blocks
        # seq.prefilled_tokens = 0  # No longer needed

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        
        for (seq, scheduled_len), token_id in zip(seqs, token_ids):
            if seq.status == SequenceStatus.RUNNING:
                # Update cached tokens after processing (for prefill phase)
                if seq.num_cached_tokens < seq.num_prompt_tokens:
                    seq.num_cached_tokens += scheduled_len
                
                # Check if prefill is complete
                if seq.num_cached_tokens >= seq.num_prompt_tokens:
                    # seq.status remains RUNNING (fully prefilled)
                    logger.debug(f"[Scheduler] Seq {seq.seq_id} prefill complete")
                else:
                    logger.debug(f"[Scheduler] Seq {seq.seq_id} prefill progress: {seq.num_cached_tokens}/{seq.num_prompt_tokens}")
            
            # For fully prefilled sequences (decode phase)
            if token_id != 0:  # Real token, not placeholder
                seq.append_token(token_id)
                logger.debug(f"[Scheduler] Appending token {token_id} to seq={seq.seq_id}")
            
            # Check if sequence is finished (only for decode phase)
            if seq.num_cached_tokens >= seq.num_prompt_tokens:
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    self.finished.append(seq)  # Track finished sequences
                    logger.info(f"[Scheduler] Seq {seq.seq_id} finished")
