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

    def schedule(self) -> deque[(Sequence,int)]:
    
        logger.debug(f"[Scheduler] Scheduling: waiting={len(self.waiting)}, running={len(self.running)}")
        
        # THIS BATCH: sequences to execute RIGHT NOW
        scheduled_seqs = deque()
        num_batched_tokens = 0
        
        # Step 1: Process existing running sequences (both prefilling and decoding)
        running_list = list(self.running)  # Convert to list to avoid modification during iteration
        for seq in running_list:
            if seq.status == SequenceStatus.DECODE:
                # Decode phase: need 1 token
                if num_batched_tokens + 1 > self.max_num_batched_tokens:
                    continue
                    
                # Check if we can append token to this sequence
                other_running = [s for s in running_list if s != seq and s.status == SequenceStatus.DECODE]
                
                while not self.block_manager.can_append(seq):
                    if other_running:
                        self.preempt(other_running.pop())
                    else:
                        self.preempt(seq)
                        break
                
                if seq.status == SequenceStatus.DECODE:  # Check if not preempted
                    self.block_manager.may_append(seq)
                    num_batched_tokens += 1
                    scheduled_seqs.append((seq, 1))
                    
            elif seq.status == SequenceStatus.PREFILL_ED:
                # Prefill complete, ready for first decode
                if num_batched_tokens + 1 > self.max_num_batched_tokens:
                    continue
                
                # Check if we can append token
                while not self.block_manager.can_append(seq):
                    other_running = [s for s in running_list if s != seq]
                    if other_running:
                        self.preempt(other_running.pop())
                    else:
                        self.preempt(seq)
                        break
                
                if seq.status == SequenceStatus.PREFILL_ED:  # Check if not preempted
                    self.block_manager.may_append(seq)
                    num_batched_tokens += 1
                    scheduled_seqs.append((seq, 1))
                    
            elif seq.status == SequenceStatus.PREFILL_ING:
                # Continue prefilling
                remaining_tokens = seq.remaining_prefill_tokens
                remaining_budget = self.max_num_batched_tokens - num_batched_tokens
                len_to_prefill = min(self.chunk_size, remaining_tokens, remaining_budget)
                logger.warning(f"PARTIALLY PREFILLING: seq_id{seq.seq_id} is prefilled {len_to_prefill} tokens")
                if len_to_prefill <= 0:
                    continue
                    
                num_batched_tokens += len_to_prefill
                scheduled_seqs.append((seq, len_to_prefill))
        
        # Step 2: Process waiting sequences (start prefill phase)
        waiting_list = list(self.waiting)  # Convert to list
        for seq in waiting_list:
            # Calculate how many tokens to prefill this chunk
            remaining_tokens = len(seq) - seq.prefilled_tokens
            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            len_to_prefill = min(self.chunk_size, remaining_tokens, remaining_budget)
            logger.warning(f" !!! FRESH START: seq_id{seq.seq_id} is prefilling {len_to_prefill} tokens and is gonna be allocated")
            if len_to_prefill <= 0:
                break
                
            if not self.block_manager.can_allocate(seq):
                break
                
            self.block_manager.allocate(seq)
            num_batched_tokens += len_to_prefill
            
            # Move sequence from waiting to running
            self.waiting.remove(seq)
            self.running.append(seq)
            
            seq.status = SequenceStatus.PREFILL_ING
            scheduled_seqs.append((seq, len_to_prefill))
        
        logger.debug(f"[Scheduler] Scheduled {len(scheduled_seqs)} sequences, {num_batched_tokens} tokens")
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        # Kick out sequence to free memory: Running -> Waiting (high priority)
        logger.info(f"[Scheduler] Preempting seq={seq.seq_id}, prefilled={seq.prefilled_tokens}/{seq.num_prompt_tokens}")
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)  # Free KV cache memory
        # Reset prefill progress since we're deallocating blocks
        seq.prefilled_tokens = 0
        self.waiting.appendleft(seq)  # Front of waiting queue (priority)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        
        for (seq, _), token_id in zip(seqs, token_ids):
            if seq.status == SequenceStatus.PREFILL_ING:
                # Update prefill progress
                chunk_processed = min(self.chunk_size, seq.remaining_prefill_tokens)
                seq.prefilled_tokens += chunk_processed
                
                # Check if prefill is complete
                if seq.prefilled_tokens >= seq.num_prompt_tokens:
                    seq.status = SequenceStatus.PREFILL_ED
                    logger.debug(f"[Scheduler] Seq {seq.seq_id} prefill complete")
                else:
                    logger.debug(f"[Scheduler] Seq {seq.seq_id} prefill progress: {seq.prefilled_tokens}/{seq.num_prompt_tokens}")
            
            elif seq.status == SequenceStatus.PREFILL_ED:
                # First decode step after prefill
                seq.status = SequenceStatus.DECODE
                seq.append_token(token_id)
                logger.debug(f"[Scheduler] Seq {seq.seq_id} started decoding, token={token_id}")
                
            elif seq.status == SequenceStatus.DECODE:
                # Continue decoding
                seq.append_token(token_id)
                logger.debug(f"[Scheduler] Appending token {token_id} to seq={seq.seq_id}")
            
            # Check if sequence is finished (only for decode phase)
            if seq.status == SequenceStatus.DECODE:
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    logger.info(f"[Scheduler] Seq {seq.seq_id} finished")
