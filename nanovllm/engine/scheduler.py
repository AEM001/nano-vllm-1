"""
Scheduler - Core batching and memory management component for nano-vllm.

This module implements the scheduling algorithm that maximizes throughput by:
- Batching multiple sequences together for efficient GPU utilization
- Managing KV cache memory allocation and deallocation
- Handling prefill (prompt processing) vs decode (token generation) phases
- Implementing preemptive scheduling when resources are constrained

Key Concepts:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Waiting Queue │───▶│   Running Batch  │───▶│   Finished      │
│                 │    │                  │    │                 │
│ - New requests  │    │ - Active seqs    │    │ - Completed     │
│ - Preempted     │    │ - KV allocated   │    │ - Memory freed  │
│ - Pending       │    │ - Being processed│    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘

Scheduling Strategy:
1. **Prefill Phase**: Process new prompts, allocate KV cache blocks
2. **Decode Phase**: Generate tokens incrementally, append to KV cache
3. **Preemption**: Free memory by moving sequences back to waiting queue
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    Batching scheduler that manages sequence execution and KV cache allocation.
    
    The scheduler implements a two-phase execution model:
    1. **Prefill**: Process new prompts by allocating KV cache and computing initial states
    2. **Decode**: Generate tokens incrementally using cached key/value states
    
    Core Responsibilities:
    - **Batch Formation**: Group sequences for efficient GPU utilization
    - **Memory Management**: Allocate/deallocate KV cache blocks via BlockManager
    - **Preemption**: Free memory by preempting running sequences when needed
    - **Fairness**: Ensure all sequences get processing time
    
    Data Structures:
    - waiting: Queue of sequences waiting to be processed
    - running: Queue of sequences currently in execution batch
    """

    def __init__(self, config: Config):
        """
        Initialize scheduler with configuration parameters.
        
        Args:
            config: Configuration containing batch sizes and memory limits
            
        Key Parameters:
        - max_num_seqs: Maximum sequences that can run simultaneously
        - max_num_batched_tokens: Maximum tokens in a single batch
        - eos: End-of-sequence token ID for completion detection
        """
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        
        # KV cache memory management
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # Sequence queues
        self.waiting: deque[Sequence] = deque()  # Sequences waiting to be processed
        self.running: deque[Sequence] = deque()  # Currently executing sequences

    def is_finished(self):
        """
        Check if all sequences have completed processing.
        
        Returns:
            bool: True if no sequences are waiting or running
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        Add a new sequence to the waiting queue.
        
        Args:
            seq: Sequence to be scheduled for processing
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        Core scheduling algorithm that forms execution batches.
        
        This method implements the two-phase scheduling strategy:
        1. **Prefill Phase**: Process waiting sequences, allocate KV cache
        2. **Decode Phase**: Process running sequences, generate new tokens
        
        Returns:
            tuple: (scheduled_sequences, is_prefill)
                - scheduled_sequences: List of sequences for this bitch
                - is_prefill: True for prefill phase, False for decode phase
        
        Scheduling Logic:
        - Prioritize prefill to get new sequences started
        - Fall back to decode if no prefill candidates
        - Preempt sequences if memory constraints require it
        """
        # ========== PREFILL PHASE ==========
        # Process new sequences from waiting queue
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # Peek at next waiting sequence
            
            # Check resource constraints
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break  # Cannot fit this sequence in current batch
            
            # Allocate resources and move to running queue
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            return scheduled_seqs, True  # Prefill batch

        # ========== DECODE PHASE ==========
        # Process existing running sequences for token generation
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # Ensure KV cache space for new token
            while not self.block_manager.can_append(seq):
                if self.running:
                    # Preempt another running sequence to free memory
                    self.preempt(self.running.pop())
                else:
                    # No other sequences to preempt, preempt this one
                    self.preempt(seq)
                    break
            else:
                # Space available, add to batch
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        assert scheduled_seqs  # Should always have at least one sequence
        # Restore running queue order (reversed to maintain fairness)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # Decode batch

    def preempt(self, seq: Sequence):
        """
        Preempt a running sequence to free memory resources.
        
        Preemption is the key mechanism for handling memory pressure:
        - Deallocates all KV cache blocks for the sequence
        - Moves sequence back to front of waiting queue (priority)
        - Sequence state is preserved, only memory is freed
        
        Args:
            seq: Running sequence to preempt
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)  # High priority for preempted sequences

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        Process generated tokens and handle sequence completion.
        
        This method is called after ModelRunner executes a batch:
        - Appends generated tokens to sequences
        - Checks for completion conditions (EOS, max tokens)
        - Frees memory for completed sequences
        - Updates sequence states
        
        Args:
            seqs: List of sequences that were processed
            token_ids: Generated tokens for each sequence
            
        Completion Conditions:
        - Generated EOS token (unless ignored)
        - Reached maximum token limit
        """
        for seq, token_id in zip(seqs, token_ids):
            # Append generated token to sequence
            seq.append_token(token_id)
            
            # Check completion conditions
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # Mark as finished and free resources
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
