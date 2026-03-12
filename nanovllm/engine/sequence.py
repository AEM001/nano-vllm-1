from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Any, Iterator, List, Tuple, Union

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Enumeration of possible states for a sequence during processing.
    
    A sequence progresses through these states as it's processed by the engine:
    - WAITING: The sequence is queued and waiting to be processed
    - RUNNING: The sequence is currently being processed
    - FINISHED: The sequence has completed processing (either naturally or due to limits)
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """Represents a single sequence of tokens being processed by the LLM engine.
    
    A Sequence contains the token IDs, sampling parameters, and state information
    for a single request through the language model. It tracks both prompt and
    completion tokens, manages block allocation for memory efficiency, and provides
    methods for token manipulation and state queries.
    
    Attributes:
        seq_id: Unique identifier for this sequence instance
        status: Current processing status (WAITING, RUNNING, FINISHED)
        token_ids: Complete list of token IDs (prompt + completion)
        last_token: The most recently generated token ID
        num_tokens: Total number of tokens in the sequence
        num_prompt_tokens: Number of tokens in the original prompt
        num_cached_tokens: Number of tokens that are cached in memory
        block_table: List of memory block indices used by this sequence
        temperature: Sampling temperature parameter
        max_tokens: Maximum number of completion tokens allowed
        ignore_eos: Whether to ignore end-of-sequence tokens
    """
    block_size = 256  # Number of tokens per memory block
    counter = count()   # Global counter for generating unique sequence IDs

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()) -> None:
        """Initialize a new Sequence.
        
        Args:
            token_ids: List of token IDs representing the initial prompt
            sampling_params: Parameters controlling the sampling behavior
        """
        self.seq_id: int = next(Sequence.counter)
        self.status: SequenceStatus = SequenceStatus.WAITING
        self.token_ids: list[int] = copy(token_ids)
        self.last_token: int = token_ids[-1]
        self.num_tokens: int = len(self.token_ids)
        self.num_prompt_tokens: int = len(token_ids)
        self.num_cached_tokens: int = 0
        self.block_table: list[int] = []
        self.temperature: float = sampling_params.temperature
        self.max_tokens: int = sampling_params.max_tokens
        self.ignore_eos: bool = sampling_params.ignore_eos

    def __len__(self) -> int:
        """Return the total number of tokens in the sequence.
        
        Returns:
            Total number of tokens (prompt + completion)
        """
        return self.num_tokens

    def __getitem__(self, key) -> Union[int, list[int]]:
        """Get token(s) from the sequence by index or slice.
        
        Args:
            key: Integer index or slice object for token selection
            
        Returns:
            Single token ID if key is int, or list of token IDs if key is slice
        """
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        """Check if the sequence has finished processing.
        
        Returns:
            True if the sequence status is FINISHED, False otherwise
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """Get the number of completion tokens generated.
        
        Returns:
            Number of tokens generated beyond the original prompt
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        """Get the token IDs for the original prompt portion.
        
        Returns:
            List of token IDs comprising the original prompt
        """
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        """Get the token IDs for the generated completion portion.
        
        Returns:
            List of token IDs comprising the generated completion
        """
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        """Get the number of memory blocks that are cached.
        
        Returns:
            Number of complete memory blocks that are cached
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        """Get the total number of memory blocks required for this sequence.
        
        Returns:
            Total number of memory blocks needed to store all tokens
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """Get the number of tokens in the last (potentially partial) memory block.
        
        Returns:
            Number of tokens stored in the final memory block
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i) -> list[int]:
        """Get the tokens in a specific memory block.
        
        Args:
            i: Index of the memory block (0-based)
            
        Returns:
            List of token IDs stored in the specified block
            
        Raises:
            AssertionError: If block index is out of valid range
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int) -> None:
        """Append a new token to the sequence.
        
        This method is typically called during the generation process to add
        newly generated tokens to the sequence.
        
        Args:
            token_id: The token ID to append
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self) -> tuple[int, int, int, list[int], Union[list[int], int]]:
        """Prepare state for pickling.
        
        Optimizes serialization by only storing essential information.
        For sequences with no completion tokens, stores all token IDs.
        For sequences with completions, only stores the last token to save memory.
        
        Returns:
            Tuple containing (num_tokens, num_prompt_tokens, num_cached_tokens, 
                           block_table, token_data) where token_data is either
                           the full token list (for prompts only) or last token
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state: tuple[int, int, int, list[int], Union[list[int], int]]) -> None:
        """Restore state from pickled data.
        
        Reconstructs the sequence object from serialized state. Handles both
        prompt-only sequences (where all tokens are stored) and completion
        sequences (where only the last token is stored).
        
        Args:
            state: Tuple containing the serialized sequence state
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
