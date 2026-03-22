from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Any, Iterator, List, Tuple, Union

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    PREFILL_ING = auto()
    PREFILL_ED=auto()
    DECODE = auto()
    FINISHED = auto()


class Sequence:

    block_size = 256  # Number of tokens per memory block
    counter = count()   # Global counter for generating unique sequence IDs

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()) -> None:

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
        self.prefilled_tokens: int = 0

    def __len__(self) -> int:

        return self.num_tokens

    def __getitem__(self, key) -> Union[int, list[int]]:
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    @property
    def is_prefilling(self) -> bool:
        return self.prefilled_tokens < self.num_prompt_tokens
    
    @property
    def remaining_prefill_tokens(self) -> int:
        return self.num_prompt_tokens - self.prefilled_tokens

    def block(self, i) -> list[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


    def __getstate__(self) -> tuple[int, int, int, list[int], Union[list[int], int]]:
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state: tuple[int, int, int, list[int], Union[list[int], int]]) -> None:
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
