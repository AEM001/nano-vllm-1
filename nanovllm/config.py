import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 60
    max_num_seqs: int = 4
    max_model_len: int = 60
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 32
    num_kvcache_blocks: int = -1
    chunk_size: int = 12
    max_num_partial_prefills: int = 3
    max_long_partial_prefills: int = 1
    long_prefill_threshold: int = 16


    def __post_init__(self):
        assert os.path.isdir(self.model)
        # assert self.kvcache_block_size % 64 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.chunk_size > 0
