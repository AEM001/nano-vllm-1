# config
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

# key observations
- chunked prefill, 
`WARNING [nanovllm.engine.scheduler]  !!! Prefill !!!: seq_id2 is prefilling 12 tokens and is gonna be allocated`
`WARNING [nanovllm.engine.scheduler] PARTIALLY PREFILLING: seq_id2 is prefilled 10 tokens(second time)`
- continuous batching
`INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 4 and 2`
`INFO [nanovllm.engine.scheduler] [Scheduler] Seq 3 finished`
`WARNING [nanovllm.engine.scheduler]  !!! Prefill !!!: seq_id4 is prefilling 7 tokens and is gonna be allocated`
- longger TTFT for longger sequences
`TTFT: 2.515s`
`TTFT: 2.630s`
