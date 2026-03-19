# nano-vllm Performance Experiment Guide

This directory is a practical experiment plan for finding the real capacity limits and bottlenecks of your GPU and this project.

## Files to study first

- `nanovllm/config.py`
- `nanovllm/engine/llm_engine.py`
- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/sequence.py`
- `nanovllm/sampling_params.py`

## What each file controls

- `config.py`
  - hard limits and global knobs
  - batching limits
  - max sequence length
  - tensor parallel size
  - GPU memory utilization target
  - KV cache block size

- `llm_engine.py`
  - end-to-end request lifecycle
  - step loop and throughput accounting
  - interaction between scheduler and model runner

- `scheduler.py`
  - how many requests can run
  - whether a request is prefilling or decoding
  - when preemption happens
  - when batching stops because of token or memory pressure

- `block_manager.py`
  - KV cache block allocation and reuse
  - free block exhaustion
  - cache reuse effectiveness

- `model_runner.py`
  - actual GPU-heavy work
  - warmup cost
  - model load cost
  - KV cache sizing formula
  - decode vs prefill execution path
  - CUDA graph capture and replay behavior

- `sequence.py`
  - how sequence length becomes block count and cache demand

- `sampling_params.py`
  - max generation length and EOS handling

## Recommended experiment order

1. `01_config_capacity_map.md`
2. `02_model_runner_gpu_capacity.md`
3. `03_scheduler_batching_limits.md`
4. `04_block_manager_kv_cache.md`
5. `05_llm_engine_throughput.md`
6. `06_sequence_and_sampling_load_shape.md`
7. `07_bottleneck_interpretation_checklist.md`

## What to record for every run

- model name
- GPU name and VRAM size
- `tensor_parallel_size`
- `max_model_len`
- `max_num_batched_tokens`
- `max_num_seqs`
- `gpu_memory_utilization`
- `kvcache_block_size`
- prompt length
- number of concurrent prompts
- completion length
- whether run finished or failed
- peak memory
- free memory after KV cache allocation
- prefill tok/s
- decode tok/s
- whether preemption occurred
- whether CUDA graphs were enabled

## Main hypotheses to test

- GPU capacity is often limited by KV cache, not only by model weights.
- Prefill bottlenecks and decode bottlenecks are different.
- `max_num_batched_tokens` can cap throughput before VRAM is fully used.
- `max_num_seqs` can cap concurrency before token budget is fully used.
- `gpu_memory_utilization` affects how many KV blocks are created and therefore how many long requests can run.
- CUDA graph mode mainly helps decode, not prefill.
- Long prompts stress prefill and KV allocation differently from long generations.
