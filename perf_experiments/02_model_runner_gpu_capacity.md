# 02 - Model Runner GPU Capacity

## Primary file

- `nanovllm/engine/model_runner.py`

## Why this file matters

This is the main GPU execution file. It controls model load, warmup, KV cache sizing, prefill preparation, decode preparation, eager execution, and CUDA graph execution.

## Exact objective

Find the true GPU-side bottleneck by separating:

- model weight residency
- warmup cost
- KV cache reservation
- prefill compute cost
- decode compute cost
- CUDA graph benefit

## High-impact functions

- `__init__`
- `warmup_model`
- `allocate_kv_cache`
- `prepare_prefill`
- `prepare_decode`
- `run_model`
- `run`
- `capture_cudagraph`

## Experiment A: startup memory budget

### Procedure

- Run startup only.
- Record:
  - `Free memory`
  - `Peak memory`
  - `Current memory`
  - `Number of KV cache blocks`
  - `Free memory after KV cache allocation`
- Repeat with different `gpu_memory_utilization` values.

### Expected outcome

- The available KV cache should scale mostly with `gpu_memory_utilization`.
- The gap between free memory before and after allocation shows how much VRAM is reserved for cache.
- If `Number of KV cache blocks` changes sharply with small config changes, your setup is near a hard memory boundary.

## Experiment B: prefill scaling

### Procedure

- Use one request at a time.
- Sweep prompt lengths: `128`, `256`, `512`, `1024`, `2048`, `4096`.
- Measure time for the first step separately from later decode steps.

### Expected outcome

- Prefill latency should grow strongly with prompt length.
- If decode speed stays similar while prefill slows down, prefill is its own bottleneck class.

## Experiment C: decode scaling

### Procedure

- Use fixed prompt length.
- Increase concurrent sequences.
- Compare decode throughput with `enforce_eager=True` vs `False`.

### Expected outcome

- Decode throughput should improve with batching up to a point.
- CUDA graph mode should help decode more than prefill.
- If eager and graph mode are similar, your bottleneck may be memory traffic or small batch sizes.

## Experiment D: graph capture coverage

### Procedure

- Disable eager mode.
- Test batch sizes around graph buckets: `1`, `2`, `4`, `8`, `16`, `32`, `48`, `64`.

### Expected outcome
n
- Batch sizes close to captured graph buckets should behave better.
- Odd sizes may map to a larger captured graph and waste some capacity.

## Experiment E: CPU-to-GPU prep cost

### Procedure

- Compare short and long prompts while logging time around:
  - `prepare_prefill`
  - `prepare_decode`
  - `run_model`
- Focus on whether tensor preparation or model compute dominates.

### Expected outcome

- For small batches, Python and tensor marshaling may dominate.
- For large prompts, GPU compute should dominate.

## What this file can prove

- Whether you are compute-bound or memory-bound.
- Whether CUDA graphs matter on your workload.
- Whether startup KV reservation is the main limiter.
- Whether prefill or decode is the real performance bottleneck.
