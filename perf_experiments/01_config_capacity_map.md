# 01 - Config Capacity Map

## Primary file

- `nanovllm/config.py`

## Why this file matters

This file defines the top-level capacity knobs that shape nearly every later bottleneck.

## Exact objective

Determine which limits are imposed by configuration before the scheduler or GPU runtime even gets a chance to scale.

## Parameters to sweep

- `max_num_batched_tokens`
  - try: `4096`, `8192`, `16384`, `32768`
- `max_num_seqs`
  - try: `1`, `2`, `4`, `8`, `16`, `32`, `64`
- `max_model_len`
  - try: `512`, `1024`, `2048`, `4096`
- `gpu_memory_utilization`
  - try: `0.7`, `0.8`, `0.9`, `0.95`
- `tensor_parallel_size`
  - try only values supported by your hardware
- `enforce_eager`
  - compare `True` and `False`

## Experiment design

### Experiment A: token-budget ceiling

- Hold prompt count fixed.
- Increase `max_num_batched_tokens`.
- Keep `max_num_seqs` high enough that it does not become the first limiter.

### Expected outcome

- Throughput should improve until another bottleneck takes over.
- Once no more gain appears, the next limit is likely compute, memory bandwidth, or scheduler behavior.

### Experiment B: concurrency ceiling

- Hold prompt length fixed.
- Increase number of concurrent requests.
- Sweep `max_num_seqs` upward.

### Expected outcome

- You should see a point where more concurrency no longer helps.
- If requests stop entering prefill, `max_num_seqs` or KV capacity is likely the cap.

### Experiment C: context-length stress

- Run short, medium, and long prompts.
- Sweep `max_model_len`.

### Expected outcome

- Larger context windows reduce effective concurrency because each sequence consumes more KV cache.
- Prefill cost grows strongly with longer prompts.

### Experiment D: memory reservation policy

- Sweep `gpu_memory_utilization`.

### Expected outcome

- Higher values should increase `num_kvcache_blocks`.
- This should allow more long-running decode requests before preemption or allocation failure.
- If set too aggressively, you may become unstable or leave too little safety margin.

## What to log

- config values for the run
- whether startup succeeds
- `Number of KV cache blocks`
- throughput
- OOM or assertion failures

## What this file can prove

- Whether your current bottleneck is self-imposed by config.
- Whether your GPU is underused because limits are too conservative.
- Whether larger context or larger concurrency is the dominant capacity killer.
