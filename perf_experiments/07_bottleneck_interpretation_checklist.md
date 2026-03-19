# 07 - Bottleneck Interpretation Checklist

## Goal

Use this checklist after each experiment to decide what actually limited the run.

## If startup fails or KV blocks are very low

### Likely bottleneck

- model weights + reserved KV cache exceed practical VRAM budget

### Most relevant file

- `nanovllm/engine/model_runner.py`

### Expected conclusion

- You are memory-bound before serving begins.
- Reduce `max_model_len`, `gpu_memory_utilization`, model size, or increase GPU capacity.

## If requests stop entering prefill early

### Likely bottleneck

- `max_num_batched_tokens` or `max_num_seqs`
- possibly `block_manager.can_allocate(seq)`

### Most relevant file

- `nanovllm/engine/scheduler.py`

### Expected conclusion

- Scheduler policy or KV capacity is capping concurrency before compute saturation.

## If prompt length hurts much more than generation length

### Likely bottleneck

- prefill compute and prefill tensor preparation

### Most relevant files

- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/llm_engine.py`

### Expected conclusion

- Your workload is prefill-bound.
- Optimize batching, prompt length, or prefill path.

## If long generations hurt more than long prompts

### Likely bottleneck

- decode-side KV growth and append pressure

### Most relevant files

- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/scheduler.py`

### Expected conclusion

- Your workload is decode-capacity-bound.
- Long tails are consuming cache and reducing concurrency.

## If throughput rises with concurrency and then flattens

### Likely bottleneck

- GPU saturation or memory bandwidth saturation

### Most relevant files

- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/llm_engine.py`

### Expected conclusion

- You found the practical throughput ceiling for this GPU and model.

## If throughput worsens sharply with mixed short and long requests

### Likely bottleneck

- scheduler policy and queue interaction

### Most relevant file

- `nanovllm/engine/scheduler.py`

### Expected conclusion

- Tail-latency behavior and fairness policy are limiting real serving quality.

## If crossing prompt lengths like 256, 512, 768 changes behavior suddenly

### Likely bottleneck

- KV block granularity effects

### Most relevant files

- `nanovllm/engine/sequence.py`
- `nanovllm/engine/block_manager.py`

### Expected conclusion

- Small workload changes are causing block-count jumps.

## If eager and CUDA graph modes are similar

### Likely bottleneck

- graph overhead is not the main issue
- compute or memory traffic may dominate

### Most relevant file

- `nanovllm/engine/model_runner.py`

### Expected conclusion

- Focus less on graph capture and more on batching, prompt shape, or memory use.

## Final summary template

After each experiment, fill this in:

- workload type:
- config values:
- prompt length:
- completion length:
- concurrency:
- prefill tok/s:
- decode tok/s:
- peak memory:
- free memory after KV allocation:
- first observed limiter:
- likely bottleneck file:
- conclusion:
