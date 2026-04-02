
The issue is clear from the logs: `seq_id12` keeps getting "PARTIALLY PREFILLING 256 tokens" infinitely — meaning `num_cached_tokens` never advances. Let me trace the code path.
Now let me check how `block_manager.allocate` and `can_allocate` work — if blocks keep getting allocated but `num_cached_tokens` doesn't advance, it could exhaust blocks or loop:

Let me check the model runner to understand the full lifecycle:



Let me first check what prompt files exist and their content, then trace the bug:



The [1024_prompts.json](cci:7://file:///home/albert/learn/nano-vllm-src/1024_prompts.json:0:0-0:0) contains very long text. Let me check the actual token counts to understand the issue:

Token counts look reasonable (~256, ~512, ~1024). The issue isn't prompt length. Let me check my theory about CUDA memory state between LLM instances:

**Root cause found.** After [llm.exit()](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/engine/llm_engine.py:43:4-47:20):
- **Peak memory stat stays at 10.15GB** (never reset)
- **Only 0.70GB free** (CUDA cache not emptied)

In [allocate_kv_cache()](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/engine/model_runner.py:172:4-218:29), the second LLM calculates:
```
available_memory = total * 0.9 - peak = 11.61 * 0.9 - 10.15 ≈ 0.3GB
```

This gives only ~43 KV cache blocks instead of ~1320. With 6 prompts of ~1024 tokens needing ~96 blocks, [can_allocate](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/engine/block_manager.py:74:4-78:56) returns False every time → **infinite loop**.




Now let me run the experiment:


It completed successfully (exit code 0). The atexit errors are harmless (double-exit). Let me check the results:


All 15 experiment runs completed successfully. Here's a summary:

## Root Cause & Fix

**Bug:** [allocate_kv_cache()](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/engine/model_runner.py:172:4-218:29) uses `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` to calculate available memory. After [llm.exit()](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/engine/llm_engine.py:43:4-47:20), the peak stat (10.15GB) was never reset, so the next LLM instance thought almost no memory was available and allocated very few KV cache blocks. The scheduler then looped infinitely because [can_allocate()](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/engine/block_manager.py:74:4-78:56) always returned False.

**Fix** in `@/home/albert/learn/nano-vllm-src/chunk_size_experiment.py:35-39`:
```python
llm.exit()
del llm
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

## Experiment Results

| Chunk Size | Prompt Len | Avg TTFT (s) | Throughput (tok/s) |
|------------|-----------|-------------|-------------------|
| 256 | 256 | 1.209 | 97.4 |
| 256 | 512 | 0.541 | 94.8 |
| 256 | 1024 | 0.977 | 72.0 |
| 512 | 256 | 0.209 | 130.1 |
| 512 | 512 | 0.453 | 102.5 |
| 512 | 1024 | 0.929 | 72.4 |
| 768 | 256 | 0.209 | 131.2 |
| 768 | 512 | 0.453 | 102.7 |
| 768 | 1024 | 0.941 | 72.6 |
| 1024 | 256 | 0.208 | 131.0 |
| 1024 | 512 | 0.452 | 102.9 |
| 1024 | 1024 | 0.996 | 72.0 |
| 2048 | 256 | 0.208 | 131.2 |
| 2048 | 512 | 0.453 | 102.5 |
| 2048 | 1024 | 0.951 | 72.0 |

Results saved to [experiment_results.json](cci:7://file:///home/albert/learn/nano-vllm-src/experiment_results.json:0:0-0:0).