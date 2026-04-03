<p align="center">
<img width="300" src="assets/logo.png">
</p>

# awesome-nano-vllm

Enhanced nano-vllm with production-grade features: **chunked prefill** and **mixed-batch execution**.

## What I Built

### Chunked Prefill

The original nano-vllm processed entire prompts in one go — a 1024-token prompt would monopolize the GPU while short requests waited. I implemented chunked prefill that splits long prompts into configurable chunks.

Key implementation details:
- `chunk_size` config controls max tokens per prefill step
- `max_num_partial_prefills` limits how many "in-progress" sequences can coexist
- `long_prefill_threshold` + `max_long_partial_prefills` prevents head-of-line blocking from very long prompts
- Sequences stay in `RUNNING` state across steps until fully prefilled

Experimental results (see `Experiments/chunk_size&prompt_length/`):
- 512+ chunk sizes reduce TTFT by 83% for short prompts
- Sweet spot around 1024 tokens for general workloads
- Diminishing returns beyond 2048 tokens

### Mixed-Batch Execution

Instead of separating prefill and decode phases, I enabled both in the same forward pass. This is the key to low-latency serving.

What the logs show:
```
INFO: prefill tokens: 10, decode tokens: 3
```

How it works:
- Scheduler prioritizes decode sequences (1 token each) for low latency
- Fills remaining `max_num_batched_tokens` budget with partial prefills
- Uses a "mask trick": prefill tokens get `mask=-1` (skip sampling), decode get `mask=0` (sample)
- Same CUDA graph replay works for both cases

The `Scheduler.schedule()` logic:
1. First, schedule all decode sequences (already prefilled)
2. If budget remains, continue partial prefills from running sequences
3. Finally, admit new sequences from waiting queue with chunked prefill
4. Preempt (kick to waiting) lowest-progress sequences if budget exceeded

### Continuous Batching

Sequences enter and exit the batch dynamically:
```
INFO: [Scheduler] Seq 3 finished
WARNING:  !!! Prefill !!!: seq_id4 is prefilling 7 tokens and is gonna be allocated
INFO: number of running and waiting seqs: 4 and 2
```

When a sequence hits EOS or `max_tokens`, it immediately frees its blocks. New sequences from the waiting queue fill those slots — all within the same scheduling step.

## Experiments

Systematic evaluation in `Experiments/`:

| Experiment | What It Tests |
|------------|---------------|
| `chunk_size&prompt_length/` | TTFT vs throughput tradeoffs across chunk sizes (256-2048) and prompt lengths |
| `mixed-batch/` | Validation that prefill+decode coexist in same batch |
| `one-long-one-short/` | Head-of-line blocking mitigation |
| `latency/` | End-to-end latency measurements |

## Core Changes

The heavy lifting happens in:

- `nanovllm/engine/scheduler.py` — `Scheduler.schedule()` with chunked prefill + continuous batching
- `nanovllm/engine/model_runner.py` — `prepare()` with mixed-batch tensor building
- `nanovllm/layers/attention.py` — `_mixed_attention()` handling both prefill and decode in one call

Git history shows the evolution: initial chunked prefill → mixed-batch support → scheduler refinements → preemptive scheduling.

## What Makes This Different

Original nano-vllm was a clean educational implementation. This version is production-oriented:

- Token-level scheduling instead of sequence-level
- Configurable partial prefill limits prevent resource starvation
- Mixed batches maximize GPU utilization without latency spikes

The sophistication isn't in complex abstractions — it's in precise token-level orchestration of simple primitives.
