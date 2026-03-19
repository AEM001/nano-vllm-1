# 04 - Block Manager KV Cache Behavior

## Primary file

- `nanovllm/engine/block_manager.py`

## Why this file matters

This file is the direct controller of KV cache block allocation, reuse, append behavior, and deallocation. If your GPU cannot hold more active context, this file is very close to the reason.

## Exact objective

Measure how efficiently KV cache blocks are used, when they run out, and whether cache reuse is actually reducing memory pressure.

## High-impact functions

- `can_allocate`
- `allocate`
- `deallocate`
- `can_append`
- `may_append`
- `compute_hash`

## Experiment A: KV block consumption by prompt length

### Procedure

- Use one request at a time.
- Sweep prompt lengths.
- For each run, estimate expected block count from `Sequence.num_blocks`.

### Expected outcome

- KV demand should increase stepwise, not smoothly, because allocation happens in blocks.
- Crossing a block boundary should cause a sudden increase in reserved cache need.

## Experiment B: free-block exhaustion threshold

### Procedure

- Increase concurrent long requests until admission stops or preemption becomes frequent.
- Record when allocation first fails.

### Expected outcome

- You should identify a practical maximum number of concurrent long contexts.
- This is one of the most important real capacity numbers for your GPU.

## Experiment C: cache reuse effectiveness

### Procedure

- Run repeated prompts with common prefixes.
- Compare against unrelated prompts of equal total length.
- Add logging around `seq.num_cached_tokens` and cache-hit paths if needed.

### Expected outcome

- Shared prefixes should reuse blocks and reduce fresh allocation.
- If not, prefix caching is not helping much for your workload.

## Experiment D: append pressure during long decode

### Procedure

- Use long generations.
- Observe when `may_append` allocates new blocks.
- Track behavior when sequences cross multiples of `block_size`.

### Expected outcome

- Every time a sequence enters a new block, cache pressure jumps.
- Long-running decode can gradually become a memory problem even if startup looked safe.

## What this file can prove

- Whether block granularity wastes memory for your request sizes.
- Whether long-tail decoding is what eventually kills concurrency.
- Whether prefix reuse is a real optimization in practice.
