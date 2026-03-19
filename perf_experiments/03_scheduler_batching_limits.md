# 03 - Scheduler Batching Limits

## Primary file

- `nanovllm/engine/scheduler.py`

## Why this file matters

This file decides which requests run now, which wait, and when memory pressure stops more work from entering the batch.

## Exact objective

Find whether throughput is being limited by scheduler policy rather than raw GPU horsepower.

## High-impact variables and paths

- `max_num_seqs`
- `max_num_batched_tokens`
- prefill path in `schedule`
- decode path in `schedule`
- `preempt`
- `postprocess`

## Experiment A: prefill admission limit

### Procedure

- Submit many requests at once.
- Use identical prompt lengths.
- Increase prompt length across runs.
- Watch when the scheduler logs that it cannot allocate more sequences.

### Expected outcome

- Short prompts should allow many requests into prefill.
- Longer prompts should hit the token budget or KV allocation limit earlier.

## Experiment B: decode concurrency limit

### Procedure

- Start several requests and let them decode together.
- Increase the number of simultaneous requests.
- Watch whether decode continues smoothly or starts preempting.

### Expected outcome

- Decode throughput should increase at first.
- After a limit, preemption or allocation pressure should flatten or hurt performance.

## Experiment C: fairness vs throughput

### Procedure

- Mix short and long requests in one run.
- Observe whether short requests finish quickly or get delayed behind long ones.

### Expected outcome

- The current policy may favor currently runnable requests rather than globally optimal latency.
- You may identify tail-latency penalties under mixed workloads.

## Experiment D: bottleneck type classification

### Procedure

- When you see `Cannot allocate sequence ...`, note whether:
  - `num_batched_tokens + len(seq)` exceeded the token cap, or
  - `block_manager.can_allocate(seq)` failed.

### Expected outcome

- If token cap hits first, your bottleneck is batching policy.
- If block allocation hits first, your bottleneck is KV capacity.

## What this file can prove

- Whether the project underuses the GPU because scheduler limits are conservative.
- Whether your system is blocked more by token budget or KV memory.
- Whether concurrency scaling is stopped by policy before hardware saturation.
