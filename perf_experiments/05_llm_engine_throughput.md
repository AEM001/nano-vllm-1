# 05 - LLM Engine Throughput

## Primary file

- `nanovllm/engine/llm_engine.py`

## Why this file matters

This file is the top-level runtime loop. It connects request submission, scheduling, model execution, and output collection. It is the best place to measure end-to-end performance rather than only isolated kernels.

## Exact objective

Measure whole-system throughput and identify which phase dominates:

- request admission
- prefill
- decode
- scheduler overhead
- model runner overhead

## High-impact functions

- `add_request`
- `step`
- `generate`

## Experiment A: single-request latency breakdown

### Procedure

- Run one prompt.
- Add temporary timing around:
  - `scheduler.schedule()`
  - `model_runner.call()`
  - `scheduler.postprocess()`
- Separate first step from later steps.

### Expected outcome

- First step should be much heavier because it is prefill.
- Later steps should represent decode latency.

## Experiment B: end-to-end throughput scaling

### Procedure

- Run with `1`, `2`, `4`, `8`, `16` concurrent prompts.
- Keep prompt length fixed.
- Compare tokens/sec and completion latency.

### Expected outcome

- Total throughput should rise at first.
- Per-request latency may worsen as concurrency increases.
- The turning point indicates your practical serving limit.

## Experiment C: short-prompt vs long-prompt service profile

### Procedure

- Compare very short prompts and very long prompts.
- Keep target completion length fixed.

### Expected outcome

- Short prompts emphasize decode throughput.
- Long prompts emphasize prefill cost and KV pressure.

## Experiment D: long-generation stress

### Procedure

- Keep prompt short.
- Increase `max_tokens` substantially.
- Use `ignore_eos=True` if you need to force long decode behavior for measurement.

### Expected outcome

- Decode throughput should stabilize after startup.
- Memory pressure may rise stepwise as more KV blocks are needed.

## What this file can prove

- Your real serving throughput, not just kernel speed.
- Whether the project bottleneck is in the Python/runtime loop or inside GPU execution.
- Whether the workload is prefill-dominated or decode-dominated.
