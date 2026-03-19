# 06 - Sequence and Sampling Load Shape

## Primary files

- `nanovllm/engine/sequence.py`
- `nanovllm/sampling_params.py`

## Why these files matter

These files define how a logical request turns into token count, block count, completion length, and stopping behavior. They do not consume most of the GPU time directly, but they strongly shape the workload seen by the scheduler and model runner.

## Exact objective

Understand how request shape changes project load.

## High-impact fields

- `Sequence.block_size`
- `Sequence.num_blocks`
- `Sequence.last_block_num_tokens`
- `SamplingParams.max_tokens`
- `SamplingParams.ignore_eos`
- `SamplingParams.temperature`

## Experiment A: completion-length pressure

### Procedure

- Hold prompt fixed.
- Sweep `max_tokens`: `64`, `128`, `256`, `512`, `1024`, `2048`.

### Expected outcome

- Longer allowed generations raise average decode duration.
- This increases steady-state KV pressure and can reduce concurrency.

## Experiment B: EOS sensitivity

### Procedure

- Compare `ignore_eos=False` and `ignore_eos=True`.
- Use the same prompt set.

### Expected outcome

- With `ignore_eos=True`, many requests will run longer and expose decode-side limits more clearly.
- This is useful for stress testing but less realistic for production behavior.

## Experiment C: prompt shape vs block shape

### Procedure

- Test prompt lengths around block boundaries.
- Good examples if `block_size=256`:
  - `255`
  - `256`
  - `257`
  - `511`
  - `512`
  - `513`

### Expected outcome

- Crossing a block boundary should have a visible effect on required block count.
- Capacity may drop suddenly at these thresholds.

## Experiment D: temperature impact on serving behavior

### Procedure

- Sweep temperature while keeping all resource-related config fixed.

### Expected outcome

- Temperature should not change memory capacity directly.
- It may change average completion length indirectly if EOS tendencies change.

## What these files can prove

- Whether your workload shape, rather than your raw GPU size, is the reason throughput is low.
- Whether small changes in prompt length are causing large jumps in memory demand.
- Whether long decode tails are self-inflicted by sampling settings.
