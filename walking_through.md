the overall walking through and further exploration

## how a request flows through the system

### from prompt to sequence

a prompt comes in — could be a string or already-tokenized IDs. `LLMEngine.add_request()` tokenizes it if needed, wraps it in a `Sequence` object. the sequence holds:
- `token_ids[]`: the actual token integers
- `num_cached_tokens`: how many tokens have KV cache allocated
- `block_table[]`: indices into the physical KV cache blocks

the sequence starts in `WAITING` state. the scheduler will pick it up when there's room.

### memory: two completely different things

i used to be confused about this — there are TWO separate storage concepts:

1. **token storage**: just the `token_ids` list in the Sequence object. simple, grows dynamically.
2. **KV cache storage**: massive pre-allocated GPU tensor with shape `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`. this stores the key/value vectors for attention.

the `BlockManager` manages #2, not #1. it hands out block IDs, tracks reference counts, and deduplicates via hashing for prefix sharing.

`block_table` is the mapping: logical token position → physical block index. if block_size=256 and you have 300 tokens, your block_table has 2 entries: `[block_id_for_0_255, block_id_for_256_299]`.

### the scheduler: making the batch

`Scheduler.schedule()` decides what runs this step. here's where it gets interesting:

**step 1: existing running sequences**
decode sequences (already prefilled) get priority — they each contribute exactly 1 token to the batch budget. if we're out of budget, we preempt (kick back to waiting, free their blocks) the ones with least progress.

partially-prefilled sequences continue their chunk. `len_to_prefill = min(chunk_size, remaining_tokens, remaining_budget)`. they stay in `RUNNING` state until fully prefilled.

**step 2: new sequences from waiting queue**
we try to pull in new sequences, but with a twist — **chunked-prefill**. if a sequence is long and we'd exceed `max_num_batched_tokens`, we only prefill a chunk this step. the sequence enters `RUNNING` state partially prefilled, continues next step. this prevents one long prompt from monopolizing the GPU.

the scheduler also limits partial prefills: `max_num_partial_prefills` and `max_long_partial_prefills` prevent too many "in-progress" sequences from clogging the pipeline.

### from scheduler to GPU: preparation

`ModelRunner.prepare()` takes the scheduled `(sequence, num_tokens)` pairs and builds the actual GPU tensors:

```
input_ids:     the tokens to process this step
positions:     each token's position in its sequence (for RoPE)
slot_mapping:  where to write K/V in physical cache = block_table[block_idx] * block_size + offset
mask:          -1 for prefill tokens (don't sample), 0 for decode tokens (sample)
cu_seqlens_*:  cumulative sequence lengths for flash attention
```

the slot_mapping is the crucial bridge — it's a flat array where each entry tells the Triton kernel exactly which slot in the giant KV cache tensor to write to.

### through the model layers

token IDs hit `Qwen3ForCausalLM.forward()`:

1. **embedding**: `VocabParallelEmbedding` converts IDs to vectors
2. **transformer layers** (repeated N times):
   - `Qwen3Attention`: 
     - `qkv_proj` linear → split to q, k, v
     - rotary embedding (RoPE) applied to q, k
     - `store_kvcache()` Triton kernel scatters k, v to cache via slot_mapping
     - `_mixed_attention()`: gathers cached K/V via block_tables, runs flash attention or fallback
   - residual + RMSNorm
   - `Qwen3MLP`: gate_up projection → SiLU activation → down projection
   - residual + RMSNorm
3. **head**: `ParallelLMHead` projects final hidden states to vocabulary logits

`compute_logits()` does the final projection. then `Sampler` samples — but only for decode sequences.

### the mask trick

here's the clever bit: prefill sequences have `mask=-1`. after `run_model()`, we filter:

```python
sample_indices = [i for i, (seq, _) in enumerate(scheduled_seqs) 
                  if seq.num_cached_tokens >= seq.num_prompt_tokens]
```

prefill sequences don't get sampled — they just update `num_cached_tokens`. decode sequences get their new token appended. this cleanly separates "compute only" from "compute + generate" without branching the model code.

### back to scheduler: postprocess

`Scheduler.postprocess()`:
- updates `num_cached_tokens` for prefilling sequences
- appends sampled tokens to completed sequences
- checks finish conditions (EOS token or max_tokens reached)
- moves finished sequences to `finished` queue, deallocates their blocks

**continuous batching** happens here implicitly: finished sequences free up slots, new ones from `waiting` can enter next step. decode gets priority because it's 1 token vs many for prefill, keeping latency low.

### paged attention in detail

the KV cache tensor is pre-allocated once: `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`. during forward pass:

1. **store**: `store_kvcache_kernel` (Triton) writes new K/V vectors at slots indicated by `slot_mapping`
2. **load**: `_mixed_attention()` converts `block_table` entries to slot indices, gathers cached K/V from previous positions

prefix sharing happens in `BlockManager.allocate()` — if a block's hash matches an existing block, we increment its ref_count instead of allocating new space. copy-on-write semantics when sequences diverge.

## what makes this elegant

the whole system operates at token-level granularity:
- memory: fixed-size blocks, not variable per sequence
- scheduling: token budget, not sequence count
- batching: mixed prefill/decode in same batch
- masking: simple integer flag controls sampling behavior

the sophistication isn't from complex abstractions — it's from precise orchestration of simple primitives.