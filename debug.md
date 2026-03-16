# 1. Allocate physical memory blocks
block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
block_manager.allocate(seq)

# 2. First call: prefill (process prompt)
token_ids_list = run_model.run([seq], is_prefill=True)

# 3. Subsequent calls: decode (generate one token at a time)
while len(generated_tokens) < sampling_params.max_tokens:
    token_ids_list = run_model.run([seq], is_prefill=False)  # <-- is_prefill=False!
    # ... append token and continue

































































All done — I ran it end-to-end in the correct env, fixed root-cause bugs, and verified clean output.

I used the saved workspace memory about your venv location ([/home/albert/learn/l-vllm/.venv](cci:9://file:///home/albert/learn/l-vllm/.venv:0:0-0:0)) to run and validate everything.

### What I fixed

1. **Example startup/runtime correctness**
   - Ensured [example.py](cci:7://file:///home/albert/learn/nano-vllm-src/example.py:0:0-0:0) uses a **local model directory** (required by nano-vllm config) and auto-downloads if missing.
   - Updated prompt formatting settings for cleaner outputs.
   - File: @/home/albert/learn/nano-vllm-src/example.py#7-33

2. **Core generation quality bug in fallback attention**
   - Fixed decode fallback masking: decode with KV cache must attend full context (not causal-masked like q_len==1 causal).
   - File: @/home/albert/learn/nano-vllm-src/nanovllm/layers/attention.py#116-145 and @/home/albert/learn/nano-vllm-src/nanovllm/layers/attention.py#248-253

3. **KV cache indexing bug in fallback paths**
   - `block_tables` hold **block IDs**, but fallback code treated them as direct token-slot IDs.
   - Added conversion from block IDs -> slot IDs and used it in both prefill/decode fallback paths.
   - File: @/home/albert/learn/nano-vllm-src/nanovllm/layers/attention.py#116-123, @/home/albert/learn/nano-vllm-src/nanovllm/layers/attention.py#192-198, @/home/albert/learn/nano-vllm-src/nanovllm/layers/attention.py#243-249

---

### Verified result

I reran with proxy + venv and got sensible outputs (e.g., correct prime numbers list), exit code `0`.

Use this command anytime:

```bash
HTTP_PROXY=http://127.0.0.1:7897/ \
HTTPS_PROXY=http://127.0.0.1:7897/ \
ALL_PROXY=socks://127.0.0.1:7897/ \
NO_PROXY=localhost,127.0.0.1,::1 \
/home/albert/learn/l-vllm/.venv/bin/python /home/albert/learn/nano-vllm-src/example.py
```