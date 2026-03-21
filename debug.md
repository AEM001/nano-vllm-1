# 1. Allocate physical memory blocks
block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
block_manager.allocate(seq)

# 2. First call: prefill (process prompt)
token_ids_list = run_model.run([seq], is_prefill=True)

# 3. Subsequent calls: decode (generate one token at a time)
while len(generated_tokens) < sampling_params.max_tokens:
    token_ids_list = run_model.run([seq], is_prefill=False)  # <-- is_prefill=False!
    # ... append token and continue

---







```
Traceback (most recent call last):
  File "/home/albert/learn/nano-vllm-src/nanovllm/engine/llm_engine.py", line 69, in exit
    self.model_runner.call("exit")#Sends a shutdown command to the main model logic (Rank 0).
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/albert/learn/nano-vllm-src/nanovllm/engine/model_runner.py", line 151, in call
    return method(*args)
           ^^^^^^^^^^^^^
  File "/home/albert/learn/nano-vllm-src/nanovllm/engine/model_runner.py", line 108, in exit
    torch.cuda.synchronize()#Blocks CPU until all GPU operations finish
    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/albert/learn/l-vllm/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py", line 1108, in synchronize
    return torch._C._cuda_synchronize()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

[rank0]:[W321 12:31:13.542863274 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```


Found the bug. In [_prefill_fallback](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/layers/attention.py:85:0-113:36), when `block_tables` is not None, it uses `k_cache` directly without flattening, but [_block_ids_to_slot_ids](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/layers/attention.py:59:0-65:45) returns physical slot indices that can exceed the number of blocks. The decode path correctly flattens the cache first, but prefill doesn't.

Now testing the fix:


## Fix verified. The output shows:

```
SCHEDULER: prefill ALLOCATED seq 6 - blocks: [0, 1], positions: 259 tokens
SCHEDULER: prefill num_batched_tokens: 259 (added 259 new tokens)

SCHEDULER: prefill ALLOCATED seq 7 - blocks: [0, 2], positions: 258 tokens
SCHEDULER: prefill num_batched_tokens: 261 (added 2 new tokens)

MODELRUNNER: !!! prepare prefill block_tables: tensor([[0, 1], ...
```

**The fix works:**
- Sequence 6: 259 new tokens (no cache)
- Sequence 7: 258 total tokens but only **2 new tokens** → 256 cached!
- The prefill branch with `block_tables` executed successfully
- No more CUDA out-of-bounds crash

**The bug was:** In [_prefill_fallback](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/layers/attention.py:85:0-116:36), when using cached tokens, the code tried to index into the raw `k_cache` tensor using physical slot IDs. But slot IDs (like 512) can exceed the number of blocks. The decode path correctly flattens the cache first with `view(-1, ...)`, but the prefill path didn't.

**The fix:** Added flattening in prefill path:
```python
flat_k = k.view(-1, k.size(-2), k.size(-1))
flat_v = v.view(-1, v.size(-2), v.size(-1))
k_i = flat_k.index_select(0, slots[:context_len])
v_i = flat_v.index_select(0, slots[:context_len])
```
---































































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