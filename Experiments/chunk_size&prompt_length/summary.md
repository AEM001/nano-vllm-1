From the experiment results, here are the key insights:

## Chunk Size Impact

**For short prompts (256 tokens):**
- 256 chunk size: 1.209s TTFT, 97.4 tok/s
- 512+ chunk size: ~0.208s TTFT, ~131 tok/s
- **Conclusion:** Larger chunks (512+) reduce TTFT by 83% and increase throughput by 34%

**For medium prompts (512 tokens):**
- All chunk sizes: ~0.45s TTFT, ~102 tok/s
- **Conclusion:** Chunk size has minimal impact when prompt ≈ chunk size

**For long prompts (1024 tokens):**
- All chunk sizes: ~0.95s TTFT, ~72 tok/s
- **Conclusion:** With very long prompts, chunk size doesn't matter much

## Prompt Length Impact

**TTFT scales with prompt length:**
- 256 tokens: ~0.21s
- 512 tokens: ~0.45s (2.1x)
- 1024 tokens: ~0.95s (4.5x)

**Throughput decreases with prompt length:**
- 256 tokens: ~131 tok/s
- 512 tokens: ~103 tok/s (21% lower)
- 1024 tokens: ~72 tok/s (45% lower)

## Optimal Configuration

**Best overall performance:** 768-2048 chunk sizes
- Consistently best TTFT for short/medium prompts
- No penalty for long prompts
- Sweet spot around 768-1024

**Practical recommendation:** Use 1024 chunk size
- Near-optimal TTFT (0.208s for 256-token prompts)
- Good throughput (131 tok/s for short prompts)
- Handles all prompt lengths well

## Why This Matters

- **TTFT is critical for user experience** - first token latency
- **Chunked prefill enables batching** - process multiple requests simultaneously
- **Larger chunks = fewer scheduling steps** = less overhead
- **But diminishing returns** after ~1024 tokens

The experiment validates that chunked prefill with 512+ chunk sizes provides significant performance benefits, especially for shorter prompts where most of the benefit comes from reduced scheduling overhead rather than actual computation speedup.