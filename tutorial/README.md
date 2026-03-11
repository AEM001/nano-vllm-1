# nano-vllm Tutorial: Build an LLM Inference Engine from Scratch

This tutorial walks you through **every core idea** in nano-vllm by letting you
run and modify the actual building blocks yourself. You will not just read — you
will experiment.

## How to run anything here

```bash
# All experiments use your project venv
/home/albert/learn/l-vllm/.venv/bin/python tutorial/01_tokenization_and_sampling.py
```

Or open any file in the IDE and run it cell by cell (each file is self-contained).

---

## Learning path (do these in order)

| File | What you learn |
|------|---------------|
| `01_tokenization_and_sampling.py` | Tokens, temperature, and how text becomes numbers |
| `02_rmsnorm_rope_activation.py` | The three core math layers every modern LLM uses |
| `03_attention_from_scratch.py` | Scaled dot-product attention — the heart of the transformer |
| `04_kvcache_and_blocks.py` | Why KV-cache exists, how blocks work, what prefix caching is |
| `05_sequence_and_scheduler.py` | How the engine manages many requests at once |
| `06_tensor_parallel_linear.py` | How large models are split across multiple GPUs |

---

## Architecture map

```
User calls LLM.generate()
        │
        ▼
  LLMEngine                    ← orchestrates the loop
    ├── Scheduler               ← decides which sequences run next
    │     └── BlockManager      ← manages KV-cache memory blocks
    └── ModelRunner             ← runs the actual GPU computation
          ├── Qwen3ForCausalLM  ← the transformer model
          │     ├── VocabParallelEmbedding
          │     ├── Qwen3DecoderLayer × N
          │     │     ├── Qwen3Attention
          │     │     │     ├── QKVParallelLinear
          │     │     │     ├── RotaryEmbedding
          │     │     │     └── Attention  ← KV-cache read/write + SDPA
          │     │     ├── Qwen3MLP
          │     │     │     ├── MergedColumnParallelLinear
          │     │     │     ├── SiluAndMul
          │     │     │     └── RowParallelLinear
          │     │     └── RMSNorm (×2)
          │     └── ParallelLMHead
          └── Sampler           ← turns logits into token ids
```

---

## Two phases every inference engine runs

**Prefill** — the prompt is processed all at once. All tokens attend to each
other. This is compute-bound (lots of matrix math).

**Decode** — one new token is generated per step. Only the new token attends to
the cached past. This is memory-bandwidth-bound (reading the KV-cache).

Everything in this codebase is designed around these two phases.

---

## Key files in the source (for reference while learning)

```
nanovllm/
  config.py                  ← all runtime settings in one dataclass
  sampling_params.py         ← temperature, max_tokens
  llm.py                     ← thin wrapper, just exposes LLMEngine as LLM
  layers/
    activation.py            ← SiluAndMul  (SwiGLU gate)
    attention.py             ← KV-cache store + SDPA fallback
    embed_head.py            ← token embedding + LM head (tensor-parallel)
    layernorm.py             ← RMSNorm (with fused residual add variant)
    linear.py                ← column/row parallel linear layers
    rotary_embedding.py      ← RoPE positional encoding
    sampler.py               ← Gumbel-max sampling
  engine/
    sequence.py              ← one request = one Sequence object
    block_manager.py         ← physical KV-cache block allocator
    scheduler.py             ← prefill/decode scheduling loop
    model_runner.py          ← GPU execution, CUDA graph capture
    llm_engine.py            ← top-level engine, multi-process coordination
  models/
    qwen3.py                 ← full Qwen3 transformer assembled from layers
  utils/
    context.py               ← thread-local state shared between layers
    loader.py                ← weight loading from HuggingFace checkpoints
```
