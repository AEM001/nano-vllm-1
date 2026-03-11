"""
Tutorial 03 — Attention from Scratch
======================================
Attention is the core computation of every transformer. This file builds it
step by step:
  1. Bare-bones scaled dot-product attention (SDPA)
  2. Causal masking (so tokens can't see the future)
  3. Multi-head attention (MHA)
  4. Grouped-query attention (GQA) — what Qwen3 actually uses
  5. How nano-vllm's Attention layer wires this together with KV-cache

Run it:
    /home/albert/learn/l-vllm/.venv/bin/python tutorial/03_attention_from_scratch.py
"""

import sys
sys.path.insert(0, ".")

import math
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Running on: {DEVICE}  dtype: {DTYPE}\n")

torch.manual_seed(42)


# =============================================================================
# PART 1 — Scaled Dot-Product Attention
# =============================================================================
# Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
#
# Intuition:
#   Q (query)  — what am I looking for?
#   K (key)    — what does each token offer?
#   V (value)  — what does each token actually say?
#   The dot product Q·Kᵀ measures similarity: high score = attend more.

print("=" * 60)
print("PART 1: Scaled dot-product attention")
print("=" * 60)

def sdpa_manual(
    q: torch.Tensor,   # (seq_q, d_k)
    k: torch.Tensor,   # (seq_k, d_k)
    v: torch.Tensor,   # (seq_k, d_v)
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.size(-1)
    scale = scale or math.sqrt(d_k)
    scores = (q.float() @ k.float().T) / scale   # (seq_q, seq_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)           # (seq_q, seq_k)
    return (weights @ v.float()).to(q.dtype)       # (seq_q, d_v)

seq_len, d_k = 6, 8
q = torch.randn(seq_len, d_k, device=DEVICE, dtype=DTYPE)
k = torch.randn(seq_len, d_k, device=DEVICE, dtype=DTYPE)
v = torch.randn(seq_len, d_k, device=DEVICE, dtype=DTYPE)

out = sdpa_manual(q, k, v)
print(f"Q, K, V shape : {q.shape}")
print(f"Output shape  : {out.shape}")

# Why divide by √d_k?
# Without scaling, the dot products grow with d_k, pushing softmax into
# near-zero gradient regions (saturation). Dividing by √d_k keeps variance ≈ 1.
print("\n-- Effect of scaling --")
d_large = 512
q_big = torch.randn(4, d_large, device=DEVICE, dtype=torch.float32)
k_big = torch.randn(4, d_large, device=DEVICE, dtype=torch.float32)
raw_scores     = (q_big @ k_big.T)
scaled_scores  = raw_scores / math.sqrt(d_large)
print(f"  d_k={d_large}  raw scores std={raw_scores.std():.2f}  scaled std={scaled_scores.std():.2f}  (want ≈ 1.0)")


# =============================================================================
# PART 2 — Causal Masking (autoregressive generation)
# =============================================================================
# During training we process the whole sequence at once, but token i should
# only be allowed to attend to tokens 0..i, never to future tokens.
# We achieve this by setting future attention scores to -inf before softmax.

print("\n" + "=" * 60)
print("PART 2: Causal masking")
print("=" * 60)

def make_causal_mask(seq_len: int, device=DEVICE) -> torch.Tensor:
    # True where we want to MASK (future positions)
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

mask = make_causal_mask(seq_len)
print("Causal mask (True = masked / forbidden):")
print(mask.int().tolist())

# Verify: after masking, each row's weights only cover past positions
out_causal = sdpa_manual(q, k, v, mask=mask)
# Manually compute attention weights to inspect
scores = (q.float() @ k.float().T) / math.sqrt(d_k)
scores_masked = scores.masked_fill(mask, float("-inf"))
weights = F.softmax(scores_masked, dim=-1)
print("\nAttention weights (rows=query, cols=key) — upper triangle is 0:")
for row in weights:
    print("  " + "  ".join(f"{w:.2f}" for w in row.tolist()))


# =============================================================================
# PART 3 — Multi-Head Attention (MHA)
# =============================================================================
# Instead of one attention operation, we run h smaller ones in parallel.
# Each "head" can learn to attend to different aspects of the input.
# Output of all heads is concatenated and projected back.

print("\n" + "=" * 60)
print("PART 3: Multi-Head Attention (MHA)")
print("=" * 60)

def multi_head_attention(
    x: torch.Tensor,      # (seq, hidden)
    W_qkv: torch.Tensor,  # (3*hidden, hidden)
    W_o: torch.Tensor,    # (hidden, hidden)
    num_heads: int,
    causal: bool = True,
) -> torch.Tensor:
    seq, hidden = x.shape
    head_dim = hidden // num_heads

    # Project to Q, K, V
    qkv = x @ W_qkv.T                           # (seq, 3*hidden)
    q, k, v = qkv.split(hidden, dim=-1)         # each: (seq, hidden)

    # Reshape into heads: (seq, num_heads, head_dim)
    q = q.view(seq, num_heads, head_dim)
    k = k.view(seq, num_heads, head_dim)
    v = v.view(seq, num_heads, head_dim)

    # Transpose to (num_heads, seq, head_dim) for batched attention
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    mask = make_causal_mask(seq) if causal else None

    # Run attention for each head
    head_outs = []
    for h in range(num_heads):
        head_outs.append(sdpa_manual(q[h], k[h], v[h], mask=mask))

    # Concatenate heads and project
    out = torch.cat(head_outs, dim=-1)          # (seq, hidden)
    return out @ W_o.T

hidden, num_heads = 32, 4
head_dim = hidden // num_heads
x = torch.randn(seq_len, hidden, device=DEVICE, dtype=DTYPE)
W_qkv = torch.randn(3 * hidden, hidden, device=DEVICE, dtype=DTYPE) * 0.02
W_o   = torch.randn(hidden, hidden, device=DEVICE, dtype=DTYPE) * 0.02

mha_out = multi_head_attention(x, W_qkv, W_o, num_heads)
print(f"Input shape  : {x.shape}")
print(f"Output shape : {mha_out.shape}")
print(f"num_heads={num_heads}, head_dim={head_dim}")

# Each head attends independently — they can specialise
print(f"\nUsing PyTorch's F.scaled_dot_product_attention for validation:")
q_t = x @ W_qkv[:hidden].T
k_t = x @ W_qkv[hidden:2*hidden].T
v_t = x @ W_qkv[2*hidden:].T
q_t = q_t.view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
k_t = k_t.view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
v_t = v_t.view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
pt_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
pt_out = pt_out.squeeze(0).transpose(0, 1).reshape(seq_len, hidden) @ W_o.T
print(f"Max diff vs PyTorch SDPA: {(mha_out.float() - pt_out.float()).abs().max().item():.2e}")


# =============================================================================
# PART 4 — Grouped-Query Attention (GQA)
# =============================================================================
# Full MHA: every head has its own Q, K, V.  Cost: num_heads × seq × d_k memory.
# GQA: multiple query heads SHARE the same K, V.  Much less KV-cache memory.
#
# Qwen3 uses GQA: e.g. 16 query heads but only 8 KV heads.
# Each group of 2 query heads shares 1 K and 1 V head.
#
# This is why nano-vllm has _repeat_kv_heads() in attention.py — when running
# GQA, it repeats K and V to match the number of query heads.

print("\n" + "=" * 60)
print("PART 4: Grouped-Query Attention (GQA)")
print("=" * 60)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads so they match query heads."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=0)  # (num_kv_heads*n_rep, seq, head_dim)

num_q_heads  = 8
num_kv_heads = 2   # GQA: 4 query heads share each KV head
n_rep        = num_q_heads // num_kv_heads
head_dim     = 16
seq          = 6

q_gqa = torch.randn(num_q_heads, seq, head_dim, device=DEVICE, dtype=DTYPE)
k_gqa = torch.randn(num_kv_heads, seq, head_dim, device=DEVICE, dtype=DTYPE)
v_gqa = torch.randn(num_kv_heads, seq, head_dim, device=DEVICE, dtype=DTYPE)

k_expanded = repeat_kv(k_gqa, n_rep)   # → (num_q_heads, seq, head_dim)
v_expanded = repeat_kv(v_gqa, n_rep)

print(f"Q heads: {num_q_heads}, KV heads: {num_kv_heads}, repeat: {n_rep}x")
print(f"K before expand: {k_gqa.shape}  →  after: {k_expanded.shape}")

mask = make_causal_mask(seq)
gqa_outs = []
for h in range(num_q_heads):
    gqa_outs.append(sdpa_manual(q_gqa[h], k_expanded[h], v_expanded[h], mask=mask))
gqa_out = torch.stack(gqa_outs, dim=0)   # (num_q_heads, seq, head_dim)
print(f"GQA output shape: {gqa_out.shape}")

# Memory saving: GQA stores num_kv_heads instead of num_q_heads per layer
print(f"\nKV-cache memory ratio GQA/MHA = {num_kv_heads}/{num_q_heads} = {num_kv_heads/num_q_heads:.2f}x")


# =============================================================================
# PART 5 — How nano-vllm wires this together
# =============================================================================
# nano-vllm's Attention.forward() does three things:
#   1. store_kvcache: write the new K,V tokens into the physical block cache
#   2. is_prefill branch: attend over the full prompt (using all KV so far)
#   3. is_decode branch: attend over cached KV for the new single token
#
# The context object (nanovllm/utils/context.py) carries metadata:
#   cu_seqlens_q/k — cumulative sequence lengths (for packed/batched sequences)
#   slot_mapping   — which physical memory slots to write K,V into
#   context_lens   — how many tokens are cached per sequence (for decode)
#   block_tables   — maps logical blocks → physical block ids

print("\n" + "=" * 60)
print("PART 5: nano-vllm Attention layer smoke test")
print("=" * 60)

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import set_context, reset_context

num_heads, head_dim, num_kv_heads = 4, 8, 2
scale = head_dim ** -0.5
attn = Attention(num_heads=num_heads, head_dim=head_dim, scale=scale, num_kv_heads=num_kv_heads)
attn = attn.to(DEVICE)

# --- Prefill (processing a batch of 2 sequences with lengths 3 and 4) ---
total_tokens = 7
q = torch.randn(total_tokens, num_heads,    head_dim, device=DEVICE, dtype=DTYPE)
k = torch.randn(total_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)
v = torch.randn(total_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE)

cu_seqlens_q = torch.tensor([0, 3, 7], dtype=torch.int32, device=DEVICE)
cu_seqlens_k = torch.tensor([0, 3, 7], dtype=torch.int32, device=DEVICE)
slot_mapping  = torch.tensor([], dtype=torch.int32, device=DEVICE)  # no cache yet

set_context(
    is_prefill=True,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=4,
    max_seqlen_k=4,
    slot_mapping=slot_mapping,
)
prefill_out = attn(q, k, v)
print(f"Prefill  — input: {q.shape}  output: {prefill_out.shape}")
reset_context()


# =============================================================================
# EXPERIMENT ZONE
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENTS")
print("=" * 60)

# Experiment A: Visualise what attention "pays attention to"
print("\nA. Attention pattern: what does each query attend to?")
seq = 5
q_exp = torch.zeros(seq, 4)
q_exp[3, 0] = 5.0   # position 3 strongly queries feature 0
k_exp = torch.zeros(seq, 4)
k_exp[1, 0] = 5.0   # position 1 strongly matches feature 0
k_exp[0, 0] = 3.0   # position 0 weakly matches

scores = (q_exp @ k_exp.T) / math.sqrt(4)
mask = make_causal_mask(seq, device="cpu")
scores = scores.masked_fill(mask, float("-inf"))
weights = F.softmax(scores, dim=-1)

print(f"Query at position 3 attends to (weights sum={weights[3].sum():.2f}):")
for i, w in enumerate(weights[3].tolist()):
    bar = "█" * int(w * 30)
    print(f"  pos {i}: {w:.3f} {bar}")

# Experiment B: What changes when you increase num_heads?
print("\nB. Effect of number of heads on parameter count")
hidden = 64
for nh in [1, 2, 4, 8, 16]:
    hd = hidden // nh
    q_params = hidden * nh * hd
    kv_params = hidden * nh * hd * 2
    print(f"  num_heads={nh:2d}  head_dim={hd:3d}  "
          f"Q params={q_params}  KV params={kv_params}  "
          f"total={q_params + kv_params}  (same! just split differently)")

# Experiment C: GQA memory savings
print("\nC. KV-cache size for different GQA configs (seq_len=4096, layers=32, fp16)")
seq_len, num_layers = 4096, 32
head_dim = 128
bytes_per_elem = 2  # fp16
for nq, nkv in [(32, 32), (32, 8), (32, 4), (32, 2), (32, 1)]:
    kv_bytes = 2 * num_layers * seq_len * nkv * head_dim * bytes_per_elem
    print(f"  Q heads={nq}, KV heads={nkv:2d}  →  KV cache = {kv_bytes / 1e9:.3f} GB")
