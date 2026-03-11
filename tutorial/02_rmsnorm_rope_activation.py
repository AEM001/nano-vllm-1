"""
Tutorial 02 — RMSNorm, RoPE, and SwiGLU Activation
=====================================================
Three mathematical building blocks that appear in every modern LLM
(Llama, Qwen, Mistral, etc.).  We build each from scratch, verify it
against the nano-vllm implementation, and run experiments.

Run it:
    /home/albert/learn/l-vllm/.venv/bin/python tutorial/02_rmsnorm_rope_activation.py
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}\n")


# =============================================================================
# PART 1 — RMSNorm
# =============================================================================
# Standard LayerNorm subtracts the mean AND divides by std.
# RMSNorm only divides by the root-mean-square — no mean subtraction.
# Result: ~same training stability, ~10% fewer operations.
#
# Formula:  RMSNorm(x) = x / sqrt(mean(x²) + ε)  * weight
#
# Source: nanovllm/layers/layernorm.py

print("=" * 60)
print("PART 1: RMSNorm")
print("=" * 60)

def rmsnorm_manual(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x / rms).to(orig_dtype) * weight

# Compare with nano-vllm's implementation
from nanovllm.layers.layernorm import RMSNorm

hidden_size = 16
torch.manual_seed(0)
x = torch.randn(4, hidden_size, device=DEVICE, dtype=torch.float16)
weight = torch.ones(hidden_size, device=DEVICE)

our_out   = rmsnorm_manual(x, weight)
theirs    = RMSNorm(hidden_size).to(DEVICE)
theirs.weight.data.fill_(1.0)
their_out = theirs(x)

print(f"Our RMSNorm output shape : {our_out.shape}")
print(f"Max diff from nano-vllm  : {(our_out.float() - their_out.float()).abs().max().item():.2e}")

# The fused residual variant: add residual THEN normalize.
# This saves one pass over memory vs doing them separately.
print("\n-- Fused residual add + RMSNorm --")
residual = torch.randn_like(x)
fused_out, new_residual = theirs(x, residual)
naive_res = x + residual
naive_norm = rmsnorm_manual(naive_res, weight)
print(f"Fused  vs naive max diff : {(fused_out.float() - naive_norm.float()).abs().max().item():.2e}")

# EXPERIMENT: what happens to the norm of the output as you scale x?
print("\n-- Experiment: RMSNorm is scale-invariant --")
for scale in [0.01, 1.0, 100.0]:
    scaled_x = x * scale
    out = rmsnorm_manual(scaled_x, weight)
    print(f"  x scaled by {scale:6.2f} → output RMS = {out.float().pow(2).mean().sqrt().item():.4f}  (always ≈ 1.0)")


# =============================================================================
# PART 2 — Rotary Positional Embedding (RoPE)
# =============================================================================
# RoPE encodes the *position* of each token by rotating the query/key vectors
# in 2D planes.  The key insight: the dot product q·k after rotation depends
# only on the *relative* position (i-j), not absolute positions.
#
# Formula for one 2D pair (x1, x2) at position p:
#   x1' = x1*cos(p*θ) - x2*sin(p*θ)
#   x2' = x2*cos(p*θ) + x1*sin(p*θ)
# where θ = 1 / (base^(2d/dim))  — gets smaller for higher dimensions
#
# Source: nanovllm/layers/rotary_embedding.py

print("\n" + "=" * 60)
print("PART 2: Rotary Positional Embedding (RoPE)")
print("=" * 60)

def build_rope_cache(head_dim: int, max_seq: int, base: float = 10000.0):
    half = head_dim // 2
    # θ_i = 1 / base^(2i / dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float) / half))
    positions = torch.arange(max_seq, dtype=torch.float)
    # outer product: shape (max_seq, half)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()   # shape (max_seq, half)
    sin = freqs.sin()
    return cos, sin

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (seq_len, num_heads, head_dim)
    x1, x2 = x.float().chunk(2, dim=-1)   # split last dim in half
    # cos/sin: (seq_len, 1, head_dim//2) for broadcasting
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1).to(x.dtype)

head_dim, max_seq = 32, 64
cos_cache, sin_cache = build_rope_cache(head_dim, max_seq)

torch.manual_seed(1)
seq_len, num_heads = 8, 4
q = torch.randn(seq_len, num_heads, head_dim)
k = torch.randn(seq_len, num_heads, head_dim)

positions = torch.arange(seq_len)
q_rot = apply_rope(q, cos_cache[positions], sin_cache[positions])
k_rot = apply_rope(k, cos_cache[positions], sin_cache[positions])

print(f"q shape before/after RoPE: {q.shape} → {q_rot.shape}")
print(f"Norms preserved (before): {q.norm(dim=-1).mean():.4f}  (after): {q_rot.norm(dim=-1).mean():.4f}")
print("(RoPE is an isometry — it preserves vector norms)")

# Key property: relative-position encoding
# The dot product q_i · k_j only depends on (i - j)
# Let's verify this numerically
def dot_at_positions(i: int, j: int):
    q_i = apply_rope(q[i:i+1], cos_cache[torch.tensor([i])], sin_cache[torch.tensor([i])])
    k_j = apply_rope(k[j:j+1], cos_cache[torch.tensor([j])], sin_cache[torch.tensor([j])])
    return (q_i * k_j).sum(dim=-1).mean().item()

print("\n-- Relative position property --")
print("dot(q@pos=2, k@pos=0) =", round(dot_at_positions(2, 0), 4))
print("dot(q@pos=5, k@pos=3) =", round(dot_at_positions(5, 3), 4))
print("(Same relative distance=2 → same dot product structure, values differ because q/k differ)")

# Compare with nano-vllm
from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb
rope = RotaryEmbedding(head_dim, head_dim, max_seq, 10000.0)
positions_t = torch.arange(seq_len)
q_nv, k_nv = rope(positions_t, q, k)
print(f"\nMax diff vs nano-vllm RoPE: {(q_rot - q_nv).abs().max().item():.2e}")


# =============================================================================
# PART 3 — SwiGLU Activation (SiluAndMul)
# =============================================================================
# Standard MLP: Linear → ReLU → Linear
# Modern LLM MLP (SwiGLU): Linear(2x) → split → SiLU(left) * right → Linear
#
# SiLU(x) = x * sigmoid(x)   — smooth version of ReLU
# The multiplication by the second half is the "gating" mechanism.
# This is why the intermediate size is usually 8/3 * hidden_size (not 4x).
#
# Source: nanovllm/layers/activation.py

print("\n" + "=" * 60)
print("PART 3: SwiGLU (SiluAndMul)")
print("=" * 60)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def swiglu_manual(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)   # split the last dimension in half
    return silu(gate) * up          # gate controls how much of up passes through

from nanovllm.layers.activation import SiluAndMul

torch.manual_seed(2)
batch, seq, d = 2, 5, 32
x_wide = torch.randn(batch, seq, d * 2, device=DEVICE)   # *2 because it will be split

our_act   = swiglu_manual(x_wide)
their_act = SiluAndMul()(x_wide)

print(f"Input shape  : {x_wide.shape}")
print(f"Output shape : {our_act.shape}  (half the last dim)")
print(f"Max diff vs nano-vllm: {(our_act - their_act).abs().max().item():.2e}")

# Why SiLU instead of ReLU?
print("\n-- SiLU vs ReLU comparison --")
x_range = torch.linspace(-4, 4, 9)
relu_out = F.relu(x_range)
silu_out = silu(x_range)
print(f"x     : {[f'{v:+.1f}' for v in x_range.tolist()]}")
print(f"ReLU  : {[f'{v:+.3f}' for v in relu_out.tolist()]}")
print(f"SiLU  : {[f'{v:+.3f}' for v in silu_out.tolist()]}")
print("Note: SiLU is negative for small negative x (tiny gradient signal),")
print("      while ReLU is dead (exactly 0). SiLU never fully dies.")


# =============================================================================
# EXPERIMENT ZONE
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENTS")
print("=" * 60)

# Experiment A: RMSNorm eps sensitivity
print("\nA. RMSNorm: effect of eps on near-zero inputs")
tiny_x = torch.tensor([[1e-8, 1e-8, 1e-8, 1e-8]], dtype=torch.float32)
for eps in [1e-8, 1e-6, 1e-4]:
    out = rmsnorm_manual(tiny_x, torch.ones(4), eps)
    print(f"  eps={eps:.0e}  output={out[0].tolist()}")

# Experiment B: RoPE — what happens with very long sequences?
print("\nB. RoPE: frequency spectrum (how θ changes per dimension)")
half = head_dim // 2
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float) / half))
print("  Dimension 0  (fastest rotation) θ =", round(inv_freq[0].item(), 6))
print(f"  Dimension {half//2} (mid)             θ =", round(inv_freq[half//2].item(), 6))
print(f"  Dimension {half-1} (slowest)         θ =", round(inv_freq[-1].item(), 8))
print("  Low dims rotate fast → encode fine-grained local position")
print("  High dims rotate slow → encode coarse long-range position")

# Experiment C: SwiGLU gating — the gate controls information flow
print("\nC. SwiGLU: gate controls how much 'up' passes through")
up   = torch.ones(1, 8)
gate = torch.tensor([[-5., -2., -1., 0., 1., 2., 5., 10.]])
out  = silu(gate) * up
print(f"  gate : {gate[0].tolist()}")
print(f"  silu : {[round(v, 3) for v in silu(gate)[0].tolist()]}")
print(f"  out  : {[round(v, 3) for v in out[0].tolist()]}")
print("  Large positive gate → output ≈ up (gate open)")
print("  Large negative gate → output ≈ 0  (gate closed)")
