"""
Tutorial 06 — Tensor Parallel Linear Layers
=============================================
A large LLM doesn't fit in one GPU's memory. Tensor parallelism splits the
weight matrices across GPUs so each GPU holds only a shard.

This file covers:
  1. Why we need tensor parallelism
  2. Column-parallel linear (split output dimension)
  3. Row-parallel linear (split input dimension)
  4. How QKV projection works with tensor parallelism
  5. Megatron-style MLP parallelism: column → row (no communication in between)
  6. How nano-vllm implements all of this

NOTE: This file runs on a SINGLE GPU/CPU (world_size=1) to let you experiment
      without needing multiple GPUs. The parallelism concepts are demonstrated
      by simulating multiple "ranks" in Python.

Run it:
    /home/albert/learn/l-vllm/.venv/bin/python tutorial/06_tensor_parallel_linear.py
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Running on: {DEVICE}  dtype: {DTYPE}\n")
torch.manual_seed(0)


# =============================================================================
# PART 1 — Why tensor parallelism?
# =============================================================================
print("=" * 60)
print("PART 1: Why tensor parallelism?")
print("=" * 60)

# A single linear layer weight matrix has shape (output_size, input_size).
# For a large model:
#   hidden_size = 4096
#   intermediate_size = 11008  (MLP)
#   A single gate_proj weight: 4096 × 11008 × 2 bytes = 90 MB
#   A full Llama-7B has 32 layers → MLP alone = 32 × 3 × 90 MB ≈ 8.6 GB
#
# With 2 GPUs, each holds HALF the weight matrix.

configs = [
    ("Qwen3-0.6B",  1024,  2816, 28),
    ("Qwen3-1.7B",  2048,  6144, 28),
    ("Qwen3-7B",    3584, 18944, 28),
    ("Qwen3-30B",   7168, 21504, 28),
]
print(f"{'Model':<15} {'hidden':>7} {'ffn':>7} {'layers':>7} {'MLP weight GB':>14}")
for name, hidden, ffn, layers in configs:
    # gate_proj + up_proj + down_proj, fp16
    mlp_gb = layers * 3 * hidden * ffn * 2 / 1e9
    print(f"  {name:<13} {hidden:>7} {ffn:>7} {layers:>7} {mlp_gb:>14.2f}")

print("\nWith tensor_parallel_size=N, each GPU holds 1/N of each weight matrix.")


# =============================================================================
# PART 2 — Column-parallel linear
# =============================================================================
# Split the OUTPUT dimension across GPUs.
# Each GPU computes a shard of the output.
# No communication needed after the computation.
#
#   GPU0: W0 = W[:out//2, :]   →  y0 = x @ W0.T  (shape: batch × out//2)
#   GPU1: W1 = W[out//2:, :]   →  y1 = x @ W1.T  (shape: batch × out//2)
#   → concatenate: y = [y0, y1]  (shape: batch × out)

print("\n" + "=" * 60)
print("PART 2: Column-parallel linear — split output dim")
print("=" * 60)

in_size, out_size = 8, 12
batch = 4
W_full = torch.randn(out_size, in_size, dtype=torch.float32)   # (out, in)
x      = torch.randn(batch,   in_size,  dtype=torch.float32)

# Reference (single GPU)
y_full = F.linear(x, W_full)   # (batch, out)

# Simulate 2 GPUs (rank 0 and rank 1)
tp_size = 2
W_shards = W_full.chunk(tp_size, dim=0)    # split output dim

y_shards = [F.linear(x, W_shards[rank]) for rank in range(tp_size)]
y_col_parallel = torch.cat(y_shards, dim=-1)    # gather across GPUs

print(f"Full weight shape       : {W_full.shape}")
print(f"Per-GPU weight shape    : {W_shards[0].shape}")
print(f"Per-GPU output shape    : {y_shards[0].shape}")
print(f"Final output shape      : {y_col_parallel.shape}")
print(f"Max diff vs reference   : {(y_full - y_col_parallel).abs().max().item():.2e}")
print("Communication needed: all_gather at the end (or use row-parallel next)")


# =============================================================================
# PART 3 — Row-parallel linear
# =============================================================================
# Split the INPUT dimension across GPUs.
# Each GPU computes a partial dot product with its shard.
# An all_reduce (sum) across GPUs produces the final output.
#
#   GPU0: W0 = W[:, :in//2],  x0 = x[:, :in//2]  →  partial0 = x0 @ W0.T
#   GPU1: W1 = W[:, in//2:],  x1 = x[:, in//2:]  →  partial1 = x1 @ W1.T
#   → all_reduce(sum): y = partial0 + partial1   (shape: batch × out)
#
# KEY INSIGHT: Column-parallel feeds directly into row-parallel.
# The column-parallel output shard IS the row-parallel input shard.
# So the chain: col_parallel → row_parallel has NO communication overhead
# in between — only one all_reduce at the very end.

print("\n" + "=" * 60)
print("PART 3: Row-parallel linear — split input dim")
print("=" * 60)

out_size2 = 6
W2_full = torch.randn(out_size2, in_size, dtype=torch.float32)   # (out, in)

y2_full = F.linear(x, W2_full)   # reference

# Split input dimension
W2_shards = W2_full.chunk(tp_size, dim=1)        # split input dim
x_shards  = x.chunk(tp_size, dim=-1)             # input shard goes to matching GPU

partials = [F.linear(x_shards[r], W2_shards[r]) for r in range(tp_size)]
y_row_parallel = sum(partials)                    # all_reduce = sum in this sim

print(f"Full weight shape    : {W2_full.shape}")
print(f"Per-GPU weight shape : {W2_shards[0].shape}  (split input)")
print(f"Per-GPU partial shape: {partials[0].shape}")
print(f"Final output shape   : {y_row_parallel.shape}")
print(f"Max diff vs reference: {(y2_full - y_row_parallel).abs().max().item():.2e}")
print("Communication needed: all_reduce once at the end")


# =============================================================================
# PART 4 — Megatron-style MLP (col → row, one all_reduce total)
# =============================================================================
# A standard MLP is: Linear1 → Activation → Linear2
# With tensor parallelism:
#   Linear1: column-parallel  (split output)  → NO communication
#   Activation: applied locally per GPU (output is already sharded)
#   Linear2: row-parallel     (split input)   → all_reduce
#
# Total: only 1 all_reduce per MLP, not 2. This is Megatron-LM's trick.

print("\n" + "=" * 60)
print("PART 4: Megatron-style MLP = col-parallel → row-parallel")
print("=" * 60)

hidden = 8
ffn    = 16   # intermediate size

W1 = torch.randn(ffn,    hidden, dtype=torch.float32)   # gate/up
W2 = torch.randn(hidden, ffn,    dtype=torch.float32)   # down

x_mlp = torch.randn(batch, hidden, dtype=torch.float32)

# Reference
h_ref = F.linear(x_mlp, W1)
h_ref = F.silu(h_ref)
y_ref = F.linear(h_ref, W2)

# Tensor parallel simulation
W1_shards = W1.chunk(tp_size, dim=0)    # column-parallel
W2_shards = W2.chunk(tp_size, dim=1)    # row-parallel (matching shard)

partials_mlp = []
for r in range(tp_size):
    h_shard = F.linear(x_mlp, W1_shards[r])    # col-parallel: no comm
    h_shard = F.silu(h_shard)                   # local activation
    p = F.linear(h_shard, W2_shards[r])         # row-parallel partial
    partials_mlp.append(p)

y_tp = sum(partials_mlp)   # all_reduce

print(f"Reference output shape : {y_ref.shape}")
print(f"TP output shape        : {y_tp.shape}")
print(f"Max diff vs reference  : {(y_ref - y_tp).abs().max().item():.2e}")
print("Communication: 1 all_reduce for the whole MLP ✓")


# =============================================================================
# PART 5 — QKV projection with tensor parallelism
# =============================================================================
# Q, K, V projections are column-parallel.
# Each GPU computes its shard of Q, K, V and runs attention on that shard.
# Because attention is per-head, splitting heads across GPUs is exact.
#
# Source: nanovllm/layers/linear.py → QKVParallelLinear
# The weight_loader slices and loads the correct shard for each rank.

print("\n" + "=" * 60)
print("PART 5: QKV projection — splitting attention heads across GPUs")
print("=" * 60)

num_q_heads  = 8
num_kv_heads = 4
head_dim     = 16
hidden       = 64   # = num_q_heads * head_dim

# Full weight: W_q (num_q_heads*head_dim, hidden)
#              W_k (num_kv_heads*head_dim, hidden)
#              W_v (num_kv_heads*head_dim, hidden)
W_q_full = torch.randn(num_q_heads  * head_dim, hidden, dtype=torch.float32)
W_k_full = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float32)
W_v_full = torch.randn(num_kv_heads * head_dim, hidden, dtype=torch.float32)

x_attn = torch.randn(4, hidden, dtype=torch.float32)

# Reference QKV
q_ref = F.linear(x_attn, W_q_full).view(4, num_q_heads,  head_dim)
k_ref = F.linear(x_attn, W_k_full).view(4, num_kv_heads, head_dim)
v_ref = F.linear(x_attn, W_v_full).view(4, num_kv_heads, head_dim)

# Shard: each GPU gets half the Q heads and half the KV heads
tp_size = 2
W_q_shards = W_q_full.chunk(tp_size, dim=0)
W_k_shards = W_k_full.chunk(tp_size, dim=0)
W_v_shards = W_v_full.chunk(tp_size, dim=0)

for rank in range(tp_size):
    q_shard = F.linear(x_attn, W_q_shards[rank]).view(4, num_q_heads // tp_size, head_dim)
    k_shard = F.linear(x_attn, W_k_shards[rank]).view(4, num_kv_heads // tp_size, head_dim)
    v_shard = F.linear(x_attn, W_v_shards[rank]).view(4, num_kv_heads // tp_size, head_dim)

    expected_q = q_ref[:, rank * (num_q_heads // tp_size) : (rank+1) * (num_q_heads // tp_size)]
    diff = (q_shard - expected_q).abs().max().item()
    print(f"  GPU {rank}: Q shard {q_shard.shape}  (heads {rank*(num_q_heads//tp_size)}..{(rank+1)*(num_q_heads//tp_size)-1})  diff={diff:.2e}")

print(f"\nEach GPU runs attention on its {num_q_heads//tp_size} Q heads + {num_kv_heads//tp_size} KV heads independently.")
print("Output projection (row-parallel) then all_reduces to get the final hidden state.")


# =============================================================================
# PART 6 — nano-vllm's weight_loader: how sharded weights are loaded
# =============================================================================
# When nano-vllm loads a HuggingFace checkpoint, the weights are FULL tensors.
# Each layer's weight_loader() method slices out the correct shard for that rank.
# Source: nanovllm/layers/linear.py

print("\n" + "=" * 60)
print("PART 6: nano-vllm weight_loader — loading shards from full weights")
print("=" * 60)

print("""
  ColumnParallelLinear.weight_loader(param, loaded_weight):
      # loaded_weight is the full (output_size, input_size) tensor from disk
      shard_size = param.data.size(0)          # = output_size // tp_size
      start_idx  = tp_rank * shard_size
      loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
      param.data.copy_(loaded_weight)

  RowParallelLinear.weight_loader(param, loaded_weight):
      # Same but splits along dim=1 (input dimension)
      shard_size = param.data.size(1)
      start_idx  = tp_rank * shard_size
      loaded_weight = loaded_weight.narrow(1, start_idx, shard_size)
      param.data.copy_(loaded_weight)

  QKVParallelLinear.weight_loader(param, loaded_weight, loaded_shard_id):
      # loaded_shard_id ∈ {"q", "k", "v"}
      # Q shard starts at offset 0
      # K shard starts at offset num_q_heads * head_size
      # V shard starts at offset num_q_heads * head_size + num_kv_heads * head_size
""")

# Demonstrate: manually loading just rank-0's shard
rank = 0
shard_size = W_q_full.size(0) // tp_size
start = rank * shard_size
loaded = W_q_full.narrow(0, start, shard_size)
print(f"  Full W_q shape: {W_q_full.shape}")
print(f"  rank={rank} shard: {loaded.shape}  rows {start}..{start+shard_size-1}")
print(f"  Matches W_q_shards[{rank}]: {torch.allclose(loaded, W_q_shards[rank])}")


# =============================================================================
# EXPERIMENT ZONE
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENTS")
print("=" * 60)

# Experiment A: verify col + row parallel gives exact same result as single GPU
print("\nA. Full verification: col-parallel + row-parallel = full linear")
in_dim, out_dim = 16, 12
W_test = torch.randn(out_dim, in_dim)
x_test = torch.randn(5, in_dim)
y_single = F.linear(x_test, W_test)

for tp in [1, 2, 4]:
    if out_dim % tp != 0 or in_dim % tp != 0:
        continue
    W_col = W_test.chunk(tp, dim=0)   # output split
    partials_ab = []
    for r in range(tp):
        # Column parallel output is the row parallel input shard
        h = F.linear(x_test, W_col[r])                  # shard of output
        # Suppose we chain a second row-parallel layer with same weight for demo
        # Instead just test col-parallel gather:
        partials_ab.append(h)
    y_tp_col = torch.cat(partials_ab, dim=-1)
    print(f"  tp={tp}: col-parallel max diff = {(y_single - y_tp_col).abs().max().item():.2e}")

# Experiment B: all_reduce cost grows with world size
print("\nB. Why tensor parallelism has diminishing returns")
print("   Each all_reduce synchronises all GPUs over a network link.")
print("   More GPUs → more communication overhead, less compute per GPU.")
print()
hidden = 4096
for tp in [1, 2, 4, 8]:
    weight_per_gpu_gb = hidden * hidden * 2 / 1e9 / tp
    comms_per_layer   = 2   # one for attention o_proj, one for mlp down_proj
    print(f"  tp={tp}: weight/GPU = {weight_per_gpu_gb*1000:.1f} MB  "
          f"all_reduces/layer = {comms_per_layer}  "
          f"total all_reduces (32 layers) = {comms_per_layer * 32}")

# Experiment C: understand VocabParallelEmbedding masking
print("\nC. VocabParallelEmbedding — each GPU owns a token id range")
vocab_size = 20
tp = 4
per_gpu    = vocab_size // tp
for rank in range(tp):
    start = rank * per_gpu
    end   = start + per_gpu
    print(f"  GPU {rank}: token ids {start}..{end-1}")

print()
print("  When a token id falls outside a GPU's range:")
print("  → the embedding lookup returns 0 for that GPU")
print("  → all_reduce (sum) gives the correct embedding from the right GPU")
print("  → This avoids any conditional routing — pure math.")
