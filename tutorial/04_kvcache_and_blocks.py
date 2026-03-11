"""
Tutorial 04 — KV-Cache and Block Management
============================================
The KV-cache is *the* reason inference engines like vLLM are fast.
Without it, every decode step re-computes all past K,V values from scratch.
With it, we store them once and reuse them forever.

This file covers:
  1. Why the KV-cache exists and what it stores
  2. Paged memory: why we split the cache into fixed-size blocks
  3. The Block and BlockManager classes from nano-vllm exactly
  4. Prefix caching: reusing KV blocks across requests with shared prefixes

Run it:
    /home/albert/learn/l-vllm/.venv/bin/python tutorial/04_kvcache_and_blocks.py
"""

import sys
sys.path.insert(0, ".")

import torch
import xxhash
import numpy as np
from collections import deque

print("=" * 60)
print("PART 1: Why does the KV-cache exist?")
print("=" * 60)

# During decode, at step t we have tokens [t0, t1, ..., t_{t-1}] cached.
# We only compute Q/K/V for the new token t_t.
# But attention needs K and V for ALL past tokens.
# → We stored them in the KV-cache at earlier steps. Read, don't recompute.
#
# Memory cost per token (fp16):
#   2 (K+V) × num_layers × num_kv_heads × head_dim × 2 bytes
# For Qwen3-0.6B: 2 × 28 × 8 × 64 × 2 = 57344 bytes ≈ 56 KB per token
#
# For a batch of 64 sequences × 4096 tokens = 14.7 GB — this fills a GPU fast!
# That's why we need smart memory management.

layers, kv_heads, head_dim = 28, 8, 64
bytes_fp16 = 2
per_token_bytes = 2 * layers * kv_heads * head_dim * bytes_fp16
print(f"KV-cache cost per token (Qwen3-0.6B): {per_token_bytes:,} bytes = {per_token_bytes/1024:.1f} KB")

for seq_len in [128, 512, 2048, 4096]:
    gb = per_token_bytes * seq_len / 1e9
    print(f"  {seq_len:5d} tokens → {gb:.4f} GB per sequence")


# =============================================================================
# PART 2 — Paged Memory (PagedAttention)
# =============================================================================
# Problem with naive caching: if you reserve max_seq_len per sequence upfront,
# you waste most of the memory (sequences rarely reach max length).
#
# Solution (from the vLLM paper): split the cache into fixed-size BLOCKS,
# just like OS virtual memory pages.
# Each block stores block_size tokens worth of K and V.
# Sequences grow block by block; unused blocks go back to the free pool.
#
# nano-vllm uses block_size=256 by default.

print("\n" + "=" * 60)
print("PART 2: Paged memory — blocks and the free pool")
print("=" * 60)

BLOCK_SIZE = 4    # small for demonstration (nano-vllm default = 256)
NUM_BLOCKS = 10

class ToyBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def __repr__(self):
        return f"Block(id={self.block_id}, rc={self.ref_count}, hash={'cached' if self.hash!=-1 else 'new'}, tokens={self.token_ids})"


class ToyBlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [ToyBlock(i) for i in range(num_blocks)]
        self.free_ids: deque[int] = deque(range(num_blocks))
        self.used_ids: set[int]   = set()

    def allocate(self) -> ToyBlock:
        assert self.free_ids, "Out of memory!"
        bid = self.free_ids.popleft()
        b = self.blocks[bid]
        b.ref_count = 1
        self.used_ids.add(bid)
        return b

    def free(self, block_id: int):
        b = self.blocks[block_id]
        b.ref_count -= 1
        if b.ref_count == 0:
            self.used_ids.remove(block_id)
            self.free_ids.append(block_id)

    def status(self):
        return f"free={list(self.free_ids)}  used={sorted(self.used_ids)}"


mgr = ToyBlockManager(NUM_BLOCKS, BLOCK_SIZE)
print(f"Initial state  : {mgr.status()}")

b0 = mgr.allocate()
b1 = mgr.allocate()
b2 = mgr.allocate()
print(f"After 3 allocs : {mgr.status()}")

mgr.free(b1.block_id)
print(f"After free b1  : {mgr.status()}")
print(f"Re-allocated   : {mgr.allocate()}")    # gets b1's slot back


# =============================================================================
# PART 3 — The actual nano-vllm Block and BlockManager
# =============================================================================
# Source: nanovllm/engine/block_manager.py

print("\n" + "=" * 60)
print("PART 3: nano-vllm BlockManager — allocate and deallocate")
print("=" * 60)

from nanovllm.engine.block_manager import Block, BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE_NV = 4   # keep small so we can inspect state easily

# Helper to show free pool
def show_state(bm: BlockManager, label: str):
    n_free = len(bm.free_block_ids)
    n_used = len(bm.used_block_ids)
    print(f"  {label}: free={n_free}  used={n_used}  used_ids={sorted(bm.used_block_ids)}")

bm = BlockManager(num_blocks=8, block_size=BLOCK_SIZE_NV)
show_state(bm, "Initial        ")

# Create a sequence with 9 tokens → needs ceil(9/4)=3 blocks
sp = SamplingParams(temperature=1.0, max_tokens=32)
seq_a = Sequence(list(range(9)), sp)
seq_a.block_size = BLOCK_SIZE_NV    # override for demo
print(f"\n  Sequence A: {len(seq_a)} tokens → needs {seq_a.num_blocks} blocks")
bm.allocate(seq_a)
show_state(bm, "After alloc A  ")
print(f"  Block table A: {seq_a.block_table}")

seq_b = Sequence(list(range(5)), sp)
seq_b.block_size = BLOCK_SIZE_NV
bm.allocate(seq_b)
show_state(bm, "After alloc B  ")

bm.deallocate(seq_a)
show_state(bm, "After free A   ")
print(f"  Block table A after free: {seq_a.block_table}")


# =============================================================================
# PART 4 — Prefix Caching (the hash trick)
# =============================================================================
# If two requests share the same prefix (e.g. a system prompt), we can reuse
# the KV blocks for that prefix instead of recomputing them.
#
# How nano-vllm does it:
#   • Each completed block gets a hash based on its token ids + the hash of the
#     previous block (so the hash is "chained" — it encodes the full prefix).
#   • On a new allocation, if a block's hash already exists in hash_to_block_id
#     AND the stored token ids match, we mark those tokens as "cached" and skip
#     recomputing them.
#   • Only FULL blocks are cached (the last partial block is not, because it
#     will keep growing).

print("\n" + "=" * 60)
print("PART 4: Prefix caching — hash chaining")
print("=" * 60)

def compute_block_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
    h = xxhash.xxh64()
    if prefix_hash != -1:
        h.update(prefix_hash.to_bytes(8, "little"))
    h.update(np.array(token_ids, dtype=np.int64).tobytes())
    return h.intdigest()

# Simulate two requests with a shared 8-token system prompt, block_size=4
BLOCK_SIZE_PC = 4
system_prompt = [10, 20, 30, 40,   50, 60, 70, 80]   # 2 full blocks
request_A     = system_prompt + [1, 2, 3, 4]           # + unique continuation
request_B     = system_prompt + [9, 8, 7, 6]           # + different continuation

def hash_sequence(tokens: list[int], block_size: int) -> list[int]:
    """Return the chain of block hashes for a token list."""
    hashes = []
    prev_hash = -1
    n_full_blocks = len(tokens) // block_size
    for i in range(n_full_blocks):
        block_tokens = tokens[i*block_size:(i+1)*block_size]
        h = compute_block_hash(block_tokens, prev_hash)
        hashes.append(h)
        prev_hash = h
    return hashes

hashes_A = hash_sequence(request_A, BLOCK_SIZE_PC)
hashes_B = hash_sequence(request_B, BLOCK_SIZE_PC)

print(f"Request A block hashes: {[hex(h)[:10] for h in hashes_A]}")
print(f"Request B block hashes: {[hex(h)[:10] for h in hashes_B]}")

shared = sum(1 for a, b in zip(hashes_A, hashes_B) if a == b)
print(f"\nShared prefix blocks: {shared} / {len(hashes_A)}")
print(f"→ Request B can reuse {shared * BLOCK_SIZE_PC} cached tokens, "
      f"skipping their prefill computation!")

# Show that different token order gives different hash
hashes_shuffled = hash_sequence([80, 70, 60, 50, 40, 30, 20, 10, 1, 2, 3, 4], BLOCK_SIZE_PC)
print(f"\nSame tokens but shuffled — hashes match? {hashes_shuffled[:2] == hashes_A[:2]}")
print("(No — order matters, the hash captures the exact sequence.)")


# =============================================================================
# PART 5 — slot_mapping: from logical position to physical GPU memory
# =============================================================================
# When writing K,V for a new token, we need to know exactly which slot in
# the physical kv_cache tensor to write to.
#
# slot = block_id * block_size + position_within_block
#
# nano-vllm computes this in model_runner.prepare_prefill() and passes it
# as context.slot_mapping to the attention layer, which calls store_kvcache().

print("\n" + "=" * 60)
print("PART 5: slot_mapping — logical → physical address")
print("=" * 60)

def compute_slot_mapping(block_table: list[int], seq_len: int, block_size: int) -> list[int]:
    slots = []
    for pos in range(seq_len):
        block_idx = pos // block_size
        offset    = pos % block_size
        physical_block = block_table[block_idx]
        slot = physical_block * block_size + offset
        slots.append(slot)
    return slots

# Example: 9 tokens in blocks [2, 5, 7]
block_table = [2, 5, 7]
block_size  = 4
seq_len     = 9

slots = compute_slot_mapping(block_table, seq_len, block_size)
print(f"block_table={block_table}  block_size={block_size}  seq_len={seq_len}")
print(f"slot_mapping={slots}")
print("\nBreakdown:")
for pos, slot in enumerate(slots):
    blk_idx = pos // block_size
    offset  = pos % block_size
    print(f"  token[{pos}] → block_table[{blk_idx}]={block_table[blk_idx]}, offset={offset} → slot {slot}")


# =============================================================================
# EXPERIMENT ZONE
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENTS")
print("=" * 60)

# Experiment A: how many blocks does a given sequence need?
print("\nA. Block count vs sequence length (block_size=256)")
BS = 256
for n in [1, 100, 256, 257, 512, 1000, 4096]:
    num_blocks = (n + BS - 1) // BS
    wasted     = num_blocks * BS - n
    print(f"  seq_len={n:5d} → {num_blocks} blocks  ({wasted:3d} slots wasted in last block)")

# Experiment B: prefix cache hit rate with a shared system prompt
print("\nB. Prefix cache hit rate vs system prompt length (block_size=4)")
unique_len = 16
for sys_len in [0, 4, 8, 12, 16, 32]:
    total = sys_len + unique_len
    n_full_sys_blocks = sys_len // BLOCK_SIZE_PC
    n_full_total      = total   // BLOCK_SIZE_PC
    hit_rate = n_full_sys_blocks / n_full_total if n_full_total > 0 else 0
    print(f"  sys_prompt={sys_len:3d} tokens, unique={unique_len} → "
          f"cache hit rate ≈ {hit_rate:.0%}")

# Experiment C: build your own mini block manager and simulate an OOM
print("\nC. Out-of-memory simulation")
small_bm = BlockManager(num_blocks=4, block_size=BLOCK_SIZE_NV)
seqs = []
for i in range(4):
    s = Sequence(list(range(BLOCK_SIZE_NV)), sp)   # exactly 1 block each
    s.block_size = BLOCK_SIZE_NV
    if small_bm.can_allocate(s):
        small_bm.allocate(s)
        seqs.append(s)
        print(f"  Allocated seq {i}: free={len(small_bm.free_block_ids)}")
    else:
        print(f"  OOM on seq {i}: not enough free blocks!")
