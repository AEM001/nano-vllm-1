"""
Tutorial 05 — Sequence, Scheduler, and the Generation Loop
============================================================
The scheduler is the "traffic controller" of nano-vllm.  It decides:
  • Which requests to run next (prefill vs decode)
  • When to preempt a request (evict its KV blocks) if memory is tight
  • When a request is finished

This file covers:
  1. The Sequence object — one request = one Sequence
  2. The two-phase loop: prefill batch → decode loop
  3. The Scheduler step by step
  4. Preemption: what happens when memory runs out
  5. Walking the full generation loop manually

Run it:
    /home/albert/learn/l-vllm/.venv/bin/python tutorial/05_sequence_and_scheduler.py
"""

import sys
sys.path.insert(0, ".")

from copy import copy
from collections import deque

print("=" * 60)
print("PART 1: The Sequence object")
print("=" * 60)

# Source: nanovllm/engine/sequence.py
# A Sequence tracks everything about a single request:
#   - token_ids          : the full token list (prompt + generated so far)
#   - status             : WAITING / RUNNING / FINISHED
#   - block_table        : which physical KV-cache blocks it owns
#   - num_cached_tokens  : how many tokens are already in the KV-cache (prefix hit)
#   - sampling params    : temperature, max_tokens, ignore_eos

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

sp = SamplingParams(temperature=1.0, max_tokens=8)
prompt_tokens = [10, 20, 30, 40, 50]
seq = Sequence(prompt_tokens, sp)

print(f"seq_id           : {seq.seq_id}")
print(f"status           : {seq.status}")
print(f"len(seq)         : {len(seq)}         ← num_tokens")
print(f"num_prompt_tokens: {seq.num_prompt_tokens}")
print(f"prompt_token_ids : {seq.prompt_token_ids}")
print(f"block_size       : {seq.block_size}")
print(f"num_blocks       : {seq.num_blocks}         ← ceil(5/256)")
print(f"last_block_num_tokens: {seq.last_block_num_tokens}")

# Simulate generating 3 tokens
EOS = 999
for tok in [100, 200, 300]:
    seq.append_token(tok)

print(f"\nAfter generating 3 tokens:")
print(f"  token_ids        : {seq.token_ids}")
print(f"  completion_token_ids: {seq.completion_token_ids}")
print(f"  num_completion_tokens: {seq.num_completion_tokens}")
print(f"  is_finished      : {seq.is_finished}")

# Check finish conditions
seq_eos = Sequence(prompt_tokens, sp)
seq_eos.append_token(EOS)
# The scheduler handles finishing, not the sequence itself — let's do it manually
if seq_eos.token_ids[-1] == EOS:
    seq_eos.status = SequenceStatus.FINISHED
print(f"\nSequence that generated EOS → is_finished: {seq_eos.is_finished}")


# =============================================================================
# PART 2 — Two-phase inference: Prefill then Decode
# =============================================================================
# PREFILL: Process the prompt. All tokens are fed in at once.
#   - GPU does one big matrix multiply for the whole prompt
#   - All K,V vectors are computed and stored in the KV-cache
#   - Output: the logit for the LAST token → sample first completion token
#
# DECODE: Generate one token at a time.
#   - Only the new token is processed (Q for new token, K/V from cache)
#   - Much less compute per step, but we repeat it many times
#   - Output: one new token per sequence per step

print("\n" + "=" * 60)
print("PART 2: Prefill vs Decode — a visual walkthrough")
print("=" * 60)

def show_prefill(prompts: list[list[int]]):
    print("\n  PREFILL batch:")
    total = 0
    for i, p in enumerate(prompts):
        print(f"    seq {i}: tokens {p}  (len={len(p)})")
        total += len(p)
    print(f"  Total tokens in prefill batch: {total}")
    print(f"  → Model processes {total} tokens in ONE forward pass")
    print(f"  → Returns 1 logit per sequence (last token position)")
    print(f"  → Sample 1 new token per sequence")

def show_decode(seqs_and_last_tokens: list[tuple[int, int]]):
    print("\n  DECODE step:")
    for seq_id, last_tok in seqs_and_last_tokens:
        print(f"    seq {seq_id}: feeds token {last_tok}")
    print(f"  → Model processes {len(seqs_and_last_tokens)} tokens (one per sequence)")
    print(f"  → Returns 1 logit per sequence")
    print(f"  → Sample 1 new token per sequence")

prompts = [[10, 20, 30], [1, 2, 3, 4, 5]]
show_prefill(prompts)

# After prefill, we enter the decode loop
show_decode([(0, 100), (1, 200)])
show_decode([(0, 101), (1, 201)])
show_decode([(0, 102)])   # seq 1 might finish earlier


# =============================================================================
# PART 3 — The Scheduler, step by step
# =============================================================================
# Source: nanovllm/engine/scheduler.py
#
# scheduler.schedule() returns (list[Sequence], is_prefill)
#
# Priority order:
#   1. If ANY sequence is waiting → schedule as many as fit → is_prefill=True
#   2. If no sequences are waiting → schedule running sequences → is_prefill=False
#
# Constraints:
#   • num_seqs ≤ max_num_seqs
#   • total tokens in prefill batch ≤ max_num_batched_tokens
#   • enough free KV-cache blocks for the sequence

print("\n" + "=" * 60)
print("PART 3: Scheduler trace")
print("=" * 60)

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.config import Config

# We need a real Config, but it requires a model path.
# Instead we'll trace the scheduler logic manually.

class ToyScheduler:
    """Simplified scheduler for demonstration — mirrors nano-vllm's logic."""

    def __init__(self, max_num_seqs=4, max_num_batched_tokens=32, num_blocks=20, block_size=4):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_manager = BlockManager(num_blocks, block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = -1
        self.step_num = 0

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self):
        self.step_num += 1
        print(f"\n  [Step {self.step_num}] waiting={len(self.waiting)}  running={len(self.running)}")

        scheduled = []
        num_seqs = 0
        num_toks = 0

        # Prefill pass: drain waiting queue
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            would_add = len(seq) - seq.num_cached_tokens
            if (num_toks + would_add > self.max_num_batched_tokens or
                    not self.block_manager.can_allocate(seq)):
                print(f"    ✗ seq {seq.seq_id} (len={len(seq)}) won't fit — stopping prefill fill")
                break
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)
            num_seqs += 1
            num_toks += would_add
            print(f"    ✓ scheduled seq {seq.seq_id} for PREFILL (len={len(seq)})")

        if scheduled:
            return scheduled, True

        # Decode pass: run all running sequences
        for seq in list(self.running):
            if self.block_manager.can_append(seq):
                self.block_manager.may_append(seq)
                scheduled.append(seq)
                print(f"    → seq {seq.seq_id} in DECODE (len={len(seq)}, cached={seq.num_cached_tokens})")
            else:
                print(f"    ✗ seq {seq.seq_id} preempted (no free blocks)")
        return scheduled, False

    def postprocess(self, seqs, fake_next_tokens):
        for seq, tok in zip(seqs, fake_next_tokens):
            seq.append_token(tok)
            if seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                print(f"    ✓ seq {seq.seq_id} FINISHED")

sp2 = SamplingParams(temperature=1.0, max_tokens=3)
sched = ToyScheduler()

# Add 3 requests
sched.add(Sequence([1, 2, 3, 4], sp2))
sched.add(Sequence([10, 20], sp2))
sched.add(Sequence([100, 200, 300], sp2))

# Step 1: prefill
seqs, is_prefill = sched.schedule()
print(f"  → is_prefill={is_prefill}, scheduled {len(seqs)} seqs")
sched.postprocess(seqs, [50] * len(seqs))

# Steps 2-5: decode loop
for _ in range(4):
    if not sched.running and not sched.waiting:
        print("\n  All sequences finished!")
        break
    seqs, is_prefill = sched.schedule()
    print(f"  → is_prefill={is_prefill}, scheduled {len(seqs)} seqs")
    fake_toks = [999] * len(seqs)   # 999 = EOS simulation (won't trigger because ignore logic differs)
    fake_toks = [50 + sched.step_num] * len(seqs)
    sched.postprocess(seqs, fake_toks)


# =============================================================================
# PART 4 — Preemption: what happens when GPU memory runs out
# =============================================================================
# When a decode step finds there's no free block for a running sequence,
# the scheduler PREEMPTS it:
#   1. Deallocate all its KV-cache blocks (free the GPU memory)
#   2. Move it back to the WAITING queue
#   3. It will be re-prefilled later (KV blocks get recomputed)
#
# This is correct but expensive. Prefix caching partially mitigates it:
# if the sequence's early blocks are still in the cache (not evicted by others),
# only the "new" part needs to be re-prefilled.

print("\n" + "=" * 60)
print("PART 4: Preemption")
print("=" * 60)

print("""
  Normal decode:
    seq A  [running] → generate token, append to KV-cache
    seq B  [running] → generate token, append to KV-cache

  When memory is tight:
    seq A  [running] → needs new block, but NO FREE BLOCKS
    → preempt seq B: deallocate its KV blocks, move back to WAITING
    → now there's a free block for seq A
    → seq B will be re-prefilled in a future step

  Cost of preemption:
    • seq B's KV blocks are LOST (must be recomputed when it runs again)
    • Re-prefill is slow (compute-intensive)
    • Prefix caching can save the early blocks if they're still in the cache
""")

# Demonstrate: preempt() in ToyScheduler (same as in nano-vllm)
def preempt(sched, seq):
    seq.status = SequenceStatus.WAITING
    sched.block_manager.deallocate(seq)
    if seq in sched.running:
        sched.running.remove(seq)
    sched.waiting.appendleft(seq)
    print(f"  Preempted seq {seq.seq_id}: blocks freed, moved back to WAITING")
    print(f"  BlockManager state: free={len(sched.block_manager.free_block_ids)}")


# =============================================================================
# PART 5 — context.py: the shared state that glues engine → layers
# =============================================================================
# nano-vllm uses a global Context object (like a thread-local) to pass
# per-forward-pass metadata to all attention layers simultaneously.
# This avoids threading the metadata through every function call.
#
# Source: nanovllm/utils/context.py

print("\n" + "=" * 60)
print("PART 5: Context — the glue between engine and layers")
print("=" * 60)

import torch
from nanovllm.utils.context import get_context, set_context, reset_context

print("Before set_context:")
ctx = get_context()
print(f"  is_prefill   = {ctx.is_prefill}")
print(f"  slot_mapping = {ctx.slot_mapping}")

device = "cuda" if torch.cuda.is_available() else "cpu"
set_context(
    is_prefill=True,
    cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32, device=device),
    cu_seqlens_k=torch.tensor([0, 3, 7], dtype=torch.int32, device=device),
    max_seqlen_q=4,
    max_seqlen_k=4,
    slot_mapping=torch.tensor([0,1,2,3,4,5,6], dtype=torch.int32, device=device),
)

ctx = get_context()
print("\nAfter set_context (prefill, 2 seqs of len 3 and 4):")
print(f"  is_prefill   = {ctx.is_prefill}")
print(f"  cu_seqlens_q = {ctx.cu_seqlens_q.tolist()}")
print(f"  max_seqlen_q = {ctx.max_seqlen_q}")
print(f"  slot_mapping = {ctx.slot_mapping.tolist()}")
print("  (Every attention layer reads these values during the forward pass)")

reset_context()
print("\nAfter reset_context: is_prefill =", get_context().is_prefill)


# =============================================================================
# EXPERIMENT ZONE
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENTS")
print("=" * 60)

# Experiment A: how does batch size affect throughput in prefill vs decode?
print("\nA. Prefill vs decode compute scaling")
print("  Prefill: compute scales as O(seq_len²) — full attention between all tokens")
print("  Decode : compute scales as O(1) per step (one token × cached KV)")
print()
for seq_len in [16, 64, 256, 1024]:
    prefill_ops = seq_len * seq_len   # attention scores
    decode_ops  = 1 * seq_len         # new token attends to seq_len cached
    print(f"  seq_len={seq_len:5d}: prefill ops ∝ {prefill_ops:8d}  decode ops ∝ {decode_ops:6d}  ratio={prefill_ops//decode_ops}")

# Experiment B: greedy scheduler vs max_num_batched_tokens constraint
print("\nB. How max_num_batched_tokens limits the prefill batch")
seqs_to_batch = [3, 7, 12, 5, 8, 2]
for limit in [10, 20, 40]:
    batch, total = [], 0
    for length in seqs_to_batch:
        if total + length <= limit:
            batch.append(length)
            total += length
        else:
            break
    print(f"  limit={limit:3d}: batched lengths={batch}  total_tokens={total}  "
          f"left_in_queue={len(seqs_to_batch)-len(batch)}")

# Experiment C: trace sequence state machine
print("\nC. Sequence state machine: WAITING → RUNNING → FINISHED")
states = ["WAITING", "RUNNING", "RUNNING", "RUNNING", "FINISHED"]
events = ["(start)", "prefilled", "decoded tok 1", "decoded tok 2", "EOS hit"]
for s, e in zip(states, events):
    print(f"  {e:25s} → {s}")
