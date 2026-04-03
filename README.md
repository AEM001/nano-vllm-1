<p align="center">
<img width="300" src="assets/logo.png">
</p>

# awesome-nano-vllm: Production-Grade vLLM Implementation

An awesome learning project that transforms a minimal vLLM implementation into a production-grade inference engine. This repository documents my journey from understanding the basics to implementing **chunked prefill**, **mixed-batch execution**, **continuous batching**, and **prefix caching**.

## What Makes This Different

| Feature | Original | This Implementation |
|---------|----------|---------------------|
| Prefill | All-or-nothing | **Chunked** - configurable chunk sizes prevent head-of-line blocking |
| Batching | Homogeneous (prefill OR decode) | **Mixed** - prefill + decode in same forward pass |
| Scheduling | Sequence-level | **Token-level** with preemptive scheduling |
| Memory | Static per-sequence | **Paged** with block-level management |
| Prefix Sharing | None | **Hash-based** deduplication |
| Observability | Minimal | **Comprehensive** timing, TTFT tracking, detailed logging |

## Key Contributions

### 1. Chunked Prefill

**Problem**: Original nano-vllm processed entire prompts at once. A 1024-token prompt would monopolize the GPU while short requests waited, causing head-of-line blocking.

**Solution**: Implemented chunked prefill that splits long prompts into configurable chunks.

```python
# Config parameters
chunk_size: int = 512                    # Max tokens per prefill chunk
max_num_partial_prefills: int = 4        # Limit in-progress sequences
max_long_partial_prefills: int = 2       # Special limit for long prompts
long_prefill_threshold: int = 512        # What counts as "long"
```

**How it works**:
1. Sequences enter `RUNNING` state even if only partially prefilled
2. `num_cached_tokens` tracks progress across scheduling steps
3. Sequences continue from where they left off in subsequent steps
4. Budget limits prevent one long prompt from starving others

**Experimental results** (`Experiments/chunk_size&prompt_length/`):
- **83% TTFT reduction** for short prompts with 512+ chunk sizes
- Optimal sweet spot around **1024 tokens** for general workloads
- Diminishing returns beyond 2048 tokens

### 2. Mixed-Batch Execution with the Mask Trick

**Problem**: Separating prefill and decode phases creates pipeline bubbles and underutilizes GPU.

**Solution**: Enable both prefill and decode in the same forward pass using a clever mask-based approach.

```
INFO: prefill tokens: 10, decode tokens: 3
```

**The Mask Trick** - How it works:
```python
# In ModelRunner.prepare():
mask = [-1] * seqlen_q   # Prefill tokens get -1 (skip sampling)
mask = [0]               # Decode tokens get 0 (sample this)

# In run_model():
sample_indices = [
    i for i, (seq, _) in enumerate(scheduled_seqs)
    if seq.num_cached_tokens >= seq.num_prompt_tokens  # Only decode sequences
]
# Prefill sequences simply don't get sampled - they just update KV cache
```

This simple integer flag lets us handle both phases in one forward pass without branching the model code. The mask flows through the attention layers and controls sampling behavior at the output layer.

**Scheduler prioritization**:
1. **First**: Decode sequences (1 token each) for low latency
2. **Second**: Continue partial prefills from running sequences
3. **Third**: Start new sequences from waiting queue
4. **Preempt** lowest-progress sequences if budget exceeded

### 3. Continuous Batching

Sequences dynamically enter and exit the batch:

```
INFO: [Scheduler] Seq 3 finished
WARNING:  !!! Prefill !!!: seq_id4 is prefilling 7 tokens and is gonna be allocated
INFO: number of running and waiting seqs: 4 and 2
```

When a sequence hits EOS or `max_tokens`:
1. Immediately freed via `block_manager.deallocate()`
2. Removed from `running` queue
3. New sequences from `waiting` fill those slots
4. All within the same scheduling step

### 4. Prefix Caching

Block-level deduplication via hashing enables automatic prefix sharing:

```
INFO: MODELRUNNER: !!! prepare prefill block_tables: tensor([[0, 1],
        [0, 2]], device='cuda:0', dtype=torch.int32)
```

Block 0 is reused by the second sequence because it shares the same prefix hash.

**How it works**:
- When a block fills, its content hash is computed
- New sequences with matching prefixes reuse existing blocks
- Copy-on-write when sequences diverge
- Reference counting tracks block usage

### 5. Comprehensive Debug Logging & Observability

This is a **learning project**, and one of the most valuable additions is the extensive instrumentation that makes nearly every internal operation visible. This visibility is crucial for understanding how a production vLLM system actually works.

**What's instrumented:**

```python
# Block table visualization - see exactly how physical memory is allocated
INFO: MODELRUNNER: !!! prepare decode block_tables: tensor([[ 0,  1],
        [ 2, -1]], device='cuda:0', dtype=torch.int32)
INFO: BLOCK_MANAGER: deallocating block 2

# Slot mapping - where each token's K/V gets stored in the cache
DEBUG: MODELRUNNER: slot_mapping for prefill: [0, 1, 2, 3, ...]

# Every scheduling decision
WARNING: [Scheduler]  !!! Prefill !!!: seq_id4 is prefilling 12 tokens and is gonna be allocated
WARNING: PARTIALLY PREFILLING: seq_id{seq.seq_id} is prefilled {len_to_prefill} tokens
WARNING: [Scheduler] Preempting seq={seq.seq_id} due to budget

# Step-by-step batch composition
INFO: prefill tokens: 10, decode tokens: 3
INFO: number of running and waiting seqs: 4 and 2

# Per-operation timing (prefill, decode, sampling)
logger.info(f"[Timing] Prefill: {num_prefill_tokens} tokens, {prefill_time_per_token*1000:.2f}ms/token")
logger.info(f"[Timing] Decode: {num_decode_tokens} tokens, {decode_time_per_token*1000:.2f}ms/token")
logger.info(f"[Timing] Sampling: {num_sampled_tokens} tokens, {sampling_time_per_token*1000:.2f}ms/token")
```

**Why this matters for learning:**
- You can see the exact block table layout after each allocation/deallocation
- Watch how sequences move through WAITING → RUNNING → FINISHED states
- Observe how the mask separates prefill from decode in real batches
- Track where every token's K/V vectors get stored in the paged cache
- Understand why certain sequences get preempted when budgets are exceeded

This level of observability turns the black box of LLM inference into something you can actually follow step-by-step.

## Experiments

I've set up a structured experimentation framework in `Experiments/` to validate and understand the implemented features:

| Experiment | Purpose |
|------------|---------|
| `chunk_size&prompt_length/` | TTFT vs throughput tradeoffs across chunk sizes (256-2048) and prompt lengths |
| `mixed-batch/` | Validation that prefill+decode coexist in same batch |
| `one-long-one-short/` | Head-of-line blocking mitigation |
| `latency/` | End-to-end latency measurements |
| `prefix-cache/` | Block sharing validation |

*This is an ongoing learning process. More experiments will be added as I continue exploring the system's behavior under different workloads.*

## Architecture Deep Dive

### Request Flow

```
Prompt → add_request() → Sequence(waiting)
                           ↓
                    Scheduler.schedule()
                           ↓
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
    Decode seqs    Partial prefills    New prefills
         ↓                 ↓                 ↓
    ModelRunner.prepare() (builds tensors)
                           ↓
              model.compute_logits()
                           ↓
              Sampler (decode only)
                           ↓
              postprocess() → finished?
```

### Key Data Structures

```python
# Sequence state tracking
class Sequence:
    token_ids: list[int]           # All tokens (prompt + generated)
    num_cached_tokens: int        # KV cache progress
    num_prompt_tokens: int        # Original prompt length
    block_table: list[int]        # Physical block indices
    status: SequenceStatus        # WAITING | RUNNING | FINISHED
    ttft: float                   # Time to first token

# Scheduler decisions
scheduled_batch: deque[(Sequence, int)]  # (sequence, num_tokens_this_step)

# GPU tensors built by ModelRunner.prepare()
input_ids:       token IDs to process
positions:       each token's position (for RoPE)
slot_mapping:    where to write K/V in physical cache
mask:           -1=prefill(skip sampling), 0=decode(sample)
cu_seqlens_q/k:  cumulative lengths for flash attention
block_tables:    physical block indices per sequence
```

### Critical Implementation Files

| File | Contribution |
|------|--------------|
| `nanovllm/engine/scheduler.py` | Chunked prefill + continuous batching + preemptive scheduling |
| `nanovllm/engine/model_runner.py` | Mixed-batch tensor preparation + comprehensive timing |
| `nanovllm/engine/llm_engine.py` | TTFT tracking + generation loop instrumentation |
| `nanovllm/layers/attention.py` | `_mixed_attention()` for unified prefill/decode |
| `nanovllm/engine/block_manager.py` | Hash-based prefix deduplication |
| `nanovllm/config.py` | New configuration parameters for all features |

## Configuration Guide

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    model="path/to/model",
    
    # Core throughput limits
    max_num_batched_tokens=2048,    # Token budget per step
    max_num_seqs=16,                # Max concurrent sequences
    
    # Chunked prefill tuning
    chunk_size=1024,                # Tokens per prefill chunk
    max_num_partial_prefills=4,     # In-progress sequence limit
    max_long_partial_prefills=2,      # Long prompt limit
    long_prefill_threshold=512,     # Long prompt definition
    
    # Memory management
    kvcache_block_size=64,          # Block granularity
    gpu_memory_utilization=0.9,     # GPU memory cap
    
    # Performance
    enforce_eager=False,            # Disable CUDA graphs for debugging
)
```

### Tuning Recommendations

**Low-latency serving** (chatbots, interactive):
```python
chunk_size=512
max_num_partial_prefills=2
max_long_partial_prefills=1
```

**High-throughput batching** (offline processing):
```python
chunk_size=2048
max_num_partial_prefills=8
max_long_partial_prefills=4
```

## The Learning Journey

This project demonstrates that production-grade LLM serving isn't about complex abstractions—it is about **precise token-level orchestration** of simple primitives:

1. **Memory**: Fixed-size blocks, not variable per sequence
2. **Scheduling**: Token budget, not sequence count
3. **Batching**: Mixed phases in same batch
4. **Masking**: Simple integer flag controls behavior

The sophistication comes from getting the orchestration right, not from adding layers of abstraction.

## Running the Code

```bash
# Basic generation
python example.py

# Benchmark throughput
python bench.py --prompts prompts.json --chunk-size 1024

# With debug logging
LOG_LEVEL=DEBUG python example.py

# Custom configuration
python -c "
from nanovllm import LLM, SamplingParams
llm = LLM('model/', chunk_size=512, max_num_partial_prefills=2)
outputs = llm.generate(['Hello world'], SamplingParams(max_tokens=100))
print(outputs[0]['text'])
"
```

## Project Structure

```
nano-vllm/
├── nanovllm/
│   ├── engine/
│   │   ├── scheduler.py          # Token-level scheduling with chunked prefill
│   │   ├── model_runner.py       # Mixed-batch execution + timing
│   │   ├── llm_engine.py         # TTFT tracking + orchestration
│   │   ├── block_manager.py      # Paged memory + prefix caching
│   │   └── sequence.py           # Sequence state management
│   ├── layers/
│   │   ├── attention.py          # Unified prefill/decode attention
│   │   ├── sampler.py            # Temperature sampling
│   │   └── ...
│   ├── models/
│   │   └── qwen3.py              # Model architecture
│   ├── config.py                 # All configuration parameters
│   └── llm.py                    # User-facing API
├── Experiments/
│   ├── chunk_size&prompt_length/ # TTFT/throughput analysis
│   ├── mixed-batch/              # Mixed execution validation
│   ├── one-long-one-short/       # Head-of-line blocking test
│   ├── latency/                  # Scheduling overhead measurement
│   └── prefix-cache/             # Block sharing validation
├── example.py                    # Basic usage example
├── bench.py                      # Throughput benchmarking
└── README.md                     # This file
```

## Acknowledgments

This builds on the excellent [nano-vllm](https://github.com/sgl-project/sglang/tree/main/examples/nano-vllm) example from the SGLang project. The original provided a clean foundation; this version adds the production features needed for real-world deployment.

---

*This is a learning project. The code prioritizes understanding over optimization—but the techniques here (chunked prefill, mixed batches, prefix caching) are exactly what powers production systems like vLLM and SGLang.*
