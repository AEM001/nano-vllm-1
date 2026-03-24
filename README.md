<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM: Extended Edition

> **Original Repository**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)  
> **This Fork**: Enhanced with **Continuous Batching**, **Chunked Prefill**, and **Mixed-Batch Execution**

## 🎯 What's New in This Fork

This repository extends the original nano-vLLM with production-grade features inspired by real vLLM:

### ✨ Major Features Added

1. **🔄 Continuous Batching** - Dynamic batch composition that allows sequences to enter/exit mid-generation
2. **📦 Chunked Prefill** - Process long prompts in chunks to avoid blocking short requests
3. **🎭 Mixed-Batch Execution** - Run prefill and decode operations in a single forward pass
4. **⚡ Preemption Support** - Evict sequences when GPU memory is tight
5. **📊 Comprehensive Logging** - Detailed scheduler and execution tracking

---

## 🚀 Quick Start

### Installation
```bash
git clone <this-repo>
cd nano-vllm-src
pip install -e .
```

### Basic Usage
```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    path="/path/to/Qwen3-0.6B/",
    chunk_size=1024,  # Chunked prefill size
    max_num_seqs=8    # Max concurrent sequences
)

prompts = [
    "Explain machine learning",
    "What is continuous batching?"
]

sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)
```

### Run Tests
```bash
# Basic continuous batching test
python test_continuous_batching.py

# Full example with multiple sequences
python example.py
```

---

## 📚 Architecture Overview

### Sequence Status Flow
```
WAITING → PARTIAL_PREFILL → FULL_PREFILL → DECODE → FINISHED
           (chunked)          (last chunk)
```

### Key Components Modified

#### 1. **Scheduler** (`nanovllm/engine/scheduler.py`)
- **Continuous batching logic**: Dynamically schedules waiting and running sequences
- **Chunked prefill**: Breaks long prompts into `chunk_size` token chunks
- **Preemption**: Evicts sequences when memory is tight
- **Mixed-batch support**: Handles PARTIAL_PREFILL + DECODE in same batch

#### 2. **ModelRunner** (`nanovllm/engine/model_runner.py`)
- **Mixed-batch preparation**: Creates masks to distinguish prefill vs decode tokens
- **Context management**: Extended with `query_mask` and `seq_mask` for routing
- **Selective sampling**: Only samples tokens from DECODE/FULL_PREFILL sequences

#### 3. **Attention Layer** (`nanovllm/layers/attention.py`)
- **Mixed attention path**: Routes prefill and decode tokens through appropriate kernels
- **Token-level routing**: Uses masks to split batch and scatter results back
- **Fallback implementations**: Python-level mixed attention for learning purposes

#### 4. **Sequence** (`nanovllm/engine/sequence.py`)
- **New statuses**: `PARTIAL_PREFILL`, `FULL_PREFILL` for chunked processing
- **Progress tracking**: `num_cached_tokens` tracks prefill progress

---

## 🔬 Experiments & Documentation

### `/Experiments/`
- **prefix-cache/**: Prefix caching experiments and analysis
- **one-long-one-short/**: Latency comparison with/without continuous batching
- Performance profiling scripts

### `/Docs/`
- **model_runner.md**: Deep dive into ModelRunner architecture
- **scheduler.md**: Scheduler logic and batching strategies
- **block_manager.md**: KV cache block management
- **qwen3.md**: Model architecture walkthrough
- **RMS_Norm.md**: RMSNorm implementation details

### Key Documents
- **CONTINUOUS_BATCHING_FIXES.md**: Detailed changelog of all fixes applied
- **debug.md**: Debugging notes and troubleshooting

---

## 🎓 How It Works: Mixed-Batch Execution

### The Problem
Original nano-vLLM processed prefill and decode separately:
```python
# Old approach
if has_prefill:
    run_prefill_batch()
if has_decode:
    run_decode_batch()
```

### The Solution
Mixed-batch execution processes both in **one forward pass**:

```python
# New approach
def prepare(seqs):
    for seq in seqs:
        if seq.status == PARTIAL_PREFILL:
            mask.append([-1] * chunk_size)  # Mark as prefill
        elif seq.status == DECODE:
            mask.append([0])  # Mark as decode
    return input_ids, positions, mask

def run_model(input_ids, positions, mask):
    hidden_states = model(input_ids, positions, mask)
    # Mask filters out partial prefill tokens before sampling
    return compute_logits(hidden_states)
```

### Attention Routing
```python
def _mixed_attention(q, k_cache, v_cache):
    # Split by mask
    prefill_idx = (query_mask == -1).nonzero()
    decode_idx = (query_mask == 0).nonzero()
    
    # Route to appropriate kernels
    if prefill_idx.numel() > 0:
        prefill_out = causal_attention(q[prefill_idx], ...)
        output[prefill_idx] = prefill_out
    
    if decode_idx.numel() > 0:
        decode_out = kv_cache_attention(q[decode_idx], ...)
        output[decode_idx] = decode_out
    
    return output
```

---

## 📊 Performance Benefits

### Continuous Batching Impact
```
Scenario: 1 long prompt (1024 tokens) + 1 short prompt (14 tokens)

Without continuous batching:
- Long prompt prefills completely (blocks GPU)
- Short prompt waits
- Total time: ~4.5s

With continuous batching:
- Both start together
- Mixed prefill + decode batches
- Total time: ~2.6s (42% faster!)
```

### Key Metrics
- **Latency**: Short prompts don't wait for long prefills
- **Throughput**: GPU utilization stays high
- **Memory**: Efficient KV cache sharing via PagedAttention

---

## 🔧 Configuration

### Scheduler Settings
```python
Config(
    chunk_size=1024,           # Prefill chunk size
    max_num_seqs=8,            # Max concurrent sequences
    max_num_batched_tokens=2048  # Token budget per step
)
```

### Tuning Guidelines
- **chunk_size**: Smaller = more batching opportunities, larger = fewer iterations
- **max_num_seqs**: Higher = more concurrency, but needs more memory
- **max_num_batched_tokens**: Controls total tokens processed per step

---

## 🆚 Comparison to Real vLLM

| Feature | This Implementation | Real vLLM |
|---------|--------------------|-----------|
| **Continuous Batching** | ✅ Python-level | ✅ CUDA-optimized |
| **Chunked Prefill** | ✅ Same approach | ✅ Same approach |
| **Mixed Batching** | ✅ Fallback kernels | ✅ PagedAttention kernel |
| **Preemption** | ✅ Basic support | ✅ Advanced strategies |
| **FlashAttention** | ⚠️ Pure prefill/decode only | ✅ Mixed-batch support |
| **CUDA Graphs** | ⚠️ Decode-only | ✅ Mixed-batch graphs |

### What's Different
- **Attention routing**: Python-level vs unified CUDA kernel
- **Performance**: Educational vs production-optimized
- **Complexity**: ~1,500 lines vs ~50,000 lines

---

## 📖 Learning Path

### For Beginners
1. Read `/Docs/model_runner.md` - Understand the execution flow
2. Run `test_continuous_batching.py` - See it in action
3. Add debug prints in `scheduler.py` - Watch batching decisions

### For Advanced Users
1. Study `_mixed_attention()` in `attention.py`
2. Implement FlashAttention for mixed batches
3. Profile with `torch.profiler` to find bottlenecks
4. Replace Python routing with custom CUDA kernel

---

## 🐛 Known Limitations

1. **No FlashAttention for mixed batches** - Falls back to slower kernels
2. **Python-level routing** - Slower than unified CUDA kernel
3. **CUDA graphs disabled for mixed batches** - Only works for pure decode
4. **Basic preemption** - Loses prefill progress when evicting

---

## 🎯 Future Improvements

- [ ] Custom CUDA kernel for mixed attention
- [ ] FlashAttention wrapper for mixed batches
- [ ] CUDA graph support for common mixed patterns
- [ ] Save/restore prefill progress during preemption
- [ ] Speculative decoding support
- [ ] Multi-LoRA batching

---

## 📝 Git History

### Major Commits
- `c2810ae` - **Mixed-batch prefill** (final working version)
- `99ad6b3` - **Chunked prefill and continuous batching** (initial success)
- `0992d26` - **Preemption support**
- `52766ef` - **Prefix cache experiments**
- `23f2ab4` - **Multi-sequence support with comprehensive logs**

---

## 🙏 Acknowledgments

- **Original nano-vLLM**: [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- **Inspiration**: [vLLM Project](https://github.com/vllm-project/vllm)
- **Papers**: 
  - [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
  - [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)

---

## 📄 License

Same as original nano-vLLM repository.

---

## 🤝 Contributing

This is a learning/research fork. Feel free to:
- Open issues for bugs or questions
- Submit PRs for improvements
- Use this code for educational purposes

---

**Status**: ✅ Fully functional continuous batching with mixed-batch execution!  
**Performance**: ~40% latency reduction for mixed workloads  
**Code Quality**: Production-ready architecture, educational implementation
