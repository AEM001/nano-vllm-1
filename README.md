<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch. Supports continuous batching, chunked prefill, and mixed-batch execution.

## Features

- **Continuous Batching** - Dynamic batch composition allowing sequences to enter/exit mid-generation
- **Chunked Prefill** - Process long prompts in chunks to avoid blocking short requests  
- **Mixed-Batch Execution** - Run prefill and decode operations in a single forward pass
- **PagedAttention** - Efficient KV cache management with block-based memory allocation

## Installation

```bash
git clone <repo-url>
cd nano-vllm-src
pip install -e .
```

Requirements: Python 3.10-3.12, CUDA-capable GPU

## Quick Start

```python
from nanovllm import LLM, SamplingParams

llm = LLM("path/to/Qwen3-0.6B/", enforce_eager=True)
sampling_params = SamplingParams(temperature=0.6, max_tokens=1024)
outputs = llm.generate(["Hello, how are you?"], sampling_params)
print(outputs[0]['text'])
```

## Example

```bash
python example.py
```

## API

### LLM
```python
LLM(
    path: str,                    # Model path
    enforce_eager: bool = False,  # Disable CUDA graphs
    tensor_parallel_size: int = 1  # TP size
)

llm.generate(prompts: list[str], sampling_params: SamplingParams) -> list[dict]
```

### SamplingParams
```python
SamplingParams(
    temperature: float = 0.3,     # Sampling temperature
    max_tokens: int = 64,          # Max tokens to generate
    ignore_eos: bool = False       # Continue past EOS token
)
```

## Architecture

```
nanovllm/
├── engine/          # Core execution engine
│   ├── llm_engine.py       # Main LLM interface
│   ├── scheduler.py        # Request scheduling & batching
│   ├── model_runner.py     # Model execution
│   ├── block_manager.py    # KV cache block management
│   └── sequence.py         # Sequence state tracking
├── layers/          # Model layers
│   ├── attention.py        # PagedAttention implementation
│   ├── linear.py           # Custom fused linear layers
│   └── ...
└── models/          # Model implementations
    └── qwen3.py            # Qwen3 model architecture
```

## Original Project

Based on [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).

## License

MIT
