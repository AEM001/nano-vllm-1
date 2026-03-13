HTTP_PROXY=http://127.0.0.1:7897/ \
HTTPS_PROXY=http://127.0.0.1:7897/ \
ALL_PROXY=socks://127.0.0.1:7897/ \
NO_PROXY=localhost,127.0.0.1,::1 \
/home/albert/learn/l-vllm/.venv/bin/python /home/albert/learn/nano-vllm-src/example.py

**Eager mode** (`enforce_eager=True`) means executing operations one-by-one as they're called, rather than optimizing them into compiled graphs.

**Eager vs. Graph mode:**

**Eager mode (what you're using):**
- Operations execute immediately when called
- Easier debugging - can inspect intermediate values
- Slower performance - no optimization
- More memory usage
- Better compatibility/less issues

**Graph/CUDA graph mode:**
- Operations are compiled into optimized execution graphs
- Much faster performance (2-10x speedup)
- Less memory overhead
- Harder debugging - can't easily inspect intermediate steps
- May have compatibility issues

**Why use eager mode:**
- Development/debugging
- When CUDA graphs cause errors
- Models that don't work well with graph optimization
- When you need to step through execution

Your code uses `enforce_eager=True` likely for compatibility or debugging purposes, at the cost of slower inference speed.

---
**`tensor_parallel_size=1`** controls how many GPUs to split the model across using tensor parallelism.

**Tensor parallelism breakdown:**
- **Size 1**: Single GPU - entire model fits on one GPU
- **Size 2**: 2 GPUs - model weights split across 2 GPUs
- **Size 4**: 4 GPUs - model weights split across 4 GPUs
- etc.

**How it works:**
- Model layers are divided into chunks
- Each GPU holds a portion of the weights
- During inference, GPUs communicate to combine results
- Allows running larger models that don't fit on single GPU

**Your setting (`tensor_parallel_size=1`):**
- Uses 1 GPU only
- No inter-GPU communication overhead
- Simpler setup
- Limited by single GPU memory

**When to increase:**
- Model too large for single GPU memory
- You have multiple available GPUs
- Willing to trade some communication overhead for larger model capacity

For your 0.6B model, `tensor_parallel_size=1` is appropriate since it's small enough to fit on a single GPU.

---
What fields(Config) does:

fields() is a function from the dataclasses module
It returns a tuple of Field objects for each field in the @dataclass

For your Config class, it returns fields for: model, max_num_batched_tokens, max_num_seqs, max_model_len, gpu_memory_utilization, tensor_parallel_size, enforce_eager, hf_config, eos, kvcache_block_size, num_kvcache_blocks

{field.name for field in fields(Config)} creates a set containing all valid field names
Result: {"model", "max_num_batched_tokens", "max_num_seqs", "max_model_len", "gpu_memory_utilization", "tensor_parallel_size", "enforce_eager", "hf_config", "eos", "kvcache_block_size", "num_kvcache_blocks"}
This is used to filter out invalid parameters before creating the Config object.

---
This line filters the input kwargs to only include valid Config parameters:

```python
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
```

**How it works:**
- `kwargs.items()` - Gets all key-value pairs passed to [__init__](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/engine/llm_engine.py:9:4-20:31)
- `if k in config_fields` - Only keeps keys that are valid Config field names
- Creates a new dictionary with only the valid parameters

**Example:**
If someone calls:
```python
LLM("model_path", tensor_parallel_size=2, invalid_param="foo", enforce_eager=True)
```

- `kwargs` = `{"tensor_parallel_size": 2, "invalid_param": "foo", "enforce_eager": True}`
- `config_fields` = `{"model", "tensor_parallel_size", "enforce_eager", ...}`
- `config_kwargs` = `{"tensor_parallel_size": 2, "enforce_eager": True}` (invalid_param filtered out)

**Purpose:** Prevents errors from invalid parameters and ensures only supported configuration options reach the Config constructor. This is a defensive programming pattern that makes the API more robust.


# nano-vLLM Configuration System Overview

## Config Class Structure
The `@dataclass Config` defines all valid configuration parameters:

**Required fields:**
- `model: str` - Path to the model directory

**Optional fields with defaults:**
- `max_num_batched_tokens: int = 16384` - Maximum tokens per batch
- `max_num_seqs: int = 512` - Maximum concurrent sequences
- `max_model_len: int = 4096` - Maximum sequence length
- `gpu_memory_utilization: float = 0.9` - GPU memory usage (90%)
- `tensor_parallel_size: int = 1` - Number of GPUs for model splitting
- `enforce_eager: bool = False` - Force eager execution mode
- `hf_config: AutoConfig | None = None` - Hugging Face config
- `eos: int = -1` - End-of-sequence token ID
- `kvcache_block_size: int = 256` - KV cache block size
- `num_kvcache_blocks: int = -1` - Number of KV cache blocks

## Parameter Validation Process

**1. Extract valid field names:**
```python
config_fields = {field.name for field in fields(Config)}
```
- Uses reflection to get all valid Config field names
- Creates set: {"model", "tensor_parallel_size", "enforce_eager", ...}

**2. Filter input parameters:**
```python
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
```
- Only keeps parameters matching valid Config fields
- Filters out invalid/unknown parameters

**3. Create Config object:**
```python
config = Config(model, **config_kwargs)
```

## Post-Initialization Validation
The [__post_init__](cci:1://file:///home/albert/learn/nano-vllm-src/nanovllm/config.py:19:4-25:64) method validates:
- Model directory exists
- KV cache block size is multiple of 256
- tensor_parallel_size between 1-8
- Loads Hugging Face config and adjusts max_model_len
- Ensures max_num_batched_tokens >= max_model_len

## Key Concepts
- **Eager mode**: Execute operations one-by-one vs optimized graphs
- **Tensor parallelism**: Split model across multiple GPUs
- **KV cache**: Stores attention keys/values for efficient generation
- **Batching**: Process multiple sequences simultaneously

---
A Batched Token is a unit of work. It represents one token position being processed for one sequence in the current step.
The max_num_batched_tokens is the sum of tokens being processed across all sequences in that single step.
Here is the formula for every iteration (step):
Current Batched Tokens
=
∑
(
Tokens to process for Sequence
𝑖
)
Current Batched Tokens=∑(Tokens to process for Sequence 
i
​
 )
3. The Two Phases (Crucial Distinction)
This is where people get confused. The number of "batched tokens" contributed by a single request changes depending on what the request is doing.
A. Decoding Phase (Generating new words)
When a sequence is already running and just generating the next word:
It only needs to process 1 token (the new one).
Example: If you have 10 users chatting, and all are waiting for the next word:
10 sequences × 1 token each = 10 Batched Tokens.
B. Prefill Phase (Reading the prompt)
When a new request first comes in, the model must read the entire prompt (e.g., "Write a story about...") to build the memory (KV Cache).
It must process all prompt tokens at once.
Example: If 1 new user joins with a 500-word prompt:
1 sequence × 500 tokens = 500 Batched Tokens.


---
Let's clarify exactly what happens in "One Step" and why max_num_batched_tokens counts only the new work, not the cached history.
**in every forward pass**

Therefore, max_num_batched_tokens is a limit on how many new tokens we are generating/updating in parallel, not how much history we are holding.

The only time max_num_batched_tokens equals the full prompt length is during the Prefill Phase (the very first step of a new request).

>max_num_batched_tokens is indeed the peak computational "workload" allowed in a single moment (step), specifically counting only the tokens that are actively being computed.

---
By limiting only the computed tokens, you manage the system based on latency, not storage.
High Throughput Scenario: If you have 100 users, each generating 1 new token (100 × 1 = 100). This consumes low compute but requires high memory bandwidth (reading 100 sequences of history).
Large Prefill Scenario: If 1 user sends a huge prompt (1000 tokens). This consumes massive compute (1000 × 1 = 1000) immediately, even if memory bandwidth is lower compared to generating small batches.
If max_num_batched_tokens counted the cached history, the scheduler would reject valid requests simply because they had long conversation histories, which would break the continuous batching mechanism entirely.

---
This code is setting up tensor parallelism for distributed model inference across multiple processes:

self.ps = [] - Empty list to hold worker processes
self.events = [] - Empty list for synchronization events
ctx = mp.get_context("spawn") - Gets multiprocessing context using "spawn" method (creates new processes rather than forks)
The loop creates worker processes for tensor parallel ranks 1 to tensor_parallel_size-1. Each worker:

Gets a synchronization event
Starts a ModelRunner process with its rank and event
Process and event are stored in the lists
Finally, it creates the main ModelRunner for rank 0 with all events.

This enables the model to be split across multiple GPU processes for parallel inference.

---
Python has a limitation called the GIL (Global Interpreter Lock) which prevents true parallel computing using standard "threads." To get around this, we use Multiprocessing. This creates entirely separate Python programs (processes) that run at the same time.


**Initializing the Main Process (Rank 0)**
Rank 0: This runs in the current process (not a new child process).
events=self.events: Rank 0 needs to hold all the event flags. Why? Because Rank 0 is the conductor. It needs to wait for Rank 1, 2, and 3 to finish their math before it can combine the results and move to the next layer.

Multiprocessing Events & Processes

*   **Process:** An independent program with **isolated memory**. It cannot see variables inside other processes.
*   **Event:** A shared **synchronization flag** (True/False) that bridges the memory gap between processes.

**The Relationship**
*   Processes use Events to **signal status** without sharing data.
*   **`.set()`**: Turns flag **True** (Signals "I am done").
*   **`.wait()`**: Pauses execution until flag is **True** (Signals "I am waiting for you").

**Why for Tensor Parallelism?**
*   **Synchronization:** Ensures all GPU workers finish Layer $N$ before *any* process starts Layer $N+1$.
*   **Safety:** Acts as a **checkpoint barrier** to prevent data corruption from mismatched calculation speeds.

Rank 0 is Hybrid (Manager + Worker)
You are partially right: The CPU runs the Python logic, but Rank 0 is special because it does two jobs at once.
Job 1: Computation (GPU 0)
Like other ranks, Rank 0 loads a slice of the model onto GPU 0 and does math.
Job 2: Coordination (CPU)
Because Rank 0 is the Main Process (the parent that spawned the others), it naturally holds the events and manages the lifecycle (start/exit).
`p.join()`
Function: Blocks the main program until process p completely terminates.
Behavior: The code pauses at this line and does nothing until the specific worker process finishes and closes.

join() = Wait for process to die.
Critical for cleanup: Prevents memory leaks and zombie processes.
Blocking: Main program pauses here until workers are gone.

---
Line 1: from transformers import AutoConfig
What it does: This imports the AutoConfig class from the Hugging Face transformers library.
What is AutoConfig?
It is a factory class.
In Hugging Face, every model (BERT, GPT-2, Llama, etc.) has a configuration file (usually config.json) that defines its architecture (e.g., number of layers, hidden size, vocabulary size).
AutoConfig allows you to load these configurations automatically without needing to know the specific model class beforehand.
Example: If you point it to a BERT model folder, it returns a BertConfig. If you point it to a GPT-2 folder, it returns a GPT2Config.

Line 2: hf_config: AutoConfig | None = None
What it does: This declares a variable named hf_config and assigns it a Type Hint.
Type Hinting (AutoConfig | None):
This syntax (using the pipe |) is valid in Python 3.10+.
It indicates that hf_config can hold one of two types:
An instance of the AutoConfig class.
The value None (null).
In older Python versions (3.9 and below), this would be written as Optional[AutoConfig] from the typing module.
Initialization (= None):
The variable is initialized as empty. This suggests a lazy loading pattern or an optional dependency. The configuration might not be needed immediately, or it will be loaded later in the code.

---
Line 50: dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
What it does:

Initializes PyTorch Distributed communication between processes
Sets up NCCL (NVIDIA Collective Communications Library) for GPU-to-GPU data transfer
Parameters explained:

"nccl": Communication backend optimized for NVIDIA GPUs
"tcp://localhost:2333": Address of the master process (rank 0) that coordinates all workers
world_size=self.world_size: Total number of processes (e.g., 4)
rank=rank: This process's ID (0,1,2,3)

