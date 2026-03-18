<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM
Here is a very detailed and sophisticated walk through of the nano-vllm codebase which is very helpful for understanding the nano-vLLM architecture.

# Preparation
- prompts
- sampling_params

---
# LLMEngine
==init==
## modelrunner
- for i in rank1~tensor_parallel_size
	- use ctx create an event
	- use ctx create several process(config, i, event)
	- start process(Instantiation of ModelRunner)

- main modelrunner (self.model_runner)
### model_runner init
- get config
- kvcache_block_size
- enforce_eager?
- world_size
- rank
- event
- nccl group
- **link to a gpu**
- **self.model**?
- **load_model**
- sampler
#### warmup_model
- max_num_batched_tokens, max_model_len
- **num_seqs**,seqs
##### run(seqs,True)
- prepare_prefill, prepare_decode,
- input_ids, locations
- run_model
##### run_model()
- directly self.model
- get graph, 
- get graph_vars and update as to the actual settings
- graph.replay()

#### allocate_kv_cache
- as for the gpu the process is in, allocate kv_cache for each layer
- **already updated the config of the num_kvcache_blocks**
- #Q what is the shape of the kv cache, (0,layer_id)
#### shared_memory and group
- synchronization and the memory
## scheduler
- max_num_seqs <mark style="background:#ff4d4f">it is super important as for multi-seqs condition</mark>
- max_num_batched_tokens (<mark style="background:#ff4d4f">I have no idea about it yet</mark>)
- queues
	- waiting
	- running
### BlockManager
#### init
- num_kvcache_blocks, block_size
- **instantiation blocks** according to the actual num of blocks the model_runner has allocated before
- hash_to_block_id diction
- free_block_ids
- used_block_ids

# .generate
## add_request
- use tokenizer to encode the prompt,
- add sampling_params, into a <mark style="background:#fff88f">sequence object</mark>
- **scheduler.add(seq)**---><mark style="background:#fff88f">waiting</mark>.append(seq)

## is_finished
- **scheduler**: no seqs in the waiting or running deque
## step
- **<mark style="background:#fff88f">get</mark> the seqs and prefill** from <mark style="background:#fff88f">Scheduler.schedule()</mark>
### schedule() get seqs and prefill
- num_seqs=0, <mark style="background:#ff4d4f">num_batched_tokens</mark>
- while waiting deque is not empty or num_seqs< max( <mark style="background:#fff88f">many seqs</mark>)
- get one seq from waiting deque
#### check
- max_num_batched_tokens
- block_manager.can_allocate()
	- can_allocate: free_block_ids >= num_blocks
	- #Q <mark style="background:#ff4d4f">since the block already exists, why allocate?</mark>
#### prefill(while)
##### block_manager.allocate(seq)
- from token_ids get hash and ***check if we can get the right <mark style="background:#fff88f">block_id </mark>through the hash***
- num_cached_tokens plus block_size
- if used
	- true: ref_count+=1
	- false: _allocate_block(block_id)
- `_allocate_block`
	- use block_id to instantiate a block<mark style="background:#ff4d4f">(repeated?)</mark>
	- update, free_block, used_block
- block_table.append
- num_batched_tokens+=len(seq)-seq.num_cached_tokens(<mark style="background:#fff88f">many seqs</mark>)

- <mark style="background:#fff88f">attention, num_cached_tokens is a property of a seq</mark>

- status: waiting---><mark style="background:#fff88f">runnning</mark>(change the deque)
- append
- **return** scheduled seqs,<mark style="background:#9254de">True</mark>(prefill batch)
---
get the seqs and pefill=True from last step

### model_runner.call(run) get token_ids
`is_prefill ==True`
#### prepare_prefill(or decode)
for
- `input_ids.extend([cached:])`
- `positions.extend([cached,seqlen])`
- seqlen_q=seqlen-cached
- seqlen_k=seqlen
- cu_seqlens_q(<mark style="background:#fff88f">many seqs</mark>)
- cu_seqlens_k
- max_seqlen_q
- max_seqlen_k
- **skip warmup phase**, block_table has been created
- **slot_mapping**.extend

<mark style="background:#ff4d4f">with cache</mark> not for first prefill time
convert to GPU and set_context
**return input_ids, positions**

- only for rank0, prepare temperatures
#### run_model(ids, posit, is_prefill)
##### directly compute_logits
- <mark style="background:#ff4d4f">what? is_prefill</mark>
##### cuda_graph
- get batch_size
- get_context
- get graph
- set graph variables
- graph.replay()
- model.compute_logits
#### compute_logits
- qwen3.py

sampler
reset_context
**return token_ids**

### scheduler.postprocess(seqs,token_ids)
for : **many seqs**
seq.append(token_id)
<mark style="background:#fff88f">not finished!</mark>

## step loop!
🚀
### Schedule() again!
#### !!!!!!!!!!Decode(while)!!!!!!!!!!!
- get a seq from running
##### memory check
- preempt other running seq or current one
##### block_manager.may_append
- check the new token and decide if it needs allocate new block or calculate hash
##### `running.extendleft(reversed(scheduled_seqs))`
return scheduled_seqs, <mark style="background:#9254de">False</mark>

## finish
- llm_engine, **scheduler**
- sequence status.FINISHED
- block_manager.deallocate(seq)
- running.remove(seq)

# tokenizer.decode
# print!!!!!!!!!!!!
