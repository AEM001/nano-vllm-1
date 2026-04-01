INFO [nanovllm.engine.model_runner] [ModelRunner] Rank 0 initialized
INFO [nanovllm.engine.model_runner] [ModelRunner] GPU device set to rank 0
`torch_dtype` is deprecated! Use `dtype` instead!
INFO [nanovllm.engine.model_runner] [ModelRunner] Model loaded on rank 0 in 0.12s
INFO [nanovllm.engine.model_runner] [ModelRunner] Allocating KV cache on rank 0...
INFO [nanovllm.engine.model_runner] [ModelRunner] GPU memory: 9.71GB free / 11.61GB total
INFO [nanovllm.engine.model_runner] [ModelRunner] Model weights: 1.42GB
INFO [nanovllm.engine.model_runner] [ModelRunner] KV cache memory: 9.03GB available, block_size=29360128 bytes
INFO [nanovllm.engine.model_runner] [ModelRunner] KV cache blocks: 330
INFO [nanovllm.engine.model_runner] [ModelRunner] KV cache allocated: torch.Size([2, 28, 330, 256, 8, 128])
Loaded 6 prompts from short_prompts.json
INFO [nanovllm.engine.llm_engine] [LLMEngine] Starting generation...
==================== INITIAL PREFILL BATCH ====================
WARNING [nanovllm.engine.scheduler]  !!! Prefill !!!: seq_id0 is prefilling 7 tokens and is gonna be allocated
WARNING [nanovllm.engine.scheduler]  !!! Prefill !!!: seq_id1 is prefilling 9 tokens and is gonna be allocated
WARNING [nanovllm.engine.scheduler]  !!! Prefill !!!: seq_id2 is prefilling 12 tokens and is gonna be allocated
WARNING [nanovllm.engine.scheduler]  !!! Prefill !!!: seq_id3 is prefilling 7 tokens and is gonna be allocated
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 4 and 2
INFO [nanovllm.engine.model_runner] the number of batched tokens: 35
INFO [nanovllm.engine.model_runner] prefill tokens: 35, decode tokens: 0
==================== MIXED BATCH: CONTINUE PREFILL + DECODE ====================
WARNING [nanovllm.engine.scheduler] PARTIALLY PREFILLING: seq_id2 is prefilled 10 tokens(second time)
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 4 and 2
INFO [nanovllm.engine.model_runner] the number of batched tokens: 13
INFO [nanovllm.engine.model_runner] prefill tokens: 10, decode tokens: 3
==================== DECODE-ONLY BATCHES ====================
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 4 and 2
INFO [nanovllm.engine.model_runner] the number of batched tokens: 4
INFO [nanovllm.engine.model_runner] prefill tokens: 0, decode tokens: 4
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 4 and 2
INFO [nanovllm.engine.model_runner] the number of batched tokens: 4
………
INFO [nanovllm.engine.model_runner] the number of batched tokens: 4
INFO [nanovllm.engine.model_runner] prefill tokens: 0, decode tokens: 4
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 4 and 2
INFO [nanovllm.engine.model_runner] the number of batched tokens: 4
INFO [nanovllm.engine.model_runner] prefill tokens: 0, decode tokens: 4
==================== SEQUENCES FINISHED, NEW ONES STARTING ====================
INFO [nanovllm.engine.scheduler] [Scheduler] Seq 0 finished
INFO [nanovllm.engine.scheduler] [Scheduler] Seq 1 finished
INFO [nanovllm.engine.scheduler] [Scheduler] Seq 3 finished
WARNING [nanovllm.engine.scheduler]  !!! Prefill !!!: seq_id4 is prefilling 7 tokens and is gonna be allocated
WARNING [nanovllm.engine.scheduler]  !!! Prefill !!!: seq_id5 is prefilling 12 tokens and is gonna be allocated
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 3 and 0
INFO [nanovllm.engine.model_runner] the number of batched tokens: 20
INFO [nanovllm.engine.model_runner] prefill tokens: 19, decode tokens: 1
==================== FINAL MIXED BATCHES ====================
INFO [nanovllm.engine.scheduler] [Scheduler] Seq 2 finished
WARNING [nanovllm.engine.scheduler] PARTIALLY PREFILLING: seq_id5 is prefilled 7 tokens
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 2 and 0
INFO [nanovllm.engine.model_runner] the number of batched tokens: 8
INFO [nanovllm.engine.model_runner] prefill tokens: 7, decode tokens: 1
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 2 and 0
INFO [nanovllm.engine.model_runner] the number of batched tokens: 2
INFO [nanovllm.engine.model_runner] prefill tokens: 0, decode tokens: 2
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 2 and 0
………
INFO [nanovllm.engine.model_runner] prefill tokens: 0, decode tokens: 2
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 2 and 0
INFO [nanovllm.engine.model_runner] the number of batched tokens: 2
INFO [nanovllm.engine.model_runner] prefill tokens: 0, decode tokens: 2
==================== SEQUENCE COMPLETION ====================
INFO [nanovllm.engine.scheduler] [Scheduler] Seq 4 finished
INFO [nanovllm.engine.scheduler] number of running and waiting seqs: 1 and 0
INFO [nanovllm.engine.model_runner] the number of batched tokens: 1
INFO [nanovllm.engine.model_runner] prefill tokens: 0, decode tokens: 1
INFO [nanovllm.engine.scheduler] [Scheduler] Seq 5 finished

================================================================================
SEQUENCE 1/6
================================================================================
Prompt: The sun sets over the horizon....

Completion (276 chars):
 The sun is casting a shadow on the ground. What is the name of the shadow? The answer is a shadow that appears when a sun is casting a shadow on the ground. This is the name of the shadow on the ground. The answer is a shadow.
Answer:

The answer is a shadow.

**Explanation:
TTFT: 2.515s

================================================================================
SEQUENCE 2/6
================================================================================
Prompt: A cat naps on the windowsill....

Completion (226 chars):
 The cat is on a long table. The cat is on a shelf. The cat is on a chair. The cat is on a shelf. The cat is on a chair. The cat is on the window sill. The cat is on the chair. The cat is on the chair. The cat is on the window
TTFT: 2.515s

================================================================================
SEQUENCE 3/6
================================================================================
Prompt: The old bookstore smelled of paper and ink, a treasure trove of forgotten stories waiting to be redi...

Completion (288 chars):
 This smells of paper and ink, a treasure trove of books and papers. The old bookstore smells of paper and ink, a treasure trove of books and papers. The old bookstore smells of paper and ink, a treasure trove of books and papers. The old bookstore smells of paper and ink, a treasure tro
TTFT: 2.630s

================================================================================
SEQUENCE 4/6
================================================================================
Prompt: Coffee brews in the morning....

Completion (213 chars):
 What is the name of the coffee that they prepare? Also, if you have a hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog, hot dog
TTFT: 2.515s

================================================================================
SEQUENCE 5/6
================================================================================
Prompt: Rain falls on the roof softly....

Completion (269 chars):
 The sun is shining brightly. The birds are singing loudly. A person is walking by the street. The night sky is dark. The wind is blowing gently. The trees are green and tall. The grass is green and green. What is the meaning of the poem?
A. The beauty of nature
B. The
TTFT: 4.564s

================================================================================
SEQUENCE 6/6
================================================================================
Prompt: As the city lights twinkled below, she wondered about the countless lives unfolding in the windows....

Completion (264 chars):
 She was not alone. The city was a place where dreams and dreamers took flight, but she was alone. She wasn't sure if she was dreaming or if she was just thinking about it. She looked up at the sky, and there was no one. No one was around. The city was silent. She
TTFT: 4.586s