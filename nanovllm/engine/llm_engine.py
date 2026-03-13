"""
Core LLM Engine for nano-vllm inference system.
- set up some configs
- initialize the list for storing process and events
- initialize several workers in rank, 0 is the main worker, and coordinate them

- whole cycle:
  - add request(for scheduler)
  - schedule
  - step, outputing sequence id and the raw token_ids, calculating the performances
  - output 

- generate: wrap the step up, adding features like tqdm progress bar
- and decode into the final text, outputting the result
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    Main engine for LLM serving.
    Handles tensor parallelism, request management, batch scheduling,
    token generation, and cleanup.
    """

    def __init__(self, model, **kwargs):
        #This code filters and validates configuration parameters before creating the Config object:
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # Tensor parallelism setup
        self.ps = []#Empty list to hold worker processes
        self.events = []#Empty list for synchronization events
        ctx = mp.get_context("spawn")#Gets multiprocessing context using "spawn" method (creates new processes rather than forks)
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)

        # Tokenizer and scheduler
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")#Sends a shutdown command to the main model logic (Rank 0).
        del self.model_runner #deletes the rank0 object from memory
        for p in self.ps:#Loops through every worker process (Rank 1, 2, 3...) and waits for them to finish.
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()#Scheduler decides what to process:


        token_ids = self.model_runner.call("run", seqs, is_prefill)#Model execution across all GPUs:
        """
        Sends batch to tensor parallel processes
        Rank 0 coordinates with ranks 1,2,3... via events
        Each process computes its portion of the model
        Returns generated tokens (one per sequence)
        """


        self.scheduler.postprocess(seqs, token_ids)#Handle results and update state:
        """
        Appends generated tokens to sequences
        Checks if sequences are finished (EOS token, max length)
        Frees KV cache memory for completed sequences
        Updates sequence statuses
        """


        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)#positive for prefill, negative for decode
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        Complete generation pipeline:
        - Submits requests
        - Runs generation loop
        - Tracks progress/throughput
        - Returns output dicts with text and token IDs
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)#prompts are a list, not prompt's tokens

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.#initialize throughput counters

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
                    
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # Before: outputs = {seq_id_3: [tokens], seq_id_1: [tokens], seq_id_2: [tokens]}
        # After: outputs = [[tokens_seq_1], [tokens_seq_2], [tokens_seq_3]]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        # Before: outputs = [[token1, token2, token3], [token4, token5]]
        # After: outputs = [
        #   {"text": "Hello world", "token_ids": [token1, token2, token3]},
        #   {"text": "How are", "token_ids": [token4, token5]}
        # ]
        if use_tqdm:
            pbar.close()
        return outputs
        
