import atexit
import logging
from dataclasses import fields
from time import perf_counter
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

# Get module-specific logger (root logger configured in example.py)
logger = logging.getLogger(__name__)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config = Config(model=model, **{k: v for k, v in kwargs.items() if v is not None})
        
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            logger.debug(f"[LLMEngine] Process {i} started")
            self.ps.append(process)
            self.events.append(event)
        logger.debug("[LLMEngine] Starting main model runner")
        self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        logger.debug("[LLMEngine] Initializing scheduler")
        self.scheduler = Scheduler(config)

        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        logger.debug("[LLMEngine] Adding request")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        logger.debug(f"[LLMEngine] Created seq={seq.seq_id} tokens={len(seq)} blocks={seq.num_blocks}")
        
        self.scheduler.add(seq)

    def step(self):
            
        logger.debug("[LLMEngine] Calling scheduler.schedule()")
        time_before_schedule=perf_counter()  
        scheduled_batch = self.scheduler.schedule()
        
        if not scheduled_batch:
            return [], 0

        logger.debug("[LLMEngine] Calling model_runner.call()")
        
        # Extract sequences and determine if this is prefill or decode
        scheduled_seqs = [seq for seq, _ in scheduled_batch]
        is_prefill = any(seq.num_cached_tokens < seq.num_prompt_tokens for seq in scheduled_seqs)
        time_after_schedule = perf_counter()

        token_ids = self.model_runner.call("run", scheduled_batch)

        logger.debug("[LLMEngine] Calling scheduler.postprocess()")
        
        # Convert batch format for postprocess
        scheduled_list = [(seq, _) for seq, _ in scheduled_batch]
        self.scheduler.postprocess(scheduled_list, token_ids)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq, _ in scheduled_batch if seq.is_finished]

        # Calculate tokens processed (positive for prefill, negative for decode)
        if is_prefill:
            num_tokens = sum(token_count for _, token_count in scheduled_batch)
        else:
            num_tokens = -len(scheduled_batch)  # Negative for decode phase
        
        logger.debug("[LLMEngine] Finished step()")
            
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams]
    ) -> list[str]:
        """
        Complete generation pipeline:
        - Submits requests
        - Runs generation loop
        - Tracks progress/throughput
        - Returns output dicts with text and token IDs
        """
        logger.info("[LLMEngine] Starting generation...")
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        seqs = {}  # Track sequences for TTFT
        prefill_throughput = decode_throughput = 0.
        step_count = 0

        logger.debug("[LLMEngine] Starting generation loop...")
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
           
            step_count += 1

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if seq_id not in seqs:
                    # Find the sequence object to get TTFT
                    for seq in self.scheduler.running + self.scheduler.finished:
                        if seq.seq_id == seq_id:
                            seqs[seq_id] = seq
                            break
                    
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        seq_list = [seqs[seq_id] for seq_id in sorted(seqs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "ttft": seq.ttft} for token_ids, seq in zip(outputs, seq_list)]
        return outputs
