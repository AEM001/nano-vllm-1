"""
LLM Engine - Core orchestration component for nano-vllm inference system.

This module implements the main engine that coordinates:
- Model loading and tensor parallelism across multiple processes
- Request scheduling and batch management
- Token generation and decoding
- Process lifecycle management

Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   LLMEngine     │───▶│   Scheduler      │───▶│   ModelRunner   │
│                 │    │                  │    │                 │
│ - Request mgmt  │    │ - Batch scheduling│    │ - Model exec    │
│ - Tokenization  │    │ - Memory mgmt     │    │ - Parallelism   │
│ - Output decode │    │ - Sequence state  │    │ - KV cache      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
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
    Main inference engine that orchestrates the entire LLM serving pipeline.
    
    The engine manages:
    1. **Tensor Parallelism**: Spawns multiple processes for distributed model execution
    2. **Request Management**: Accepts and queues generation requests
    3. **Batch Scheduling**: Coordinates efficient batching of requests
    4. **Token Generation**: Drives the step-by-step generation process
    5. **Resource Cleanup**: Ensures proper shutdown of all processes
    
    Process Flow:
    User Request → Tokenization → Scheduler → ModelRunner → Token Decoding → User Output
    """

    def __init__(self, model, **kwargs):
        """
        Initialize the LLM engine with tensor parallelism support.
        
        Args:
            model (str): Model name/path for loading
            **kwargs: Configuration parameters passed to Config
            
        Process Setup:
        - Main process (rank 0): Handles coordination and primary model execution
        - Worker processes (rank 1..N-1): Handle tensor-parallel model shards
        - Events: Synchronization primitives for inter-process communication
        """
        # Extract only valid Config fields from kwargs
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        # Initialize tensor parallelism infrastructure
        self.ps = []      # Worker process handles
        self.events = []  # Synchronization events
        ctx = mp.get_context("spawn")  # Use spawn context for CUDA safety
        
        # Spawn worker processes for tensor parallel ranks 1..N-1
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # Synchronization event for this worker
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # Initialize main model runner (rank 0) with worker events
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # Load tokenizer and set EOS token
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        # Initialize request scheduler
        self.scheduler = Scheduler(config)
        
        # Register cleanup handler for graceful shutdown
        atexit.register(self.exit)

    def exit(self):
        """
        Gracefully shutdown all processes and clean up resources.
        
        Shutdown Sequence:
        1. Signal main model runner to exit
        2. Delete main model runner to free CUDA memory
        3. Wait for all worker processes to complete
        """
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        Add a new generation request to the scheduler.
        
        Args:
            prompt: Either a text string or list of token IDs
            sampling_params: Parameters controlling generation behavior
            
        Process:
        1. Convert string prompts to token IDs using tokenizer
        2. Create Sequence object with prompt and sampling params
        3. Submit to scheduler for batch processing
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        Execute one generation step for the current batch.
        
        This is the core execution loop that handles both prefill (prompt processing)
        and decode (token generation) phases.
        
        Returns:
            tuple: (finished_sequences, token_count)
                - finished_sequences: List of (seq_id, token_ids) for completed requests
                - token_count: Positive for prefill tokens, negative for decode tokens
        
        Step Flow:
        1. **Schedule**: Get next batch of sequences from scheduler
        2. **Execute**: Run model inference via ModelRunner
        3. **Postprocess**: Update sequence states with generated tokens
        4. **Collect**: Extract finished sequences for output
        """
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """
        Check if all generation requests are completed.
        
        Returns:
            bool: True if no active sequences remain
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        High-level generation interface that handles the complete generation pipeline.
        
        This method orchestrates the entire generation process from request submission
        to final output decoding, with progress tracking and throughput monitoring.
        
        Args:
            prompts: List of text prompts or pre-tokenized sequences
            sampling_params: Single params object or list (one per prompt)
            use_tqdm: Whether to show progress bar and throughput metrics
            
        Returns:
            List of dictionaries containing:
                - "text": Decoded generated text
                - "token_ids": Raw token ID sequences
            
        Generation Pipeline:
        1. **Setup**: Initialize progress tracking and parameter validation
        2. **Submit**: Add all requests to the scheduler
        3. **Execute Loop**: Repeatedly call step() until all sequences finish
        4. **Monitor**: Track prefill/decode throughput in real-time
        5. **Collect**: Gather finished outputs and decode to text
        6. **Cleanup**: Close progress bar and return results
        """
        # Initialize progress tracking
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # Normalize sampling parameters to list format
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # Submit all requests to scheduler
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # Generation loop with throughput monitoring
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        while not self.is_finished():
            # Time the generation step for throughput calculation
            t = perf_counter()
            output, num_tokens = self.step()
            
            # Update throughput metrics for progress display
            if use_tqdm:
                if num_tokens > 0:  # Prefill phase
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:  # Decode phase
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # Collect finished sequences
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # Convert outputs to ordered list and decode to text
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        # Clean up progress tracking
        if use_tqdm:
            pbar.close()
        return outputs
