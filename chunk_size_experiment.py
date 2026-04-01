#!/usr/bin/env python3
"""
Chunk Size Tuning Experiment for nano-vLLM

This script measures the impact of different prefill chunk sizes on:
- TTFT (Time To First Token)
- Throughput (sequences per second)

Usage:
    python chunk_size_experiment.py
"""

import os
import sys
import json
import time
import logging
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("chunk_exp")

# Suppress noisy third-party loggers
for noisy_logger in ["transformers", "huggingface_hub", "torch.distributed"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment run."""
    chunk_size: int
    prompt_length: int
    num_prompts: int = 4
    max_tokens: int = 64
    temperature: float = 0.6


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    ttfts: List[float]
    total_time: float
    actual_prompt_lengths: List[int]
    success: bool = True
    error: str = ""
    
    @property
    def avg_ttft(self) -> float:
        return sum(self.ttfts) / len(self.ttfts) if self.ttfts else 0.0
    
    @property
    def throughput(self) -> float:
        return self.config.num_prompts / self.total_time if self.total_time > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "chunk_size": self.config.chunk_size,
            "prompt_length": self.config.prompt_length,
            "num_prompts": self.config.num_prompts,
            "avg_ttft": self.avg_ttft,
            "throughput": self.throughput,
            "total_time": self.total_time,
            "ttfts": self.ttfts,
            "actual_prompt_lengths": self.actual_prompt_lengths,
            "success": self.success,
            "error": self.error
        }


def generate_experiment_script(model_path: str, config: ExperimentConfig) -> str:
    """Generate the Python script to run a single experiment in isolation."""
    return f'''
import json
import time
import torch
import logging
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

# Disable torch.compile to avoid Dynamo errors
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()

# Suppress logging
logging.basicConfig(level=logging.WARNING)

def generate_prompt(tokenizer, target_length: int, prompt_idx: int) -> str:
    """Generate a prompt of approximately target_length tokens."""
    segments = []
    segment_idx = 0
    while True:
        segment = (
            "Prompt " + str(prompt_idx) +
            " segment " + str(segment_idx) +
            ": The future of artificial intelligence is promising. "
        )
        segments.append(segment)
        prompt = "".join(segments)
        if len(tokenizer.encode(prompt)) >= target_length:
            break
        segment_idx += 1
    
    # Trim to exact length
    tokens = tokenizer.encode(prompt)
    if len(tokens) > target_length:
        tokens = tokens[:target_length]
        prompt = tokenizer.decode(tokens)
    return prompt

def run():
    tokenizer = AutoTokenizer.from_pretrained("{model_path}")
    prompts = [generate_prompt(tokenizer, {config.prompt_length}, i) 
               for i in range({config.num_prompts})]
    actual_lengths = [len(tokenizer.encode(p)) for p in prompts]
    
    llm = LLM(
        "{model_path}",
        enforce_eager=True,
        chunk_size={config.chunk_size},
        max_num_batched_tokens=4096,
        max_num_seqs=16,
        max_model_len=4096,
    )
    
    sampling_params = SamplingParams(
        temperature={config.temperature},
        max_tokens={config.max_tokens}
    )
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start
    
    ttfts = [o.get('ttft', -1) for o in outputs]
    
    result = {{
        'ttfts': ttfts,
        'total_time': total_time,
        'actual_prompt_lengths': actual_lengths,
        'success': True,
        'error': ''
    }}
    print("RESULT_JSON:" + json.dumps(result))

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        result = {{
            'ttfts': [],
            'total_time': 0,
            'actual_prompt_lengths': [],
            'success': False,
            'error': str(e) + "\\n" + traceback.format_exc()
        }}
        print("RESULT_JSON:" + json.dumps(result))
'''


def run_single_experiment(model_path: str, config: ExperimentConfig) -> ExperimentResult:
    """Execute a single experiment configuration in an isolated subprocess."""
    script = generate_experiment_script(model_path, config)
    
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        timeout=300  # 5 minute timeout
    )
    
    # Parse RESULT_JSON from output
    for line in reversed(result.stdout.strip().split('\n')):
        if line.startswith('RESULT_JSON:'):
            try:
                data = json.loads(line[12:])
                return ExperimentResult(
                    config=config,
                    ttfts=data.get('ttfts', []),
                    total_time=data.get('total_time', 0.0),
                    actual_prompt_lengths=data.get('actual_prompt_lengths', []),
                    success=data.get('success', False),
                    error=data.get('error', '')
                )
            except json.JSONDecodeError:
                continue
    
    # Failed to parse output
    error_msg = f"Failed to parse result. stdout: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
    return ExperimentResult(
        config=config,
        ttfts=[],
        total_time=0.0,
        actual_prompt_lengths=[],
        success=False,
        error=error_msg
    )


def run_experiment_suite(
    model_path: str,
    chunk_sizes: List[int],
    prompt_lengths: List[int],
    num_prompts: int = 4
) -> List[ExperimentResult]:
    """Run a full suite of experiments."""
    configs = [
        ExperimentConfig(chunk_size=cs, prompt_length=pl, num_prompts=num_prompts)
        for cs in chunk_sizes
        for pl in prompt_lengths
    ]
    
    logger.info(f"Starting experiment suite with {len(configs)} configurations")
    logger.info(f"Chunk sizes: {chunk_sizes}")
    logger.info(f"Prompt lengths: {prompt_lengths}")
    
    results: List[ExperimentResult] = []
    
    for i, config in enumerate(configs, 1):
        logger.info(f"[{i}/{len(configs)}] chunk_size={config.chunk_size}, "
                   f"prompt_length={config.prompt_length}")
        
        result = run_single_experiment(model_path, config)
        
        if result.success:
            logger.info(f"  Success: TTFT={result.avg_ttft:.3f}s, "
                       f"Throughput={result.throughput:.2f} seq/s")
            results.append(result)
        else:
            logger.error(f"  Failed: {result.error[:150]}")
            results.append(result)  # Include failed results for analysis
    
    return results


def print_detailed_results(results: List[ExperimentResult]):
    """Print a detailed table of all experiment results."""
    print("\n" + "=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)
    print(f"{'Chunk':>8} {'Prompt':>8} {'Actual':>8} {'TTFT(s)':>10} {'Throughput':>12} {'Time(s)':>10} {'Status':>10}")
    print("-" * 100)
    
    for r in sorted(results, key=lambda x: (x.config.chunk_size, x.config.prompt_length)):
        avg_actual = sum(r.actual_prompt_lengths) / len(r.actual_prompt_lengths) if r.actual_prompt_lengths else 0
        status = "OK" if r.success else "FAIL"
        print(f"{r.config.chunk_size:>8} {r.config.prompt_length:>8} {avg_actual:>8.0f} "
              f"{r.avg_ttft:>10.3f} {r.throughput:>12.2f} {r.total_time:>10.3f} {status:>10}")
    
    print("=" * 100)


def print_summary_by_chunk(results: List[ExperimentResult]):
    """Print summary statistics grouped by chunk size."""
    by_chunk = defaultdict(list)
    for r in results:
        if r.success:
            by_chunk[r.config.chunk_size].append(r)
    
    if not by_chunk:
        print("\nNo successful results to summarize.")
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY BY CHUNK SIZE")
    print("=" * 80)
    print(f"{'Chunk Size':>12} {'Experiments':>12} {'Avg TTFT(s)':>14} {'Avg Throughput':>16}")
    print("-" * 80)
    
    for chunk_size in sorted(by_chunk.keys()):
        chunk_results = by_chunk[chunk_size]
        avg_ttft = sum(r.avg_ttft for r in chunk_results) / len(chunk_results)
        avg_throughput = sum(r.throughput for r in chunk_results) / len(chunk_results)
        print(f"{chunk_size:>12} {len(chunk_results):>12} {avg_ttft:>14.3f} {avg_throughput:>16.2f}")
    
    print("=" * 80)


def print_summary_by_prompt(results: List[ExperimentResult]):
    """Print summary statistics grouped by prompt length."""
    by_prompt = defaultdict(list)
    for r in results:
        if r.success:
            by_prompt[r.config.prompt_length].append(r)
    
    if not by_prompt:
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY BY PROMPT LENGTH")
    print("=" * 80)
    print(f"{'Prompt Len':>12} {'Experiments':>12} {'Avg TTFT(s)':>14} {'Avg Throughput':>16}")
    print("-" * 80)
    
    for prompt_length in sorted(by_prompt.keys()):
        prompt_results = by_prompt[prompt_length]
        avg_ttft = sum(r.avg_ttft for r in prompt_results) / len(prompt_results)
        avg_throughput = sum(r.throughput for r in prompt_results) / len(prompt_results)
        print(f"{prompt_length:>12} {len(prompt_results):>12} {avg_ttft:>14.3f} {avg_throughput:>16.2f}")
    
    print("=" * 80)


def save_results(results: List[ExperimentResult], output_path: str = "experiment_results.json"):
    """Save results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "results": [r.to_dict() for r in results]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def verify_model(path: str) -> bool:
    """Verify that the model exists at the given path."""
    config_file = Path(path) / "config.json"
    if not config_file.exists():
        logger.error(f"Model not found at {path}")
        logger.error("Please ensure the model is downloaded.")
        return False
    return True


def main():
    """Main entry point for the experiment."""
    model_path = "/home/albert/learn/models/Qwen3-0.6B/"
    
    if not verify_model(model_path):
        return 1
    
    # Experiment parameters
    chunk_sizes = [128, 256, 512, 1024]
    prompt_lengths = [256, 512, 1024]
    num_prompts = 4
    
    logger.info("=" * 60)
    logger.info("Chunk Size Tuning Experiment")
    logger.info("=" * 60)
    
    # Run experiments
    start_time = time.time()
    results = run_experiment_suite(model_path, chunk_sizes, prompt_lengths, num_prompts)
    suite_time = time.time() - start_time
    
    # Print results
    print_detailed_results(results)
    print_summary_by_chunk(results)
    print_summary_by_prompt(results)
    
    # Save results
    save_results(results)
    
    # Final summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"Experiment complete: {successful}/{len(results)} configurations succeeded")
    logger.info(f"Total time: {suite_time:.1f}s")
    
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
