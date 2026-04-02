import os
import logging
import json
import argparse
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys

def setup_logging(level=logging.INFO):
    """Setup clean logging format for nano-vllm."""
    formatter = logging.Formatter(
        fmt='%(levelname)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('nano_vllm.log', mode='w')
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)
    
    # Quiet down some noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Setup logging
setup_logging(logging.INFO)


def load_prompts(json_file="short_prompts.json"):
    """Load prompts from JSON file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {json_file}")
        return prompts
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return []


def main():
    model_id = "Qwen/Qwen3-0.6B"
    path = "/home/albert/learn/models/Qwen3-0.6B/"
    os.makedirs(path, exist_ok=True)
    if not os.path.isfile(os.path.join(path, "config.json")):
        snapshot_download(
            repo_id=model_id,
            local_dir=path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)  # disable cuda graphs, use eager execution

    sampling_params = SamplingParams(temperature=0.6)

    # Load prompts from JSON file
    prompts = load_prompts("short_prompts.json")
    if not prompts:
        print("No prompts loaded, exiting...")
        return

    import time
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    # end_time = time.time()
    # total_time = end_time - start_time
    
    # Calculate token throughput metrics
    # total_prompt_tokens = sum(len(tokenizer.encode(p)) if isinstance(p, str) else len(p) for p in prompts)
    # total_generated_tokens = sum(len(output['token_ids']) for output in outputs)
    # total_tokens = total_prompt_tokens + total_generated_tokens
    
    # print(f"\n{'='*60}")
    # print(f"PERFORMANCE METRICS")
    # print(f"{'='*60}")
    # print(f"Total time: {total_time:.2f} seconds")
    # print(f"Number of sequences: {len(outputs)}")
    # print(f"\nSequence throughput: {len(outputs)/total_time:.2f} seq/sec")
    # print(f"Token throughput: {total_tokens/total_time:.2f} tokens/sec (input+output)")
    # print(f"Generation throughput: {total_generated_tokens/total_time:.2f} tokens/sec (output only)")
    # print(f"{'='*60}\n")
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        prompt_text = tokenizer.decode(prompt) if isinstance(prompt, list) else prompt
        print(f"\n{'='*80}")
        print(f"SEQUENCE {i+1}/{len(prompts)}")
        print(f"{'='*80}")
        print(f"Prompt: {prompt_text[:100]}...")
        print(f"\nCompletion ({len(output['text'])} chars):")
        print(output['text'][:500] + "..." if len(output['text']) > 500 else output['text'])
        print(f"TTFT: {output.get('ttft', -1):.3f}s")
        # print(f"\nTokens generated: {len(output['token_ids'])}")
        # print(f"Prompt tokens: {len(tokenizer.encode(prompt)) if isinstance(prompt, str) else len(prompt)}")


if __name__ == "__main__":
    main()
