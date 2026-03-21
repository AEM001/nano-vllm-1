import os
import logging
import json
import torch
import torch.distributed as dist
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

def load_prompts(file_path):
    """Load prompts from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Handle both array of strings and array of objects with "text" field
        if data and isinstance(data[0], dict) and 'text' in data[0]:
            prompts = [item['text'] for item in data]
        else:
            prompts = data
    return prompts

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
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT A: PREFILL THROUGHPUT vs BATCH_SIZE")
    print(f"{'='*80}")
    
    # Test different max_num_batched_tokens values
    batch_sizes = [2048, 4096, 8192, 16384, 32768]
    
    # Load prompts from file
    prompts = load_prompts('perf_experiments/01/prompts.json')
    
    print(f"Testing {len(batch_sizes)} different max_num_batched_tokens values")
    print(f"Using {len(prompts)} prompts with total {sum(len(tokenizer.encode(p)) for p in prompts)} tokens")
    print(f"{'='*80}\n")
    
    # Create LLM once with the largest batch size
    print("Creating LLM with largest batch size...")
    llm = LLM(path, 
              enforce_eager=True, 
              tensor_parallel_size=1,
              max_num_batched_tokens=max(batch_sizes),
              max_model_len=2048,  # Set reasonable model length
              max_num_seqs=25)  # High enough to not limit
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n--- Testing max_num_batched_tokens = {batch_size} ---")
        
        # Update the batch size for this test
        llm.scheduler.max_num_batched_tokens = batch_size
        
        sampling_params = [SamplingParams(temperature=0.6, max_tokens=4096)]
        
        import time
        start_time = time.time()
        
        outputs = llm.generate(prompts, sampling_params)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate token throughput metrics - focus on prefill performance
        total_prompt_tokens = sum(len(tokenizer.encode(p)) if isinstance(p, str) else len(p) for p in prompts)
        total_generated_tokens = sum(len(output['token_ids']) for output in outputs)
        
        # Prefill throughput (input tokens processed during prefill phase)
        prefill_throughput = total_prompt_tokens / total_time
        
        # Store results
        result = {
            'batch_size': batch_size,
            'total_time': total_time,
            'prefill_throughput': prefill_throughput,
            'total_prompt_tokens': total_prompt_tokens,
            'total_generated_tokens': total_generated_tokens,
            'num_sequences': len(outputs)
        }
        results.append(result)
        
        print(f"  Time: {total_time:.2f}s, Prefill: {prefill_throughput:.2f} tokens/s")
    
    # Final comparison table
    print(f"\n{'='*80}")
    print(f"EXPERIMENT A: RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Batch Size':<12} {'Time (s)':<10} {'Prefill (t/s)':<12} {'Generated':<10} {'Seqs':<5}")
    print(f"{'-'*80}")
    
    for result in results:
        print(f"{result['batch_size']:<12} {result['total_time']:<10.2f} {result['prefill_throughput']:<12.2f} "
              f"{result['total_generated_tokens']:<10} {result['num_sequences']:<5}")
    
    # Find optimal batch size
    best_result = max(results, key=lambda x: x['prefill_throughput'])
    print(f"\n{'='*80}")
    print(f"OPTIMAL BATCH SIZE: {best_result['batch_size']}")
    print(f"Prefill throughput: {best_result['prefill_throughput']:.2f} tokens/s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
