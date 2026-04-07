import os
import logging
import json
import argparse
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys
from datetime import datetime

def setup_logging(level=logging.INFO):
    """Setup clean logging format for nano-vllm."""
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'nano_vllm_{timestamp}.log'
    
    formatter = logging.Formatter(
        fmt='%(levelname)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    
    # File handler with timestamped filename
    file_handler = logging.FileHandler(log_filename, mode='w')
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

    sampling_params = SamplingParams(temperature=1.0, max_tokens=128, ignore_eos=False)

    # Load prompts from JSON file
    prompts = load_prompts("short_prompts.json")
    if not prompts:
        print("No prompts loaded, exiting...")
        return

    import time
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Save generated text as JSON array like short_prompts.json
    generated_prompts = []
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        # Only save the generated completion text (256 tokens), not the prompt
        generated_prompts.append(output['text'])
    
    # Save as JSON file
    output_file = "generated_prompts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_prompts, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated prompts saved to: {output_file}")
    print(f"Total sequences generated: {len(outputs)}")
    
    # Display results with token counts
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        prompt_text = tokenizer.decode(prompt) if isinstance(prompt, list) else prompt
        prompt_tokens = len(tokenizer.encode(prompt)) if isinstance(prompt, str) else len(prompt)
        completion_tokens = len(output['token_ids'])
        total_tokens = prompt_tokens + completion_tokens
        
        print(f"\n{'='*80}")
        print(f"SEQUENCE {i+1}/{len(outputs)}")
        print(f"{'='*80}")
        print(f"Total tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
        print(f"Prompt: {prompt_text[:100]}...")
        print(f"\nCompletion ({len(output['text'])} chars):")
        print(output['text'][:500] + "..." if len(output['text']) > 500 else output['text'])


if __name__ == "__main__":
    main()
