import os
import logging
import json
import argparse
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nano_vllm.log', mode='w')
    ]
)


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

    sampling_params = SamplingParams(temperature=1.0, max_tokens=512, ignore_eos=True)

    # Load prompts from JSON file
    prompts = load_prompts("short_prompts.json")
    if not prompts:
        print("No prompts loaded, exiting...")
        return

    import time
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        prompt_text = tokenizer.decode(prompt) if isinstance(prompt, list) else prompt
        prompt_tokens = len(tokenizer.encode(prompt)) if isinstance(prompt, str) else len(prompt)
        completion_tokens = len(output['token_ids'])
        total_tokens = prompt_tokens + completion_tokens
        
        print(f"\nTotal tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
        print(f"Prompt: {prompt_text[:100]}...")
        print(f"\nCompletion ({len(output['text'])} chars):")
        print(output['text'][:500] + "..." if len(output['text']) > 500 else output['text'])


if __name__ == "__main__":
    main()
