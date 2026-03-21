import os
import logging
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys

# Configure logging to both console and file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('nano_vllm_debug.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)


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

    sampling_params = [SamplingParams(temperature=0.6, max_tokens=16),
                       SamplingParams(temperature=0.6, max_tokens=16)]

    shared_prefix = [83482] * 256
    suffix_a = tokenizer.encode(" neural networks summary", add_special_tokens=False)
    suffix_b = tokenizer.encode(" california summary", add_special_tokens=False)
    prompts = [
        shared_prefix + suffix_a,
        shared_prefix + suffix_b,
    ]
    
 
    
    import time
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== Generation completed in {total_time:.2f} seconds ===")
    print(f"Average time per sequence: {total_time/len(prompts):.2f} seconds")
    print(f"Throughput: {len(prompts)/total_time:.2f} sequences/second\n")
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        prompt_text = tokenizer.decode(prompt) if isinstance(prompt, list) else prompt
        print(f"\n{'='*80}")
        print(f"SEQUENCE {i+1}/{len(prompts)}")
        print(f"{'='*80}")
        print(f"Prompt: {prompt_text[:100]}...")
        print(f"\nCompletion ({len(output['text'])} chars):")
        print(output['text'][:500] + "..." if len(output['text']) > 500 else output['text'])
        print(f"\nTokens generated: {len(output['token_ids'])}")


if __name__ == "__main__":
    main()
