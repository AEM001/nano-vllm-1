import os
import logging
import sys
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def setup_logging(level=logging.INFO):
    """Setup clean logging format for nano-vllm."""
    formatter = logging.Formatter(
        fmt='%(levelname)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []
    root_logger.addHandler(console)
    
    logging.getLogger("transformers").setLevel(logging.WARNING)

setup_logging(logging.INFO)

def main():
    path = "/home/albert/learn/models/Qwen3-0.6B/"
    
    if not os.path.isdir(path):
        print(f"Model path {path} does not exist!")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, chunk_size=110, max_num_batched_tokens=128, max_model_len=100)
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=50)
    
    # Test with 2 prompts of different lengths
    prompts = [
    "The rapid advancement of artificial intelligence has fundamentally transformed how we approach complex problem solving in modern society. From healthcare diagnostics to financial modeling, algorithms now process vast datasets with unprecedented speed and accuracy. However, this technological leap brings significant ethical considerations regarding privacy, bias, and employment displacement that policymakers must address urgently. As we integrate these systems deeper into daily life, ensuring transparency and accountability becomes paramount for maintaining public trust. The future landscape will likely depend on our ability to balance innovation with robust regulatory frameworks that protect individual rights while fostering continued growth and development in the global digital economy.",
    "Running continuous batching tests requires careful monitoring of system memory and latency metrics.",
    "is porn harmful or ok?"
]
    
    print(f"\n{'='*60}")
    print(f"Testing Continuous Batching")
    print(f"{'='*60}")
    print(f"Prompt 1 length: {len(tokenizer.encode(prompts[0]))} tokens")
    print(f"Prompt 2 length: {len(tokenizer.encode(prompts[1]))} tokens")
    print(f"Prompt 3 length: {len(tokenizer.encode(prompts[2]))} tokens")

    print(f"{'='*60}\n")
    
    import time
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Number of sequences: {len(outputs)}")
    print(f"{'='*60}\n")
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\nSequence {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Generated: {output['text']}")
        print(f"Tokens: {len(output['token_ids'])}")

if __name__ == "__main__":
    main()
