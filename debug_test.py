import os
import logging
import sys
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def setup_logging(level=logging.DEBUG):
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

setup_logging(logging.WARNING)

def main():
    path = "/home/albert/learn/models/Qwen3-0.6B/"
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, chunk_size=110, max_num_batched_tokens=128, max_model_len=100)
    
    sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
    
    # Simple short prompt
    prompt = "Hello, my name is"
    
    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {tokenizer.encode(prompt)}")
    
    outputs = llm.generate([prompt], sampling_params)
    
    print(f"\nGenerated text: {outputs[0]['text']}")
    print(f"Generated token IDs: {outputs[0]['token_ids']}")
    print(f"Decoded tokens: {[tokenizer.decode([tid]) for tid in outputs[0]['token_ids']]}")

if __name__ == "__main__":
    main()
