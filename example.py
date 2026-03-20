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

    sampling_params = [SamplingParams(temperature=0.6, max_tokens=500),
                       SamplingParams(temperature=0.6, max_tokens=40)]
    
    # Multi-sequence prompts - different topics for concurrent processing
    prompts = [
        "In the realm of artificial intelligence and machine learning, neural networks have emerged as a powerful paradigm for solving complex problems. These computational models, inspired by the biological neural networks in animal brains, consist of interconnected layers of artificial neurons that process and transform information through weighted connections. The fundamental building blocks of neural networks include input layers that receive raw data, hidden layers that perform intermediate computations, and output layers that produce final predictions or classifications. Deep learning architectures, which contain multiple hidden layers, have revolutionized fields such as computer vision, natural language processing, and speech recognition. Training these networks typically involves optimization algorithms like gradient descent and backpropagation, which adjust the network's parameters to minimize a loss function that measures the difference between predicted and actual outputs. Regularization techniques such as dropout, batch normalization, and weight decay help prevent overfitting and improve generalization performance. Modern neural network architectures like Transformers have further advanced the field by employing self-attention mechanisms that capture long-range dependencies in sequential data, enabling breakthroughs in language understanding and generation tasks. The scalability of these models has led to the development of large language models with billions of parameters, capable of performing a wide range of natural language tasks with remarkable accuracy and coherence.And what is the difference between inference and training?",
        "introduce California"
        # "Describe the major events of World War II, including the key battles, political developments, and the war's impact on the modern world.",
        # "What is artificial intelligence and machine learning? Explain the difference between supervised, unsupervised, and reinforcement learning.",
        # "Write about climate change: its causes, effects on ecosystems and human society, and potential solutions to mitigate global warming.",
        # "Explain the human digestive system, including the function of each organ and the process of nutrient absorption.",
        # "Discuss the history and evolution of the internet, from ARPANET to modern web technologies and social media platforms."
    ]
    
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]#This code transforms plain text prompts into properly formatted chat messages for the model
    
 
    
    import time
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== Generation completed in {total_time:.2f} seconds ===")
    print(f"Average time per sequence: {total_time/len(prompts):.2f} seconds")
    print(f"Throughput: {len(prompts)/total_time:.2f} sequences/second\n")
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n{'='*80}")
        print(f"SEQUENCE {i+1}/{len(prompts)}")
        print(f"{'='*80}")
        print(f"Prompt: {prompt[:100]}...")
        print(f"\nCompletion ({len(output['text'])} chars):")
        print(output['text'][:500] + "..." if len(output['text']) > 500 else output['text'])
        print(f"\nTokens generated: {len(output['token_ids'])}")


if __name__ == "__main__":
    main()
