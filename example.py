import os
import logging
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging at the application entry point
logging.basicConfig(level=logging.DEBUG,format='%(levelname)s: %(message)s')


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

    sampling_params = SamplingParams(temperature=0.6, max_tokens=4096, ignore_eos=True)
    prompts = [
        "Please provide a comprehensive and detailed introduction to the United States, covering its geography, history, government, economy, culture, and major achievements. Write at least 1000 words."
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
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
