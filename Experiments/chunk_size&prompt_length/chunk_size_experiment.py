import time
import json
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

def load_prompts(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def call_model(prompt_file: str, chunk_size: int = 512):
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
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, chunk_size=chunk_size)
    sampling_params = SamplingParams(temperature=0.6)
    prompts = load_prompts(prompt_file)
    
    # Measure total time and collect metrics
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()
    
    # Clean up LLM to free the distributed process group and CUDA state
    llm.exit()
    del llm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    total_time = end_time - start_time
    
    # Extract TTFT and calculate throughput
    ttfts = [out.get('ttft', 0) for out in outputs]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    
    # Calculate throughput: total tokens / total time
    total_tokens = sum(len(out.get('token_ids', [])) for out in outputs)
    throughput = total_tokens / total_time if total_time > 0 else 0
    
    result = {
        'chunk_size': chunk_size,
        'prompt_file': prompt_file,
        'num_prompts': len(prompts),
        'avg_ttft': avg_ttft,
        'min_ttft': min(ttfts) if ttfts else 0,
        'max_ttft': max(ttfts) if ttfts else 0,
        'total_time': total_time,
        'total_tokens': total_tokens,
        'throughput_tokens_per_sec': throughput,
        'ttfts': ttfts
    }
    
    print(f"[chunk_size={chunk_size}, prompts={prompt_file}] "
          f"Avg TTFT: {avg_ttft:.3f}s, Throughput: {throughput:.1f} tokens/sec")
    
    return result

def generate_in_batches():
    chunk_sizes = [256, 512, 768, 1024, 2048]
    prompt_lengths = [256, 512, 1024]
    
    all_results = []
    
    for cs in chunk_sizes:
        for pl in prompt_lengths:
            prompt_file = f"{pl}_prompts.json"
            if not os.path.exists(prompt_file):
                print(f"Warning: {prompt_file} not found, skipping...")
                continue
            
            print(f"\nRunning: chunk_size={cs}, prompt_length={pl}")
            result = call_model(prompt_file, cs)
            all_results.append(result)
    
    # Save all results to JSON
    with open('experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Chunk Size':<12} {'Prompt Len':<12} {'Avg TTFT':<12} {'Throughput':<15}")
    print("-"*60)
    for r in all_results:
        print(f"{r['chunk_size']:<12} {r['prompt_file']:<12} {r['avg_ttft']:<12.3f} {r['throughput_tokens_per_sec']:<15.1f}")
    
    print(f"\nResults saved to: experiment_results.json")
    return all_results

if __name__ == "__main__":
    generate_in_batches()
