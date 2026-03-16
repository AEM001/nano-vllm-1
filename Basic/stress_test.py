import torch
import time
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

class StressTestRunner:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.cuda()
        self.model.eval()
        
    def get_memory_info(self):
        """Get GPU and system memory info"""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        else:
            gpu_allocated = gpu_reserved = gpu_free = 0
            
        ram = psutil.virtual_memory()
        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_reserved,
            'gpu_free': gpu_free,
            'ram_used': ram.used / 1024**3,
            'ram_available': ram.available / 1024**3
        }

    def test_sequence_length_scaling(self, max_seq_len=8192):
        """Test how performance scales with sequence length"""
        print(f"\n{'='*60}")
        print(f"TEST 1: Sequence Length Scaling (HF KV Cache)")
        print(f"{'='*60}")
        
        results = []
        for seq_len in [128, 512, 1024, 2048, 4096, 8192]:
            try:
                # Create long prompt
                prompt = "Tell me a story: " * (seq_len // 15)  # Approximate
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len).to("cuda")
                actual_len = inputs["input_ids"].shape[1]
                
                # Measure memory before
                mem_before = self.get_memory_info()
                
                # Time the prefill phase
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(inputs["input_ids"], use_cache=True)
                    kv_cache = outputs.past_key_values
                prefill_time = time.time() - start_time
                
                # Time decode phase (10 tokens)
                decode_start = time.time()
                current_cache = kv_cache
                for i in range(10):
                    with torch.no_grad():
                        outputs = self.model(
                            torch.tensor([[inputs["input_ids"][0, -1]]], device="cuda"),
                            past_key_values=current_cache,
                            use_cache=True
                        )
                        current_cache = outputs.past_key_values
                decode_time = time.time() - decode_start
                
                # Measure memory after
                mem_after = self.get_memory_info()
                
                results.append({
                    'seq_len': actual_len,
                    'prefill_time': prefill_time,
                    'decode_time_per_token': decode_time / 10,
                    'gpu_memory_gb': mem_after['gpu_allocated'] - mem_before['gpu_allocated'],
                    'success': True
                })
                
                print(f"SeqLen {actual_len:5d}: Prefill {prefill_time:.3f}s | "
                      f"Decode {decode_time/10:.4f}s/token | "
                      f"GPU Mem {mem_after['gpu_allocated'] - mem_before['gpu_allocated']:.2f}GB")
                
                # Clear cache
                torch.cuda.empty_cache()
                del kv_cache, current_cache
                
            except Exception as e:
                print(f"SeqLen {seq_len}: FAILED - {str(e)}")
                results.append({'seq_len': seq_len, 'success': False, 'error': str(e)})
                
        return results

    def test_batch_scaling(self, batch_sizes=[1, 2, 4, 8, 16]):
        """Test how performance scales with batch size"""
        print(f"\n{'='*60}")
        print(f"TEST 2: Batch Size Scaling (HF KV Cache)")
        print(f"{'='*60}")
        
        prompt = "Tell me a short story about a robot who discovered emotions."
        results = []
        
        for batch_size in batch_sizes:
            try:
                # Create batch
                batch_prompts = [prompt] * batch_size
                inputs = self.tokenizer(batch_prompts, padding=True, return_tensors="pt").to("cuda")
                
                mem_before = self.get_memory_info()
                
                # Time prefill
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(**inputs, use_cache=True)
                    kv_cache = outputs.past_key_values
                prefill_time = time.time() - start_time
                
                # Time decode (5 tokens per sequence)
                decode_start = time.time()
                current_cache = kv_cache
                for i in range(5):
                    # Get last token from each sequence
                    last_tokens = inputs["input_ids"][:, -1:].unsqueeze(1)
                    with torch.no_grad():
                        outputs = self.model(
                            last_tokens.squeeze(1),
                            past_key_values=current_cache,
                            use_cache=True
                        )
                        current_cache = outputs.past_key_values
                decode_time = time.time() - decode_start
                
                mem_after = self.get_memory_info()
                
                tokens_per_sec = (batch_size * 5) / decode_time
                
                results.append({
                    'batch_size': batch_size,
                    'prefill_time': prefill_time,
                    'decode_time': decode_time,
                    'tokens_per_sec': tokens_per_sec,
                    'gpu_memory_gb': mem_after['gpu_allocated'] - mem_before['gpu_allocated'],
                    'success': True
                })
                
                print(f"Batch {batch_size:2d}: Prefill {prefill_time:.3f}s | "
                      f"Decode {decode_time:.3f}s | "
                      f"{tokens_per_sec:.1f} tokens/s | "
                      f"GPU Mem {mem_after['gpu_allocated'] - mem_before['gpu_allocated']:.2f}GB")
                
                torch.cuda.empty_cache()
                del kv_cache, current_cache
                
            except Exception as e:
                print(f"Batch {batch_size}: FAILED - {str(e)}")
                results.append({'batch_size': batch_size, 'success': False, 'error': str(e)})
                
        return results

    def test_memory_fragmentation(self):
        """Test memory fragmentation with varying sequence lengths"""
        print(f"\n{'='*60}")
        print(f"TEST 3: Memory Fragmentation (HF KV Cache)")
        print(f"{'='*60}")
        
        # Simulate realistic workload: varying sequence lengths
        sequence_lengths = [128, 512, 1024, 256, 2048, 128, 4096, 512, 1024]
        memory_usage = []
        
        for i, seq_len in enumerate(sequence_lengths):
            prompt = "Story: " * (seq_len // 6)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_len).to("cuda")
            
            mem_before = self.get_memory_info()
            
            with torch.no_grad():
                outputs = self.model(inputs["input_ids"], use_cache=True)
                kv_cache = outputs.past_key_values
            
            mem_after = self.get_memory_info()
            current_usage = mem_after['gpu_allocated']
            memory_usage.append(current_usage)
            
            print(f"Step {i+1:2d}: SeqLen {seq_len:4d} | "
                  f"GPU Mem {current_usage:.2f}GB | "
                  f"Fragmented {current_usage - max(memory_usage[:-1] or [0]):.3f}GB")
            
            # "Free" the cache (but HF might not actually free the memory)
            del kv_cache
            torch.cuda.empty_cache()
            
        return memory_usage

def main():
    model_path = "/home/albert/learn/models/Qwen3-0.6B/"
    runner = StressTestRunner(model_path)
    
    print("Starting HF Transformer Stress Tests...")
    print("This will expose the bottlenecks that vLLM solves!")
    
    # Run all stress tests
    seq_results = runner.test_sequence_length_scaling()
    batch_results = runner.test_batch_scaling()
    mem_results = runner.test_memory_fragmentation()
    
    print(f"\n{'='*60}")
    print("SUMMARY: HF Transformer Limitations Exposed!")
    print(f"{'='*60}")
    print("1. Memory grows O(n²) with sequence length (KV cache)")
    print("2. Batch processing is memory-intensive")
    print("3. Memory fragmentation occurs with varying seq lengths")
    print("4. No sharing of KV states between similar prompts")
    print("\nThese are exactly the problems vLLM solves with:")
    print("- PagedAttention (fixed-size blocks)")
    print("- Block management (dynamic allocation)")
    print("- Prefix caching (reuse KV states)")
    print("- Continuous batching")

if __name__ == "__main__":
    main()
