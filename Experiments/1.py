import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


class SimpleModelRunner:
    
    def __init__(self, model_path):
        """Load model and tokenizer using standard HuggingFace"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Move to GPU and set to eval
        self.model.cuda()
        self.model.eval()
        
        # Initialize KV cache storage
        self.kv_cache = None
        
    def generate(self, prompt, max_tokens=1000, temperature=0.6):
        """Generate text from prompt - using KV cache for efficiency"""
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        
        # Simple timing
        start_time = time.time()
        
        # Phase 1: Prefill - process all prompt tokens at once
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            self.kv_cache = outputs.past_key_values
            logits = outputs.logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated_ids = input_ids[0].tolist()
        generated_ids.append(next_token)
        
        # Phase 2: Decode - one token at a time using KV cache
        for step in range(max_tokens - 1):
            # Stop at EOS
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Get next token using cached K,V
            with torch.no_grad():
                outputs = self.model(
                    torch.tensor([[next_token]], device="cuda"),#only feed one generated token
                    past_key_values=self.kv_cache,#use cached K,V
                    use_cache=True
                )
                self.kv_cache = outputs.past_key_values
                logits = outputs.logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids.append(next_token)
            
        # End timing
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTotal generation time: {total_time:.2f} seconds")
             
        # Decode and return
        return self.tokenizer.decode(generated_ids)

def main():
    model_path = "/home/albert/learn/models/Qwen3-0.6B/"
    
    # Create simple model runner
    runner = SimpleModelRunner(model_path)
    
    # Generate text
    prompt = "Tell me a long love story:"
    
    # Format prompt with chat template (the way model was trained)
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = runner.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking output
    )
    
    result = runner.generate(formatted_prompt, max_tokens=200, temperature=0.6)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")

if __name__ == "__main__":
    main()
