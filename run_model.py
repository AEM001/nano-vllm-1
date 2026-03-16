from nanovllm.engine.model_runner import ModelRunner
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.block_manager import BlockManager
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os
import torch

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

    sampling_params = SamplingParams(temperature=0.6, max_tokens=2560)
    prompt = "tell me a funny story"
    
    # Format prompt with chat template
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Disable thinking output
    )
    
    # Encode the formatted prompt
    token_ids = tokenizer.encode(formatted_prompt)
    
    # Create sequence and model runner
    seq = Sequence(token_ids, sampling_params)
    config = Config(path, enforce_eager=True, max_model_len=2048)
    run_model = ModelRunner(config, 0, [])
    
    # Initialize BlockManager and allocate physical memory blocks, what file of the whole project has done such thing?
    block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
    if not block_manager.can_allocate(seq):
        raise RuntimeError("Cannot allocate blocks for sequence")
    block_manager.allocate(seq)
    
    # Run prefill phase (process the prompt)
    token_ids_list = run_model.run([seq], is_prefill=True)
    new_token = token_ids_list[0] if isinstance(token_ids_list, list) else token_ids_list.item()
    
    # Add generated token to sequence and allocate block if needed
    seq.append_token(new_token)
    block_manager.may_append(seq)
    
    # Collect generated tokens
    generated_tokens = [new_token]
    
    # Generate up to max_tokens using decode phase
    while len(generated_tokens) < sampling_params.max_tokens:
        # Decode phase: generate one token at a time
        token_ids_list = run_model.run([seq], is_prefill=False)# how does the model know if a token has been cached?
        new_token = token_ids_list[0] if isinstance(token_ids_list, list) else token_ids_list.item()
        
        # Check for EOS
        if new_token == config.eos:
            break
            
        # Add token to sequence and manage blocks
        seq.append_token(new_token)
        generated_tokens.append(new_token)
        
        # Manage block allocation - must hash full blocks before allocating new ones
        if len(seq) % config.kvcache_block_size == 0:
            # Block just became full - hash it
            block_manager.may_append(seq)
        elif len(seq) % config.kvcache_block_size == 1 and len(seq) > 1:
            # Started a new block - need to allocate it
            if block_manager.can_append(seq):
                block_manager.may_append(seq)
            else:
                break
    
    # Print full response
    full_response = tokenizer.decode(generated_tokens)
    print(f"\nFull response ({len(generated_tokens)} tokens):")
    print(full_response)

if __name__ == "__main__":
    main()