#!/usr/bin/env python3
"""
Advanced multi-sequence examples for nano-vllm
Demonstrates different batching strategies and concurrent processing patterns
"""

import os
import logging
import time
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def example_1_concurrent_sequences():
    """Example 1: Process multiple sequences concurrently"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Concurrent Multi-Sequence Processing")
    print("="*80)
    
    model_path = "/home/albert/learn/models/Qwen3-0.6B/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create LLM with optimized multi-sequence settings
    llm = LLM(model_path, 
              enforce_eager=True, 
              tensor_parallel_size=1,
              max_num_batched_tokens=24800,
              max_num_seqs=7)
    
    # Different prompts for concurrent processing
    prompts = [
        "Explain photosynthesis in detail.",
        "What are the main causes of climate change?",
        "Describe the structure of DNA.",
        "How does the internet work?",
        "What is machine learning?",
        "Explain the solar system.",
        "What causes earthquakes?"
    ]
    
    # Format for chat template
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    ) for prompt in prompts]
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    
    print(f"Processing {len(prompts)} sequences concurrently...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")
    print(f"Throughput: {len(prompts)/(end_time - start_time):.2f} sequences/sec")
    
    for i, output in enumerate(outputs):
        print(f"\nSeq {i+1}: {output['text'][:100]}...")

def example_2_staggered_arrival():
    """Example 2: Simulate sequences arriving at different times"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Staggered Sequence Arrival")
    print("="*80)
    
    model_path = "/home/albert/learn/models/Qwen3-0.6B/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    llm = LLM(model_path, 
              enforce_eager=True, 
              tensor_parallel_size=1,
              max_num_batched_tokens=24800,
              max_num_seqs=7)
    
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # Simulate requests arriving over time
    all_outputs = {}
    batch_size = 3
    
    for batch_num in range(batch_size):
        print(f"\n--- Batch {batch_num + 1} ---")
        
        # New prompts for this batch
        new_prompts = [
            f"Write a short story about topic {batch_num * 3 + 1}.",
            f"Explain concept {batch_num * 3 + 2} briefly.",
            f"Describe item {batch_num * 3 + 3}."
        ]
        
        # Format prompts
        formatted_prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        ) for prompt in new_prompts]
        
        # Add to existing requests (simulating staggered arrival)
        for prompt in formatted_prompts:
            llm.add_request(prompt, sampling_params)
        
        # Generate one step to show batching
        if not llm.is_finished():
            outputs, num_tokens = llm.step()
            print(f"Step completed, tokens: {num_tokens}")
    
    # Complete remaining generation
    outputs = llm.generate([], sampling_params)  # Empty prompts - just finish existing
    
    print("All sequences completed!")

def example_3_different_lengths():
    """Example 3: Mix of short and long sequences"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Mixed Sequence Lengths")
    print("="*80)
    
    model_path = "/home/albert/learn/models/Qwen3-0.6B/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    llm = LLM(model_path, 
              enforce_eager=True, 
              tensor_parallel_size=1,
              max_num_batched_tokens=24800,
              max_num_seqs=7)
    
    # Mix of short and long prompts
    prompts = [
        "What is 2+2?",  # Very short
        "Explain quantum entanglement in simple terms.",  # Medium
        "Write a detailed analysis of the economic impact of artificial intelligence on global markets, including historical context, current trends, and future projections. Discuss both opportunities and challenges.",  # Very long
        "Briefly describe photosynthesis.",  # Short
        "Compare and contrast the political systems of democracy and authoritarianism, providing historical examples and analyzing their effectiveness in different cultural contexts.",  # Long
        "Hello, how are you?",  # Very short
        "Provide a comprehensive overview of climate change, including its causes, effects on ecosystems and human society, potential solutions, and the role of international cooperation in addressing this global challenge."  # Very long
    ]
    
    # Format prompts
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    ) for prompt in prompts]
    
    # Different sampling params for different sequence types
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
    
    print(f"Processing {len(prompts)} sequences with varying lengths...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        prompt_type = "Short" if len(prompt) < 200 else "Long" if len(prompt) > 500 else "Medium"
        print(f"Seq {i+1} ({prompt_type}): {len(output['text'])} chars generated")

def example_4_stress_test():
    """Example 4: Stress test with maximum concurrent sequences"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Stress Test - Max Concurrent Sequences")
    print("="*80)
    
    model_path = "/home/albert/learn/models/Qwen3-0.6B/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    llm = LLM(model_path, 
              enforce_eager=True, 
              tensor_parallel_size=1,
              max_num_batched_tokens=24800,
              max_num_seqs=7)
    
    # Maximum number of concurrent sequences
    prompts = [f"Generate a creative story about number {i+1}." for i in range(7)]
    
    # Format prompts
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    ) for prompt in prompts]
    
    sampling_params = SamplingParams(temperature=0.8, max_tokens=300)
    
    print(f"Stress testing with {len(prompts)} concurrent sequences...")
    print(f"Max sequences configured: {llm.scheduler.max_num_seqs}")
    
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")
    print(f"Max concurrent throughput achieved: {len(prompts)/(end_time - start_time):.2f} sequences/sec")
    
    for i, output in enumerate(outputs):
        print(f"Sequence {i+1}: {len(output['text'])} chars")

if __name__ == "__main__":
    print("Multi-Sequence Examples for nano-vllm")
    print("=====================================")
    
    # Run all examples
    example_1_concurrent_sequences()
    example_2_staggered_arrival()
    example_3_different_lengths()
    example_4_stress_test()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
