# settings    
    sampling_params = [SamplingParams(temperature=0.6, max_tokens=500),
                       SamplingParams(temperature=0.6, max_tokens=40)]
    
    # Multi-sequence prompts - different topics for concurrent processing
    prompts = [
        "In the realm of artificial intelligence and machine learning, neural networks have emerged as a powerful paradigm for solving complex problems. These computational models, inspired by the biological neural networks in animal brains, consist of interconnected layers of artificial neurons that process and transform information through weighted connections. The fundamental building blocks of neural networks include input layers that receive raw data, hidden layers that perform intermediate computations, and output layers that produce final predictions or classifications. Deep learning architectures, which contain multiple hidden layers, have revolutionized fields such as computer vision, natural language processing, and speech recognition. Training these networks typically involves optimization algorithms like gradient descent and backpropagation, which adjust the network's parameters to minimize a loss function that measures the difference between predicted and actual outputs. Regularization techniques such as dropout, batch normalization, and weight decay help prevent overfitting and improve generalization performance. Modern neural network architectures like Transformers have further advanced the field by employing self-attention mechanisms that capture long-range dependencies in sequential data, enabling breakthroughs in language understanding and generation tasks. The scalability of these models has led to the development of large language models with billions of parameters, capable of performing a wide range of natural language tasks with remarkable accuracy and coherence.And what is the difference between inference and training?",
        "introduce California"

# things to observe and results
- model_runner, prepare_decode and the prepare_block_table, if the -1 padding would work
```
INFO: MODELRUNNER: !!! prepare decode block_tables: tensor([[ 0,  1],
        [ 2, -1]], device='cuda:0', dtype=torch.int32)
INFO: MODELRUNNER: block_tables shape: torch.Size([2, 2])
INFO: MODELRUNNER: block_tables content: [[0, 1], [2, -1]]
```
- what does the block_table look like
- what if one sequence finishes early?
```
INFO: BLOCK_MANAGER: deallocating block 2
INFO: MODELRUNNER: !!! prepare decode block_tables: tensor([[0, 1]], device='cuda:0', dtype=torch.int32)
INFO: MODELRUNNER: block_tables shape: torch.Size([1, 2])
INFO: MODELRUNNER: block_tables content: [[0, 1]]
```
- what if a block is full?
```
DEBUG: MODELRUNNER: prepare decode context_lens: [512]
DEBUG: BLOCK_MANAGER: allocating a new block
DEBUG: MODELRUNNER: prepare decode context_lens: [513]
INFO: MODELRUNNER: !!! prepare decode block_tables: tensor([[0, 1, 3]], device='cuda:0', dtype=torch.int32)
INFO: MODELRUNNER: block_tables shape: torch.Size([1, 3])
INFO: MODELRUNNER: block_tables content: [[0, 1, 3]]
```

- basic loop components. regular, don't record here


