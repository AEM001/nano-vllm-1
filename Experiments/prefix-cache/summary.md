- when one block is full it would calculate the hash of it
- when seqs share EXACTLY the same block, they would share the same hash
-  that way, the cached tokens would be used

```
INFO: MODELRUNNER: !!! prepare prefill block_tables: tensor([[0, 1],
        [0, 2]], device='cuda:0', dtype=torch.int32)
```
the block 0 is reused by the second sequence