## Hash Function

Why Hierarchical Hashing?
Block 0: [12,34,56,78] → hash = hash([12,34,56,78])
Block 1: [90,91,92,93] → hash = hash(hash(block0), [90,91,92,93])
Block 2: [94,95,96,97] → hash = hash(hash(block1), [94,95,96,97])
This creates a chain of hashes where each block's hash depends on all previous blocks, allowing the entire sequence to be uniquely identified.

```python
# First block (no prefix)
tokens1 = [12, 34, 56, 78]
hash1 = BlockManager.compute_hash(tokens1)  # prefix = -1
# Result: hash1 = 1234567890

# Second block (uses previous block's hash as prefix)
tokens2 = [90, 91, 92, 93]  
hash2 = BlockManager.compute_hash(tokens2, hash1)  # prefix = 1234567890
# Result: hash2 = 9876543210

# Same tokens but different prefix = different hash
tokens3 = [90, 91, 92, 93]
hash3 = BlockManager.compute_hash(tokens3, 5555555555)  # different prefix
# Result: hash3 = 1111111111 (different from hash2)
```


## may_append
```python
# Initial state: block_size = 4
seq = Sequence([12, 34])  # 2 tokens, 1 partial block
# Block 0: [12,34], hash = -1

# Append token 56 (now 3 tokens)
seq.append_token(56)
# Scenario 3: Still partial, no action needed
# Block 0: [12,34,56], hash = -1

# Append token 78 (now 4 tokens) 
seq.append_token(78)
# Scenario 2: Block just became full!
# Compute hash for [12,34,56,78] → hash = 1234
# Block 0: [12,34,56,78], hash = 1234, cached

# Append token 90 (now 5 tokens)
seq.append_token(90)  
# Scenario 1: Need new block!
# Allocate Block 1: [90], hash = -1
# Block 0: [12,34,56,78], hash = 1234 (cached)
# Block 1: [90], hash = -1
```

---
## allocate

The cache_miss variable tracks whether we need to allocate new blocks:
### **Cache Miss Meaning**

A **cache miss** occurs when the BlockManager cannot find an existing block with the same token sequence, so it must allocate a new block instead of reusing memory.

### **When Cache Miss Happens**

```python
# Look for existing block with this hash
block_id = self.hash_to_block_id.get(h, -1)

# Cache miss if:
# 1. Hash not found in cache (block_id == -1)
# 2. OR hash collision - tokens don't match
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    cache_miss = True
```

### **Cache Miss vs Cache Hit Examples**

#### **Cache Hit** (Memory Saved)
```python
# First sequence allocates blocks
seq1 = Sequence([12, 34, 56, 78])  # "Hello world"
# Block 0: [12,34,56,78] → hash=1234, allocated as block 0
# Cache mapping: {1234: 0}

# Second sequence with same tokens
seq2 = Sequence([12, 34, 56, 78])  # Same "Hello world"
# Look up hash=1234 → found block_id=0
# Verify tokens match → YES
# cache_miss = False (CACHE HIT!)
# Result: Reuse block 0, increment ref_count to 2
```

#### **Cache Miss** (New Block Needed)
```python
# Different sequence
seq3 = Sequence([99, 100, 101, 102])  # "Goodbye"
# Compute hash → 5678
# Look up hash=5678 → not found (block_id = -1)
# cache_miss = True (CACHE MISS!)
# Result: Must allocate new block from free_block_ids
```

#### **Hash Collision Cache Miss**
```python
# Rare case: different tokens produce same hash
seq4 = Sequence([88, 89, 90, 91])  # Hash collision! = 1234
# Look up hash=1234 → found block_id=0
# Verify tokens match → NO! [88,89,90,91] != [12,34,56,78]
# cache_miss = True (CACHE MISS due to collision)
# Result: Allocate new block despite hash match
```

#### **Performance Impact**

| Scenario | Memory Usage | Speed | Result |
|----------|-------------|-------|--------|
| **Cache Hit** | Shared (efficient) | Fast (O(1) lookup) | ✅ Memory saved |
| **Cache Miss** | New allocation | Slower (allocation) | ❌ More memory used |

#### **Why Cache Miss Detection Matters**

1. **Memory Efficiency**: Cache hits reuse existing blocks
2. **Performance**: Avoid allocating duplicate memory
3. **Scalability**: Handle many concurrent requests with limited memory

The entire caching system is designed to **minimize cache misses** by reusing blocks for identical token sequences, which is crucial for LLM inference where many requests may have similar prompt prefixes.