from collections import deque
import xxhash
import numpy as np
from typing import Optional

from nanovllm.engine.sequence import Sequence


class Block:
    """Represents a memory block storing token sequences with caching capabilities.
    
    A Block stores a fixed-size sequence of tokens and tracks reference counting
    for memory management. It supports hash-based caching to avoid duplicate
    storage of identical token sequences.
    
    Attributes:
        block_id: Unique identifier for this block
        ref_count: Number of sequences currently referencing this block
        hash: Hash value of the token sequence for caching, -1 if not cached
        token_ids: List of token IDs stored in this block
    """

    def __init__(self, block_id: int) -> None:
        """Initialize a new Block.
        
        Args:
            block_id: Unique identifier for this block
        """
        self.block_id: int = block_id
        self.ref_count: int = 0
        self.hash: int = -1
        self.token_ids: list[int] = []

    def update(self, hash: int, token_ids: list[int]) -> None:
        """Update the block with new token data and hash.
        
        Args:
            hash: Hash value of the token sequence
            token_ids: List of token IDs to store in this block
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self) -> None:
        """Reset the block to its initial state.
        
        Resets the block for reuse, setting ref_count to 1 (for the new owner)
        and clearing cached data.
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Manages memory blocks for efficient token storage and caching.
    
    The BlockManager handles allocation, deallocation, and caching of memory blocks
    containing token sequences. It uses hash-based caching to avoid storing duplicate
    token sequences and manages reference counting for shared blocks.
    
    Attributes:
        block_size: Number of tokens each block can hold
        blocks: List of all available blocks
        hash_to_block_id: Mapping from hash values to block IDs for caching
        free_block_ids: Queue of available block IDs for allocation
        used_block_ids: Set of currently allocated block IDs
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        """Initialize the BlockManager.
        
        Args:
            num_blocks: Total number of blocks to manage
            block_size: Number of tokens each block can hold
        """
        self.block_size: int = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """Compute hash for token sequence with optional prefix.
        
        Uses xxhash64 for fast hashing of token sequences. The prefix allows
        for hierarchical hashing of consecutive blocks.
        
        Args:
            token_ids: List of token IDs to hash
            prefix: Optional prefix hash for hierarchical hashing
            
        Returns:
            64-bit hash value of the token sequence
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Allocate a specific block for use.
        
        Args:
            block_id: ID of the block to allocate
            
        Returns:
            The allocated Block object
            
        Raises:
            AssertionError: If the block is already in use
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} is already in use"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> None:
        """Deallocate a block, making it available for reuse.
        
        Args:
            block_id: ID of the block to deallocate
            
        Raises:
            AssertionError: If the block is still referenced
        """
        assert self.blocks[block_id].ref_count == 0, f"Block {block_id} still has references"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if there are enough free blocks for the sequence.
        
        Args:
            seq: Sequence to check allocation for
            
        Returns:
            True if enough free blocks are available, False otherwise
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        """Allocate blocks for a sequence, using cached blocks when possible.
        
        Attempts to reuse cached blocks with identical token sequences to reduce
        memory usage. Falls back to allocating new blocks when cache misses occur.
        
        Args:
            seq: Sequence to allocate blocks for
            
        Raises:
            AssertionError: If the sequence already has allocated blocks
        """
        assert not seq.block_table, "Sequence already has allocated blocks"
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        """Deallocate all blocks used by a sequence.
        
        Decrements reference counts and frees blocks that are no longer
        referenced by any sequence.
        
        Args:
            seq: Sequence to deallocate blocks for
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check if the sequence can append a new token.
        
        Returns True if either the sequence has space in its last block
        or there's a free block available for allocation.
        
        Args:
            seq: Sequence to check
            
        Returns:
            True if the sequence can append, False otherwise
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence) -> None:
        """Handle block allocation when a sequence appends a token.
        
        Manages block allocation and hashing when sequences grow beyond
        their current block boundaries.
        
        Args:
            seq: Sequence that may need additional block allocation
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1, "Previous block should be hashed before allocating new block"
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1, "Block should not be hashed when full"
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1, "Partial block should not be hashed"
