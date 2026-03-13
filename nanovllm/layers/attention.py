# ============================================================================
# ATTENTION LAYER IMPLEMENTATION
# ============================================================================
# This file implements the attention mechanism for nano-vllm, supporting:
# 1. Flash Attention (with fallback to PyTorch implementation)
# 2. KV Cache management for efficient inference
# 3. Prefill and decode phases
# 4. Multi-query attention (MQA) and grouped-query attention (GQA)
# ============================================================================

import torch
from torch import nn
import triton
import triton.language as tl

# Flash Attention imports with graceful fallback
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context


# ============================================================================
# KV CACHE STORAGE KERNEL (Triton)
# ============================================================================
# Custom Triton kernel for efficiently storing key-value pairs in cache
# This is performance-critical for inference speed
# ============================================================================

@triton.jit
def store_kvcache_kernel(
    key_ptr,           # Pointer to input key tensor
    key_stride,        # Stride for key tensor (elements per sequence)
    value_ptr,         # Pointer to input value tensor  
    value_stride,      # Stride for value tensor (elements per sequence)
    k_cache_ptr,       # Pointer to key cache storage
    v_cache_ptr,       # Pointer to value cache storage
    slot_mapping_ptr,  # Pointer to cache slot mapping for each token
    D: tl.constexpr,   # Total dimension: num_heads * head_dim (compile-time)
):
    """
    Stores key-value pairs in KV cache at specified slots.
    Each program instance handles one token position.
    """
    idx = tl.program_id(0)  # Get token index for this program instance
    slot = tl.load(slot_mapping_ptr + idx)  # Get cache slot for this token
    if slot == -1: return  # Skip invalid slots
    
    # Calculate memory offsets for key and value loading
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # Load key and value vectors for this token
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # Calculate cache storage offsets and store
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


# ============================================================================
# KV CACHE STORAGE WRAPPER FUNCTION
# ============================================================================

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    Stores new key-value pairs in the KV cache using the Triton kernel.
    
    Args:
        key: New keys to store [N, num_heads, head_dim]
        value: New values to store [N, num_heads, head_dim]  
        k_cache: Key cache tensor [num_slots, num_heads, head_dim]
        v_cache: Value cache tensor [num_slots, num_heads, head_dim]
        slot_mapping: Cache slot indices for each token [N]
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # Total dimension per token
    
    # Memory layout assertions for optimal performance
    assert key.stride(-1) == 1 and value.stride(-1) == 1  # Last dimension contiguous
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # Head dimension contiguous
    assert k_cache.stride(1) == D and v_cache.stride(1) == D  # Cache layout optimized
    assert slot_mapping.numel() == N  # One slot per token
    
    # Launch Triton kernel with one program per token
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _repeat_kv_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Expands KV heads to match query heads for Grouped-Query Attention (GQA).
    For example: num_kv_heads=8, num_heads=32 -> repeat each KV head 4 times.
    
    Args:
        x: Key or value tensor [seq_len, num_kv_heads, head_dim]
        num_heads: Target number of query heads
        
    Returns:
        Tensor with expanded KV heads [seq_len, num_heads, head_dim]
    """
    if x.size(1) == num_heads:
        return x  # Already correct size (standard attention)
    assert num_heads % x.size(1) == 0  # Must divide evenly for GQA
    repeat = num_heads // x.size(1)
    return x.repeat_interleave(repeat, dim=1)


def _block_ids_to_slot_ids(block_ids: torch.Tensor, context_len: int, block_size: int) -> torch.Tensor:
    block_ids = block_ids[block_ids >= 0].to(dtype=torch.int64)
    if block_ids.numel() == 0:
        return block_ids
    slot_offsets = torch.arange(block_size, device=block_ids.device, dtype=torch.int64)
    slot_ids = block_ids.unsqueeze(1) * block_size + slot_offsets.unsqueeze(0)
    return slot_ids.reshape(-1)[:context_len]


def _causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Computes causal attention using PyTorch's optimized SDPA.
    Used as fallback when Flash Attention is not available.
    
    Args:
        q: Query tensor [seq_len, num_heads, head_dim]
        k: Key tensor [seq_len, num_heads, head_dim]  
        v: Value tensor [seq_len, num_heads, head_dim]
        scale: Scaling factor for attention scores (1/sqrt(head_dim))
        
    Returns:
        Attention output [seq_len, num_heads, head_dim]
    """
    # PyTorch SDPA expects [batch, heads, seq_len, head_dim] format
    q_t = q.transpose(0, 1)  # [num_heads, seq_len, head_dim]
    k_t = k.transpose(0, 1)  # [num_heads, seq_len, head_dim]
    v_t = v.transpose(0, 1)  # [num_heads, seq_len, head_dim]
    
    # Add batch dimension for SDPA and compute causal attention
    o = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, is_causal=is_causal, scale=scale
    )
    return o.transpose(0, 1)  # Back to [seq_len, num_heads, head_dim]


# ============================================================================
# PREFILL PHASE FALLBACK IMPLEMENTATION
# ============================================================================

def _prefill_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, num_heads: int) -> torch.Tensor:
    """
    PyTorch fallback for prefill phase (processing multiple tokens at once).
    Handles variable sequence lengths and prefix caching.
    
    Args:
        q: Queries from all sequences [total_tokens, num_heads, head_dim]
        k: Keys from all sequences [total_tokens, num_kv_heads, head_dim]
        v: Values from all sequences [total_tokens, num_kv_heads, head_dim]
        scale: Attention scaling factor
        num_heads: Number of query heads
        
    Returns:
        Attention outputs [total_tokens, num_heads, head_dim]
    """
    context = get_context()
    outputs = []
    num_seqs = context.cu_seqlens_q.numel() - 1  # Number of sequences
    
    # Process each sequence individually due to variable lengths
    for i in range(num_seqs):
        # Extract token ranges for this sequence
        q_start = context.cu_seqlens_q[i].item()
        q_end = context.cu_seqlens_q[i + 1].item()
        k_start = context.cu_seqlens_k[i].item()
        k_end = context.cu_seqlens_k[i + 1].item()
        
        # Get queries for this sequence
        q_i = q[q_start:q_end]
        
        # Handle prefix caching: use cached KV if available
        if context.block_tables is not None:
            context_len = k_end - k_start
            block_size = k.size(1)
            slots = _block_ids_to_slot_ids(context.block_tables[i], context_len, block_size)
            # Retrieve cached KV from specific slots
            k_i = k.index_select(0, slots[:context_len])
            v_i = v.index_select(0, slots[:context_len])
        else:
            # No caching: use fresh KV
            k_i = k[k_start:k_end]
            v_i = v[k_start:k_end]
        
        # Expand KV heads for GQA if needed
        k_i = _repeat_kv_heads(k_i, num_heads)
        v_i = _repeat_kv_heads(v_i, num_heads)
        
        # Compute attention for this sequence
        outputs.append(_causal_attention(q_i, k_i, v_i, scale))
    
    return torch.cat(outputs, dim=0)  # Concatenate all sequence outputs


# ============================================================================
# DECODE PHASE FALLBACK IMPLEMENTATION  
# ============================================================================

def _decode_fallback(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, scale: float, num_heads: int) -> torch.Tensor:
    """
    PyTorch fallback for decode phase (generating one token at a time).
    Retrieves cached KV pairs and computes attention for each sequence.
    
    Args:
        q: Single token queries [num_seqs, num_heads, head_dim]
        k_cache: Cached keys [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Cached values [num_blocks, block_size, num_kv_heads, head_dim]
        scale: Attention scaling factor
        num_heads: Number of query heads
        
    Returns:
        Attention outputs [num_seqs, num_heads, head_dim]
    """
    context = get_context()
    outputs = []
    num_seqs = q.size(0)
    
    # Flatten cache for easier indexing: [total_slots, num_kv_heads, head_dim]
    flat_k = k_cache.view(-1, k_cache.size(-2), k_cache.size(-1))
    flat_v = v_cache.view(-1, v_cache.size(-2), v_cache.size(-1))
    
    # Process each sequence independently
    for i in range(num_seqs):
        context_len = context.context_lens[i].item()  # Length of this sequence
        block_size = k_cache.size(1)
        slots = _block_ids_to_slot_ids(context.block_tables[i], context_len, block_size)
        
        # Retrieve cached KV for this sequence's context
        k_i = flat_k.index_select(0, slots[:context_len])
        v_i = flat_v.index_select(0, slots[:context_len])
        
        # Expand KV heads for GQA if needed
        k_i = _repeat_kv_heads(k_i, num_heads)
        v_i = _repeat_kv_heads(v_i, num_heads)
        
        # Compute attention for this single token
        # In decode with KV cache, each query token is the latest token and
        # should attend to the full cached context. Using is_causal=True with
        # q_len=1 and k_len=context would incorrectly mask most keys.
        outputs.append(_causal_attention(q[i:i + 1], k_i, v_i, scale, is_causal=False))
    
    return torch.cat(outputs, dim=0)  # Concatenate all sequence outputs


# ============================================================================
# MAIN ATTENTION MODULE
# ============================================================================

class Attention(nn.Module):
    """
    Multi-Head Attention module with KV caching and Flash Attention support.
    
    This module handles both prefill (processing many tokens) and decode 
    (generating one token) phases, with automatic fallback to PyTorch 
    implementations when Flash Attention is unavailable.
    
    Key Features:
    - Flash Attention 2 integration with graceful fallback
    - KV Cache for efficient inference
    - Grouped-Query Attention (GQA) support
    - Prefix caching for faster prefill
    - Variable sequence length handling
    """

    def __init__(
        self,
        num_heads,          # Number of query heads
        head_dim,           # Dimension per head
        scale,              # Attention scaling factor (1/sqrt(head_dim))
        num_kv_heads,       # Number of key/value heads (for GQA)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        
        # KV cache storage (initialized later during model setup)
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Forward pass handling both prefill and decode phases.
        
        Args:
            q: Query tensor [total_tokens, num_heads, head_dim]
            k: Key tensor [total_tokens, num_kv_heads, head_dim]  
            v: Value tensor [total_tokens, num_kv_heads, head_dim]
            
        Returns:
            Attention output [total_tokens, num_heads, head_dim]
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # Store new KV pairs in cache if cache exists
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # Check if Flash Attention is available
        use_flash_attn = flash_attn_varlen_func is not None and flash_attn_with_kvcache is not None
        
        if context.is_prefill:
            # ===== PREFILL PHASE: Process multiple tokens =====
            if context.block_tables is not None:    # Prefix caching enabled
                k, v = k_cache, v_cache  # Use cached KV pairs
                
            if use_flash_attn:
                # Use Flash Attention for optimal performance
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q, 
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k, 
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale, 
                    causal=True, 
                    block_table=context.block_tables
                )
            else:
                # Fallback to PyTorch implementation
                o = _prefill_fallback(q, k, v, self.scale, self.num_heads)
                
        else:
            # ===== DECODE PHASE: Generate single token =====
            if use_flash_attn:
                # Flash Attention with KV cache for decode
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,  # Add seq_len dimension
                    cache_seqlens=context.context_lens, 
                    block_table=context.block_tables, 
                    softmax_scale=self.scale, 
                    causal=True
                )
                o = o.squeeze(1)  # Remove seq_len dimension
            else:
                # Fallback to PyTorch implementation
                o = _decode_fallback(q, k_cache, v_cache, self.scale, self.num_heads)
                
        return o
