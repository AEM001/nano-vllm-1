import torch
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def _repeat_kv_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.size(1) == num_heads:
        return x
    assert num_heads % x.size(1) == 0
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
    q_t = q.transpose(0, 1)
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    
    o = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, is_causal=is_causal, scale=scale
    )
    return o.transpose(0, 1)


def _prefill_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, num_heads: int) -> torch.Tensor:
    context = get_context()
    outputs = []
    num_seqs = context.cu_seqlens_q.numel() - 1
    
    for i in range(num_seqs):
        q_start = context.cu_seqlens_q[i].item()
        q_end = context.cu_seqlens_q[i + 1].item()
        k_start = context.cu_seqlens_k[i].item()
        k_end = context.cu_seqlens_k[i + 1].item()
        
        q_i = q[q_start:q_end]
        
        if context.block_tables is not None:
            context_len = k_end - k_start
            block_size = k.size(1)
            # Flatten cache before indexing (same as decode path)
            flat_k = k.view(-1, k.size(-2), k.size(-1))
            flat_v = v.view(-1, v.size(-2), v.size(-1))
            slots = _block_ids_to_slot_ids(context.block_tables[i], context_len, block_size)
            k_i = flat_k.index_select(0, slots[:context_len])
            v_i = flat_v.index_select(0, slots[:context_len])
        else:
            k_i = k[k_start:k_end]
            v_i = v[k_start:k_end]
        
        k_i = _repeat_kv_heads(k_i, num_heads)
        v_i = _repeat_kv_heads(v_i, num_heads)
        
        outputs.append(_causal_attention(q_i, k_i, v_i, scale))
    
    return torch.cat(outputs, dim=0)


def _decode_fallback(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, scale: float, num_heads: int) -> torch.Tensor:
    context = get_context()
    outputs = []
    num_seqs = q.size(0)
    
    flat_k = k_cache.view(-1, k_cache.size(-2), k_cache.size(-1))
    flat_v = v_cache.view(-1, v_cache.size(-2), v_cache.size(-1))
    
    for i in range(num_seqs):
        context_len = context.context_lens[i].item()
        block_size = k_cache.size(1)
        slots = _block_ids_to_slot_ids(context.block_tables[i], context_len, block_size)
        
        k_i = flat_k.index_select(0, slots[:context_len])
        v_i = flat_v.index_select(0, slots[:context_len])
        
        k_i = _repeat_kv_heads(k_i, num_heads)
        v_i = _repeat_kv_heads(v_i, num_heads)
        
        outputs.append(_causal_attention(q[i:i + 1], k_i, v_i, scale, is_causal=False))
    
    return torch.cat(outputs, dim=0)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        if k_cache.numel() and v_cache.numel() and context.slot_mapping.numel() > 0:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        use_flash_attn = flash_attn_varlen_func is not None and flash_attn_with_kvcache is not None
        
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
                
            if use_flash_attn:
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
                o = _prefill_fallback(q, k, v, self.scale, self.num_heads)
                
        else:
            if use_flash_attn:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context.context_lens, 
                    block_table=context.block_tables, 
                    softmax_scale=self.scale, 
                    causal=True
                )
                o = o.squeeze(1)
            else:
                o = _decode_fallback(q, k_cache, v_cache, self.scale, self.num_heads)
                
        return o
