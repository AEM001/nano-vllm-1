import torch
from torch import nn
import triton
import triton.language as tl
import logging

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context
logger = logging.getLogger(__name__)
      
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


def _mixed_prefill_fallback(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    num_heads: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    num_seqs = block_tables.size(0)
    flat_k = k_cache.view(-1, k_cache.size(-2), k_cache.size(-1))
    flat_v = v_cache.view(-1, k_cache.size(-2), k_cache.size(-1))

    for i in range(num_seqs):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        k_start = cu_seqlens_k[i].item()
        k_end = cu_seqlens_k[i + 1].item()
        context_len = k_end - k_start
        block_size = k_cache.size(1)
        slots = _block_ids_to_slot_ids(block_tables[i], context_len, block_size)
        # logger.info(f"mixed prefill: slots={slots}")
        k_i = flat_k.index_select(0, slots[:context_len])
        v_i = flat_v.index_select(0, slots[:context_len])
        k_i = _repeat_kv_heads(k_i, num_heads)
        v_i = _repeat_kv_heads(v_i, num_heads)
        outputs.append(_causal_attention(q[q_start:q_end], k_i, v_i, scale))

    return torch.cat(outputs, dim=0)


def _mixed_decode_fallback(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    num_heads: int,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    flat_k = k_cache.view(-1, k_cache.size(-2), k_cache.size(-1))
    flat_v = v_cache.view(-1, v_cache.size(-2), v_cache.size(-1))

    for i in range(q.size(0)):
        context_len = context_lens[i].item()
        block_size = k_cache.size(1)

        slots = _block_ids_to_slot_ids(block_tables[i], context_len, block_size)
        k_i = flat_k.index_select(0, slots[:context_len])
        v_i = flat_v.index_select(0, slots[:context_len])
        k_i = _repeat_kv_heads(k_i, num_heads)
        v_i = _repeat_kv_heads(v_i, num_heads)
        outputs.append(_causal_attention(q[i:i + 1], k_i, v_i, scale, is_causal=False))

    return torch.cat(outputs, dim=0)


def _mixed_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    num_heads: int,
) -> torch.Tensor:
    context = get_context()
    query_mask = context.query_mask
    seq_mask = context.seq_mask
    output = torch.empty_like(q)

    prefill_token_idx = (query_mask == -1).nonzero(as_tuple=False).flatten()
    decode_token_idx = (query_mask == 0).nonzero(as_tuple=False).flatten()
    prefill_seq_idx = (seq_mask == -1).nonzero(as_tuple=False).flatten()
    decode_seq_idx = (seq_mask == 0).nonzero(as_tuple=False).flatten()

    if prefill_token_idx.numel() > 0:
        prefill_q = q.index_select(0, prefill_token_idx)
        prefill_block_tables = context.block_tables.index_select(0, prefill_seq_idx)
        # Filter cu_seqlens to only prefill sequences and rebuild cumulative
        prefill_indices = torch.cat([prefill_seq_idx, prefill_seq_idx[-1:] + 1])
        prefill_cu_seqlens_q = context.cu_seqlens_q.index_select(0, prefill_indices)
        prefill_cu_seqlens_k = context.cu_seqlens_k.index_select(0, prefill_indices)
        # Rebase to start from 0
        prefill_cu_seqlens_q = prefill_cu_seqlens_q - prefill_cu_seqlens_q[0]
        prefill_cu_seqlens_k = prefill_cu_seqlens_k - prefill_cu_seqlens_k[0]
        prefill_output = _mixed_prefill_fallback(
            prefill_q,
            k_cache,
            v_cache,
            scale,
            num_heads,
            prefill_cu_seqlens_q,
            prefill_cu_seqlens_k,
            prefill_block_tables,
        )#mainly just recompute the actual number of prefill,not directly getting from context
        output.index_copy_(0, prefill_token_idx, prefill_output)

    if decode_token_idx.numel() > 0:
        decode_q = q.index_select(0, decode_token_idx)
        decode_block_tables = context.block_tables.index_select(0, decode_seq_idx)
        decode_output = _mixed_decode_fallback(
            decode_q,
            k_cache,
            v_cache,
            scale,
            num_heads,
            context.context_lens,
            decode_block_tables,
        )
        output.index_copy_(0, decode_token_idx, decode_output)

    return output


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
        
        # Always use mixed attention path
        return _mixed_attention(q, k_cache, v_cache, self.scale, self.num_heads)
