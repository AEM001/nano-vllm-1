import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config
import logging
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


logger = logging.getLogger(__name__)

class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:

        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        #in GQA, total_num_heads is the number of query heads
        assert self.total_num_heads % tp_size == 0#number of query heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        #in GQA, total_num_kv_heads is the number of key/value heads
        assert self.total_num_kv_heads % tp_size == 0#number of key/value heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5# Attention scores = Q · K^T / √(d_k)
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )

        # Input: Takes the concatenated attention outputs from all heads
        # Output: Projects back to the model's hidden dimension
        # Function: Transforms the attention result back to the same dimension as the input
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        # Each attention head gets normalized independently
        # Normalization happens before position encoding
        # With bias: Skip normalization (bias provides centering)
        # Without bias: Apply RMSNorm for stability

        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)



    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        # Purpose: Combined query/key/value projection in single linear layer
        # Input: hidden_states (dimension: hidden_size)
        # Output: Combined QKV tensor (dimension: q_size + kv_size + kv_size)
        qkv = self.qkv_proj(hidden_states)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)

        output = self.o_proj(o.flatten(1, -1))# Project concatenated heads back to hidden_size
        return output

# The MLP processes the attention output to add non-linear transformations 
# and increase model capacity.
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # Combined projection: Single linear layer 
        # that outputs both gate and up projections
        # for SwiGLU to work well
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # Projection back: 
        # Reduces dimension from intermediate_size to hidden_size
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()

        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Pre-Norm Architecture
        # Qwen3 uses pre-norm (normalize before attention/MLP, not after):
        # input → layernorm → attention → residual_add → layernorm → mlp → residual_add
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # logger.info("Qwen3Model initialized")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        logger.debug(f"[Qwen3Model] Input: {input_ids.size(0)} tokens, positions: {positions.size(0)}")
        hidden_states = self.embed_tokens(input_ids)
        logger.debug(f"[Qwen3Model] After embedding: {hidden_states.shape}")
        residual = None
        for i, layer in enumerate(self.layers):
            pre_layer_shape = hidden_states.shape
            hidden_states, residual = layer(positions, hidden_states, residual)
            logger.debug(f"[Qwen3Model] Layer {i}: {pre_layer_shape} -> {hidden_states.shape}")
            
        hidden_states, _ = self.norm(hidden_states, residual)
        logger.debug(f"[Qwen3Model] After norm: {hidden_states.shape}")
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)#final linear layer for vocabulary prediction
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

        # logger.info("Qwen3ForCausalLM initialized")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask: list[list[int]] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        # logger.info("got hidden states, ready to deal with mask")
        if mask is None:
            return hidden_states

        if isinstance(mask, torch.Tensor):
            # logger.warning(f"torch.Tensor mask is {mask}")
            flat_mask = mask.reshape(-1).to(hidden_states.device)
        else:
            flat_mask = torch.tensor(
                [value for group in mask for value in group],
                dtype=torch.int32,
                device=hidden_states.device,
            )

        if flat_mask.numel() != hidden_states.size(0):
            raise ValueError(
                f"Mask size {flat_mask.numel()} does not match hidden states {hidden_states.size(0)}"
            )

        keep = flat_mask.ne(-1)
        if keep.all():
            return hidden_states
        return hidden_states[keep]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # logger.info("computing logits")
        return self.lm_head(hidden_states)
