from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,  # Input tensor to apply rotary embeddings to
    cos: torch.Tensor,  # Cosine values for rotary embeddings
    sin: torch.Tensor,  # Sine values for rotary embeddings
) -> torch.Tensor:  # Output tensor with rotary embeddings applied
    """Apply rotary embeddings to the input tensor."""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation."""

    def __init__(
        self,
        head_size: int,  # Size of each attention head
        rotary_dim: int,  # Dimensionality for rotary embeddings (must equal head_size)
        max_position_embeddings: int,  # Maximum sequence length for position embeddings
        base: float,  # Base value for frequency calculation
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,  # Position indices for the tokens
        query: torch.Tensor,  # Query tensor to apply rotary embeddings to
        key: torch.Tensor,  # Key tensor to apply rotary embeddings to
    ) -> tuple[torch.Tensor, torch.Tensor]:  # Tuple of (query, key) with rotary embeddings applied
        """Apply rotary embeddings to query and key tensors."""
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,  # Size of each attention head
    rotary_dim: int,  # Dimensionality for rotary embeddings
    max_position: int,  # Maximum sequence length for position embeddings
    base: float,  # Base value for frequency calculation
    rope_scaling: dict | None = None,  # Optional rope scaling configuration (not supported)
):  # Returns a RotaryEmbedding instance
    """Get a cached RotaryEmbedding instance."""
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
