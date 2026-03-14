import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
"""
x, y = x.chunk(2, dim=-1)
Splits the input tensor into two equal parts along the last dimension.

Example: if input shape = (B, S, 2D), after chunk you get two tensors each of shape (B, S, D).


F.silu(x)
Applies the SiLU (Swish) activation function to the first half: SiLU(x) = x * σ(x)

* y
Performs element-wise multiplication between the SiLU-activated first half and the second half y.

Return the result.
"""