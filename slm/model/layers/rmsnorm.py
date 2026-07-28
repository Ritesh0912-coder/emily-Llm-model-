"""
RMSNorm — Root Mean Square Layer Normalisation.

RMSNorm is a simplified variant of LayerNorm that removes the mean-centering
step and re-centering bias. It achieves comparable performance to LayerNorm
with fewer operations, and is used by LLaMA, Mistral, and other modern LLMs.

Formula:
    RMSNorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x²) + eps)

Reference:
    Zhang & Sennrich (2019) — "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.

    Unlike standard LayerNorm, RMSNorm does not subtract the mean before
    normalising, which makes it faster while achieving comparable accuracy.
    A single learnable scale parameter (``weight``) is applied element-wise
    after normalisation.

    Args:
        d_model: Dimensionality of the input features (last dimension of input tensor).
        eps: Small constant added to the denominator for numerical stability.

    Shape:
        - Input:  ``(*, d_model)``
        - Output: ``(*, d_model)``  (same as input)

    Example:
        >>> norm = RMSNorm(512)
        >>> x = torch.randn(2, 16, 512)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([2, 16, 512])
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Learnable scale vector, initialised to ones
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalisation to the input tensor.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)`` or any shape
               where the last dimension equals ``d_model``.

        Returns:
            Normalised tensor of the same shape as ``x``.
        """
        # Compute RMS across last dimension: sqrt(mean(x²) + eps)
        # rsqrt(a) = 1 / sqrt(a) — fused operation, faster on GPU
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"
