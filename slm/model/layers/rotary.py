"""
Rotary Position Embeddings (RoPE) for Emily SLM.

RoPE encodes positional information by rotating query and key vectors in
attention using position-dependent rotation matrices. Unlike additive
positional embeddings, RoPE:

  - Has no additional parameters
  - Naturally supports relative position encoding
  - Generalises to sequences longer than seen during training
  - Enables efficient KV-cache usage during generation

Reference:
    Su et al. (2021) — "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes cosine and sine frequency tables for sequences up to
    ``max_seq_len`` tokens. The tables are extended automatically if a
    longer sequence is requested at forward time.

    Args:
        dim: Head dimension (``d_model // n_heads``). Must be even.
        base: Base for frequency computation (default: 10000, as in the paper).
        max_seq_len: Initial cache size. Extended lazily if exceeded.

    Example:
        >>> rope = RotaryEmbedding(dim=64)
        >>> cos, sin = rope(seq_len=128, device=torch.device("cpu"))
        >>> cos.shape
        torch.Size([128, 64])
    """

    def __init__(self, dim: int, base: int = 10_000, max_seq_len: int = 4096) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # inv_freq: (dim/2,) — reused across positions
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cache for initial max_seq_len
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._cached_len: int = 0

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        """Compute and cache cos/sin tables up to seq_len."""
        if seq_len <= self._cached_len and self._cos_cached is not None:
            return  # Cache is still valid

        self._cached_len = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        # Outer product → (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate to get (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    def forward(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (cos, sin) frequency tables for positions 0 … seq_len-1.

        Args:
            seq_len: Number of positions needed.
            device: Compute device.

        Returns:
            Tuple of ``(cos, sin)`` tensors, each of shape ``(seq_len, dim)``.
        """
        self._build_cache(seq_len, device)
        assert self._cos_cached is not None and self._sin_cached is not None
        return (
            self._cos_cached[:seq_len].to(device),
            self._sin_cached[:seq_len].to(device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of the last dimension to implement the RoPE rotation.

    Given x = [x1, x2] where x1 and x2 each have size dim/2, returns
    [-x2, x1] concatenated along the last dimension.

    Args:
        x: Tensor of shape ``(..., dim)``. ``dim`` must be even.

    Returns:
        Rotated tensor of the same shape.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    The rotation is applied independently to each query/key head:

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

    Args:
        q: Query tensor of shape ``(batch, n_heads, seq_len, head_dim)``.
        k: Key tensor of shape ``(batch, n_kv_heads, seq_len, head_dim)``.
        cos: Cosine table of shape ``(seq_len, head_dim)`` (from RotaryEmbedding).
        sin: Sine table of shape ``(seq_len, head_dim)`` (from RotaryEmbedding).

    Returns:
        Tuple ``(q_rotated, k_rotated)`` with the same shapes as inputs.

    Note:
        ``cos`` and ``sin`` are broadcast over batch and head dimensions
        automatically via unsqueeze on dimensions 0 and 1.
    """
    # Reshape cos/sin to broadcast: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
