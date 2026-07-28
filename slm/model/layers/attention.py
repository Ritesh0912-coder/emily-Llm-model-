"""
Causal Self-Attention with KV Cache, RoPE, Flash Attention, and GQA.

Implements two attention variants:
- ``CausalSelfAttention``: Standard Multi-Head Attention (MHA).
- ``GroupedQueryAttention``: GQA — fewer KV heads than query heads (LLaMA-3 style).

Both support:
- Rotary Position Embeddings (RoPE)
- KV Cache for efficient autoregressive decoding
- Flash Attention via ``torch.nn.functional.scaled_dot_product_attention``
- Causal masking (lower-triangular)
- Attention dropout

References:
    - Vaswani et al. (2017) — Attention Is All You Need
    - Su et al. (2021) — RoPE
    - Ainslie et al. (2023) — GQA: Training Generalized Multi-Query Transformer Models
    - Dao et al. (2022) — FlashAttention
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.model.layers.rotary import RotaryEmbedding, apply_rotary_emb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

@dataclass
class KVCache:
    """
    Key-Value cache for efficient autoregressive decoding.

    During generation, we avoid recomputing K and V for already-processed
    tokens by caching them and appending new tokens incrementally.

    Attributes:
        keys: Cached key tensor ``(batch, n_kv_heads, cached_len, head_dim)``.
        values: Cached value tensor ``(batch, n_kv_heads, cached_len, head_dim)``.
        current_len: Number of tokens currently cached.
    """

    keys: torch.Tensor
    values: torch.Tensor
    current_len: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.current_len = self.keys.shape[2]

    def update(
        self, new_k: torch.Tensor, new_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key/value tensors and return the full cache.

        Args:
            new_k: New keys ``(batch, n_kv_heads, new_len, head_dim)``.
            new_v: New values ``(batch, n_kv_heads, new_len, head_dim)``.

        Returns:
            Tuple ``(full_keys, full_values)`` with all cached tokens.
        """
        self.keys = torch.cat([self.keys, new_k], dim=2)
        self.values = torch.cat([self.values, new_v], dim=2)
        self.current_len = self.keys.shape[2]
        return self.keys, self.values

    @classmethod
    def empty(
        cls,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "KVCache":
        """Create an empty (zero-length) KV cache."""
        empty_k = torch.zeros(batch_size, n_kv_heads, 0, head_dim, device=device, dtype=dtype)
        empty_v = torch.zeros(batch_size, n_kv_heads, 0, head_dim, device=device, dtype=dtype)
        return cls(keys=empty_k, values=empty_v)


# ---------------------------------------------------------------------------
# Causal Self-Attention  (standard MHA)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE and KV Cache.

    The attention is strictly causal — each position can only attend to
    positions ≤ its own index. This is enforced via an upper-triangular mask
    (or via ``is_causal=True`` in Flash Attention mode).

    Architecture:
        Q = x W_q     K = x W_k     V = x W_v
        Scores = Q Kᵀ / √head_dim   (+ causal mask)
        Attn = softmax(Scores) · V
        Out = Attn W_o

    Args:
        config: ``ModelConfig`` instance supplying ``d_model``, ``n_heads``,
            ``dropout``, ``use_rope``, ``rope_base``, and ``max_seq_len``.

    Shape:
        - Input ``x``:  ``(batch, seq_len, d_model)``
        - Output:       ``(batch, seq_len, d_model)``

    Example:
        >>> from slm.config import ModelConfig
        >>> attn = CausalSelfAttention(ModelConfig.tiny())
        >>> x = torch.randn(2, 16, 128)
        >>> out, cache = attn(x)
        >>> out.shape
        torch.Size([2, 16, 128])
    """

    def __init__(self, config: object) -> None:  # config: ModelConfig
        super().__init__()
        self.d_model: int = config.d_model  # type: ignore[attr-defined]
        self.n_heads: int = config.n_heads  # type: ignore[attr-defined]
        self.head_dim: int = self.d_model // self.n_heads
        self.dropout_p: float = config.dropout  # type: ignore[attr-defined]
        self.use_flash: bool = config.attention_type == "flash"  # type: ignore[attr-defined]

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        # Projection matrices — no bias (following LLaMA/Mistral convention)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout_p)
        self.resid_dropout = nn.Dropout(self.dropout_p)

        # Rotary embeddings (optional)
        self.rotary: Optional[RotaryEmbedding] = None
        if getattr(config, "use_rope", True):
            self.rotary = RotaryEmbedding(
                dim=self.head_dim,
                base=getattr(config, "rope_base", 10_000),
                max_seq_len=getattr(config, "max_seq_len", 4096),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise projection weights with scaled normal distribution."""
        std = 0.02
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        """
        Compute causal self-attention.

        Args:
            x: Input tensor ``(batch, seq_len, d_model)``.
            mask: Optional additive attention mask ``(seq_len, seq_len)`` or
                ``(batch, 1, seq_len, seq_len)``. Positions to mask should be
                set to ``-inf``.
            kv_cache: Optional KV cache for incremental decoding. When provided,
                only the new token(s) in ``x`` are projected; K/V are read from
                the cache and updated.

        Returns:
            Tuple ``(output, updated_kv_cache)`` where ``output`` has the same
            shape as ``x`` and ``updated_kv_cache`` is the updated cache (or
            ``None`` if no cache was passed).
        """
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.rotary is not None:
            offset = kv_cache.current_len if kv_cache is not None else 0
            cos, sin = self.rotary(T + offset, x.device)
            # Slice cos/sin for new tokens only
            cos_new = cos[offset: offset + T]
            sin_new = sin[offset: offset + T]
            q, k = apply_rotary_emb(q, k, cos_new, sin_new)

        # Update KV cache
        if kv_cache is not None:
            k_full, v_full = kv_cache.update(k, v)
        else:
            k_full, v_full = k, v

        # Attention computation
        if self.use_flash:
            # Flash Attention (PyTorch 2.0+ SDPA)
            # is_causal only valid when q and k have same seq length (prefill)
            is_causal = (kv_cache is None)
            out = F.scaled_dot_product_attention(
                q, k_full, v_full,
                attn_mask=None if is_causal else mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Manual scaled dot-product attention
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # (B, H, T, T_full)

            # Apply causal mask (upper triangle = -inf)
            T_full = k_full.shape[2]
            if mask is not None:
                scores = scores + mask
            else:
                # Build causal mask on-the-fly
                causal_mask = torch.full(
                    (T, T_full), float("-inf"), device=x.device, dtype=scores.dtype
                )
                causal_mask = torch.triu(causal_mask, diagonal=T_full - T + 1)
                scores = scores + causal_mask

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v_full)

        # Reshape back to (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, kv_cache

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"head_dim={self.head_dim}, use_flash={self.use_flash}"
        )


# ---------------------------------------------------------------------------
# Grouped Query Attention  (GQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    GQA uses fewer KV heads than query heads, reducing the KV cache size
    during generation without significantly degrading quality. When
    ``n_kv_heads == n_heads``, this reduces to standard MHA.

    The K and V tensors from each KV head are shared across a group of
    ``n_heads // n_kv_heads`` query heads via ``repeat_interleave``.

    Reference:
        Ainslie et al. (2023) — "GQA: Training Generalized Multi-Query
        Transformer Models from Multi-Head Checkpoints"
        https://arxiv.org/abs/2305.13245

    Args:
        config: ``ModelConfig`` with ``d_model``, ``n_heads``, ``n_kv_heads``,
            ``dropout``, ``use_rope``, ``rope_base``, ``max_seq_len``.

    Example:
        >>> from slm.config import ModelConfig
        >>> cfg = ModelConfig.small()  # n_heads=8, n_kv_heads=4
        >>> gqa = GroupedQueryAttention(cfg)
        >>> x = torch.randn(2, 32, 256)
        >>> out, _ = gqa(x)
        >>> out.shape
        torch.Size([2, 32, 256])
    """

    def __init__(self, config: object) -> None:
        super().__init__()
        self.d_model: int = config.d_model  # type: ignore[attr-defined]
        self.n_heads: int = config.n_heads  # type: ignore[attr-defined]
        self.n_kv_heads: int = getattr(config, "n_kv_heads", config.n_heads)  # type: ignore
        self.head_dim: int = self.d_model // self.n_heads
        self.n_groups: int = self.n_heads // self.n_kv_heads
        self.dropout_p: float = config.dropout  # type: ignore[attr-defined]
        self.use_flash: bool = config.attention_type == "flash"  # type: ignore[attr-defined]

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout_p)
        self.resid_dropout = nn.Dropout(self.dropout_p)

        self.rotary: Optional[RotaryEmbedding] = None
        if getattr(config, "use_rope", True):
            self.rotary = RotaryEmbedding(
                dim=self.head_dim,
                base=getattr(config, "rope_base", 10_000),
                max_seq_len=getattr(config, "max_seq_len", 4096),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        """
        Compute grouped-query attention.

        Args:
            x: ``(batch, seq_len, d_model)``
            mask: Optional additive mask.
            kv_cache: Optional KV cache for incremental decoding.

        Returns:
            ``(output, updated_kv_cache)``
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            offset = kv_cache.current_len if kv_cache is not None else 0
            cos, sin = self.rotary(T + offset, x.device)
            cos_new = cos[offset: offset + T]
            sin_new = sin[offset: offset + T]
            q, k = apply_rotary_emb(q, k, cos_new, sin_new)

        if kv_cache is not None:
            k_full, v_full = kv_cache.update(k, v)
        else:
            k_full, v_full = k, v

        # Expand KV heads to match Q heads via repeat_interleave
        k_exp = k_full.repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, T_full, head_dim)
        v_exp = v_full.repeat_interleave(self.n_groups, dim=1)

        if self.use_flash:
            is_causal = (kv_cache is None)
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=None if is_causal else mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
            T_full = k_exp.shape[2]
            if mask is not None:
                scores = scores + mask
            else:
                causal_mask = torch.full(
                    (T, T_full), float("-inf"), device=x.device, dtype=scores.dtype
                )
                causal_mask = torch.triu(causal_mask, diagonal=T_full - T + 1)
                scores = scores + causal_mask
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v_exp)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.resid_dropout(self.out_proj(out))
        return out, kv_cache

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, n_groups={self.n_groups}"
        )
