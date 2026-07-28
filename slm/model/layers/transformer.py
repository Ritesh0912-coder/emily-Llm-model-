"""
Transformer Decoder Block for Emily SLM.

Each ``TransformerBlock`` implements one layer of the decoder with:
- Pre-normalization (RMSNorm applied before each sub-layer)
- Causal self-attention (MHA or GQA, based on config)
- Feed-forward network (SwiGLU or GeLU, based on config)
- Residual connections around both sub-layers

Pre-norm architecture:
    x = x + Attn(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

This is more stable during training than post-norm and is used by all
modern LLMs (LLaMA, Mistral, Falcon, etc.).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from slm.model.layers.attention import CausalSelfAttention, GroupedQueryAttention, KVCache
from slm.model.layers.mlp import SwiGLU, GeLUMLP
from slm.model.layers.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single decoder-only Transformer block with pre-norm architecture.

    Selects the attention and FFN variant based on the provided config:
    - Attention: ``GroupedQueryAttention`` when ``n_kv_heads < n_heads``,
      else ``CausalSelfAttention``.
    - FFN: ``SwiGLU`` when ``use_swiglu=True``, else ``GeLUMLP``.
    - Norm: ``RMSNorm`` when ``use_rms_norm=True``, else ``nn.LayerNorm``.

    Args:
        config: ``ModelConfig`` instance with architecture hyperparameters.

    Shape:
        - Input ``x``:  ``(batch, seq_len, d_model)``
        - Output:       ``(batch, seq_len, d_model)``

    Example:
        >>> from slm.config import ModelConfig
        >>> block = TransformerBlock(ModelConfig.tiny())
        >>> x = torch.randn(2, 16, 128)
        >>> out, cache = block(x)
        >>> out.shape
        torch.Size([2, 16, 128])
    """

    def __init__(self, config: object) -> None:
        super().__init__()
        d_model: int = config.d_model  # type: ignore[attr-defined]
        use_rms_norm: bool = getattr(config, "use_rms_norm", True)
        use_swiglu: bool = getattr(config, "use_swiglu", True)
        n_heads: int = config.n_heads  # type: ignore[attr-defined]
        n_kv_heads: int = getattr(config, "n_kv_heads", n_heads)

        # ----- Normalisation -----
        if use_rms_norm:
            self.attn_norm: nn.Module = RMSNorm(d_model)
            self.ff_norm: nn.Module = RMSNorm(d_model)
        else:
            self.attn_norm = nn.LayerNorm(d_model)
            self.ff_norm = nn.LayerNorm(d_model)

        # ----- Attention -----
        if n_kv_heads < n_heads:
            self.attn: nn.Module = GroupedQueryAttention(config)
        else:
            self.attn = CausalSelfAttention(config)

        # ----- Feed-Forward -----
        d_ff: int = config.d_ff  # type: ignore[attr-defined]
        dropout: float = config.dropout  # type: ignore[attr-defined]
        if use_swiglu:
            self.ff: nn.Module = SwiGLU(d_model, d_ff, dropout=dropout)
        else:
            self.ff = GeLUMLP(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass through the transformer block.

        Args:
            x: Hidden states ``(batch, seq_len, d_model)``.
            mask: Optional additive attention mask.
            kv_cache: Optional KV cache for incremental decoding.

        Returns:
            Tuple ``(hidden_states, updated_kv_cache)`` where ``hidden_states``
            has the same shape as the input ``x``.
        """
        # Pre-norm + attention + residual
        attn_out, kv_cache = self.attn(self.attn_norm(x), mask=mask, kv_cache=kv_cache)
        x = x + attn_out

        # Pre-norm + FFN + residual
        x = x + self.ff(self.ff_norm(x))

        return x, kv_cache
