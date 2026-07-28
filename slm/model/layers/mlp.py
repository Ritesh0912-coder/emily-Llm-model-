"""
Feed-Forward Network (FFN) layers for Emily SLM.

Two implementations are provided:

- ``SwiGLU``: Gated Linear Unit with SiLU activation, used by LLaMA, Mistral.
  Outperforms GeLU on most benchmarks while being parameter-efficient.
  Formula: ``down_proj(silu(gate_proj(x)) * up_proj(x))``

- ``GeLUMLP``: Standard GPT-2 style MLP with GeLU activation.
  Formula: ``fc2(gelu(fc1(x)))``

References:
    - Shazeer (2020) — "GLU Variants Improve Transformer" (SwiGLU)
      https://arxiv.org/abs/2002.05202
    - Hendrycks & Gimpel (2016) — "Gaussian Error Linear Units (GELUs)"
      https://arxiv.org/abs/1606.08415
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Combines the Swish (SiLU) activation with a Gated Linear Unit (GLU):

        output = down_proj(silu(gate_proj(x)) ⊙ up_proj(x))

    where ⊙ is element-wise multiplication. This gating mechanism allows the
    network to selectively pass information, resulting in superior performance
    compared to standard GeLU FFNs.

    The inner dimension ``d_ff`` should be roughly ``8/3 × d_model`` (rounded
    to a multiple of 64 for hardware efficiency). The config system accepts the
    raw ``d_ff`` value; this class uses it directly.

    Args:
        d_model: Input/output feature dimension.
        d_ff: Inner (hidden) dimension of the FFN.
        dropout: Dropout probability applied to the output.

    Shape:
        - Input:  ``(batch, seq_len, d_model)``
        - Output: ``(batch, seq_len, d_model)``

    Example:
        >>> ffn = SwiGLU(d_model=512, d_ff=2048)
        >>> x = torch.randn(2, 64, 512)
        >>> out = ffn(x)
        >>> out.shape
        torch.Size([2, 64, 512])
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Three projection matrices (no bias, following LLaMA)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights with small normal distribution."""
        for layer in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward transformation.

        Args:
            x: Input tensor ``(batch, seq_len, d_model)``.

        Returns:
            Output tensor ``(batch, seq_len, d_model)``.
        """
        # gate_proj produces the gating signal; up_proj produces the value
        gate = F.silu(self.gate_proj(x))   # (B, T, d_ff)
        value = self.up_proj(x)             # (B, T, d_ff)
        hidden = gate * value               # element-wise gating
        return self.dropout(self.down_proj(hidden))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"

    @staticmethod
    def recommended_d_ff(d_model: int, multiple_of: int = 64) -> int:
        """
        Compute the recommended ``d_ff`` for SwiGLU (≈ 8/3 × d_model).

        The result is rounded up to the nearest multiple of ``multiple_of``
        for hardware alignment.

        Args:
            d_model: Model hidden dimension.
            multiple_of: Round up to this multiple (default 64).

        Returns:
            Recommended ``d_ff`` value.

        Example:
            >>> SwiGLU.recommended_d_ff(512)
            1408
        """
        d_ff_raw = int(8 * d_model / 3)
        return multiple_of * ((d_ff_raw + multiple_of - 1) // multiple_of)


class GeLUMLP(nn.Module):
    """
    Standard GPT-2 style MLP with GeLU activation.

    Architecture:
        fc1: d_model → d_ff   (with bias)
        gelu activation
        fc2: d_ff → d_model   (with bias)
        dropout

    Args:
        d_model: Input/output feature dimension.
        d_ff: Inner hidden dimension (typically 4 × d_model).
        dropout: Dropout probability applied after the output projection.

    Shape:
        - Input:  ``(batch, seq_len, d_model)``
        - Output: ``(batch, seq_len, d_model)``

    Example:
        >>> mlp = GeLUMLP(d_model=512, d_ff=2048)
        >>> x = torch.randn(2, 64, 512)
        >>> out = mlp(x)
        >>> out.shape
        torch.Size([2, 64, 512])
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeLU MLP transformation.

        Args:
            x: Input tensor ``(batch, seq_len, d_model)``.

        Returns:
            Output tensor ``(batch, seq_len, d_model)``.
        """
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"
