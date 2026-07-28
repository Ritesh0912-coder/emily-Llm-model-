"""
Embedding layers for Emily SLM.

Provides:
- ``TokenEmbedding``: Maps token IDs to dense vectors, scaled by √d_model.
- ``SinusoidalPositionalEmbedding``: Fixed sin/cos positional encodings (Vaswani 2017).
- ``LearnablePositionalEmbedding``: Trainable positional encodings (GPT-2 style).

When RoPE is enabled (``use_rope=True`` in config), positional information is
injected inside the attention mechanism rather than here. In that case, only
``TokenEmbedding`` is used at the input stage.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with √d_model scaling.

    Converts discrete token IDs into continuous ``d_model``-dimensional vectors.
    The output is scaled by ``sqrt(d_model)`` following the original Transformer
    paper convention, which helps balance the magnitude of embeddings against
    positional encodings.

    Args:
        vocab_size: Total vocabulary size (number of unique tokens).
        d_model: Embedding dimension.
        padding_idx: Token ID used for padding (its embedding is forced to zero
            and receives no gradient). Pass ``None`` to disable.

    Shape:
        - Input:  ``(batch, seq_len)``  — integer token IDs
        - Output: ``(batch, seq_len, d_model)``

    Example:
        >>> emb = TokenEmbedding(vocab_size=32000, d_model=512)
        >>> ids = torch.randint(0, 32000, (2, 64))
        >>> out = emb(ids)
        >>> out.shape
        torch.Size([2, 64, 512])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # Initialise with small normal values (std = 0.02, as in GPT-2)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[padding_idx])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs and apply √d_model scaling.

        Args:
            input_ids: Long tensor of shape ``(batch, seq_len)``.

        Returns:
            Float tensor of shape ``(batch, seq_len, d_model)``.
        """
        return self.embedding(input_ids) * self.scale

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, d_model={self.d_model}"


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embeddings (Vaswani et al., 2017).

    Positional encodings are precomputed and not updated during training.
    Even indices use ``sin``, odd indices use ``cos``:

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Embedding dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.

    Shape:
        - Output: ``(seq_len, d_model)``  (no batch dimension)

    Example:
        >>> pe = SinusoidalPositionalEmbedding(d_model=512, max_seq_len=1024)
        >>> pos = pe(seq_len=64, device=torch.device("cpu"))
        >>> pos.shape
        torch.Size([64, 512])
    """

    def __init__(self, d_model: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.d_model = d_model

        # Precompute the encoding table once
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with .to(device) but is not a parameter
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Return positional encodings for the given sequence length.

        Args:
            seq_len: Number of positions to return.
            device: Target device (the buffer is moved here automatically).

        Returns:
            Tensor of shape ``(seq_len, d_model)``.
        """
        return self.pe[:seq_len].to(device)  # type: ignore[index]


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings (GPT-2 style).

    Each position up to ``max_seq_len`` gets its own trainable embedding vector.
    Used when RoPE is disabled (``use_rope=False``).

    Args:
        d_model: Embedding dimension.
        max_seq_len: Maximum sequence length (determines embedding table size).

    Shape:
        - Output: ``(seq_len, d_model)``

    Example:
        >>> pe = LearnablePositionalEmbedding(d_model=512, max_seq_len=1024)
        >>> pos = pe(seq_len=64, device=torch.device("cpu"))
        >>> pos.shape
        torch.Size([64, 512])
    """

    def __init__(self, d_model: int, max_seq_len: int = 4096) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Return positional embeddings for positions 0 … seq_len-1.

        Args:
            seq_len: Number of positions (must be ≤ ``max_seq_len``).
            device: Target device.

        Returns:
            Tensor of shape ``(seq_len, d_model)``.

        Raises:
            ValueError: If ``seq_len`` exceeds ``max_seq_len``.
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_seq_len={self.max_seq_len}"
