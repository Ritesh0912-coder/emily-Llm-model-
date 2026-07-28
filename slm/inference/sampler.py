"""
Sampling utilities for Emily SLM inference.

Standalone sampling functions decoupled from the model, useful for
testing and custom generation pipelines.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Sampler:
    """
    Token sampling strategies for language model decoding.

    All methods operate on raw logit tensors and return a sampled token ID.

    Example:
        >>> logits = torch.randn(32000)
        >>> token_id = Sampler.sample(logits, temperature=0.8, top_k=50, top_p=0.9)
    """

    @staticmethod
    def greedy(logits: torch.Tensor) -> int:
        """Return the token with the highest logit (argmax)."""
        return int(logits.argmax(dim=-1).item())

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> int:
        """
        Sample a token using temperature, top-k, and top-p filtering.

        Args:
            logits: Raw logits tensor ``(vocab_size,)``.
            temperature: Scaling factor. ``> 1`` = more random, ``< 1`` = sharper.
            top_k: Keep only top-k tokens. ``0`` disables.
            top_p: Nucleus probability. ``1.0`` disables.

        Returns:
            Sampled integer token ID.
        """
        if logits.dim() != 1:
            raise ValueError(f"Expected 1-D logits, got shape {logits.shape}")

        logits = logits.clone().float()

        # Temperature
        if temperature != 1.0 and temperature > 0.0:
            logits /= temperature

        # Top-K
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            top_vals, _ = torch.topk(logits, k)
            logits[logits < top_vals[-1]] = float("-inf")

        # Top-P (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def beam(logits: torch.Tensor, beam_scores: torch.Tensor, num_beams: int) -> torch.Tensor:
        """
        Return top-num_beams (score, token_id) pairs for beam search.

        Args:
            logits: ``(vocab_size,)`` raw logits.
            beam_scores: Current beam cumulative log-probs ``(num_beams,)``.
            num_beams: Beam width.

        Returns:
            Top ``(num_beams,)`` token indices.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        expanded = beam_scores.unsqueeze(-1) + log_probs.unsqueeze(0)  # (beams, vocab)
        top_k_vals, top_k_idx = torch.topk(expanded.view(-1), k=num_beams)
        return top_k_idx
