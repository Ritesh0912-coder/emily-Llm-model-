"""
Evaluation engine for Emily SLM.

Computes perplexity on a validation set and runs optional benchmark tasks.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from slm.dataset.collator import CausalLMCollator
from slm.utils.device import get_device, get_dtype

logger = logging.getLogger(__name__)


class EmilyEvaluator:
    """
    Evaluator for Emily SLM.

    Computes:
    - Perplexity (PPL) on a held-out dataset
    - Token accuracy
    - Bits-per-character (BPC) estimate

    Args:
        model: Trained ``EmilySLM`` model.
        tokenizer: ``EmilyTokenizer`` instance.
        device: Compute device. Auto-detected if ``None``.

    Example:
        >>> evaluator = EmilyEvaluator(model, tokenizer)
        >>> metrics = evaluator.evaluate(val_dataset, batch_size=16)
        >>> print(f"PPL: {metrics['perplexity']:.2f}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: object,
        device: Optional[torch.device] = None,
        amp_dtype: str = "bfloat16",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or get_device()
        self.amp_dtype = get_dtype(amp_dtype)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        num_workers: int = 0,
        max_batches: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Evaluate perplexity and accuracy on a dataset.

        Args:
            dataset: Evaluation ``Dataset`` producing ``{"input_ids", "labels"}``.
            batch_size: Evaluation batch size.
            num_workers: DataLoader worker processes.
            max_batches: Limit evaluation to this many batches (``None`` = all).

        Returns:
            Dict with keys: ``perplexity``, ``loss``, ``accuracy``, ``bpc``.
        """
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        collator = CausalLMCollator(pad_token_id=pad_id)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
        )

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        n_batches = 0

        for batch in loader:
            if max_batches is not None and n_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            use_amp = self.device.type == "cuda"
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_amp):
                out = self.model(input_ids)
                logits = out["logits"]  # (B, T, V)

            # Loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid_mask = labels.view(-1) != -100
            n_valid = valid_mask.sum().item()

            total_loss += loss.item()
            total_tokens += n_valid

            # Accuracy
            preds = logits.view(-1, logits.size(-1)).argmax(-1)
            correct = (preds[valid_mask] == labels.view(-1)[valid_mask]).sum().item()
            total_correct += correct
            n_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))
        accuracy = total_correct / max(total_tokens, 1)
        bpc = avg_loss / math.log(2)  # bits per token (approx bits per char)

        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "bpc": bpc,
            "total_tokens": total_tokens,
        }
        logger.info(
            f"Evaluation | loss={avg_loss:.4f} | ppl={perplexity:.2f} | "
            f"acc={accuracy:.4f} | bpc={bpc:.4f}"
        )
        return metrics

    @torch.no_grad()
    def evaluate_text(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> dict[str, float]:
        """
        Evaluate directly on a list of text strings.

        Args:
            texts: List of text strings to evaluate.
            batch_size: Processing batch size.

        Returns:
            Same metrics dict as ``evaluate()``.
        """
        from slm.dataset.loader import TextDataset
        dataset = TextDataset(texts, self.tokenizer, max_length=512)
        return self.evaluate(dataset, batch_size=batch_size)
