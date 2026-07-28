"""
Data collator for causal language model training.

Pads sequences to the same length within a batch and creates
labels with -100 masking for pad positions.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


class CausalLMCollator:
    """
    Collate variable-length token sequences into padded batches.

    Labels are identical to input_ids shifted by 1 (handled in the Dataset),
    with pad positions replaced by ``-100`` so they are ignored by
    ``nn.CrossEntropyLoss``.

    Args:
        pad_token_id: ID of the padding token.
        max_length: Optional maximum sequence length. Sequences longer than
            this are truncated. If ``None``, no truncation is applied.

    Example:
        >>> collator = CausalLMCollator(pad_token_id=0)
        >>> batch = collator([{"input_ids": t1, "labels": l1}, ...])
        >>> batch["input_ids"].shape
        torch.Size([B, max_seq_len])
    """

    def __init__(self, pad_token_id: int = 0, max_length: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate a list of sample dicts into a padded batch.

        Args:
            samples: Each sample must have ``"input_ids"`` and ``"labels"``
                as 1-D ``LongTensor``s.

        Returns:
            Dict with:
            - ``"input_ids"``:      ``(B, L)``  long tensor
            - ``"labels"``:         ``(B, L)``  long tensor, -100 at pad positions
            - ``"attention_mask"``: ``(B, L)``  bool tensor, True = real token
        """
        input_ids_list = [s["input_ids"] for s in samples]
        labels_list = [s["labels"] for s in samples]

        if self.max_length is not None:
            input_ids_list = [ids[: self.max_length] for ids in input_ids_list]
            labels_list = [lbl[: self.max_length] for lbl in labels_list]

        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(self.pad_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
