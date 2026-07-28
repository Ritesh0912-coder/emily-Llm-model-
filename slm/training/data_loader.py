"""
data_loader.py — DataLoader factory helpers for EmilyTrainer.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from slm.dataset.collator import CausalLMCollator


def build_train_loader(
    dataset: Dataset,
    batch_size: int,
    pad_token_id: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_seq_len: int | None = None,
) -> DataLoader:
    """Build a training DataLoader with shuffle and drop_last."""
    collator = CausalLMCollator(pad_token_id=pad_token_id, max_length=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True,
    )


def build_val_loader(
    dataset: Dataset,
    batch_size: int,
    pad_token_id: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_seq_len: int | None = None,
) -> DataLoader:
    """Build a validation DataLoader without shuffle."""
    collator = CausalLMCollator(pad_token_id=pad_token_id, max_length=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
