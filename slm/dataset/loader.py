"""
Dataset loader for Emily SLM.

Supports loading from:
- Raw text files (.txt, .md)
- JSONL files ({"text": "..."} per line)
- HuggingFace datasets
- Pre-tokenised binary files (numpy mmap)
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


class TokenisedDataset(Dataset):
    """
    Memory-mapped dataset over a pre-tokenised binary file.

    Binary format: flat array of uint16 token IDs written with numpy.
    Sequences of length ``seq_len + 1`` are extracted with stride 1
    (or stride ``seq_len`` for non-overlapping windows).

    Args:
        path: Path to the ``.bin`` file.
        seq_len: Sequence length for each sample (input length = seq_len,
            target = input shifted by 1).
        overlapping: If ``True``, use stride-1 windows; else stride=seq_len.
    """

    def __init__(self, path: str | Path, seq_len: int, overlapping: bool = False) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self.seq_len = seq_len
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.stride = 1 if overlapping else seq_len
        self.n_samples = max(0, (len(self.data) - seq_len - 1) // self.stride)
        logger.info(
            f"TokenisedDataset loaded | path={path} | tokens={len(self.data):,} | "
            f"samples={self.n_samples:,} | seq_len={seq_len}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.stride
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "labels": y}


class TextDataset(Dataset):
    """
    In-memory dataset built from raw text strings.

    Each text is tokenised on construction. Sequences longer than
    ``max_length`` are split into non-overlapping chunks.

    Args:
        texts: List of raw text strings.
        tokenizer: ``EmilyTokenizer`` instance for encoding.
        max_length: Maximum token length per sample.
    """

    def __init__(self, texts: list[str], tokenizer: object, max_length: int = 1024) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)  # type: ignore
            # Split into chunks of max_length+1 (for x, y pairs)
            for i in range(0, max(1, len(ids) - max_length), max_length):
                chunk = ids[i : i + max_length + 1]
                if len(chunk) < 2:
                    continue
                # Pad to max_length+1 if needed
                if len(chunk) <= max_length:
                    chunk = chunk + [tokenizer.pad_token_id] * (max_length + 1 - len(chunk))  # type: ignore
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.samples.append({"input_ids": x, "labels": y})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


class DatasetLoader:
    """
    Unified dataset loader supporting multiple input formats.

    Example:
        >>> loader = DatasetLoader(tokenizer, seq_len=256)
        >>> dataset = loader.from_text_files(["data/raw/corpus.txt"])
        >>> dataset = loader.from_jsonl("data/raw/train.jsonl")
        >>> dataset = loader.from_binary("datasets/tokenized/train.bin")
    """

    def __init__(self, tokenizer: object, seq_len: int = 1024) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def from_binary(self, path: str | Path, overlapping: bool = False) -> TokenisedDataset:
        """Load from a pre-tokenised numpy binary file."""
        return TokenisedDataset(path, self.seq_len, overlapping=overlapping)

    def from_text_files(self, paths: list[str | Path]) -> TextDataset:
        """Load and tokenise plain text files."""
        texts: list[str] = []
        for p in paths:
            p = Path(p)
            if not p.exists():
                logger.warning(f"Text file not found, skipping: {p}")
                continue
            texts.append(p.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(texts)} text files")
        return TextDataset(texts, self.tokenizer, max_length=self.seq_len)

    def from_jsonl(
        self, path: str | Path, text_key: str = "text"
    ) -> TextDataset:
        """Load from a JSONL file where each line has a text field."""
        path = Path(path)
        texts: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if text_key in obj:
                    texts.append(obj[text_key])
        logger.info(f"Loaded {len(texts)} texts from {path}")
        return TextDataset(texts, self.tokenizer, max_length=self.seq_len)

    def from_strings(self, texts: list[str]) -> TextDataset:
        """Build a dataset from an in-memory list of strings."""
        return TextDataset(texts, self.tokenizer, max_length=self.seq_len)

    @staticmethod
    def tokenise_and_save(
        texts: list[str],
        tokenizer: object,
        output_path: str | Path,
        val_ratio: float = 0.05,
    ) -> tuple[Path, Path]:
        """
        Tokenise texts and write train/val binary files.

        Args:
            texts: Raw text strings.
            tokenizer: EmilyTokenizer instance.
            output_path: Directory to write train.bin and val.bin.
            val_ratio: Fraction of tokens held out for validation.

        Returns:
            Tuple of (train_path, val_path).
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        all_ids: list[int] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)  # type: ignore
            all_ids.extend(ids)

        arr = np.array(all_ids, dtype=np.uint16)
        n_val = max(1, int(len(arr) * val_ratio))
        val_arr = arr[:n_val]
        train_arr = arr[n_val:]

        train_path = output_path / "train.bin"
        val_path = output_path / "val.bin"
        train_arr.tofile(train_path)
        val_arr.tofile(val_path)

        logger.info(
            f"Saved | train={len(train_arr):,} tokens → {train_path} | "
            f"val={len(val_arr):,} tokens → {val_path}"
        )
        return train_path, val_path
