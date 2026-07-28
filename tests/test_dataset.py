"""
Tests for dataset loading, preprocessing, and collation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slm.dataset.loader import TokenisedDataset, TextDataset, DatasetLoader
from slm.dataset.preprocessor import TextPreprocessor
from slm.dataset.collator import CausalLMCollator


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class _FakeTok:
    """Minimal fake tokenizer for testing."""
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def encode(self, text, add_special_tokens=True, **kwargs):
        return [self.bos_token_id] + [ord(c) % 100 + 3 for c in text[:20]] + [self.eos_token_id]


# ---------------------------------------------------------------------------
# TokenisedDataset
# ---------------------------------------------------------------------------

class TestTokenisedDataset:
    def test_load_and_length(self, tmp_path):
        n_tokens = 1000
        arr = np.arange(n_tokens, dtype=np.uint16)
        path = tmp_path / "data.bin"
        arr.tofile(path)

        ds = TokenisedDataset(path, seq_len=64)
        assert len(ds) > 0

    def test_item_shapes(self, tmp_path):
        arr = np.arange(200, dtype=np.uint16)
        path = tmp_path / "data.bin"
        arr.tofile(path)

        ds = TokenisedDataset(path, seq_len=32)
        item = ds[0]
        assert item["input_ids"].shape == (32,)
        assert item["labels"].shape == (32,)

    def test_labels_shifted_by_one(self, tmp_path):
        arr = np.arange(100, dtype=np.uint16)
        path = tmp_path / "data.bin"
        arr.tofile(path)

        ds = TokenisedDataset(path, seq_len=10)
        item = ds[0]
        # input[i+1] should equal label[i]
        assert torch.equal(item["input_ids"][1:], item["labels"][:-1])

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            TokenisedDataset("/nonexistent/path.bin", seq_len=64)


# ---------------------------------------------------------------------------
# TextDataset
# ---------------------------------------------------------------------------

class TestTextDataset:
    def test_creates_samples(self):
        tok = _FakeTok()
        texts = ["Hello world! " * 5, "Emily SLM is great! " * 5]
        ds = TextDataset(texts, tok, max_length=32)
        assert len(ds) > 0

    def test_item_shapes(self):
        tok = _FakeTok()
        ds = TextDataset(["test text"] * 10, tok, max_length=16)
        for i in range(len(ds)):
            item = ds[i]
            assert "input_ids" in item
            assert "labels" in item
            assert item["input_ids"].shape[0] == 16


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------

class TestDatasetLoader:
    def test_from_binary(self, tmp_path):
        arr = np.arange(500, dtype=np.uint16)
        path = tmp_path / "train.bin"
        arr.tofile(path)

        tok = _FakeTok()
        loader = DatasetLoader(tok, seq_len=32)
        ds = loader.from_binary(path)
        assert len(ds) > 0

    def test_from_strings(self):
        tok = _FakeTok()
        loader = DatasetLoader(tok, seq_len=16)
        ds = loader.from_strings(["hello world"] * 5)
        assert len(ds) > 0

    def test_from_text_files(self, tmp_path):
        f = tmp_path / "corpus.txt"
        f.write_text("Hello world! " * 50)
        tok = _FakeTok()
        loader = DatasetLoader(tok, seq_len=16)
        ds = loader.from_text_files([str(f)])
        assert len(ds) > 0

    def test_from_jsonl(self, tmp_path):
        import json
        f = tmp_path / "data.jsonl"
        with open(f, "w") as fp:
            for i in range(10):
                fp.write(json.dumps({"text": f"Sample text number {i} " * 5}) + "\n")
        tok = _FakeTok()
        loader = DatasetLoader(tok, seq_len=16)
        ds = loader.from_jsonl(str(f))
        assert len(ds) > 0

    def test_tokenise_and_save(self, tmp_path):
        tok = _FakeTok()
        texts = ["hello world " * 20] * 10
        train_p, val_p = DatasetLoader.tokenise_and_save(texts, tok, tmp_path, val_ratio=0.1)
        assert train_p.exists()
        assert val_p.exists()
        assert train_p.stat().st_size > 0


# ---------------------------------------------------------------------------
# TextPreprocessor
# ---------------------------------------------------------------------------

class TestTextPreprocessor:
    def test_removes_short_texts(self):
        pp = TextPreprocessor(min_length=50)
        texts = ["short", "a" * 100]
        result = pp.process(texts)
        assert len(result) == 1
        assert result[0] == "a" * 100

    def test_deduplication(self):
        pp = TextPreprocessor(min_length=5, dedup=True)
        texts = ["hello world duplicate"] * 5
        result = pp.process(texts)
        assert len(result) == 1

    def test_url_removal(self):
        pp = TextPreprocessor(min_length=5, remove_urls=True)
        texts = ["visit https://example.com for more info about stuff"]
        result = pp.process(texts)
        assert "https://example.com" not in result[0]

    def test_whitespace_normalisation(self):
        pp = TextPreprocessor(min_length=5)
        texts = ["hello   world  test"]
        result = pp.process(texts)
        assert "  " not in result[0]

    def test_reset_dedup(self):
        pp = TextPreprocessor(min_length=5, dedup=True)
        texts = ["hello world duplicate text here"]
        pp.process(texts)
        pp.reset_dedup()
        result = pp.process(texts)
        assert len(result) == 1  # Should accept again after reset


# ---------------------------------------------------------------------------
# CausalLMCollator
# ---------------------------------------------------------------------------

class TestCausalLMCollator:
    def test_output_keys(self):
        collator = CausalLMCollator(pad_token_id=0)
        samples = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([5, 6]),    "labels": torch.tensor([6, 7])},
        ]
        batch = collator(samples)
        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch

    def test_padding(self):
        collator = CausalLMCollator(pad_token_id=0)
        samples = [
            {"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([2, 3, 4, 5])},
            {"input_ids": torch.tensor([1, 2]),        "labels": torch.tensor([2, 3])},
        ]
        batch = collator(samples)
        # All rows must have the same length (length of longest = 4)
        assert batch["input_ids"].shape == (2, 4)
        assert batch["labels"].shape == (2, 4)

    def test_label_padding_uses_minus100(self):
        collator = CausalLMCollator(pad_token_id=0)
        samples = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([1]),        "labels": torch.tensor([2])},
        ]
        batch = collator(samples)
        # Padded positions in labels should be -100
        assert (batch["labels"][1, 1:] == -100).all()

    def test_attention_mask(self):
        collator = CausalLMCollator(pad_token_id=0)
        samples = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([5, 6]),    "labels": torch.tensor([6, 7])},
        ]
        batch = collator(samples)
        # Second row: positions 0,1 are real (True), position 2 is pad (False)
        assert batch["attention_mask"][1, 0].item() is True
        assert batch["attention_mask"][1, 2].item() is False

    def test_max_length_truncation(self):
        collator = CausalLMCollator(pad_token_id=0, max_length=3)
        samples = [
            {"input_ids": torch.tensor([1, 2, 3, 4, 5]), "labels": torch.tensor([2, 3, 4, 5, 6])},
        ]
        batch = collator(samples)
        assert batch["input_ids"].shape[1] == 3
