"""
Tests for device utilities, logging setup, and checkpoint save/load.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from slm.utils.device import (
    get_device,
    get_dtype,
    count_parameters,
    format_parameters,
    set_seed,
    memory_summary,
)
from slm.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    get_latest_checkpoint,
    checkpoint_filename,
)
from slm.config import ModelConfig, TrainingConfig, EmilyConfig


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_valid_device_type(self):
        device = get_device()
        assert device.type in ("cpu", "cuda", "mps")


class TestGetDtype:
    @pytest.mark.parametrize("dtype_str,expected", [
        ("bfloat16", torch.bfloat16),
        ("float16",  torch.float16),
        ("float32",  torch.float32),
    ])
    def test_valid_dtype(self, dtype_str: str, expected: torch.dtype):
        assert get_dtype(dtype_str) == expected

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError):
            get_dtype("float8")


class TestCountParameters:
    def test_simple_linear(self):
        model = nn.Linear(10, 5)
        assert count_parameters(model) == 10 * 5 + 5  # weight + bias

    def test_trainable_only(self):
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        assert count_parameters(model, trainable_only=True) == 0
        assert count_parameters(model, trainable_only=False) == 55


class TestFormatParameters:
    @pytest.mark.parametrize("n,expected", [
        (125_000_000, "125.0M"),
        (1_300_000_000, "1.3B"),
        (47_000, "47.0K"),
        (999, "999"),
    ])
    def test_formatting(self, n: int, expected: str):
        assert format_parameters(n) == expected


class TestSetSeed:
    def test_reproducible_tensors(self):
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_different_seeds_differ(self):
        set_seed(1)
        t1 = torch.randn(100)
        set_seed(2)
        t2 = torch.randn(100)
        assert not torch.allclose(t1, t2)


class TestMemorySummary:
    def test_returns_dict(self):
        summary = memory_summary()
        assert isinstance(summary, dict)
        assert "allocated_gb" in summary
        assert "total_gb" in summary


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestCheckpoints:
    def test_save_and_load(self, tmp_path):
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = ModelConfig.tiny()

        ckpt_path = tmp_path / "step-0000001.pt"
        save_checkpoint(model, optimizer, None, step=1, loss=2.5, config=config, path=ckpt_path)
        assert ckpt_path.exists()

        # Modify model weights
        with torch.no_grad():
            model.fc.weight.fill_(999.0)

        # Load back
        state = load_checkpoint(ckpt_path, model, optimizer)
        assert state["step"] == 1
        assert abs(state["loss"] - 2.5) < 1e-6
        assert not torch.all(model.fc.weight == 999.0)

    def test_list_checkpoints_empty(self, tmp_path):
        result = list_checkpoints(tmp_path)
        assert result == []

    def test_list_checkpoints_sorted(self, tmp_path):
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cfg = ModelConfig.tiny()
        for step in [3000, 1000, 2000]:
            save_checkpoint(model, optimizer, None, step=step, loss=1.0, config=cfg,
                            path=tmp_path / checkpoint_filename(step))
        ckpts = list_checkpoints(tmp_path)
        steps = [int(p.stem.split("-")[1]) for p in ckpts]
        assert steps == sorted(steps)

    def test_get_latest_checkpoint(self, tmp_path):
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cfg = ModelConfig.tiny()
        for step in [1000, 2000, 3000]:
            save_checkpoint(model, optimizer, None, step=step, loss=1.0, config=cfg,
                            path=tmp_path / checkpoint_filename(step))
        latest = get_latest_checkpoint(tmp_path)
        assert latest is not None
        assert "0003000" in latest.name

    def test_get_latest_nonexistent_dir(self, tmp_path):
        result = get_latest_checkpoint(tmp_path / "nonexistent")
        assert result is None

    def test_load_nonexistent_raises(self, tmp_path):
        model = _TinyModel()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "missing.pt", model)

    def test_checkpoint_filename_format(self):
        name = checkpoint_filename(1000)
        assert name == "step-0001000.pt"
        name2 = checkpoint_filename(1_000_000)
        assert name2 == "step-1000000.pt"
