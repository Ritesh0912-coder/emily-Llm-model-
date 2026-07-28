"""
Tests for slm.config — ModelConfig, TrainingConfig, EmilyConfig, and factories.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from slm.config import ModelConfig, TrainingConfig, DatasetConfig, TokenizerConfig, EmilyConfig


class TestModelConfig:
    def test_defaults_are_sensible(self):
        cfg = ModelConfig()
        assert cfg.d_model > 0
        assert cfg.n_heads > 0
        assert cfg.n_layers > 0
        assert cfg.vocab_size > 0
        assert cfg.d_model % cfg.n_heads == 0

    def test_validation_raises_on_bad_head_ratio(self):
        with pytest.raises(ValueError, match="d_model"):
            ModelConfig(d_model=127, n_heads=4)  # 127 % 4 != 0

    def test_validation_raises_on_invalid_attention_type(self):
        with pytest.raises(ValueError, match="attention_type"):
            ModelConfig(attention_type="invalid")

    def test_validation_raises_on_bad_dropout(self):
        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=1.5)

    def test_head_dim_property(self):
        cfg = ModelConfig(d_model=512, n_heads=8)
        assert cfg.head_dim == 64

    def test_n_groups_property(self):
        cfg = ModelConfig(d_model=256, n_heads=8, n_kv_heads=4)
        assert cfg.n_groups == 2

    def test_to_dict_roundtrip(self):
        cfg = ModelConfig.tiny()
        d = cfg.to_dict()
        cfg2 = ModelConfig.from_dict(d)
        assert cfg.d_model == cfg2.d_model
        assert cfg.n_heads == cfg2.n_heads
        assert cfg.vocab_size == cfg2.vocab_size

    def test_from_dict_ignores_unknown_keys(self):
        d = ModelConfig.tiny().to_dict()
        d["nonexistent_key"] = "some_value"
        cfg = ModelConfig.from_dict(d)
        assert cfg.d_model == 128


class TestModelConfigFactories:
    @pytest.mark.parametrize("factory,expected_d_model", [
        ("tiny",   128),
        ("small",  256),
        ("base",   512),
        ("medium", 768),
        ("large",  1024),
    ])
    def test_factory_methods(self, factory: str, expected_d_model: int):
        cfg = getattr(ModelConfig, factory)()
        assert cfg.d_model == expected_d_model
        assert cfg.d_model % cfg.n_heads == 0  # valid head ratio

    def test_all_factories_pass_validation(self):
        for name in ["tiny", "small", "base", "medium", "large"]:
            cfg = getattr(ModelConfig, name)()
            assert isinstance(cfg, ModelConfig)


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate > 0
        assert cfg.min_lr < cfg.learning_rate
        assert cfg.max_steps > 0

    def test_invalid_lr(self):
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-1e-4, min_lr=-1e-5)

    def test_min_lr_must_be_less_than_lr(self):
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=1e-4, min_lr=1e-3)  # min > lr

    def test_invalid_amp_dtype(self):
        with pytest.raises(ValueError):
            TrainingConfig(amp_dtype="float8")


class TestEmilyConfig:
    def test_from_dict(self):
        d = {
            "model": {"d_model": 128, "n_heads": 4, "n_kv_heads": 4, "n_layers": 2,
                      "vocab_size": 4096, "context_length": 256, "d_ff": 512},
            "training": {"learning_rate": 3e-4, "min_lr": 3e-5},
            "tokenizer": {},
            "dataset": {},
        }
        cfg = EmilyConfig.from_dict(d)
        assert cfg.model.d_model == 128
        assert cfg.training.learning_rate == 3e-4

    def test_from_yaml(self, tmp_path: Path):
        import yaml
        yaml_content = {
            "model": {"name": "test", "d_model": 128, "n_heads": 4, "n_kv_heads": 4,
                      "n_layers": 2, "vocab_size": 1024, "context_length": 64, "d_ff": 256,
                      "dropout": 0.1, "use_rope": True, "rope_base": 10000,
                      "attention_type": "standard", "use_rms_norm": True, "use_swiglu": True,
                      "tie_embeddings": True, "max_seq_len": 64},
            "training": {"learning_rate": 1e-3, "min_lr": 1e-4, "max_steps": 100,
                         "batch_size": 4, "gradient_accumulation_steps": 1,
                         "eval_interval": 10, "save_interval": 50, "log_interval": 10,
                         "warmup_steps": 5, "weight_decay": 0.1, "max_grad_norm": 1.0,
                         "optimizer": "adamw", "scheduler": "cosine",
                         "use_amp": False, "amp_dtype": "float32", "seed": 0,
                         "compile": False, "checkpoint_dir": "ckpt", "log_dir": "logs",
                         "wandb_project": "test", "wandb_run_name": "test", "wandb_enabled": False},
            "tokenizer": {"model_type": "bpe", "vocab_size": 1024, "model_path": "tok.json"},
            "dataset": {"train_path": "train.bin", "val_path": "val.bin", "max_seq_len": 64},
        }
        cfg_path = tmp_path / "test.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(yaml_content, f)

        cfg = EmilyConfig.from_yaml(cfg_path)
        assert cfg.model.d_model == 128
        assert cfg.model.n_layers == 2

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            EmilyConfig.from_yaml("nonexistent/config.yaml")

    def test_to_yaml_roundtrip(self, tmp_path: Path):
        cfg = EmilyConfig.tiny()
        out_path = tmp_path / "out.yaml"
        cfg.to_yaml(out_path)
        assert out_path.exists()
        cfg2 = EmilyConfig.from_yaml(out_path)
        assert cfg.model.d_model == cfg2.model.d_model
        assert cfg.training.max_steps == cfg2.training.max_steps

    def test_tiny_factory(self):
        cfg = EmilyConfig.tiny()
        assert cfg.model.d_model == 128
        assert cfg.model.n_layers == 2
