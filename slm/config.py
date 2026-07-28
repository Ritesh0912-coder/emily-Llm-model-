"""
Emily SLM Configuration System.

Provides dataclasses for all model, training, dataset, and tokenizer
configuration. Supports loading from YAML files, dict serialization,
and factory methods for standard sizes.

Example:
    >>> config = EmilyConfig.from_yaml("configs/tiny.yaml")
    >>> model_cfg = config.model
    >>> print(model_cfg.d_model)
    128
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Configuration for the Emily SLM transformer architecture.

    Attributes:
        name: Human-readable model name.
        vocab_size: Size of the token vocabulary.
        context_length: Maximum sequence length (used for positional encodings).
        d_model: Hidden dimension size (embedding dimension).
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value attention heads (for GQA; set equal to
            n_heads for standard MHA).
        n_layers: Number of transformer decoder blocks.
        d_ff: Feed-forward network inner dimension.
        dropout: Dropout probability applied in attention and FFN.
        attention_type: Attention implementation — "standard" uses manual SDPA,
            "flash" uses torch.nn.functional.scaled_dot_product_attention.
        tie_embeddings: Whether to tie the token embedding weights to the LM head.
        use_rms_norm: Use RMSNorm instead of LayerNorm.
        use_swiglu: Use SwiGLU activation in FFN instead of GeLU.
        use_rope: Apply Rotary Position Embeddings (RoPE).
        rope_base: Base for RoPE frequency computation (default: 10000).
        max_seq_len: Maximum sequence length for RoPE cache precomputation.
    """

    name: str = "emily-base"
    vocab_size: int = 32000
    context_length: int = 1024
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    attention_type: str = "standard"
    tie_embeddings: bool = True
    use_rms_norm: bool = True
    use_swiglu: bool = True
    use_rope: bool = True
    rope_base: int = 10000
    max_seq_len: int = 1024

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}). "
                f"Got d_model % n_heads = {self.d_model % self.n_heads}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) "
                f"for Grouped Query Attention."
            )
        if self.attention_type not in ("standard", "flash"):
            raise ValueError(
                f"attention_type must be 'standard' or 'flash', got {self.attention_type!r}"
            )
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        # Sync max_seq_len with context_length
        if self.max_seq_len != self.context_length:
            self.max_seq_len = self.context_length

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    @property
    def n_groups(self) -> int:
        """Number of query head groups per KV head (for GQA)."""
        return self.n_heads // self.n_kv_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        """
        Create a ModelConfig from a dictionary.

        Args:
            d: Dictionary with configuration key-value pairs.

        Returns:
            Constructed ModelConfig instance.
        """
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""
        return asdict(self)

    # ----- Factory methods -----

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Tiny model (~1M params) — for unit testing and rapid iteration."""
        return cls(
            name="emily-tiny",
            vocab_size=4096,
            context_length=256,
            d_model=128,
            n_heads=4,
            n_kv_heads=4,
            n_layers=2,
            d_ff=512,
            dropout=0.1,
            attention_type="standard",
            tie_embeddings=True,
            use_rms_norm=True,
            use_swiglu=True,
            use_rope=True,
            rope_base=10000,
            max_seq_len=256,
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small model (~10M params) — for consumer GPU training."""
        return cls(
            name="emily-small",
            vocab_size=16384,
            context_length=512,
            d_model=256,
            n_heads=8,
            n_kv_heads=4,
            n_layers=4,
            d_ff=1024,
            dropout=0.1,
            attention_type="standard",
            tie_embeddings=True,
            use_rms_norm=True,
            use_swiglu=True,
            use_rope=True,
            rope_base=10000,
            max_seq_len=512,
        )

    @classmethod
    def base(cls) -> "ModelConfig":
        """Base model (~85M params) — baseline research model."""
        return cls(
            name="emily-base",
            vocab_size=32000,
            context_length=1024,
            d_model=512,
            n_heads=8,
            n_kv_heads=4,
            n_layers=8,
            d_ff=2048,
            dropout=0.1,
            attention_type="flash",
            tie_embeddings=True,
            use_rms_norm=True,
            use_swiglu=True,
            use_rope=True,
            rope_base=10000,
            max_seq_len=1024,
        )

    @classmethod
    def medium(cls) -> "ModelConfig":
        """Medium model (~117M params) — GPT-2 scale."""
        return cls(
            name="emily-medium",
            vocab_size=32000,
            context_length=2048,
            d_model=768,
            n_heads=12,
            n_kv_heads=4,
            n_layers=12,
            d_ff=3072,
            dropout=0.1,
            attention_type="flash",
            tie_embeddings=True,
            use_rms_norm=True,
            use_swiglu=True,
            use_rope=True,
            rope_base=10000,
            max_seq_len=2048,
        )

    @classmethod
    def large(cls) -> "ModelConfig":
        """Large model (~350M params) — high-capability model."""
        return cls(
            name="emily-large",
            vocab_size=32000,
            context_length=2048,
            d_model=1024,
            n_heads=16,
            n_kv_heads=8,
            n_layers=24,
            d_ff=4096,
            dropout=0.0,
            attention_type="flash",
            tie_embeddings=True,
            use_rms_norm=True,
            use_swiglu=True,
            use_rope=True,
            rope_base=10000,
            max_seq_len=2048,
        )


# ---------------------------------------------------------------------------
# TokenizerConfig
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    """
    Configuration for the Emily BPE tokenizer.

    Attributes:
        model_type: Tokenizer type (currently only "bpe" supported).
        vocab_size: Target vocabulary size for BPE training.
        model_path: Path to saved tokenizer JSON file.
        pad_token: Padding token string.
        eos_token: End-of-sequence token string.
        bos_token: Beginning-of-sequence token string.
        unk_token: Unknown token string.
    """

    model_type: str = "bpe"
    vocab_size: int = 32000
    model_path: str = "checkpoints/emily-base/tokenizer.json"
    pad_token: str = "<|pad|>"
    eos_token: str = "<|eos|>"
    bos_token: str = "<|bos|>"
    unk_token: str = "<|unk|>"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenizerConfig":
        """Create a TokenizerConfig from a dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Configuration for the Emily SLM training pipeline.

    Attributes:
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Steps before optimizer update.
        max_steps: Total training steps.
        eval_interval: Run validation every N steps.
        save_interval: Save checkpoint every N steps.
        log_interval: Log metrics every N steps.
        learning_rate: Peak learning rate for the scheduler.
        min_lr: Minimum LR at the end of cosine decay.
        warmup_steps: Linear warmup steps.
        weight_decay: L2 regularization coefficient.
        max_grad_norm: Gradient clipping max norm.
        optimizer: Optimizer name ("adamw").
        scheduler: LR scheduler name ("cosine" | "linear" | "constant").
        use_amp: Enable automatic mixed precision.
        amp_dtype: AMP dtype ("bfloat16" | "float16").
        seed: Random seed for reproducibility.
        compile: Use torch.compile() for the model.
        checkpoint_dir: Directory to save checkpoints.
        log_dir: Directory for TensorBoard logs.
        wandb_project: W&B project name.
        wandb_run_name: W&B run name.
        wandb_enabled: Enable Weights & Biases logging.
    """

    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    eval_interval: int = 2000
    save_interval: int = 5000
    log_interval: int = 100
    learning_rate: float = 1.5e-4
    min_lr: float = 1.5e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    seed: int = 42
    compile: bool = False
    checkpoint_dir: str = "checkpoints/emily-base"
    log_dir: str = "logs/emily-base"
    wandb_project: str = "emily-slm"
    wandb_run_name: str = "emily-base"
    wandb_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {self.min_lr}")
        if self.min_lr >= self.learning_rate:
            raise ValueError(
                f"min_lr ({self.min_lr}) must be less than learning_rate ({self.learning_rate})"
            )
        if self.amp_dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"amp_dtype must be one of bfloat16/float16/float32")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """
    Configuration for dataset loading and processing.

    Attributes:
        train_path: Path to tokenized training binary file.
        val_path: Path to tokenized validation binary file.
        max_seq_len: Maximum sequence length for batching.
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
    """

    train_path: str = "datasets/tokenized/train.bin"
    val_path: str = "datasets/tokenized/val.bin"
    max_seq_len: int = 1024
    num_workers: int = 4
    pin_memory: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetConfig":
        """Create from dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# EmilyConfig  (root config)
# ---------------------------------------------------------------------------

@dataclass
class EmilyConfig:
    """
    Root configuration for the Emily SLM platform.

    Combines ModelConfig, TokenizerConfig, TrainingConfig, and DatasetConfig
    into a single object that can be loaded from a YAML file.

    Example:
        >>> config = EmilyConfig.from_yaml("configs/tiny.yaml")
        >>> print(config.model.d_model)
        128
        >>> config.to_yaml("my_config.yaml")
    """

    model: ModelConfig = field(default_factory=ModelConfig.base)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EmilyConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            EmilyConfig with all sub-configs populated.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If YAML structure is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Config file must be a YAML mapping, got {type(raw)}")

        model_cfg = ModelConfig.from_dict(raw.get("model", {}))
        tokenizer_cfg = TokenizerConfig.from_dict(raw.get("tokenizer", {}))
        training_cfg = TrainingConfig.from_dict(raw.get("training", {}))
        dataset_cfg = DatasetConfig.from_dict(raw.get("dataset", {}))

        logger.info(f"Loaded config from {path} (model={model_cfg.name})")
        return cls(
            model=model_cfg,
            tokenizer=tokenizer_cfg,
            training=training_cfg,
            dataset=dataset_cfg,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmilyConfig":
        """Create EmilyConfig from a nested dictionary."""
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            tokenizer=TokenizerConfig.from_dict(d.get("tokenizer", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
            dataset=DatasetConfig.from_dict(d.get("dataset", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to nested dictionary."""
        return {
            "model": self.model.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "training": self.training.to_dict(),
            "dataset": self.dataset.to_dict(),
        }

    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Destination path for the YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config saved to {path}")

    # ----- Convenience factory methods -----

    @classmethod
    def tiny(cls) -> "EmilyConfig":
        """Create a tiny config for testing."""
        mc = ModelConfig.tiny()
        return cls(
            model=mc,
            tokenizer=TokenizerConfig(
                vocab_size=mc.vocab_size,
                model_path=f"checkpoints/{mc.name}/tokenizer.json",
            ),
            training=TrainingConfig(
                batch_size=8,
                max_steps=10000,
                learning_rate=3e-4,
                min_lr=3e-5,
                checkpoint_dir=f"checkpoints/{mc.name}",
                log_dir=f"logs/{mc.name}",
                wandb_run_name=mc.name,
            ),
            dataset=DatasetConfig(max_seq_len=mc.context_length),
        )
