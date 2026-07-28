"""
Emily SLM — Custom GPT-style decoder-only transformer.

A production-quality small language model built from scratch in PyTorch.
"""

__version__ = "0.1.0"
__author__ = "Emily AI Team"
__license__ = "MIT"

from slm.config import EmilyConfig, ModelConfig, TrainingConfig, DatasetConfig, TokenizerConfig

__all__ = [
    "__version__",
    "__author__",
    "EmilyConfig",
    "ModelConfig",
    "TrainingConfig",
    "DatasetConfig",
    "TokenizerConfig",
]
