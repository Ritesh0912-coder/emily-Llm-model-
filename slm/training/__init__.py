"""Training pipeline package for Emily SLM."""

from slm.training.trainer import EmilyTrainer
from slm.training.optimizer import build_optimizer, build_scheduler
from slm.training.callbacks import (
    TrainingCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
)

__all__ = [
    "EmilyTrainer",
    "build_optimizer",
    "build_scheduler",
    "TrainingCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "TensorBoardCallback",
]
