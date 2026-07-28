"""
Training callbacks for Emily SLM.

Callbacks follow a simple event-based protocol. The ``EmilyTrainer``
calls the appropriate method at each event point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TrainingCallback:
    """Base class for all training callbacks. Override any method as needed."""

    def on_train_begin(self, trainer: Any) -> None: ...
    def on_train_end(self, trainer: Any) -> None: ...
    def on_step_begin(self, trainer: Any, step: int) -> None: ...
    def on_step_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None: ...
    def on_eval_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None: ...


class CheckpointCallback(TrainingCallback):
    """
    Save model checkpoints at regular intervals and keep only the best N.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        save_interval: Save every N steps.
        keep_last_n: Keep only the N most recent checkpoints. ``0`` keeps all.
        save_best: Also save a ``best.pt`` checkpoint based on val loss.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_interval: int = 1000,
        keep_last_n: int = 3,
        save_best: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_loss = float("inf")
        self._saved: list[Path] = []

    def on_step_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None:
        if step % self.save_interval != 0:
            return
        self._save(trainer, step, logs.get("train_loss", 0.0))

    def on_eval_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None:
        val_loss = logs.get("val_loss", float("inf"))
        if self.save_best and val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save_best(trainer, step, val_loss)

    def _save(self, trainer: Any, step: int, loss: float) -> None:
        from slm.utils.checkpoint import save_checkpoint, checkpoint_filename
        path = self.checkpoint_dir / checkpoint_filename(step)
        save_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            step=step,
            loss=loss,
            config=trainer.config,
            path=path,
        )
        self._saved.append(path)
        # Prune old checkpoints
        if self.keep_last_n > 0:
            while len(self._saved) > self.keep_last_n:
                old = self._saved.pop(0)
                if old.exists():
                    old.unlink()
                    logger.debug(f"Removed old checkpoint: {old}")

    def _save_best(self, trainer: Any, step: int, loss: float) -> None:
        best_dir = self.checkpoint_dir / "best"
        try:
            trainer.model.save_pretrained(best_dir)
        except AttributeError:
            # Compiled model — unwrap
            orig = getattr(trainer.model, "_orig_mod", trainer.model)
            orig.save_pretrained(best_dir)
        logger.info(f"New best checkpoint saved (val_loss={loss:.4f}) → {best_dir}")


class EarlyStoppingCallback(TrainingCallback):
    """
    Stop training when validation loss stops improving.

    Args:
        patience: Number of evaluations with no improvement before stopping.
        min_delta: Minimum change to count as improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_evals = 0
        self.should_stop = False

    def on_eval_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None:
        val_loss = logs.get("val_loss", float("inf"))
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_evals = 0
        else:
            self.bad_evals += 1
            logger.info(
                f"EarlyStopping: no improvement for {self.bad_evals}/{self.patience} evals"
            )
            if self.bad_evals >= self.patience:
                self.should_stop = True
                logger.warning(f"Early stopping triggered at step {step}")


class TensorBoardCallback(TrainingCallback):
    """
    Log training metrics to TensorBoard.

    Args:
        log_dir: Directory for TensorBoard event files.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self._writer: Any = None

    def on_train_begin(self, trainer: Any) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info(f"TensorBoard logging → {self.log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install tensorboard.")

    def on_step_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None:
        if self._writer is None:
            return
        for key, value in logs.items():
            self._writer.add_scalar(f"train/{key}", value, step)

    def on_eval_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None:
        if self._writer is None:
            return
        for key, value in logs.items():
            self._writer.add_scalar(f"eval/{key}", value, step)

    def on_train_end(self, trainer: Any) -> None:
        if self._writer is not None:
            self._writer.close()


class WandbCallback(TrainingCallback):
    """
    Log training metrics to Weights & Biases.

    Args:
        project: W&B project name.
        run_name: W&B run name.
        config: Config dict to log to the run.
    """

    def __init__(
        self, project: str, run_name: str, config: Optional[dict] = None
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.run_config = config or {}

    def on_train_begin(self, trainer: Any) -> None:
        try:
            import wandb
            wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.run_config,
            )
            logger.info(f"W&B run initialised: {self.run_name}")
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb")

    def on_step_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None:
        try:
            import wandb
            wandb.log({"step": step, **{f"train/{k}": v for k, v in logs.items()}})
        except Exception:
            pass

    def on_eval_end(self, trainer: Any, step: int, logs: dict[str, float]) -> None:
        try:
            import wandb
            wandb.log({"step": step, **{f"eval/{k}": v for k, v in logs.items()}})
        except Exception:
            pass

    def on_train_end(self, trainer: Any) -> None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
