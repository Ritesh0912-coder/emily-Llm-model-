"""
Checkpoint utilities for Emily SLM.

Provides save/load routines for model weights, optimizer state, and training
metadata. Supports checkpoint discovery and resumption.

Example:
    >>> from slm.utils.checkpoint import save_checkpoint, load_checkpoint
    >>> save_checkpoint(model, optimizer, scheduler, step=1000, loss=2.5,
    ...                 config=config, path="checkpoints/step-1000.pt")
    >>> meta = load_checkpoint("checkpoints/step-1000.pt", model, optimizer)
    >>> print(meta["step"])
    1000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    step: int,
    loss: float,
    config: Any,
    path: str | Path,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Save a complete training checkpoint to disk.

    The checkpoint contains:
    - ``model_state_dict``: Model weights.
    - ``optimizer_state_dict``: Optimizer state (moments, etc.).
    - ``scheduler_state_dict``: LR scheduler state (if provided).
    - ``step``: Current training step.
    - ``loss``: Last recorded validation loss.
    - ``config``: Serialised model config dict.
    - Any additional keys provided in ``extra``.

    Args:
        model: The model whose weights to save.
        optimizer: The optimizer whose state to save.
        scheduler: Optional LR scheduler (its state is saved if not ``None``).
        step: Current global training step.
        loss: Current validation loss value.
        config: Config object with a ``to_dict()`` method (EmilyConfig or ModelConfig).
        path: Destination file path (e.g., ``"checkpoints/step-1000.pt"``).
        extra: Optional dictionary of additional key-value pairs to store.

    Raises:
        OSError: If the destination directory cannot be created or written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle DataParallel / DistributedDataParallel wrappers
    raw_model = model.module if hasattr(model, "module") else model

    state: dict[str, Any] = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if hasattr(config, "to_dict"):
        state["config"] = config.to_dict()
    else:
        state["config"] = config

    if extra:
        state.update(extra)

    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path} (step={step}, loss={loss:.4f})")


def load_checkpoint(
    path: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a training checkpoint and restore model/optimizer/scheduler state.

    Args:
        path: Path to the checkpoint ``.pt`` file.
        model: Model to load weights into. If ``None``, weights are not loaded
            but the checkpoint dictionary is still returned.
        optimizer: Optimizer to restore state into. If ``None``, skipped.
        scheduler: LR scheduler to restore state into. If ``None``, skipped.
        device: Target device for loading tensors. Defaults to CPU if ``None``.
        strict: Whether ``load_state_dict`` should enforce strict key matching.

    Returns:
        The full checkpoint dictionary containing at minimum:
        ``{"step": int, "loss": float, "config": dict}``.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint format is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device or torch.device("cpu")
    state: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        raw_model = model.module if hasattr(model, "module") else model
        missing, unexpected = raw_model.load_state_dict(
            state["model_state_dict"], strict=strict
        )
        if missing:
            logger.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    step = state.get("step", 0)
    loss = state.get("loss", float("inf"))
    logger.info(f"Checkpoint loaded ← {path} (step={step}, loss={loss:.4f})")
    return state


def get_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """
    Find the most recent checkpoint in a directory.

    Checkpoints are expected to be ``.pt`` files. The most recent is determined
    by the integer step number embedded in the filename
    (e.g., ``step-001000.pt`` → step 1000) or, failing that, by file modification
    time.

    Args:
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        Path to the latest checkpoint, or ``None`` if no checkpoints exist.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
        return None

    return checkpoints[-1]  # list_checkpoints returns sorted ascending


def list_checkpoints(checkpoint_dir: str | Path) -> list[Path]:
    """
    Return all checkpoint files in a directory, sorted by step number.

    Args:
        checkpoint_dir: Directory to search.

    Returns:
        Sorted list of checkpoint :class:`Path` objects (ascending step order).
        Returns an empty list if the directory does not exist or is empty.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    ckpts = [p for p in checkpoint_dir.glob("*.pt") if p.is_file()]
    if not ckpts:
        return []

    def _step_key(p: Path) -> int:
        """Extract step number from filename like step-001000.pt."""
        name = p.stem  # e.g., "step-001000" or "best" or "checkpoint-5000"
        for part in name.split("-"):
            if part.isdigit():
                return int(part)
        # Fallback: use modification time as sort key
        return int(p.stat().st_mtime)

    return sorted(ckpts, key=_step_key)


def checkpoint_filename(step: int, prefix: str = "step") -> str:
    """
    Generate a zero-padded checkpoint filename.

    Args:
        step: Training step number.
        prefix: Filename prefix (default: ``"step"``).

    Returns:
        Filename string, e.g., ``"step-001000.pt"``.
    """
    return f"{prefix}-{step:07d}.pt"
