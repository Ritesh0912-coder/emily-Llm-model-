"""
Optimizer and LR scheduler factory for Emily SLM.

Builds AdamW with weight-decay parameter groups (no decay on biases/norms)
and cosine/linear/constant LR schedules with linear warmup.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from slm.config import TrainingConfig

logger = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """
    Build an AdamW optimizer with separate parameter groups.

    Parameters that should NOT have weight decay:
    - All bias vectors
    - LayerNorm / RMSNorm weight vectors (1-D)
    - Embedding weights

    All other parameters (2-D+ matrices) get ``weight_decay``.

    Args:
        model: The model to optimise.
        config: ``TrainingConfig`` with ``learning_rate``, ``weight_decay``.

    Returns:
        Configured ``AdamW`` optimizer.
    """
    decay_params: list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),  # Use fused kernel on CUDA
    )

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    logger.info(
        f"Optimizer: AdamW | lr={config.learning_rate:.2e} | "
        f"decay params={n_decay:,} | no-decay params={n_no_decay:,}"
    )
    return optimizer


def build_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
    num_training_steps: int | None = None,
) -> LRScheduler:
    """
    Build a learning-rate scheduler with linear warmup.

    Supported schedulers (``config.scheduler``):
    - ``"cosine"``   — cosine decay from peak LR to ``min_lr``
    - ``"linear"``   — linear decay from peak LR to ``min_lr``
    - ``"constant"`` — constant LR after warmup

    Args:
        optimizer: Optimizer to attach the scheduler to.
        config: ``TrainingConfig`` with scheduler settings.
        num_training_steps: Total steps (required for linear/cosine).

    Returns:
        ``LambdaLR`` scheduler.
    """
    warmup_steps = config.warmup_steps
    max_steps = num_training_steps or config.max_steps
    min_lr_ratio = config.min_lr / config.learning_rate

    def cosine_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    def linear_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    def constant_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    schedule_fn = {
        "cosine": cosine_schedule,
        "linear": linear_schedule,
        "constant": constant_schedule,
    }.get(config.scheduler)

    if schedule_fn is None:
        raise ValueError(
            f"Unknown scheduler: {config.scheduler!r}. "
            f"Choose from: cosine, linear, constant"
        )

    logger.info(
        f"Scheduler: {config.scheduler} | warmup={warmup_steps} | "
        f"max_steps={max_steps} | min_lr={config.min_lr:.2e}"
    )
    return LambdaLR(optimizer, lr_lambda=schedule_fn)
