"""
Emily SLM Training Engine.

The ``EmilyTrainer`` orchestrates the full training loop:
- Gradient accumulation
- Mixed precision (BF16/FP16) via torch.autocast + GradScaler
- Gradient clipping
- Evaluation on validation set
- Callback dispatch
- torch.compile() support
- Checkpoint resumption

Example:
    >>> from slm.config import EmilyConfig
    >>> from slm.model import EmilySLM
    >>> from slm.training import EmilyTrainer
    >>> config = EmilyConfig.from_yaml("configs/tiny.yaml")
    >>> model = EmilySLM(config.model)
    >>> trainer = EmilyTrainer(model, config, train_dataset, val_dataset)
    >>> trainer.train()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from slm.config import EmilyConfig, TrainingConfig
from slm.dataset.collator import CausalLMCollator
from slm.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback,
    TrainingCallback,
)
from slm.training.optimizer import build_optimizer, build_scheduler
from slm.utils.device import get_device, get_dtype, set_seed, format_parameters, count_parameters
from slm.utils.checkpoint import load_checkpoint, get_latest_checkpoint

logger = logging.getLogger(__name__)


class EmilyTrainer:
    """
    Training engine for Emily SLM.

    Args:
        model: ``EmilySLM`` model to train.
        config: ``EmilyConfig`` with training, model, and dataset settings.
        train_dataset: Training ``Dataset``.
        val_dataset: Optional validation ``Dataset``.
        callbacks: List of ``TrainingCallback`` instances. If not provided,
            defaults are built from the config.
        device: Compute device. Auto-detected if ``None``.
        resume_from: Path to a checkpoint to resume from.
    """

    def __init__(
        self,
        model: nn.Module,
        config: EmilyConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        callbacks: Optional[list[TrainingCallback]] = None,
        device: Optional[torch.device] = None,
        resume_from: Optional[str | Path] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_cfg: TrainingConfig = config.training
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device or get_device()

        set_seed(self.train_cfg.seed)

        # Move model to device
        self.model = self.model.to(self.device)

        # Compile model (PyTorch 2.0+)
        if self.train_cfg.compile:
            logger.info("Compiling model with torch.compile()…")
            self.model = torch.compile(self.model)  # type: ignore

        # Build optimizer and scheduler
        self.optimizer = build_optimizer(self.model, self.train_cfg)
        self.scheduler = build_scheduler(self.optimizer, self.train_cfg)

        # AMP (automatic mixed precision)
        self.use_amp = self.train_cfg.use_amp and self.device.type == "cuda"
        self.amp_dtype = get_dtype(self.train_cfg.amp_dtype)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp and self.amp_dtype == torch.float16)

        # Collator and data loaders
        pad_token_id: int = 0  # default fallback
        model_cfg = getattr(model, "config", None)
        if model_cfg is not None:
            # Try to read from tokenizer config in EmilyConfig
            pad_token_id = getattr(config.tokenizer, "pad_token_id", 0)
        self.collator = CausalLMCollator(pad_token_id=pad_token_id, max_length=config.dataset.max_seq_len)

        # Use drop_last=False so small datasets are never silently emptied
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.train_cfg.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
            collate_fn=self.collator,
            drop_last=False,
        )
        self.val_loader: Optional[DataLoader] = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.train_cfg.batch_size * 2,
                shuffle=False,
                num_workers=config.dataset.num_workers,
                pin_memory=config.dataset.pin_memory,
                collate_fn=self.collator,
            )

        # Callbacks — build defaults if not provided
        if callbacks is None:
            callbacks = self._build_default_callbacks()
        self.callbacks = callbacks

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Resume from checkpoint
        if resume_from:
            self._resume(resume_from)

        n_params = count_parameters(self.model)
        logger.info(
            f"EmilyTrainer ready | device={self.device} | params={format_parameters(n_params)} | "
            f"amp={self.use_amp}({self.train_cfg.amp_dtype}) | "
            f"steps={self.train_cfg.max_steps:,} | "
            f"batch_size={self.train_cfg.batch_size} | "
            f"grad_accum={self.train_cfg.gradient_accumulation_steps}"
        )

    # ------------------------------------------------------------------
    # Default callbacks
    # ------------------------------------------------------------------

    def _build_default_callbacks(self) -> list[TrainingCallback]:
        cfg = self.train_cfg
        callbacks: list[TrainingCallback] = [
            CheckpointCallback(
                checkpoint_dir=cfg.checkpoint_dir,
                save_interval=cfg.save_interval,
                keep_last_n=3,
                save_best=True,
            ),
            TensorBoardCallback(log_dir=cfg.log_dir),
        ]
        if cfg.wandb_enabled:
            from slm.training.callbacks import WandbCallback
            callbacks.append(
                WandbCallback(
                    project=cfg.wandb_project,
                    run_name=cfg.wandb_run_name,
                    config=self.config.to_dict(),
                )
            )
        return callbacks

    # ------------------------------------------------------------------
    # Checkpoint resumption
    # ------------------------------------------------------------------

    def _resume(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            ckpt = get_latest_checkpoint(path)
            if ckpt is None:
                logger.warning(f"No checkpoints found in {path}; starting fresh")
                return
            path = ckpt
        state = load_checkpoint(path, self.model, self.optimizer, self.scheduler, self.device)
        self.global_step = state.get("step", 0)
        self.best_val_loss = state.get("loss", float("inf"))
        logger.info(f"Resumed from step {self.global_step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """
        Run the full training loop.

        Returns:
            Dictionary with final training metrics.
        """
        cfg = self.train_cfg
        for cb in self.callbacks:
            cb.on_train_begin(self)

        self.model.train()
        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        accum_steps = 0
        t0 = time.perf_counter()

        self.optimizer.zero_grad()

        while self.global_step < cfg.max_steps:
            # Fetch next batch — reset iterator safely when epoch ends
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    logger.error("DataLoader is empty — check dataset size and batch_size")
                    break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward + loss
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                out = self.model(input_ids)
                logits = out["logits"]  # (B, T, vocab)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / cfg.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            accum_loss += loss.item()
            accum_steps += 1

            # Gradient update after accumulation
            if accum_steps == cfg.gradient_accumulation_steps:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                step_loss = accum_loss
                accum_loss = 0.0
                accum_steps = 0

                # Logging
                if self.global_step % cfg.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.perf_counter() - t0
                    tok_per_sec = (
                        cfg.batch_size * cfg.gradient_accumulation_steps
                        * input_ids.shape[1] * cfg.log_interval / max(elapsed, 1e-6)
                    )
                    logs = {
                        "train_loss": step_loss,
                        "learning_rate": lr,
                        "tokens_per_sec": tok_per_sec,
                        "step": self.global_step,
                    }
                    logger.info(
                        f"step={self.global_step:>7,} | loss={step_loss:.4f} | "
                        f"lr={lr:.2e} | tok/s={tok_per_sec:,.0f}"
                    )
                    for cb in self.callbacks:
                        cb.on_step_end(self, self.global_step, logs)
                    t0 = time.perf_counter()

                # Evaluation
                if self.global_step % cfg.eval_interval == 0 and self.val_loader is not None:
                    val_metrics = self.evaluate()
                    for cb in self.callbacks:
                        cb.on_eval_end(self, self.global_step, val_metrics)
                    # Check early stopping
                    for cb in self.callbacks:
                        if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
                            logger.info("Early stopping — exiting training loop")
                            break
                    self.model.train()

        for cb in self.callbacks:
            cb.on_train_end(self)

        logger.info(f"Training complete at step {self.global_step}")
        return {"final_step": self.global_step, "best_val_loss": self.best_val_loss}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, n_batches: int = 50) -> dict[str, float]:
        """
        Evaluate on the validation DataLoader.

        Args:
            n_batches: Maximum number of validation batches to evaluate.

        Returns:
            Dict with ``"val_loss"`` and ``"val_perplexity"``.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        n = 0

        for i, batch in enumerate(self.val_loader):
            if i >= n_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                out = self.model(input_ids)
                logits = out["logits"]
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        import math
        perplexity = math.exp(min(avg_loss, 20))  # cap at exp(20) to avoid inf

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss

        metrics = {"val_loss": avg_loss, "val_perplexity": perplexity}
        logger.info(f"Eval step={self.global_step} | val_loss={avg_loss:.4f} | ppl={perplexity:.2f}")
        return metrics
