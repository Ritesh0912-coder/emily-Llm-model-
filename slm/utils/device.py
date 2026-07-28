"""
Device and hardware utilities for Emily SLM.

Provides helpers for device detection (CUDA / MPS / CPU), dtype selection,
GPU memory reporting, parameter counting, and reproducible seed setting.

Example:
    >>> from slm.utils.device import get_device, set_seed, format_parameters
    >>> device = get_device()
    >>> set_seed(42)
    >>> print(format_parameters(125_000_000))
    '125.0M'
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Auto-detect the best available compute device.

    Priority order:
    1. CUDA GPU (if available and ``prefer_cuda=True``)
    2. Apple Silicon MPS (if available)
    3. CPU (always available fallback)

    Args:
        prefer_cuda: Prefer CUDA over MPS when both are available.

    Returns:
        The selected :class:`torch.device`.
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"Using CUDA device: {device_name} ({total_mem:.1f} GB)")
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple MPS device")
        return torch.device("mps")

    logger.warning("No GPU found — falling back to CPU. Training will be very slow.")
    return torch.device("cpu")


def get_dtype(amp_dtype: str = "bfloat16") -> torch.dtype:
    """
    Resolve an AMP dtype string to a :class:`torch.dtype`.

    Args:
        amp_dtype: One of ``"bfloat16"``, ``"float16"``, or ``"float32"``.

    Returns:
        Corresponding :class:`torch.dtype`.

    Raises:
        ValueError: If ``amp_dtype`` is not recognised.
    """
    mapping: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if amp_dtype not in mapping:
        raise ValueError(
            f"amp_dtype must be one of {list(mapping.keys())}, got {amp_dtype!r}"
        )
    return mapping[amp_dtype]


def memory_summary() -> dict[str, float]:
    """
    Return a summary of GPU memory usage.

    Returns:
        Dictionary with keys:
        - ``allocated_gb``: Currently allocated GPU memory in GB.
        - ``reserved_gb``: Reserved (cached) GPU memory in GB.
        - ``total_gb``: Total GPU VRAM in GB.
        - ``free_gb``: Estimated free GPU memory in GB.

        Returns zeros for all keys if no CUDA GPU is available.
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0}

    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    total = props.total_memory / 1024 ** 3
    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "total_gb": round(total, 3),
        "free_gb": round(total - reserved, 3),
    }


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model: The model to inspect.
        trainable_only: If ``True``, count only trainable parameters.
            If ``False``, count all parameters (including frozen ones).

    Returns:
        Total parameter count as an integer.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameters(n: int) -> str:
    """
    Format a parameter count into a human-readable string.

    Args:
        n: Raw parameter count.

    Returns:
        Formatted string such as ``"125.0M"``, ``"1.3B"``, or ``"47K"``.

    Example:
        >>> format_parameters(125_000_000)
        '125.0M'
        >>> format_parameters(1_300_000_000)
        '1.3B'
        >>> format_parameters(47_000)
        '47.0K'
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def set_seed(seed: int) -> None:
    """
    Set random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Covers:
    - Python's built-in ``random`` module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA deterministic algorithms (may reduce performance)

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Force deterministic CUDA operations (slight performance cost)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug(f"Random seed set to {seed}")


def get_world_size() -> int:
    """
    Return the distributed training world size.

    Returns:
        World size if ``torch.distributed`` is initialised, else 1.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def get_rank() -> int:
    """
    Return the distributed training rank of this process.

    Returns:
        Rank if ``torch.distributed`` is initialised, else 0.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    """Return ``True`` if this is the main (rank 0) process."""
    return get_rank() == 0
