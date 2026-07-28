#!/usr/bin/env python
"""
Training script for Emily SLM.

Usage:
    python scripts/train.py --config configs/tiny.yaml
    python scripts/train.py --config configs/base.yaml --resume checkpoints/emily-base
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.cli.main import train

if __name__ == "__main__":
    train(standalone_mode=True)
