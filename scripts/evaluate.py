#!/usr/bin/env python
"""
Evaluation script for Emily SLM.

Usage:
    python scripts/evaluate.py \
        --model checkpoints/emily-tiny/best \
        --tokenizer checkpoints/emily-tiny/tokenizer.json \
        --data datasets/tokenized/val.bin
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.cli.main import evaluate

if __name__ == "__main__":
    evaluate(standalone_mode=True)
