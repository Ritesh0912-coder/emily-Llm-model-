#!/usr/bin/env python
"""
Data preparation script for Emily SLM.

Cleans, deduplicates, trains a BPE tokenizer, and writes
binary train/val datasets.

Usage:
    python scripts/prepare_data.py \
        --input data/raw/corpus.txt \
        --config configs/tiny.yaml \
        --output datasets/tokenized \
        --train-tokenizer
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from slm.cli.main import prepare_data

if __name__ == "__main__":
    prepare_data(standalone_mode=True)
