#!/usr/bin/env python
"""
Export script for Emily SLM.

Exports a trained model to ONNX or applies INT8 quantisation.

Usage:
    # ONNX export
    python scripts/export.py --model checkpoints/emily-tiny/best --format onnx

    # Dynamic INT8 quantisation
    python scripts/export.py --model checkpoints/emily-tiny/best --format int8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from slm.model import EmilySLM
from slm.utils.logging import setup_logger
from slm.utils.device import count_parameters, format_parameters

import logging
setup_logger("slm", level=logging.INFO)
logger = logging.getLogger(__name__)


def export_onnx(model: EmilySLM, output_path: Path, seq_len: int = 32) -> None:
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randint(0, model.config.vocab_size, (1, seq_len))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
    )
    logger.info(f"ONNX model saved → {output_path}")


def export_int8(model: EmilySLM, output_path: Path) -> None:
    """Apply dynamic INT8 quantisation and save."""
    model.eval()
    quantised = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantised.state_dict(), output_path)
    logger.info(f"INT8 quantised model saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Emily SLM")
    parser.add_argument("--model", required=True, help="Model directory")
    parser.add_argument("--format", choices=["onnx", "int8", "torchscript"], default="onnx")
    parser.add_argument("--output", default=None, help="Output path (default: <model>/export/)")
    parser.add_argument("--seq-len", default=32, type=int, help="Dummy sequence length for ONNX")
    args = parser.parse_args()

    model_dir = Path(args.model)
    output_dir = Path(args.output) if args.output else model_dir / "export"

    model = EmilySLM.from_pretrained(model_dir)
    logger.info(f"Loaded: {model} | params={format_parameters(count_parameters(model))}")

    if args.format == "onnx":
        export_onnx(model, output_dir / "model.onnx", seq_len=args.seq_len)
    elif args.format == "int8":
        export_int8(model, output_dir / "model_int8.pt")
    elif args.format == "torchscript":
        model.eval()
        dummy = torch.randint(0, model.config.vocab_size, (1, args.seq_len))
        scripted = torch.jit.trace(model, (dummy,))
        out_path = output_dir / "model.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(out_path))
        logger.info(f"TorchScript model saved → {out_path}")


if __name__ == "__main__":
    main()
