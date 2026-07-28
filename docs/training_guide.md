# Emily SLM — Training Guide

## Prerequisites

```bash
# Install dependencies
pip install -e ".[dev]"

# Or install core packages manually
pip install torch numpy tokenizers pyyaml rich click fastapi uvicorn pydantic pytest
```

---

## Step 1 — Prepare Your Data

Place raw `.txt` or `.jsonl` files in `data/raw/`.

JSONL format (one document per line):
```json
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "Emily is a language model built from scratch."}
```

Run the preparation script:
```bash
emily-prepare \
    --input data/raw/ \
    --config configs/tiny.yaml \
    --output datasets/tokenized \
    --train-tokenizer \
    --val-ratio 0.05
```

This will:
1. Clean and deduplicate all text
2. Train a BPE tokenizer (vocab_size from config)
3. Tokenise everything and write `train.bin` / `val.bin`
4. Save `tokenizer.json` + `tokenizer.config.json`

---

## Step 2 — Choose a Config

Configs live in `configs/`. Start with `tiny.yaml` for rapid iteration:

```yaml
model:
  name: emily-tiny
  d_model: 128
  n_heads: 4
  n_kv_heads: 4
  n_layers: 2
  vocab_size: 4096
  context_length: 512
  d_ff: 512
  dropout: 0.1
  use_rope: true
  use_rms_norm: true
  use_swiglu: true
  tie_embeddings: true

training:
  learning_rate: 3.0e-4
  min_lr: 3.0e-5
  max_steps: 10000
  batch_size: 8
  gradient_accumulation_steps: 4
  eval_interval: 500
  save_interval: 1000
  warmup_steps: 100
  scheduler: cosine
  use_amp: false       # set true on CUDA
  amp_dtype: bfloat16
```

---

## Step 3 — Train

```bash
emily-train --config configs/tiny.yaml
```

With checkpoint resumption:
```bash
emily-train --config configs/tiny.yaml --resume checkpoints/emily-tiny
```

Override dataset directory:
```bash
emily-train --config configs/tiny.yaml --data datasets/tokenized
```

**Training output:**
```
step=   1,000 | loss=4.2831 | lr=2.94e-04 | tok/s=12,450
step=   2,000 | loss=3.9102 | lr=2.88e-04 | tok/s=12,520
Eval step=2000 | val_loss=4.1023 | ppl=60.48
New best checkpoint saved (val_loss=4.1023)
```

Checkpoints are saved to `checkpoints/emily-tiny/`:
- `step-0001000.pt` — periodic checkpoint
- `best.pt` — best validation loss checkpoint
- `config.yaml` — model config snapshot

---

## Step 4 — Evaluate

```bash
emily-eval \
    --model checkpoints/emily-tiny/best \
    --tokenizer checkpoints/emily-tiny/tokenizer.json \
    --data datasets/tokenized/val.bin \
    --batch-size 16
```

Output:
```
Results:
  loss:        3.8421
  perplexity:  46.70
  accuracy:    0.3812
  bpc:         5.5432
  total_tokens: 48,320
```

---

## Step 5 — Chat

**Interactive REPL:**
```bash
emily-chat \
    --model checkpoints/emily-tiny/best \
    --tokenizer checkpoints/emily-tiny/tokenizer.json \
    --temperature 0.8 \
    --max-tokens 256
```

**REST API server:**
```bash
emily-chat \
    --model checkpoints/emily-tiny/best \
    --tokenizer checkpoints/emily-tiny/tokenizer.json \
    --serve --port 8000
```

**Gradio demo:**
```bash
python slm/frontend/demo.py \
    --model checkpoints/emily-tiny/best \
    --tokenizer checkpoints/emily-tiny/tokenizer.json \
    --port 7860
```

---

## Step 6 — Export

```bash
# ONNX (for production inference)
python scripts/export.py \
    --model checkpoints/emily-tiny/best \
    --format onnx

# INT8 dynamic quantisation (smaller model, faster CPU inference)
python scripts/export.py \
    --model checkpoints/emily-tiny/best \
    --format int8

# TorchScript (portable serialisation)
python scripts/export.py \
    --model checkpoints/emily-tiny/best \
    --format torchscript
```

---

## Scaling Tips

| Goal | Recommendation |
|------|---------------|
| Fast iteration | Use `tiny` config, short context (256), small batch |
| Better quality | Scale to `base` or `medium` with GQA (`n_kv_heads=4`) |
| CUDA training | Set `use_amp: true`, `amp_dtype: bfloat16`, `compile: true` |
| Large datasets | Use binary `.bin` files via `DatasetLoader.tokenise_and_save()` |
| Multi-GPU | Wrap model in `torch.nn.DataParallel` or `DistributedDataParallel` |
| Memory pressure | Reduce `batch_size`, increase `gradient_accumulation_steps` |
| Slow convergence | Increase `warmup_steps`, lower `learning_rate` |

---

## Hyperparameter Reference

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| `learning_rate` | 1e-4 – 6e-4 | Peak LR; higher = faster but unstable |
| `min_lr` | 10% of LR | Floor LR at end of cosine decay |
| `warmup_steps` | 1–5% of max_steps | Linear LR warmup |
| `weight_decay` | 0.01 – 0.1 | L2 regularisation on weight matrices |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `dropout` | 0.0 – 0.2 | 0.0 for large models, ~0.1 for small |
| `batch_size` × `grad_accum` | 64 – 2048 tokens | Effective batch size |

---

## Running Tests

```bash
# Full suite
python -m pytest tests/ -v

# Single module
python -m pytest tests/test_model.py -v

# With coverage
python -m pytest tests/ --cov=slm --cov-report=term-missing
```

Current status: **128/128 passing**
