# Emily SLM — Architecture Guide

## Overview

Emily SLM is a **GPT-style decoder-only transformer** built entirely from scratch in PyTorch. It is designed as the central reasoning engine of the Emily AI Platform. Every component — from the tokenizer to the training loop — is custom-engineered without wrapping external inference APIs.

---

## Model Architecture

```
Input Token IDs  (B, T)
        │
        ▼
 ┌─────────────────────┐
 │   TokenEmbedding    │  × √d_model scaling
 └─────────────────────┘
        │
        ▼  (+ LearnablePositionalEmbedding if use_rope=False)
        │
        ▼
 ┌──────────────────────────────────────────────────────┐
 │               TransformerBlock  × n_layers           │
 │                                                      │
 │   x = x + Attention(RMSNorm(x))   ← Pre-Norm        │
 │   x = x + FFN(RMSNorm(x))         ← Pre-Norm        │
 └──────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │      RMSNorm        │  Final layer normalisation
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │   Linear LM Head   │  d_model → vocab_size
 └─────────────────────┘   (weight-tied to embedding)
        │
        ▼
 Logits  (B, T, vocab_size)
```

---

## Key Design Choices

### Pre-Norm Residuals
RMSNorm is applied **before** each sub-layer (attention, FFN), not after. This stabilises gradients in deep networks and is used by LLaMA, Mistral, and Falcon.

```
x = x + SubLayer(Norm(x))   ✅ Pre-Norm (Emily SLM)
x = Norm(x + SubLayer(x))   ❌ Post-Norm (original GPT-2)
```

### Rotary Position Embeddings (RoPE)
Position information is encoded by rotating Q and K vectors before the attention dot product. Unlike absolute embeddings, RoPE generalises naturally to sequences longer than those seen during training.

```
q_rot, k_rot = apply_rotary_emb(q, k, cos[offset:offset+T], sin[offset:offset+T])
```

### SwiGLU Feed-Forward Network
The FFN uses a gated activation (SiLU × linear) instead of GeLU. This gives the network a selective information gate, improving quality without adding parameters.

```
output = down_proj(silu(gate_proj(x)) ⊙ up_proj(x))
```

The recommended inner dim is `⌈8/3 × d_model⌉` rounded to a multiple of 64.

### Grouped Query Attention (GQA)
When `n_kv_heads < n_heads`, KV projections are shared across groups of query heads. This reduces the KV cache memory footprint during generation by a factor of `n_heads / n_kv_heads`.

```
# n_heads=8, n_kv_heads=4 → 2 query heads share each KV head
k_expanded = k.repeat_interleave(n_groups, dim=1)
```

### KV Cache
During autoregressive decoding, K and V tensors are cached and incrementally extended — avoiding O(T²) recomputation at each step.

```
# Prefill
_, cache = attn(prompt_tokens, kv_cache=KVCache.empty(...))

# Decode (one token at a time)
out, cache = attn(new_token, kv_cache=cache)
```

### Weight Tying
The LM head shares weights with the token embedding matrix (`tie_embeddings=True`), reducing parameter count and improving training efficiency.

---

## Model Scales

| Scale  | d_model | Layers | Heads | KV Heads | d_ff  | Parameters |
|--------|---------|--------|-------|----------|-------|------------|
| Tiny   | 128     | 2      | 4     | 4        | 512   | ~1M        |
| Small  | 256     | 4      | 8     | 4        | 1024  | ~7M        |
| Base   | 512     | 8      | 8     | 8        | 2048  | ~50M       |
| Medium | 768     | 12     | 12    | 4        | 3072  | ~120M      |
| Large  | 1024    | 24     | 16    | 8        | 4096  | ~350M      |

---

## Component Map

```
Emily-SLM/
├── slm/
│   ├── config.py              ← ModelConfig, TrainingConfig, EmilyConfig
│   ├── model/
│   │   ├── layers/
│   │   │   ├── rmsnorm.py     ← RMSNorm
│   │   │   ├── embedding.py   ← TokenEmbedding, LearnablePositionalEmbedding
│   │   │   ├── rotary.py      ← RotaryEmbedding, apply_rotary_emb
│   │   │   ├── attention.py   ← CausalSelfAttention, GroupedQueryAttention, KVCache
│   │   │   ├── mlp.py         ← SwiGLU, GeLUMLP
│   │   │   └── transformer.py ← TransformerBlock
│   │   └── model.py           ← EmilySLM (full model + generate + beam_search)
│   ├── tokenizer/
│   │   ├── special_tokens.py  ← SpecialTokens dataclass
│   │   ├── tokenizer.py       ← EmilyTokenizer (BPE, train/encode/decode/save/load)
│   │   └── chat_template.py   ← ChatTemplate (format_messages, apply)
│   ├── dataset/
│   │   ├── loader.py          ← TokenisedDataset, TextDataset, DatasetLoader
│   │   ├── preprocessor.py    ← TextPreprocessor (clean, dedup, filter)
│   │   └── collator.py        ← CausalLMCollator
│   ├── training/
│   │   ├── trainer.py         ← EmilyTrainer (full training loop)
│   │   ├── optimizer.py       ← build_optimizer, build_scheduler
│   │   ├── callbacks.py       ← Checkpoint, EarlyStopping, TensorBoard, WandB
│   │   └── data_loader.py     ← build_train_loader, build_val_loader
│   ├── evaluation/
│   │   └── evaluator.py       ← EmilyEvaluator (PPL, accuracy, BPC)
│   ├── inference/
│   │   ├── engine.py          ← EmilyInferenceEngine (generate, stream, chat)
│   │   └── sampler.py         ← Sampler (greedy, top-k/p, beam)
│   ├── api/
│   │   ├── app.py             ← FastAPI app (generate, chat, tokenize, health)
│   │   └── schemas.py         ← Pydantic request/response schemas
│   ├── cli/
│   │   └── main.py            ← Click CLI (train, eval, chat, prepare)
│   ├── frontend/
│   │   └── demo.py            ← Gradio demo (chat + generation tabs)
│   └── utils/
│       ├── device.py          ← get_device, count_parameters, set_seed
│       ├── logging.py         ← setup_logger (rich formatting)
│       └── checkpoint.py      ← save/load/list checkpoints
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── prepare_data.py
│   └── export.py              ← ONNX / INT8 / TorchScript export
├── tests/                     ← 128 tests, 100% pass rate
├── configs/                   ← tiny / small / base / medium / large .yaml
└── pyproject.toml
```

---

## Training Flow

```
emily-prepare --input data/raw/ --config configs/tiny.yaml --train-tokenizer
      ↓  (writes datasets/tokenized/train.bin + val.bin + tokenizer.json)

emily-train --config configs/tiny.yaml
      ↓  (runs EmilyTrainer loop)
      │   → gradient accumulation
      │   → AMP (bfloat16/float16)
      │   → grad clipping
      │   → cosine LR with warmup
      │   → checkpoint every N steps
      │   → eval PPL on val set
      ↓

checkpoints/emily-tiny/
    ├── step-0001000.pt
    ├── best.pt
    └── config.yaml
```

---

## Inference Flow

```python
from slm.inference.engine import EmilyInferenceEngine

engine = EmilyInferenceEngine(
    model_path="checkpoints/emily-tiny/best",
    tokenizer_path="checkpoints/emily-tiny/tokenizer.json",
)

# Simple generation
text = engine.generate("Once upon a time", max_new_tokens=100)

# Streaming
for chunk in engine.stream("Tell me about AI", max_new_tokens=200):
    print(chunk, end="", flush=True)

# Chat
reply = engine.chat([
    {"role": "system", "content": "You are Emily, a helpful AI."},
    {"role": "user",   "content": "What is a transformer?"},
])
```

---

## API Usage

```bash
# Launch the REST API server
EMILY_MODEL_PATH=checkpoints/emily-tiny/best \
EMILY_TOKENIZER_PATH=checkpoints/emily-tiny/tokenizer.json \
uvicorn slm.api.app:app --host 0.0.0.0 --port 8000

# Or via CLI
emily-chat --model checkpoints/emily-tiny/best \
           --tokenizer checkpoints/emily-tiny/tokenizer.json \
           --serve --port 8000
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/model` | Model metadata |
| POST | `/v1/generate` | Text generation |
| POST | `/v1/generate/stream` | Streaming SSE generation |
| POST | `/v1/chat` | Chat-style generation |
| POST | `/v1/tokenize` | Tokenise text |

---

## Chat Format

```
<|bos|>
<|system|>You are Emily, a helpful AI assistant.<|eos|>
<|user|>What is 2 + 2?<|eos|>
<|assistant|>4.<|eos|>
<|user|>And 3 + 3?<|eos|>
<|assistant|>               ← model generates from here
```
