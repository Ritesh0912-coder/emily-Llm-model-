"""
Emily SLM FastAPI Application.

Provides a production-ready REST API with endpoints for:
- POST /v1/generate    — text generation
- POST /v1/chat        — chat-style generation
- POST /v1/tokenize    — tokenisation
- GET  /v1/model       — model info
- GET  /health         — health check

Run with:
    uvicorn slm.api.app:app --host 0.0.0.0 --port 8000

Or via CLI:
    emily-chat --serve --model checkpoints/emily-tiny/best
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from slm import __version__
from slm.api.schemas import (
    ChatRequest, ChatResponse, ChatMessage,
    GenerateRequest, GenerateResponse,
    HealthResponse, ModelInfoResponse,
    TokenizeRequest, TokenizeResponse,
)
from slm.utils.device import format_parameters, count_parameters

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — loaded once at startup
# ---------------------------------------------------------------------------

_engine: Optional[object] = None   # EmilyInferenceEngine
_model_name: str = "emily"
_device_str: str = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model on startup, clean up on shutdown."""
    global _engine, _model_name, _device_str

    model_path = os.environ.get("EMILY_MODEL_PATH", "")
    tokenizer_path = os.environ.get("EMILY_TOKENIZER_PATH", "")

    if model_path and tokenizer_path:
        try:
            from slm.inference.engine import EmilyInferenceEngine
            _engine = EmilyInferenceEngine(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
            )
            raw_model = getattr(_engine, "model", None)
            if raw_model is not None:
                _model_name = getattr(getattr(raw_model, "config", None), "name", "emily")
            device = getattr(_engine, "device", None)
            _device_str = str(device) if device else "cpu"
            logger.info(f"Model loaded: {_model_name} on {_device_str}")
        except Exception as exc:
            logger.warning(f"Failed to load model: {exc}. API will run without a model.")

    yield  # Application runs here

    _engine = None
    logger.info("Emily API shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Emily SLM API",
    description=(
        "REST API for Emily SLM — a custom GPT-style decoder-only transformer. "
        "Provides text generation, chat, and tokenisation endpoints."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_engine() -> object:
    """Return the inference engine or raise a 503 if not loaded."""
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Set EMILY_MODEL_PATH and EMILY_TOKENIZER_PATH "
                "environment variables before starting the server."
            ),
        )
    return _engine


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        device=_device_str,
        model_loaded=_engine is not None,
    )


@app.get("/v1/model", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Return metadata about the loaded model."""
    engine = _require_engine()
    model = engine.model  # type: ignore[attr-defined]
    cfg = model.config
    n_params = count_parameters(model)
    return ModelInfoResponse(
        name=cfg.name,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        parameters=format_parameters(n_params),
        version=__version__,
    )


@app.post("/v1/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate text from a prompt.

    Returns generated text (not including the prompt).
    """
    engine = _require_engine()

    try:
        text = engine.generate(  # type: ignore[attr-defined]
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )
    except Exception as exc:
        logger.exception("Generation error")
        raise HTTPException(status_code=500, detail=str(exc))

    # Rough token count estimate
    tokenizer = engine.tokenizer  # type: ignore[attr-defined]
    tokens_generated = len(tokenizer.encode(text, add_special_tokens=False))

    return GenerateResponse(
        text=text,
        tokens_generated=tokens_generated,
        model=_model_name,
    )


@app.post("/v1/generate/stream", tags=["Generation"])
async def generate_stream(request: GenerateRequest) -> StreamingResponse:
    """
    Stream generated tokens as Server-Sent Events (SSE).

    Each event contains a single decoded token fragment.
    """
    engine = _require_engine()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            for chunk in engine.stream(  # type: ignore[attr-defined]
                prompt=request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat-style generation from a list of messages.

    Formats messages using Emily's chat template and returns the assistant reply.
    """
    engine = _require_engine()

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        reply = engine.chat(  # type: ignore[attr-defined]
            messages=messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(exc))

    tokenizer = engine.tokenizer  # type: ignore[attr-defined]
    tokens_generated = len(tokenizer.encode(reply, add_special_tokens=False))

    return ChatResponse(
        message=ChatMessage(role="assistant", content=reply),
        tokens_generated=tokens_generated,
        model=_model_name,
    )


@app.post("/v1/tokenize", response_model=TokenizeResponse, tags=["Tokenizer"])
async def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    """
    Tokenise text and return token IDs.
    """
    engine = _require_engine()
    tokenizer = engine.tokenizer  # type: ignore[attr-defined]

    ids = tokenizer.encode(request.text, add_special_tokens=request.add_special_tokens)
    tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]

    return TokenizeResponse(token_ids=ids, tokens=tokens, count=len(ids))
