"""API schemas for Emily SLM REST API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for /v1/generate endpoint."""
    prompt: str = Field(..., description="Input text prompt", min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Max new tokens")
    temperature: float = Field(default=0.8, ge=0.01, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=0, le=1000, description="Top-K sampling cutoff")
    top_p: float = Field(default=0.9, ge=0.01, le=1.0, description="Nucleus sampling threshold")
    do_sample: bool = Field(default=True, description="Enable sampling (False = greedy)")
    stream: bool = Field(default=False, description="Stream tokens as SSE")


class GenerateResponse(BaseModel):
    """Response body for /v1/generate endpoint."""
    text: str = Field(..., description="Generated text (prompt not included)")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    model: str = Field(..., description="Model name")


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Message role: system | user | assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for /v1/chat endpoint."""
    messages: list[ChatMessage] = Field(..., description="Conversation history", min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=1.2, ge=0.01, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, ge=0.01, le=1.0)
    repetition_penalty: float = Field(default=1.5, ge=1.0, le=3.0)


class ChatResponse(BaseModel):
    """Response body for /v1/chat endpoint."""
    message: ChatMessage = Field(..., description="Generated assistant reply")
    tokens_generated: int
    model: str


class TokenizeRequest(BaseModel):
    """Request body for /v1/tokenize endpoint."""
    text: str = Field(..., description="Text to tokenise")
    add_special_tokens: bool = Field(default=True)


class TokenizeResponse(BaseModel):
    """Response body for /v1/tokenize endpoint."""
    token_ids: list[int]
    tokens: list[str] = Field(default_factory=list)
    count: int


class ModelInfoResponse(BaseModel):
    """Response body for /v1/model endpoint."""
    name: str
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    parameters: str
    version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    device: str
    model_loaded: bool
