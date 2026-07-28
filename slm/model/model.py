"""
Emily SLM — Full Decoder-Only Transformer Model.

The ``EmilySLM`` class assembles all layers into a complete GPT-style
language model and exposes a high-level API for:

- Forward pass (training)
- Autoregressive generation with temperature, top-k, top-p sampling
- Streaming (token-by-token) generation
- Beam search
- Saving / loading pretrained weights
- torch.compile() support

Architecture:
    TokenEmbedding(vocab_size → d_model)
    [+ LearnablePositionalEmbedding if not use_rope]
    TransformerBlock × n_layers
    RMSNorm (final)
    Linear LM Head (d_model → vocab_size)
    [weight-tied to TokenEmbedding if tie_embeddings]
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Generator, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config import ModelConfig
from slm.model.layers.attention import KVCache
from slm.model.layers.embedding import TokenEmbedding, LearnablePositionalEmbedding
from slm.model.layers.rmsnorm import RMSNorm
from slm.model.layers.transformer import TransformerBlock
from slm.utils.device import count_parameters, format_parameters

logger = logging.getLogger(__name__)


class EmilySLM(nn.Module):
    """
    Emily Small Language Model.

    A decoder-only transformer trained for causal language modelling.
    Implements the full forward pass, autoregressive generation with
    multiple sampling strategies, KV cache, and checkpoint I/O.

    Args:
        config: ``ModelConfig`` defining the architecture hyperparameters.

    Attributes:
        config: The ``ModelConfig`` used to construct this model.
        token_embedding: Token embedding layer.
        pos_embedding: Learnable positional embedding (only when ``use_rope=False``).
        layers: Ordered list of ``TransformerBlock`` modules.
        norm: Final RMSNorm applied before the LM head.
        lm_head: Linear projection from ``d_model`` to ``vocab_size``.

    Example:
        >>> config = ModelConfig.tiny()
        >>> model = EmilySLM(config)
        >>> print(model)
        EmilySLM(params=1.2M, layers=2, heads=4, d_model=128)
        >>> ids = torch.randint(0, config.vocab_size, (1, 16))
        >>> out = model(ids)
        >>> out["logits"].shape
        torch.Size([1, 16, 4096])
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )

        # Positional embedding (only used when RoPE is disabled)
        self.pos_embedding: Optional[LearnablePositionalEmbedding] = None
        if not config.use_rope:
            self.pos_embedding = LearnablePositionalEmbedding(
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
            )

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final normalisation
        if config.use_rms_norm:
            self.norm: nn.Module = RMSNorm(config.d_model)
        else:
            self.norm = nn.LayerNorm(config.d_model)

        # Language model head — projects to vocabulary logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with LM head
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.embedding.weight

        # Initialise all weights
        self.apply(self._init_weights)
        # Re-apply scaled init to residual projections (GPT-2 convention)
        self._apply_scaled_init()

        n_params = count_parameters(self)
        logger.info(
            f"EmilySLM initialised | params={format_parameters(n_params)} | "
            f"layers={config.n_layers} | heads={config.n_heads} | "
            f"d_model={config.d_model} | vocab={config.vocab_size}"
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """Apply standard GPT-2 weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_scaled_init(self) -> None:
        """
        Scale residual projection weights by 1/√(2 × n_layers).

        This follows the GPT-2 paper recommendation to prevent the residual
        stream from growing too large in deep networks.
        """
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for name, param in self.named_parameters():
            if "out_proj" in name or "down_proj" in name:
                nn.init.normal_(param, mean=0.0, std=0.02 * scale)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[Optional[KVCache]]] = None,
    ) -> dict[str, torch.Tensor | list[Optional[KVCache]]]:
        """
        Full forward pass — used during training and prefill.

        Args:
            input_ids: Token IDs ``(batch, seq_len)``.
            attention_mask: Optional boolean mask ``(batch, seq_len)`` where
                ``True`` indicates valid tokens. Currently unused in the
                default causal masking path but reserved for future use.
            kv_caches: Optional list of per-layer ``KVCache`` objects for
                incremental decoding. Pass ``None`` for training.

        Returns:
            Dictionary with:
            - ``"logits"``: Unnormalised log-probs ``(batch, seq_len, vocab_size)``.
            - ``"kv_caches"``: Updated list of per-layer caches (or ``None``).
        """
        B, T = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # (B, T, d_model)

        # Add positional embeddings if not using RoPE
        if self.pos_embedding is not None:
            offset = 0
            if kv_caches is not None and kv_caches[0] is not None:
                offset = kv_caches[0].current_len
            positions = torch.arange(offset, offset + T, device=input_ids.device)
            pos_emb = self.pos_embedding.embedding(positions)  # (T, d_model)
            x = x + pos_emb.unsqueeze(0)

        # Pass through transformer blocks
        updated_caches: list[Optional[KVCache]] = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = layer(x, mask=None, kv_cache=cache_i)
            updated_caches.append(updated_cache)

        # Final normalisation + LM head
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return {"logits": logits, "kv_caches": updated_caches}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        generated_ids: Optional[torch.Tensor] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample the next token from logits using temperature, top-k, and top-p.

        Processing order:
        1. Apply temperature scaling.
        2. Apply top-k filtering (keep only top-k highest logits).
        3. Apply top-p (nucleus) filtering (keep smallest set summing to ≥ p).
        4. Sample from the resulting distribution.

        Args:
            logits: Raw logits for a single position ``(vocab_size,)`` or
                ``(batch, vocab_size)``.
            temperature: Scales logits before softmax. ``1.0`` = no scaling,
                ``< 1.0`` = sharper distribution, ``> 1.0`` = flatter.
            top_k: Keep only the top-k highest-probability tokens. ``0`` disables.
            top_p: Nucleus sampling threshold. ``1.0`` disables.

        Returns:
            Sampled token ID tensor ``(batch, 1)`` or ``(1,)`` if unbatched.
        """
        # Ensure batched: (batch, vocab_size)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Step 0 — repetition penalty (penalise tokens already generated)
        if repetition_penalty != 1.0 and generated_ids is not None:
            for b in range(logits.size(0)):
                for token_id in generated_ids[b].tolist():
                    if logits[b, token_id] < 0:
                        logits[b, token_id] *= repetition_penalty
                    else:
                        logits[b, token_id] /= repetition_penalty

        # Step 1 — temperature
        if temperature != 1.0 and temperature > 0.0:
            logits = logits / temperature

        # Step 2 — top-k
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_vals, _ = torch.topk(logits, k, dim=-1)
            threshold = topk_vals[..., -1, None]  # min value in top-k
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Step 3 — top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above top_p
            sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits = sorted_logits.masked_fill(sorted_remove, float("-inf"))
            # Scatter back to original order
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

        # Step 4 — sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        return next_token

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        stream: bool = False,
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor | Generator[int, None, None]:
        """
        Autoregressive token generation.

        When ``stream=False`` (default), generates all tokens and returns the
        complete sequence. When ``stream=True``, returns a generator that yields
        one new token ID at a time.

        Args:
            input_ids: Prompt token IDs ``(batch, seq_len)`` or ``(seq_len,)``.
                Single sequences are automatically batched.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (ignored when ``do_sample=False``).
            top_k: Top-k cutoff (``0`` to disable).
            top_p: Nucleus sampling threshold (``1.0`` to disable).
            do_sample: If ``True``, sample; if ``False``, use greedy decoding.
            eos_token_id: Stop generation when this token is produced.
            pad_token_id: Pad token ID (unused in this implementation).
            stream: If ``True``, return a generator instead of a tensor.

        Returns:
            - When ``stream=False``: ``(batch, original_len + new_tokens)`` tensor.
            - When ``stream=True``: Generator yielding ``int`` token IDs.
        """
        if stream:
            return self._generate_stream(
                input_ids, max_new_tokens, temperature, top_k, top_p,
                do_sample, eos_token_id, repetition_penalty,
            )
        return self._generate_full(
            input_ids, max_new_tokens, temperature, top_k, top_p,
            do_sample, eos_token_id, repetition_penalty,
        )

    def _generate_full(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        eos_token_id: Optional[int],
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor:
        """Non-streaming generation returning the full token tensor."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids.clone()
        kv_caches: list[Optional[KVCache]] = [None] * self.config.n_layers

        for _ in range(max_new_tokens):
            # Only feed the last token when cache is populated
            if kv_caches[0] is not None:
                context = generated[:, -1:]
            else:
                context = generated

            out = self.forward(context, kv_caches=kv_caches)
            logits = out["logits"][:, -1, :]  # (batch, vocab_size)
            kv_caches = out["kv_caches"]  # type: ignore[assignment]

            if do_sample:
                next_token = self._sample_next_token(
                    logits, temperature, top_k, top_p,
                    generated_ids=generated,
                    repetition_penalty=repetition_penalty,
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        eos_token_id: Optional[int],
        repetition_penalty: float = 1.3,
    ) -> Generator[int, None, None]:
        """Streaming generation yielding one token ID at a time."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids.clone()
        kv_caches: list[Optional[KVCache]] = [None] * self.config.n_layers

        for _ in range(max_new_tokens):
            context = generated[:, -1:] if kv_caches[0] is not None else generated
            out = self.forward(context, kv_caches=kv_caches)
            logits = out["logits"][:, -1, :]
            kv_caches = out["kv_caches"]  # type: ignore[assignment]

            if do_sample:
                next_token = self._sample_next_token(
                    logits, temperature, top_k, top_p,
                    generated_ids=generated,
                    repetition_penalty=repetition_penalty,
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            token_id: int = next_token[0, 0].item()  # type: ignore[assignment]
            generated = torch.cat([generated, next_token], dim=1)
            yield token_id

            if eos_token_id is not None and token_id == eos_token_id:
                return

    # ------------------------------------------------------------------
    # Beam Search
    # ------------------------------------------------------------------

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        num_beams: int = 4,
        eos_token_id: Optional[int] = None,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Beam search decoding.

        Maintains ``num_beams`` candidate sequences and selects the one with
        the highest cumulative log-probability at the end.

        Args:
            input_ids: Prompt ``(1, seq_len)`` — beam search requires batch=1.
            max_new_tokens: Maximum tokens to generate.
            num_beams: Number of beams.
            eos_token_id: Stop beam when EOS is produced.
            length_penalty: Exponent applied to sequence length when scoring.
                ``< 1.0`` favours shorter sequences, ``> 1.0`` favours longer.

        Returns:
            Best sequence ``(1, total_len)``.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError("Beam search requires batch_size=1")

        device = input_ids.device
        vocab_size = self.config.vocab_size

        # Initialise beams: (num_beams, seq_len)
        beams = input_ids.repeat(num_beams, 1)
        beam_scores = torch.zeros(num_beams, device=device)  # log-probs
        completed: list[tuple[float, torch.Tensor]] = []

        kv_caches_list: list[list[Optional[KVCache]]] = [
            [None] * self.config.n_layers for _ in range(num_beams)
        ]

        for step in range(max_new_tokens):
            all_next_scores: list[torch.Tensor] = []
            all_next_tokens: list[torch.Tensor] = []

            for b in range(num_beams):
                context = beams[b : b + 1, -1:] if kv_caches_list[b][0] is not None else beams[b : b + 1]
                out = self.forward(context, kv_caches=kv_caches_list[b])
                logits = out["logits"][0, -1, :]  # (vocab_size,)
                kv_caches_list[b] = out["kv_caches"]  # type: ignore[assignment]

                log_probs = F.log_softmax(logits, dim=-1)
                next_scores = beam_scores[b] + log_probs  # (vocab_size,)
                all_next_scores.append(next_scores)
                all_next_tokens.append(torch.arange(vocab_size, device=device))

            # Flatten and select top num_beams
            flat_scores = torch.cat(all_next_scores)  # (num_beams * vocab_size,)
            top_scores, top_indices = torch.topk(flat_scores, num_beams, sorted=True)

            beam_idx = top_indices // vocab_size
            token_idx = top_indices % vocab_size

            new_beams = []
            new_scores = []
            new_caches: list[list[Optional[KVCache]]] = []

            for rank in range(num_beams):
                b = beam_idx[rank].item()
                tok = token_idx[rank].item()
                score = top_scores[rank].item()

                new_seq = torch.cat(
                    [beams[b], torch.tensor([tok], device=device)]
                )  # (seq_len+1,)

                if eos_token_id is not None and tok == eos_token_id:
                    length = new_seq.shape[0]
                    normalised = score / (length ** length_penalty)
                    completed.append((normalised, new_seq.unsqueeze(0)))
                else:
                    new_beams.append(new_seq)
                    new_scores.append(score)
                    new_caches.append(kv_caches_list[b])  # type: ignore[arg-type]

            if not new_beams:
                break

            beams = torch.stack(new_beams)
            beam_scores = torch.tensor(new_scores, device=device)
            kv_caches_list = new_caches  # type: ignore[assignment]

        # If no beam completed (no EOS), add remaining beams
        for b in range(beams.shape[0]):
            length = beams[b].shape[0]
            score = beam_scores[b].item() / (length ** length_penalty)
            completed.append((score, beams[b].unsqueeze(0)))

        # Return best completed sequence
        best = max(completed, key=lambda t: t[0])
        return best[1]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of model parameters."""
        return count_parameters(self, trainable_only=trainable_only)

    def __repr__(self) -> str:
        n = count_parameters(self)
        return (
            f"EmilySLM("
            f"params={format_parameters(n)}, "
            f"layers={self.config.n_layers}, "
            f"heads={self.config.n_heads}, "
            f"d_model={self.config.d_model}, "
            f"vocab={self.config.vocab_size}"
            f")"
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str | Path) -> None:
        """
        Save model weights and config to a directory.

        Creates two files:
        - ``model.pt``: Model ``state_dict``.
        - ``config.yaml``: Serialised ``ModelConfig``.

        Args:
            path: Destination directory (created if it does not exist).
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), save_dir / "model.pt")

        # Save config
        import yaml
        with open(save_dir / "config.yaml", "w") as f:
            yaml.safe_dump({"model": self.config.to_dict()}, f)

        logger.info(f"Model saved → {save_dir}")

    @classmethod
    def from_pretrained(cls, path: str | Path, device: Optional[torch.device] = None) -> "EmilySLM":
        """
        Load a model from a saved directory.

        Args:
            path: Directory containing ``model.pt`` and ``config.yaml``.
            device: Target device. Defaults to CPU if not specified.

        Returns:
            Loaded ``EmilySLM`` instance.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        import yaml

        load_dir = Path(path)
        config_path = load_dir / "config.yaml"
        weights_path = load_dir / "model.pt"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        with open(config_path) as f:
            raw = yaml.safe_load(f)
        config = ModelConfig.from_dict(raw.get("model", raw))

        model = cls(config)
        map_location = device or torch.device("cpu")
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        if device is not None:
            model = model.to(device)

        logger.info(f"Model loaded ← {load_dir}")
        return model

    @classmethod
    def from_config(cls, config: ModelConfig) -> "EmilySLM":
        """Convenience alias for ``EmilySLM(config)``."""
        return cls(config)
