"""
Inference engine for Emily SLM.

Provides a high-level ``EmilyInferenceEngine`` that loads a model,
manages device placement, and exposes a clean generation API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional

import torch

from slm.utils.device import get_device, get_dtype

logger = logging.getLogger(__name__)


class EmilyInferenceEngine:
    """
    High-level inference engine for Emily SLM.

    Handles model loading, device management, and provides clean
    generate / chat / stream APIs for production use.

    Args:
        model_path: Path to a saved model directory (containing
            ``model.pt`` and ``config.yaml``).
        tokenizer_path: Path to a saved tokenizer JSON file.
        device: Compute device. Auto-detected if ``None``.
        amp_dtype: AMP dtype string (``"bfloat16"``, ``"float16"``, ``"float32"``).

    Example:
        >>> engine = EmilyInferenceEngine(
        ...     model_path="checkpoints/emily-tiny/best",
        ...     tokenizer_path="checkpoints/emily-tiny/tokenizer.json",
        ... )
        >>> response = engine.generate("Hello, Emily!")
        >>> print(response)
    """

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        device: Optional[torch.device] = None,
        amp_dtype: str = "bfloat16",
    ) -> None:
        self.device = device or get_device()
        self.amp_dtype = get_dtype(amp_dtype)
        self.use_amp = self.device.type == "cuda"

        # Load tokenizer
        from slm.tokenizer import EmilyTokenizer
        self.tokenizer = EmilyTokenizer.load(tokenizer_path)
        logger.info(f"Tokenizer loaded (vocab_size={len(self.tokenizer)})")

        # Load model
        from slm.model import EmilySLM
        self.model = EmilySLM.from_pretrained(model_path, device=self.device)
        self.model.eval()
        logger.info(f"Model loaded: {self.model}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.2,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        skip_special_tokens: bool = True,
        repetition_penalty: float = 1.5,
    ) -> str:
        """
        Generate text from a text prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K cutoff.
            top_p: Nucleus sampling threshold.
            do_sample: Greedy if ``False``, sampled if ``True``.
            skip_special_tokens: Strip special tokens from output.
            repetition_penalty: Penalty for repeating tokens. >1.0 reduces repetition.

        Returns:
            Generated text string (prompt not included).
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)
        prompt_len = len(input_ids)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                stream=False,
                repetition_penalty=repetition_penalty,
            )

        new_ids = output_ids[0, prompt_len:].tolist()
        return self.tokenizer.decode(new_ids, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """
        Stream generated text token-by-token.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K cutoff.
            top_p: Nucleus sampling threshold.

        Yields:
            Decoded text fragment for each generated token.
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        token_gen = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            stream=True,
        )

        for token_id in token_gen:
            yield self.tokenizer.decode([token_id], skip_special_tokens=True)

    @torch.no_grad()
    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 1.2,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.5,
    ) -> str:
        """
        Chat-style generation from a message list.

        Args:
            messages: List of ``{"role": str, "content": str}`` dicts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus threshold.
            top_k: Top-K cutoff.
            repetition_penalty: Penalty for repeating tokens.

        Returns:
            Assistant reply string.
        """
        from slm.tokenizer import ChatTemplate
        prompt = ChatTemplate.format_messages(messages, self.tokenizer, add_generation_prompt=True)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
