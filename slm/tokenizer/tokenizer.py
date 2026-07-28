"""
Emily BPE Tokenizer.

Wraps the HuggingFace ``tokenizers`` library to provide a production-quality
Byte-Pair Encoding (BPE) tokenizer with:

- Vocabulary training from raw text or files
- Full special token support (pad, eos, bos, unk, system, user, assistant, tool_call, tool_result)
- Encoding with optional BOS/EOS injection, truncation, and padding
- Batch encoding
- Streaming tokenisation
- Chat template formatting
- Save / load to a single JSON file

Example:
    >>> tokenizer = EmilyTokenizer.train(texts, vocab_size=4096)
    >>> ids = tokenizer.encode("Hello, Emily!")
    >>> text = tokenizer.decode(ids)
    >>> tokenizer.save("checkpoints/tiny/tokenizer.json")
    >>> tokenizer2 = EmilyTokenizer.load("checkpoints/tiny/tokenizer.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, Iterable, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from slm.tokenizer.special_tokens import SpecialTokens

logger = logging.getLogger(__name__)


class EmilyTokenizer:
    """
    BPE tokenizer for Emily SLM.

    Wraps ``tokenizers.Tokenizer`` and exposes a clean, typed API
    tailored for language model training and inference.

    Attributes:
        special_tokens: ``SpecialTokens`` instance used by this tokenizer.

    Example:
        >>> texts = ["Hello world!", "Emily is a language model."]
        >>> tok = EmilyTokenizer.train(texts, vocab_size=1000)
        >>> ids = tok.encode("Hello world!")
        >>> tok.decode(ids)
        'Hello world!'
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        special_tokens: Optional[SpecialTokens] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.special_tokens = special_tokens or SpecialTokens()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        """Token ID for the padding token."""
        return self._tokenizer.token_to_id(self.special_tokens.pad_token)

    @property
    def eos_token_id(self) -> int:
        """Token ID for the end-of-sequence token."""
        return self._tokenizer.token_to_id(self.special_tokens.eos_token)

    @property
    def bos_token_id(self) -> int:
        """Token ID for the beginning-of-sequence token."""
        return self._tokenizer.token_to_id(self.special_tokens.bos_token)

    @property
    def unk_token_id(self) -> int:
        """Token ID for the unknown token."""
        return self._tokenizer.token_to_id(self.special_tokens.unk_token)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 32_000,
        special_tokens: Optional[SpecialTokens] = None,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> "EmilyTokenizer":
        """
        Train a new BPE tokenizer on a list of text strings.

        Args:
            texts: List of raw text strings to train on.
            vocab_size: Target vocabulary size (including special tokens).
            special_tokens: Custom ``SpecialTokens``; uses defaults if ``None``.
            min_frequency: Minimum pair frequency for BPE merges.
            show_progress: Show a progress bar during training.

        Returns:
            Trained ``EmilyTokenizer`` instance.
        """
        st = special_tokens or SpecialTokens()
        tokenizer = Tokenizer(BPE(unk_token=st.unk_token))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=st.all_tokens,
            min_frequency=min_frequency,
            show_progress=show_progress,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)

        instance = cls(tokenizer, st)
        logger.info(
            f"Tokenizer trained | vocab_size={instance.vocab_size} | "
            f"texts={len(texts)}"
        )
        return instance

    @classmethod
    def train_from_files(
        cls,
        files: list[str],
        vocab_size: int = 32_000,
        special_tokens: Optional[SpecialTokens] = None,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> "EmilyTokenizer":
        """
        Train a new BPE tokenizer from text files.

        Args:
            files: List of file paths (plain text, one document per file).
            vocab_size: Target vocabulary size.
            special_tokens: Custom ``SpecialTokens``; uses defaults if ``None``.
            min_frequency: Minimum pair frequency for BPE merges.
            show_progress: Show a progress bar during training.

        Returns:
            Trained ``EmilyTokenizer`` instance.
        """
        st = special_tokens or SpecialTokens()
        tokenizer = Tokenizer(BPE(unk_token=st.unk_token))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=st.all_tokens,
            min_frequency=min_frequency,
            show_progress=show_progress,
        )
        tokenizer.train(files, trainer=trainer)

        instance = cls(tokenizer, st)
        logger.info(
            f"Tokenizer trained from {len(files)} files | vocab_size={instance.vocab_size}"
        )
        return instance

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
        padding_side: str = "right",
    ) -> list[int]:
        """
        Encode a single text string to a list of token IDs.

        Args:
            text: Input text to tokenise.
            add_special_tokens: Prepend BOS and append EOS when ``True``.
            max_length: Maximum sequence length. Required when ``truncation=True``.
            truncation: Truncate to ``max_length`` when ``True``.
            padding: Pad to ``max_length`` when ``True``.
            padding_side: ``"right"`` (default) or ``"left"``.

        Returns:
            List of integer token IDs.
        """
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids: list[int] = encoding.ids

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]

        if padding and max_length is not None and len(ids) < max_length:
            pad_count = max_length - len(ids)
            pad_ids = [self.pad_token_id] * pad_count
            ids = ids + pad_ids if padding_side == "right" else pad_ids + ids

        return ids

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
    ) -> list[list[int]]:
        """
        Encode a batch of text strings.

        Args:
            texts: List of input strings.
            add_special_tokens: Add BOS/EOS tokens.
            max_length: Maximum length per sequence.
            truncation: Truncate to ``max_length``.
            padding: Pad all sequences to ``max_length``.

        Returns:
            List of token ID lists.
        """
        return [
            self.encode(
                t,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
            )
            for t in texts
        ]

    def encode_pair(
        self,
        text1: str,
        text2: str,
        add_special_tokens: bool = True,
    ) -> tuple[list[int], list[int]]:
        """
        Encode two texts separately (e.g., prompt and completion for SFT).

        Args:
            text1: First text (e.g., instruction).
            text2: Second text (e.g., response).
            add_special_tokens: Add BOS/EOS to each sequence.

        Returns:
            Tuple of ``(ids1, ids2)``.
        """
        return (
            self.encode(text1, add_special_tokens=add_special_tokens),
            self.encode(text2, add_special_tokens=add_special_tokens),
        )

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token IDs back to text.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: Remove special tokens from the output.

        Returns:
            Decoded text string.
        """
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        token_ids_list: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            token_ids_list: List of token ID lists.
            skip_special_tokens: Remove special tokens.

        Returns:
            List of decoded text strings.
        """
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids_list]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream_encode(self, text: str) -> Generator[int, None, None]:
        """
        Yield token IDs one at a time from the encoded text.

        Useful for streaming tokenisation of large documents without
        materialising the full token list in memory.

        Args:
            text: Input text to tokenise.

        Yields:
            Integer token IDs in order.
        """
        ids = self.encode(text, add_special_tokens=False)
        yield from ids

    def stream_decode(self, token_ids: Iterable[int]) -> Generator[str, None, None]:
        """
        Decode token IDs incrementally, yielding text chunks.

        Each yielded chunk corresponds to one token decoded individually.
        Note: Multi-byte characters may produce ``Â» `` artefacts if decoded
        one token at a time; use ``decode()`` for clean output.

        Args:
            token_ids: Iterable of integer token IDs.

        Yields:
            Decoded text fragment for each token.
        """
        for tid in token_ids:
            yield self._tokenizer.decode([tid], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Padding
    # ------------------------------------------------------------------

    def pad_sequence(
        self,
        token_ids: list[int],
        max_length: int,
        pad_left: bool = False,
    ) -> list[int]:
        """
        Pad or truncate a token ID sequence to ``max_length``.

        Args:
            token_ids: Input token ID list.
            max_length: Target length.
            pad_left: Pad on the left side when ``True``.

        Returns:
            Padded/truncated list of length ``max_length``.
        """
        current_len = len(token_ids)
        if current_len >= max_length:
            return token_ids[:max_length]

        pad_count = max_length - current_len
        pad_ids = [self.pad_token_id] * pad_count
        return pad_ids + token_ids if pad_left else token_ids + pad_ids

    # ------------------------------------------------------------------
    # Chat formatting
    # ------------------------------------------------------------------

    def format_chat(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format a list of chat messages into a single string using special tokens.

        Format:
            <|bos|><|system|>system content<|eos|>
            <|user|>user content<|eos|>
            <|assistant|>assistant content<|eos|>
            [<|assistant|>]  ← if add_generation_prompt=True

        Args:
            messages: List of ``{"role": str, "content": str}`` dicts.
                Valid roles: ``"system"``, ``"user"``, ``"assistant"``.
            add_generation_prompt: Append the assistant role token at the end
                to prime the model for generation.

        Returns:
            Formatted string ready for tokenisation.

        Raises:
            ValueError: If an unknown role is encountered.
        """
        st = self.special_tokens
        role_tokens = {
            "system": st.system_token,
            "user": st.user_token,
            "assistant": st.assistant_token,
            "tool": st.tool_call_token,
        }

        parts: list[str] = [st.bos_token]
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role not in role_tokens:
                raise ValueError(
                    f"Unknown role {role!r}. Valid roles: {list(role_tokens.keys())}"
                )
            parts.append(f"{role_tokens[role]}{content}{st.eos_token}")

        if add_generation_prompt:
            parts.append(st.assistant_token)

        return "".join(parts)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the tokenizer to disk.

        Creates two files:
        - ``<path>``                   — the tokenizers library JSON (unchanged)
        - ``<path>.config.json``       — Emily special tokens sidecar

        This two-file approach avoids corrupting the tokenizers library
        JSON format, which does not accept unknown top-level keys.

        Args:
            path: Destination file path (e.g., ``"checkpoints/tiny/tokenizer.json"``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the core tokenizers state (untouched)
        self._tokenizer.save(str(path))

        # Save Emily special tokens to a sidecar file
        sidecar = path.with_suffix(".config.json")
        with open(sidecar, "w") as f:
            json.dump({"emily_special_tokens": self.special_tokens.to_dict()}, f, indent=2)

        logger.info(f"Tokenizer saved → {path} + {sidecar.name}")

    @classmethod
    def load(cls, path: str | Path) -> "EmilyTokenizer":
        """
        Load a tokenizer from files created by :meth:`save`.

        Reads ``<path>`` for the core vocabulary and ``<path>.config.json``
        for the Emily special tokens sidecar (optional — falls back to defaults).

        Args:
            path: Path to the tokenizer JSON file.

        Returns:
            Loaded ``EmilyTokenizer`` instance.

        Raises:
            FileNotFoundError: If the tokenizer JSON file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        tokenizer = Tokenizer.from_file(str(path))

        # Load special tokens from sidecar if it exists
        sidecar = path.with_suffix(".config.json")
        st = SpecialTokens()
        if sidecar.exists():
            with open(sidecar) as f:
                data = json.load(f)
            st_dict = data.get("emily_special_tokens", {})
            if st_dict:
                st = SpecialTokens.from_dict(st_dict)

        logger.info(f"Tokenizer loaded ← {path} (vocab_size={tokenizer.get_vocab_size()})")
        return cls(tokenizer, st)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"EmilyTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"pad={self.pad_token_id}, "
            f"eos={self.eos_token_id}, "
            f"bos={self.bos_token_id}"
            f")"
        )
