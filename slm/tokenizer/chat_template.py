"""
Chat template formatting for Emily SLM.

Provides ``ChatTemplate`` — a stateless utility class for converting
OpenAI-style message lists into the Emily prompt format and tokenising them.

Emily chat format:
    <|bos|>
    <|system|>{system content}<|eos|>
    <|user|>{user content}<|eos|>
    <|assistant|>{assistant content}<|eos|>
    <|assistant|>          ← generation prompt (optional)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slm.tokenizer.tokenizer import EmilyTokenizer

VALID_ROLES = {"system", "user", "assistant", "tool"}


class ChatTemplate:
    """
    Stateless chat template formatter for Emily SLM.

    Converts a list of OpenAI-style message dicts into:
    - A formatted string using Emily special tokens
    - A list of token IDs ready for inference

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are Emily, a helpful AI."},
        ...     {"role": "user",   "content": "What is 2 + 2?"},
        ... ]
        >>> text = ChatTemplate.format_messages(messages, tokenizer)
        >>> ids  = ChatTemplate.apply_chat_template(text, tokenizer)
    """

    @staticmethod
    def validate_messages(messages: list[dict[str, str]]) -> None:
        """
        Validate that all messages have required keys and valid roles.

        Args:
            messages: List of message dicts.

        Raises:
            ValueError: If a message is malformed or has an invalid role.
        """
        for i, msg in enumerate(messages):
            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role' key: {msg}")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content' key: {msg}")
            role = msg["role"].lower()
            if role not in VALID_ROLES:
                raise ValueError(
                    f"Message {i} has invalid role {role!r}. "
                    f"Valid roles: {sorted(VALID_ROLES)}"
                )

    @staticmethod
    def format_messages(
        messages: list[dict[str, str]],
        tokenizer: "EmilyTokenizer",
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format a list of chat messages into a string using Emily special tokens.

        Args:
            messages: List of ``{"role": str, "content": str}`` dicts.
            tokenizer: ``EmilyTokenizer`` instance (provides special token strings).
            add_generation_prompt: Append ``<|assistant|>`` at the end to prime
                the model for generation.

        Returns:
            Formatted string ready for tokenisation.

        Raises:
            ValueError: If any message is malformed.
        """
        ChatTemplate.validate_messages(messages)
        return tokenizer.format_chat(messages, add_generation_prompt=add_generation_prompt)

    @staticmethod
    def apply_chat_template(
        text: str,
        tokenizer: "EmilyTokenizer",
        max_length: int | None = None,
        truncation: bool = True,
    ) -> list[int]:
        """
        Tokenise a pre-formatted chat string.

        Args:
            text: Formatted chat string (output of ``format_messages``).
            tokenizer: ``EmilyTokenizer`` to tokenise with.
            max_length: Optional maximum token length.
            truncation: Truncate to ``max_length`` when ``True``.

        Returns:
            List of token IDs.
        """
        return tokenizer.encode(
            text,
            add_special_tokens=False,  # Special tokens already embedded in text
            max_length=max_length,
            truncation=truncation,
        )

    @staticmethod
    def build_and_encode(
        messages: list[dict[str, str]],
        tokenizer: "EmilyTokenizer",
        add_generation_prompt: bool = True,
        max_length: int | None = None,
    ) -> list[int]:
        """
        Format messages and tokenise in one call.

        Args:
            messages: List of chat message dicts.
            tokenizer: ``EmilyTokenizer`` instance.
            add_generation_prompt: Append the assistant role token.
            max_length: Optional truncation length.

        Returns:
            List of token IDs.
        """
        text = ChatTemplate.format_messages(messages, tokenizer, add_generation_prompt)
        return ChatTemplate.apply_chat_template(text, tokenizer, max_length=max_length)
