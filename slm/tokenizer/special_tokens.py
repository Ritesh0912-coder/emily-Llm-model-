"""
Special tokens for Emily SLM.

Defines the full set of special tokens used by the Emily tokenizer,
including standard language model tokens (BOS, EOS, PAD, UNK) and
Emily-specific tokens for structured chat formatting and tool calling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpecialTokens:
    """
    Special token definitions for the Emily BPE tokenizer.

    Includes standard LM tokens and Emily-specific tokens for:
    - Chat message role demarcation (system, user, assistant)
    - Tool calling protocol (tool_call, tool_result)

    All token strings use the ``<|...|>`` convention to minimise collision
    with natural text.

    Attributes:
        pad_token: Padding token — used to fill shorter sequences in a batch.
        eos_token: End-of-sequence token — signals the end of generation.
        bos_token: Beginning-of-sequence token — prepended to every sequence.
        unk_token: Unknown token — substituted for out-of-vocabulary sub-words.
        system_token: Marks the start of a system prompt in chat format.
        user_token: Marks the start of a user message in chat format.
        assistant_token: Marks the start of an assistant response in chat format.
        tool_call_token: Marks a structured tool call in the assistant turn.
        tool_result_token: Marks the result returned by a tool.

    Example:
        >>> st = SpecialTokens()
        >>> print(st.all_tokens)
        ['<|pad|>', '<|eos|>', '<|bos|>', '<|unk|>', '<|system|>',
         '<|user|>', '<|assistant|>', '<|tool_call|>', '<|tool_result|>']
    """

    pad_token: str = "<|pad|>"
    eos_token: str = "<|eos|>"
    bos_token: str = "<|bos|>"
    unk_token: str = "<|unk|>"
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    tool_call_token: str = "<|tool_call|>"
    tool_result_token: str = "<|tool_result|>"

    @property
    def all_tokens(self) -> list[str]:
        """Return all special tokens as an ordered list."""
        return [
            self.pad_token,
            self.eos_token,
            self.bos_token,
            self.unk_token,
            self.system_token,
            self.user_token,
            self.assistant_token,
            self.tool_call_token,
            self.tool_result_token,
        ]

    def to_dict(self) -> dict[str, str]:
        """Serialise to a plain dictionary mapping role → token string."""
        return {
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "unk_token": self.unk_token,
            "system_token": self.system_token,
            "user_token": self.user_token,
            "assistant_token": self.assistant_token,
            "tool_call_token": self.tool_call_token,
            "tool_result_token": self.tool_result_token,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "SpecialTokens":
        """Reconstruct from a serialised dictionary."""
        return cls(
            pad_token=d.get("pad_token", "<|pad|>"),
            eos_token=d.get("eos_token", "<|eos|>"),
            bos_token=d.get("bos_token", "<|bos|>"),
            unk_token=d.get("unk_token", "<|unk|>"),
            system_token=d.get("system_token", "<|system|>"),
            user_token=d.get("user_token", "<|user|>"),
            assistant_token=d.get("assistant_token", "<|assistant|>"),
            tool_call_token=d.get("tool_call_token", "<|tool_call|>"),
            tool_result_token=d.get("tool_result_token", "<|tool_result|>"),
        )
