"""
Tests for EmilyTokenizer and ChatTemplate.
"""

from __future__ import annotations

import pytest

from slm.tokenizer.special_tokens import SpecialTokens
from slm.tokenizer.tokenizer import EmilyTokenizer
from slm.tokenizer.chat_template import ChatTemplate


CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Emily is a small language model built from scratch in PyTorch.",
    "Transformers use attention mechanisms to process sequences.",
    "Machine learning requires large datasets and compute.",
    "Hello world! This is a test of the tokenizer system.",
] * 20  # repeat to have enough frequency for BPE merges


@pytest.fixture(scope="module")
def tokenizer() -> EmilyTokenizer:
    """Train a tiny tokenizer once for the whole module."""
    return EmilyTokenizer.train(CORPUS, vocab_size=512)


class TestSpecialTokens:
    def test_all_tokens_non_empty(self):
        st = SpecialTokens()
        for tok in st.all_tokens:
            assert tok and isinstance(tok, str)

    def test_all_tokens_unique(self):
        st = SpecialTokens()
        assert len(st.all_tokens) == len(set(st.all_tokens))

    def test_to_dict_roundtrip(self):
        st = SpecialTokens()
        d = st.to_dict()
        st2 = SpecialTokens.from_dict(d)
        assert st.all_tokens == st2.all_tokens

    def test_correct_count(self):
        st = SpecialTokens()
        assert len(st.all_tokens) == 9  # pad, eos, bos, unk, system, user, assistant, tool_call, tool_result


class TestEmilyTokenizer:
    def test_vocab_size(self, tokenizer: EmilyTokenizer):
        assert tokenizer.vocab_size >= 9  # at minimum the special tokens

    def test_special_token_ids_are_valid(self, tokenizer: EmilyTokenizer):
        assert tokenizer.pad_token_id >= 0
        assert tokenizer.eos_token_id >= 0
        assert tokenizer.bos_token_id >= 0
        assert tokenizer.unk_token_id >= 0

    def test_special_token_ids_are_unique(self, tokenizer: EmilyTokenizer):
        ids = [
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.unk_token_id,
        ]
        assert len(ids) == len(set(ids))

    def test_encode_returns_list_of_ints(self, tokenizer: EmilyTokenizer):
        ids = tokenizer.encode("Hello world!")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_adds_bos_eos(self, tokenizer: EmilyTokenizer):
        ids = tokenizer.encode("Hello", add_special_tokens=True)
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id

    def test_decode_roundtrip(self, tokenizer: EmilyTokenizer):
        text = "Hello world"
        ids = tokenizer.encode(text, add_special_tokens=False)
        recovered = tokenizer.decode(ids, skip_special_tokens=True)
        assert "Hello" in recovered

    def test_truncation(self, tokenizer: EmilyTokenizer):
        text = "word " * 50
        ids = tokenizer.encode(text, max_length=10, truncation=True)
        assert len(ids) <= 10

    def test_padding(self, tokenizer: EmilyTokenizer):
        ids = tokenizer.encode("Hi", max_length=20, padding=True, add_special_tokens=False)
        assert len(ids) == 20
        assert tokenizer.pad_token_id in ids

    def test_batch_encode_matches_individual(self, tokenizer: EmilyTokenizer):
        texts = ["Hello world", "Emily SLM"]
        batch = tokenizer.encode_batch(texts, add_special_tokens=False)
        for i, text in enumerate(texts):
            single = tokenizer.encode(text, add_special_tokens=False)
            assert batch[i] == single

    def test_stream_encode_matches_encode(self, tokenizer: EmilyTokenizer):
        text = "The quick brown fox"
        ids_batch = tokenizer.encode(text, add_special_tokens=False)
        ids_stream = list(tokenizer.stream_encode(text))
        assert ids_batch == ids_stream

    def test_pad_sequence_right(self, tokenizer: EmilyTokenizer):
        ids = [1, 2, 3]
        padded = tokenizer.pad_sequence(ids, max_length=6, pad_left=False)
        assert len(padded) == 6
        assert padded[:3] == ids
        assert all(p == tokenizer.pad_token_id for p in padded[3:])

    def test_pad_sequence_left(self, tokenizer: EmilyTokenizer):
        ids = [1, 2, 3]
        padded = tokenizer.pad_sequence(ids, max_length=6, pad_left=True)
        assert len(padded) == 6
        assert padded[3:] == ids

    def test_pad_sequence_truncation(self, tokenizer: EmilyTokenizer):
        ids = list(range(10))
        padded = tokenizer.pad_sequence(ids, max_length=5)
        assert len(padded) == 5

    def test_save_load_roundtrip(self, tokenizer: EmilyTokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        tokenizer.save(str(path))
        assert path.exists()

        tok2 = EmilyTokenizer.load(str(path))
        assert tok2.vocab_size == tokenizer.vocab_size
        assert tok2.eos_token_id == tokenizer.eos_token_id

        text = "Hello world"
        ids1 = tokenizer.encode(text, add_special_tokens=False)
        ids2 = tok2.encode(text, add_special_tokens=False)
        assert ids1 == ids2

    def test_len(self, tokenizer: EmilyTokenizer):
        assert len(tokenizer) == tokenizer.vocab_size

    def test_repr(self, tokenizer: EmilyTokenizer):
        r = repr(tokenizer)
        assert "EmilyTokenizer" in r

    def test_format_chat(self, tokenizer: EmilyTokenizer):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        text = tokenizer.format_chat(messages, add_generation_prompt=True)
        assert "<|system|>" in text
        assert "<|user|>" in text
        assert "<|assistant|>" in text
        assert "You are helpful." in text
        assert "Hello!" in text


class TestChatTemplate:
    def test_validate_messages_valid(self, tokenizer: EmilyTokenizer):
        messages = [{"role": "user", "content": "Hi"}]
        ChatTemplate.validate_messages(messages)  # Should not raise

    def test_validate_messages_invalid_role(self):
        with pytest.raises(ValueError, match="role"):
            ChatTemplate.validate_messages([{"role": "unknown", "content": "x"}])

    def test_validate_messages_missing_content(self):
        with pytest.raises(ValueError, match="content"):
            ChatTemplate.validate_messages([{"role": "user"}])

    def test_format_messages(self, tokenizer: EmilyTokenizer):
        messages = [{"role": "user", "content": "What is 2+2?"}]
        text = ChatTemplate.format_messages(messages, tokenizer, add_generation_prompt=True)
        assert "<|user|>" in text
        assert "2+2" in text
        assert text.endswith("<|assistant|>")

    def test_apply_chat_template_returns_list_of_ints(self, tokenizer: EmilyTokenizer):
        messages = [{"role": "user", "content": "Hello"}]
        text = ChatTemplate.format_messages(messages, tokenizer)
        ids = ChatTemplate.apply_chat_template(text, tokenizer)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_build_and_encode(self, tokenizer: EmilyTokenizer):
        messages = [{"role": "user", "content": "Hi Emily"}]
        ids = ChatTemplate.build_and_encode(messages, tokenizer)
        assert len(ids) > 0
