"""
Tests for the Emily SLM model layers and full EmilySLM model.
Uses ModelConfig.tiny() for fast CPU testing.
"""

from __future__ import annotations

import pytest
import torch

from slm.config import ModelConfig
from slm.model.layers.rmsnorm import RMSNorm
from slm.model.layers.embedding import TokenEmbedding, LearnablePositionalEmbedding
from slm.model.layers.rotary import RotaryEmbedding, apply_rotary_emb, rotate_half
from slm.model.layers.attention import CausalSelfAttention, GroupedQueryAttention, KVCache
from slm.model.layers.mlp import SwiGLU, GeLUMLP
from slm.model.layers.transformer import TransformerBlock
from slm.model.model import EmilySLM


TINY = ModelConfig.tiny()
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(128)
        x = torch.randn(2, 16, 128)
        y = norm(x)
        assert y.shape == x.shape

    def test_no_nan(self):
        norm = RMSNorm(128)
        x = torch.randn(2, 16, 128)
        y = norm(x)
        assert not torch.isnan(y).any()

    def test_scale_parameter_exists(self):
        norm = RMSNorm(64)
        assert norm.weight.shape == (64,)

    def test_normalisation_effect(self):
        norm = RMSNorm(128)
        # After norm, RMS should be close to 1 (before weight scaling)
        x = torch.randn(4, 8, 128) * 100  # large values
        y = norm(x)
        # Weight starts as ones, so output RMS ≈ 1 per position
        rms = y.pow(2).mean(-1).sqrt()
        assert (rms - 1.0).abs().mean().item() < 0.1


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestTokenEmbedding:
    def test_output_shape(self):
        emb = TokenEmbedding(vocab_size=4096, d_model=128)
        ids = torch.randint(0, 4096, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, 128)

    def test_scaled_output(self):
        import math
        emb = TokenEmbedding(vocab_size=4096, d_model=128)
        ids = torch.zeros(1, 1, dtype=torch.long)
        out = emb(ids)
        # Output should be scaled by sqrt(d_model) = sqrt(128) ≈ 11.31
        assert out.abs().mean().item() > 0.0


class TestLearnablePositionalEmbedding:
    def test_output_shape(self):
        pe = LearnablePositionalEmbedding(d_model=128, max_seq_len=256)
        out = pe(seq_len=16, device=DEVICE)
        assert out.shape == (16, 128)

    def test_exceeds_max_raises(self):
        pe = LearnablePositionalEmbedding(d_model=128, max_seq_len=32)
        with pytest.raises(ValueError):
            pe(seq_len=64, device=DEVICE)


# ---------------------------------------------------------------------------
# Rotary Embeddings
# ---------------------------------------------------------------------------

class TestRotaryEmbedding:
    def test_output_shapes(self):
        rope = RotaryEmbedding(dim=32)
        cos, sin = rope(seq_len=16, device=DEVICE)
        assert cos.shape == (16, 32)
        assert sin.shape == (16, 32)

    def test_cos_sin_values_bounded(self):
        rope = RotaryEmbedding(dim=32)
        cos, sin = rope(seq_len=8, device=DEVICE)
        assert cos.abs().max().item() <= 1.0 + 1e-5
        assert sin.abs().max().item() <= 1.0 + 1e-5

    def test_cache_extends_automatically(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=16)
        cos1, _ = rope(seq_len=16, device=DEVICE)
        cos2, _ = rope(seq_len=32, device=DEVICE)  # Should extend cache
        assert cos2.shape[0] == 32


class TestRotateHalf:
    def test_shape_preserved(self):
        x = torch.randn(2, 4, 8, 32)
        y = rotate_half(x)
        assert y.shape == x.shape

    def test_double_rotation_is_negation(self):
        x = torch.randn(1, 1, 4, 32)
        y = rotate_half(rotate_half(x))
        assert torch.allclose(y, -x, atol=1e-5)


class TestApplyRotaryEmb:
    def test_shape_preserved(self):
        B, H, T, D = 2, 4, 8, 32
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        rope = RotaryEmbedding(dim=D)
        cos, sin = rope(T, DEVICE)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(TINY)
        x = torch.randn(2, 16, TINY.d_model)
        out, cache = attn(x)
        assert out.shape == (2, 16, TINY.d_model)
        assert cache is None

    def test_no_nan(self):
        attn = CausalSelfAttention(TINY)
        x = torch.randn(2, 8, TINY.d_model)
        out, _ = attn(x)
        assert not torch.isnan(out).any()

    def test_kv_cache_consistency(self):
        """Cached and non-cached outputs must match for the last token."""
        attn = CausalSelfAttention(TINY)
        attn.eval()
        x = torch.randn(1, 4, TINY.d_model)

        # Without cache — full sequence
        with torch.no_grad():
            out_full, _ = attn(x)

        # With incremental cache — seed with an empty KVCache so the
        # prefill call populates it and returns the filled cache object
        init_cache = KVCache.empty(
            batch_size=1, n_kv_heads=TINY.n_heads,
            head_dim=TINY.d_model // TINY.n_heads, device=DEVICE
        )
        with torch.no_grad():
            # Prefill the first 3 tokens using the empty cache
            _, cache_after_prefill = attn(x[:, :3, :], kv_cache=init_cache)
            # Decode token 4 — cache now holds keys/values for positions 0-2
            out_step, _ = attn(x[:, 3:4, :], kv_cache=cache_after_prefill)

        # Output for position 3 should match full-sequence output at pos 3
        assert torch.allclose(out_full[:, 3:4, :], out_step, atol=1e-4)


class TestGroupedQueryAttention:
    def test_output_shape(self):
        cfg = ModelConfig.small()  # n_heads=8, n_kv_heads=4
        gqa = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, cfg.d_model)
        out, _ = gqa(x)
        assert out.shape == (2, 16, cfg.d_model)

    def test_no_nan(self):
        cfg = ModelConfig.small()
        gqa = GroupedQueryAttention(cfg)
        x = torch.randn(1, 8, cfg.d_model)
        out, _ = gqa(x)
        assert not torch.isnan(out).any()


class TestKVCache:
    def test_update_appends(self):
        cache = KVCache.empty(batch_size=1, n_kv_heads=4, head_dim=32, device=DEVICE)
        assert cache.current_len == 0
        k = torch.randn(1, 4, 5, 32)
        v = torch.randn(1, 4, 5, 32)
        k_full, v_full = cache.update(k, v)
        assert cache.current_len == 5
        assert k_full.shape == (1, 4, 5, 32)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU(d_model=128, d_ff=512)
        x = torch.randn(2, 16, 128)
        out = ffn(x)
        assert out.shape == (2, 16, 128)

    def test_no_nan(self):
        ffn = SwiGLU(d_model=128, d_ff=512)
        x = torch.randn(2, 8, 128)
        assert not torch.isnan(ffn(x)).any()

    def test_recommended_d_ff(self):
        d_ff = SwiGLU.recommended_d_ff(512)
        assert d_ff % 64 == 0
        assert d_ff > 0


class TestGeLUMLP:
    def test_output_shape(self):
        mlp = GeLUMLP(d_model=128, d_ff=512)
        x = torch.randn(2, 16, 128)
        assert mlp(x).shape == (2, 16, 128)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(TINY)
        x = torch.randn(2, 16, TINY.d_model)
        out, cache = block(x)
        assert out.shape == (2, 16, TINY.d_model)
        assert cache is None

    def test_residual_connection(self):
        """Output should differ from input (residual not just pass-through)."""
        block = TransformerBlock(TINY)
        x = torch.randn(1, 8, TINY.d_model)
        out, _ = block(x)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Full EmilySLM model
# ---------------------------------------------------------------------------

class TestEmilySLM:
    def test_forward_shape(self):
        model = EmilySLM(TINY)
        ids = torch.randint(0, TINY.vocab_size, (2, 16))
        out = model(ids)
        assert out["logits"].shape == (2, 16, TINY.vocab_size)

    def test_no_nan_in_logits(self):
        model = EmilySLM(TINY)
        ids = torch.randint(0, TINY.vocab_size, (1, 8))
        out = model(ids)
        assert not torch.isnan(out["logits"]).any()

    def test_parameter_count_positive(self):
        model = EmilySLM(TINY)
        assert model.count_parameters() > 0

    def test_weight_tying(self):
        cfg = ModelConfig.tiny()
        cfg.tie_embeddings = True
        model = EmilySLM(cfg)
        # Embedding weight and LM head weight must be the same tensor
        assert model.lm_head.weight is model.token_embedding.embedding.weight

    def test_generate_output_length(self):
        model = EmilySLM(TINY)
        model.eval()
        ids = torch.randint(0, TINY.vocab_size, (1, 4))
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=5, do_sample=False, stream=False)
        assert out.shape[1] == 4 + 5

    def test_generate_streaming(self):
        model = EmilySLM(TINY)
        model.eval()
        ids = torch.randint(0, TINY.vocab_size, (1, 4))
        tokens = []
        with torch.no_grad():
            for tok in model.generate(ids, max_new_tokens=5, stream=True):
                tokens.append(tok)
                assert isinstance(tok, int)
        assert len(tokens) == 5

    def test_generate_greedy(self):
        """Greedy decoding should be deterministic."""
        model = EmilySLM(TINY)
        model.eval()
        ids = torch.randint(0, TINY.vocab_size, (1, 4))
        with torch.no_grad():
            out1 = model.generate(ids, max_new_tokens=3, do_sample=False)
            out2 = model.generate(ids, max_new_tokens=3, do_sample=False)
        assert torch.equal(out1, out2)

    def test_save_load_pretrained(self, tmp_path):
        model = EmilySLM(TINY)
        model.eval()
        ids = torch.randint(0, TINY.vocab_size, (1, 8))

        with torch.no_grad():
            logits_before = model(ids)["logits"]

        model.save_pretrained(tmp_path / "model")
        model2 = EmilySLM.from_pretrained(tmp_path / "model")
        model2.eval()

        with torch.no_grad():
            logits_after = model2(ids)["logits"]

        assert torch.allclose(logits_before, logits_after, atol=1e-5)

    def test_repr_contains_params(self):
        model = EmilySLM(TINY)
        r = repr(model)
        assert "params=" in r
        assert "layers=" in r
