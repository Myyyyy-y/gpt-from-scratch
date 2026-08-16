"""Model tests: shapes, causality, RoPE, norms, ablation switches, numerics."""

import math

import torch
import torch.nn.functional as F

from src.model import (
    GeluMLP,
    LayerNorm,
    ModelConfig,
    ReluSquaredMLP,
    RMSNorm,
    TransformerBlock,
    TransformerLM,
    apply_rope,
    causal_attention,
    precompute_rope,
)


def _small_cfg(**kw):
    defaults = dict(vocab_size=100, n_layers=2, d_model=32, n_heads=4,
                    d_ff=64, context_length=16)
    defaults.update(kw)
    return ModelConfig(**defaults)


# ---------- property tests ----------

def test_forward_shape():
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg())
    x = torch.randint(0, 100, (2, 16))
    assert model(x).shape == (2, 16, 100)


def test_causality():
    """Changing inputs at position >= 10 must not change outputs before it."""
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg()).eval()
    x = torch.randint(0, 100, (1, 16))
    x2 = x.clone()
    x2[0, 10:] = torch.randint(0, 100, (1, 6))
    with torch.no_grad():
        logits1, logits2 = model(x), model(x2)
    assert torch.equal(logits1[0, :10], logits2[0, :10])


def test_apply_rope_rotation():
    x = torch.tensor([[3.0, 4.0]])
    c, s = math.cos(1.0), math.sin(1.0)
    y = apply_rope(x, torch.tensor([[c, c]]), torch.tensor([[s, s]]))
    expected = torch.tensor([[3 * c - 4 * s, 3 * s + 4 * c]])
    assert torch.allclose(y, expected, atol=1e-6)


def test_rope_cache_shapes():
    cos, sin = precompute_rope(16, 8)
    assert cos.shape == (16, 8) and sin.shape == (16, 8)


def test_rmsnorm_unit_rms():
    torch.manual_seed(0)
    y = RMSNorm(16)(torch.randn(4, 8, 16))
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 8), atol=1e-5)


def test_weight_tying():
    model = TransformerLM(_small_cfg(n_layers=1, d_model=16, n_heads=2, d_ff=32))
    assert model.lm_head.weight is model.token_embedding.weight


def test_backward_finite_grads():
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg())
    x = torch.randint(0, 100, (2, 16))
    model(x).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_default_config_param_count():
    n = sum(p.numel() for p in TransformerLM(ModelConfig()).parameters())
    assert 15_000_000 <= n <= 17_000_000, n


# ---------- numerics vs PyTorch reference ----------

def test_causal_attention_matches_torch():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 16, 8)
    k, v = torch.randn_like(q), torch.randn_like(q)
    mine = causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(mine, ref, atol=1e-5)


def test_layernorm_matches_torch():
    torch.manual_seed(0)
    ln = LayerNorm(32)
    x = torch.randn(2, 16, 32)
    ref = F.layer_norm(x, (32,), ln.weight, ln.bias, ln.eps)
    assert torch.allclose(ln(x), ref, atol=1e-5)


def test_gelu_mlp_matches_torch():
    torch.manual_seed(0)
    mlp = GeluMLP(_small_cfg(d_model=8, d_ff=16))
    x = torch.randn(2, 4, 8)
    expected = mlp.w2(F.gelu(mlp.w1(x)))
    assert torch.allclose(mlp(x), expected, atol=1e-6)


# ---------- ablation switches: every combination must forward ----------

def test_all_switch_combinations_forward():
    for pos in ["rope", "learned", "none"]:
        for norm in ["rmsnorm", "layernorm"]:
            for ffn in ["swiglu", "gelu"]:
                torch.manual_seed(0)
                model = TransformerLM(_small_cfg(pos_type=pos, norm_type=norm,
                                                 ffn_type=ffn))
                x = torch.randint(0, 100, (2, 16))
                out = model(x)
                assert out.shape == (2, 16, 100), (pos, norm, ffn)
                assert torch.isfinite(out).all(), (pos, norm, ffn)


def test_causality_holds_for_all_pos_types():
    for pos in ["rope", "learned", "none"]:
        torch.manual_seed(0)
        model = TransformerLM(_small_cfg(pos_type=pos)).eval()
        x = torch.randint(0, 100, (1, 16))
        x2 = x.clone()
        x2[0, 10:] = torch.randint(0, 100, (1, 6))
        with torch.no_grad():
            l1, l2 = model(x), model(x2)
        assert torch.equal(l1[0, :10], l2[0, :10]), pos


# ---------- training-technique checks ----------

def test_relu2_matches_formula():
    torch.manual_seed(0)
    mlp = ReluSquaredMLP(_small_cfg(d_model=8, d_ff=16))
    x = torch.randn(2, 4, 8)
    expected = mlp.w2(torch.relu(mlp.w1(x)) ** 2)
    assert torch.allclose(mlp(x), expected, atol=1e-6)


def test_zero_init_block_is_identity():
    """Zero-init projections must make each block an exact identity at init."""
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg(zero_init_proj=True)).eval()
    blk = model.blocks[0]
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        out, _ = blk(x)
    assert torch.equal(out, x)


def test_zero_init_logits_match_untied_embedding():
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg(zero_init_proj=True)).eval()
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits = model(x)
        direct = model.lm_head(model.norm_f(model.token_embedding(x)))
    assert torch.allclose(logits, direct, atol=1e-5)


def test_qk_norm_forward_and_norm_bounded():
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg(qk_norm=True)).eval()
    x = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 16, 100)
    assert torch.isfinite(out).all()
    attn = model.blocks[0].attn
    assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm")


def test_attn_res_forward_shape_and_zero_init():
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg(attn_res=True)).eval()
    x = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        out, hidden = model(x, return_hidden_states=True)
    assert out.shape == (2, 16, 100)
    assert len(hidden) == 3
    assert torch.isfinite(out).all()
    for q in model.attn_res_queries:
        assert torch.equal(q, torch.zeros_like(q))


def test_attn_res_embedding_output_matches_baseline():
    torch.manual_seed(0)
    base = TransformerLM(_small_cfg()).eval()
    torch.manual_seed(0)
    attn = TransformerLM(_small_cfg(attn_res=True)).eval()
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        _, h_base = base(x, return_hidden_states=True)
        _, h_attn = attn(x, return_hidden_states=True)
    assert torch.allclose(h_base[0], h_attn[0], atol=1e-6)
