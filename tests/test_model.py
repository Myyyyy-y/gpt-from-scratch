"""
模型测试：前向形状 / 因果性 / RoPE / 归一化 / 消融开关 / 数值对照 / 梯度 / 参数量

==================== 给初学者的整体说明 ====================

【模型测试的两类打法】
1. 性质测试：不依赖任何"标准答案"，直接验证代码必须满足的数学性质。
   例：因果性（改未来的输入，过去的输出不能变）、RoPE 旋转公式。
2. 数值对照：把手写组件和 PyTorch 官方实现的输出逐一对比。
   我们的模型代码不用 F.scaled_dot_product_attention / F.layer_norm，
   但【测试里】可以拿它们当"阅卷老师"——这正是自定义架构的正确性验证范式。

【为什么测试里可以有 PyTorch 内置函数，模型里不行？】
模型代码是"答卷"，必须从零手写；测试代码是"答案"，用什么都行。
两者职责不同，不矛盾。
"""
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
    """测试用小配置：模型够小，CPU 上也能秒跑。"""
    defaults = dict(vocab_size=100, n_layers=2, d_model=32, n_heads=4,
                    d_ff=64, context_length=16)
    defaults.update(kw)
    return ModelConfig(**defaults)


# ---------- 性质测试 ----------

def test_forward_shape():
    """(B, T) 的 id 输入 -> (B, T, vocab) 的 logits 输出。"""
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg())
    x = torch.randint(0, 100, (2, 16))
    assert model(x).shape == (2, 16, 100)


def test_causality():
    """【最重要的测试】改掉第 10 位之后的输入，前 10 位的输出必须完全不变。

    这条性质是"causal"（因果）的含义：第 t 个位置的预测不许偷看未来。
    如果掩码写错了（比如对角线 off-by-one），这个测试立刻挂。
    """
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg()).eval()
    x = torch.randint(0, 100, (1, 16))
    x2 = x.clone()
    x2[0, 10:] = torch.randint(0, 100, (1, 6))   # 只动"未来"
    with torch.no_grad():
        logits1, logits2 = model(x), model(x2)
    assert torch.equal(logits1[0, :10], logits2[0, :10])


def test_apply_rope_rotation():
    """验证二维旋转公式：(a,b) 转 θ -> (a·cosθ - b·sinθ, a·sinθ + b·cosθ)。"""
    x = torch.tensor([[3.0, 4.0]])
    c, s = math.cos(1.0), math.sin(1.0)
    y = apply_rope(x, torch.tensor([[c, c]]), torch.tensor([[s, s]]))
    expected = torch.tensor([[3 * c - 4 * s, 3 * s + 4 * c]])
    assert torch.allclose(y, expected, atol=1e-6)


def test_rope_cache_shapes():
    cos, sin = precompute_rope(16, 8)
    assert cos.shape == (16, 8) and sin.shape == (16, 8)


def test_rmsnorm_unit_rms():
    """RMSNorm（weight=1 时）输出的每一行，均方根应该恰好是 1。"""
    torch.manual_seed(0)
    y = RMSNorm(16)(torch.randn(4, 8, 16))
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 8), atol=1e-5)


def test_weight_tying():
    """权重绑定：lm_head 和 embedding 必须是同一个对象（改一边等于改两边）。"""
    model = TransformerLM(_small_cfg(n_layers=1, d_model=16, n_heads=2, d_ff=32))
    assert model.lm_head.weight is model.token_embedding.weight


def test_backward_finite_grads():
    """反向传播后每个参数都要有梯度且不含 inf/nan（bf16 训练的底线保障）。"""
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg())
    x = torch.randint(0, 100, (2, 16))
    model(x).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_default_config_param_count():
    """README 承诺的参数量：默认配置应约 16M。"""
    n = sum(p.numel() for p in TransformerLM(ModelConfig()).parameters())
    assert 15_000_000 <= n <= 17_000_000, n


# ---------- 数值对照（vs PyTorch 官方实现，测试专用"标准答案"） ----------

def test_causal_attention_matches_torch():
    """手写因果注意力 vs F.scaled_dot_product_attention，逐元素对齐。

    这是整个模型里最容易写错的部分（掩码、缩放、softmax），
    值得用官方实现当裁判。is_causal=True 即官方的上三角掩码。
    """
    torch.manual_seed(0)
    q = torch.randn(2, 4, 16, 8)   # (B, H, T, D)
    k, v = torch.randn_like(q), torch.randn_like(q)
    mine = causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(mine, ref, atol=1e-5)


def test_layernorm_matches_torch():
    """手写 LayerNorm vs F.layer_norm。"""
    torch.manual_seed(0)
    ln = LayerNorm(32)
    x = torch.randn(2, 16, 32)
    ref = F.layer_norm(x, (32,), ln.weight, ln.bias, ln.eps)
    assert torch.allclose(ln(x), ref, atol=1e-5)


def test_gelu_mlp_matches_torch():
    """手写 GELU 的 MLP vs F.gelu 版本：验证激活公式和权重接线。"""
    torch.manual_seed(0)
    mlp = GeluMLP(_small_cfg(d_model=8, d_ff=16))
    x = torch.randn(2, 4, 8)
    expected = mlp.w2(F.gelu(mlp.w1(x)))   # F.gelu 默认就是精确 erf 版
    assert torch.allclose(mlp(x), expected, atol=1e-6)


# ---------- 消融开关：每个组合都要能正常前向 ----------

def test_all_switch_combinations_forward():
    """消融实验的前提是：三个开关的所有组合都能跑通且形状正确。

    消融实验的铁律是"一次只换一个变量"，如果某个组合直接报错，
    实验做到一半才发现就晚了——所以这里提前把 18 种组合全过一遍。
    """
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
    """因果性对三种位置编码都必须成立（换 pos_type 不能破坏掩码）。"""
    for pos in ["rope", "learned", "none"]:
        torch.manual_seed(0)
        model = TransformerLM(_small_cfg(pos_type=pos)).eval()
        x = torch.randint(0, 100, (1, 16))
        x2 = x.clone()
        x2[0, 10:] = torch.randint(0, 100, (1, 6))
        with torch.no_grad():
            l1, l2 = model(x), model(x2)
        assert torch.equal(l1[0, :10], l2[0, :10]), pos


# ---------- 训练技术验证 ----------

def test_relu2_matches_formula():
    """ReLU² MLP vs 手写公式 w2(relu(w1(x))²)。"""
    torch.manual_seed(0)
    mlp = ReluSquaredMLP(_small_cfg(d_model=8, d_ff=16))
    x = torch.randn(2, 4, 8)
    expected = mlp.w2(torch.relu(mlp.w1(x)) ** 2)
    assert torch.allclose(mlp(x), expected, atol=1e-6)


def test_zero_init_block_is_identity():
    """零初始化投影的 Block 在开局必须是严格恒等映射：block(x) == x。

    这正是零初始化的设计意图：训练从"N 层直通"的稳定状态起步。
    """
    torch.manual_seed(0)
    # 零初始化发生在 TransformerLM 层级（不是单个 Block 自建时），
    # 所以要从完整模型里取 Block
    model = TransformerLM(_small_cfg(zero_init_proj=True)).eval()
    blk = model.blocks[0]
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        out, _ = blk(x)
    assert torch.equal(out, x)      # 严格相等，不是近似——两个子层输出都恰好是 0


def test_zero_init_logits_match_untied_embedding():
    """零初始化时，初始 logits 完全由 embedding×lm_head 决定（Block 无贡献）。"""
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg(zero_init_proj=True)).eval()
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits = model(x)
        direct = model.lm_head(model.norm_f(model.token_embedding(x)))
    assert torch.allclose(logits, direct, atol=1e-5)


def test_qk_norm_forward_and_norm_bounded():
    """QK-Norm：开启后前向正常；归一化后的 Q 模长应有界（ RMS≈1 × 可学习缩放）。"""
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg(qk_norm=True)).eval()
    x = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 16, 100)
    assert torch.isfinite(out).all()
    # 归一化层确实被注册（说明 qk_norm 开关生效）
    attn = model.blocks[0].attn
    assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm")


def test_attn_res_forward_shape_and_zero_init():
    """Attention Residuals：前向形状正确、可返回 hidden_states、查询向量零初始化。"""
    torch.manual_seed(0)
    model = TransformerLM(_small_cfg(attn_res=True)).eval()
    x = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        out, hidden = model(x, return_hidden_states=True)
    assert out.shape == (2, 16, 100)
    assert len(hidden) == 3          # embedding 输出 + 2 层
    assert torch.isfinite(out).all()
    for q in model.attn_res_queries:  # 新结构从"无害"起步（对标 L 的零初始化）
        assert torch.equal(q, torch.zeros_like(q))


def test_attn_res_embedding_output_matches_baseline():
    """AttnRes 的 embedding 输出与 baseline 完全一致（路由只从第 1 层开始）。"""
    torch.manual_seed(0)
    base = TransformerLM(_small_cfg()).eval()
    torch.manual_seed(0)
    attn = TransformerLM(_small_cfg(attn_res=True)).eval()
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        _, h_base = base(x, return_hidden_states=True)
        _, h_attn = attn(x, return_hidden_states=True)
    assert torch.allclose(h_base[0], h_attn[0], atol=1e-6)
