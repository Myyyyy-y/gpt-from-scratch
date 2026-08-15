"""
从零实现的 decoder-only Transformer 语言模型（LLaMA 风格）

==================== 给初学者的整体说明 ====================

【模型在干什么？】
输入一串 token id，输出每个位置上"下一个 token 是什么"的预测分数（logits）：
  (B, T) 的 id  ->  (B, T, vocab_size) 的 logits
训练时拿 logits 和"真实的下一个 token"算交叉熵损失，反向传播更新参数。

【数据在模型里的旅程】
  id (B, T)
    -> Embedding 查表：每个 id 变成一个 d_model 维向量      (B, T, d_model)
    -> [可选] 加上可学习位置向量（pos_type="learned" 时）
    -> 经过 n_layers 个 TransformerBlock（形状不变，信息在层间提炼）
    -> 最后一层归一化
    -> lm_head 线性层：每个位置投影到 vocab_size 维           (B, T, vocab_size)

【一个 Block 里有什么？】（pre-norm 结构，现代标准做法）
  x = x + Attention(Norm(x))     # 注意力子层：token 之间互相"看"，交换信息
  x = x + FFN(Norm(x))           # 前馈子层：每个 token 独立过一个小 MLP
  "残差连接"（x = x + ...）让梯度可以直通底层，是深层网络能训练的关键。
  "pre-norm" 指归一化放在子层【之前】，比放在之后（post-norm）训练更稳定。

【三个可切换开关——为消融实验设计】
  norm_type: "rmsnorm" | "layernorm"   归一化方式
  ffn_type:  "swiglu"  | "gelu"        前馈网络
  pos_type:  "rope" | "learned" | "none"   位置编码
每个开关只换一个组件，其余完全相同——这才满足消融实验"控制变量"的要求。

【"从零"的边界】
只用 nn.Module / nn.Parameter / 基础张量运算；LayerNorm、softmax、注意力
全部手写。不用 nn.Transformer / nn.MultiheadAttention /
F.scaled_dot_product_attention（它只在【测试】里当"标准答案"对照用）。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """模型的全部超参数，集中在一处，消融实验只改这里。

    默认值对应 README 的 baseline：~16M 参数。
      Embedding:        8192 × 384                    ≈ 3.1M（与 lm_head 共享）
      每层注意力:       4 × 384²                       ≈ 0.59M
      每层 SwiGLU:      3 × 384 × 1344                 ≈ 1.55M
      6 层 + 收尾 norm  合计                            ≈ 16M
    """
    vocab_size: int = 8192
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1344            # SwiGLU 隐层维度；若用 gelu MLP 建议改成 4×d_model=1536
    context_length: int = 256
    dropout: float = 0.0        # 小模型小数据，默认不用 dropout
    tie_weights: bool = True    # lm_head 与 token embedding 共享权重（GPT-2 做法，省参数）
    # --- 消融开关 ---
    norm_type: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    ffn_type: str = "swiglu"    # "swiglu" | "gelu"
    pos_type: str = "rope"      # "rope" | "learned" | "none"


# ---------- 归一化 ----------

class RMSNorm(nn.Module):
    """RMSNorm：只除以"均方根"，不减均值（比 LayerNorm 少一步，更快）。

    公式：y = x / sqrt(mean(x²) + eps) * weight
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))   # 可学习的逐维缩放，初始为 1（=不缩放）
        self.eps = eps                                # 防止除以 0 的小常数

    def forward(self, x):
        # 【数值稳定】bf16 半精度下算 mean(x²) 会损失精度（平方和容易溢出/下溢），
        # 所以先升到 float32 算完归一化，再降回原精度。这是参考项目的标准做法。
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        # rsqrt(t) = 1/sqrt(t)，一条指令搞定，比先 sqrt 再除快
        return (x * rms).to(dtype) * self.weight


class LayerNorm(nn.Module):
    """LayerNorm：减均值、除标准差、再缩放平移（GPT-2 用的就是这个）。

    公式：y = (x - mean) / sqrt(var + eps) * weight + bias
    手写而不用 nn.LayerNorm——它只在测试里充当对照的"标准答案"。
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()                                  # 同 RMSNorm，半精度下先升精度
        mean = x.mean(-1, keepdim=True)
        # unbiased=False：方差除以 N 而不是 N-1（深度学习惯例）
        var = x.var(-1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return (y.to(dtype)) * self.weight + self.bias


def make_norm(norm_type, dim):
    """小工厂：按配置名归一化层。消融实验换 norm_type 时只改这里。"""
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return LayerNorm(dim)
    raise ValueError(f"未知 norm_type: {norm_type}")


# ---------- 手写 softmax（注意力的核心零件） ----------

def softmax(x, dim=-1):
    """数值稳定的 softmax，不用 F.softmax。

    陷阱：softmax 里有 exp(x)，x 稍大（如 88）就会溢出成 inf。
    解法：先减去最大值再 exp——softmax(x) == softmax(x - c) 对任意常数 c 成立
    （分子分母同乘 e^{-c} 约掉了），减完后最大输入是 0，exp 结果 ∈ (0, 1]，不会溢出。
    """
    x = x - x.max(dim=dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


# ---------- 旋转位置编码 RoPE ----------

def precompute_rope(context_length, head_dim, base=10000.0, device=None):
    """预计算每个位置的 cos/sin 表，形状都是 (context_length, head_dim)。

    【RoPE 直觉】不给向量"加"位置信息，而是把 q/k 向量按位置"旋转"一个角度：
    位置 t 的向量旋转 t·θ 度。两个位置的向量做点积时，角度差自然体现
    相对距离——注意力因此能感知"隔了多远"，而不是"在第几位"。
    """
    # 每两个维度共用一个频率：第 i 对的角速度 θ_i = base^(-2i/head_dim)
    # 低维转得慢（感知长距离），高维转得快（感知短距离），像钟表的时针秒针
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(context_length, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)           # (T, head_dim/2)：外积 = 每个位置×每个频率
    emb = torch.cat([freqs, freqs], dim=-1)    # (T, head_dim)：复制一份，配合 rotate_half 写法
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """对 q 或 k 施加旋转。x: (..., T, head_dim)，cos/sin: (T, head_dim)。

    二维旋转公式：把向量 (x1, x2) 旋转 θ 后得到
      (x1·cosθ - x2·sinθ,  x1·sinθ + x2·cosθ)
    写成向量形式就是  x·cos + rotate_half(x)·sin，
    其中 rotate_half(x) = (-x2, x1)（前、后半维互换并取负）。
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


# ---------- 手写因果注意力核心 ----------

def causal_attention(q, k, v):
    """缩放点积注意力 + 因果掩码。q/k/v: (B, H, T, D) -> (B, H, T, D)。

    三步：
      1. scores = q @ kᵀ / √D     每个 query 对每个 key 的"相关度"打分
                                  除以 √D 防止分数随维度增大而过大（softmax 会饱和）
      2. 因果掩码：把"未来"位置（上三角）的分数设成 -inf，
         softmax 后这些位置权重=0 —— 每个 token 只能看自己及左边，不许偷看未来
      3. softmax 归一化成权重，加权求和 v
    """
    D = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)          # (B, H, T, T)
    T = q.shape[-2]
    # torch.triu(..., diagonal=1)：取上三角（不含对角线），即"未来"的位置
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ v


# ---------- 因果多头注意力 ----------

class CausalSelfAttention(nn.Module):
    """多头 = 把 d_model 切成 n_heads 份，并行做 n_heads 个小注意力再拼回。

    不同的头可以学不同的关注模式（有的关注语法、有的关注指代……）。
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.use_rope = (config.pos_type == "rope")
        # q/k/v 三个投影合并成一次大矩阵乘（3C 宽），比三次小矩阵乘快
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        if self.use_rope:
            cos, sin = precompute_rope(config.context_length, self.head_dim)
            # persistent=False：cos/sin 是"算出来的常量"，不进 state_dict——
            # 否则 checkpoint 白白多存两份大表，换 context_length 后加载还会报错
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, positions=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)               # 各 (B, T, C)
        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)：把头维度提到前面，每个头独立做注意力
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if positions is None:
                positions = torch.arange(T, device=x.device)
            # positions 查表而不是内部 arange：为以后 KV cache 预留
            # （增量 decode 时新 token 的位置不从 0 开始）
            cos = self.cos[positions].to(dtype=x.dtype)      # (T, D)
            sin = self.sin[positions].to(dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        y = causal_attention(q, k, v)                        # (B, H, T, D)
        # 拼回 (B, T, C)：transpose 后内存不连续，先 contiguous 才能 view
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(self.dropout(y))


# ---------- 前馈网络（两种可切换） ----------

class SwiGLU(nn.Module):
    """SwiGLU（LLaMA 用）：w2( SiLU(w1(x)) * w3(x) )。

    相比普通 MLP 多了一个"门控"分支：w1 支经过 SiLU 激活后，与 w3 支
    逐元素相乘——相当于给信息流通加了一个可学习的"阀门"。
    三个矩阵，所以同等 d_ff 下比 GELU-MLP 多 50% 参数（调 d_ff 来对齐总参数量）。
    """

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)    # gate
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)    # up
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)    # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        # SiLU(x) = x·sigmoid(x)，PyTorch 内置就是这条公式，无黑盒


class GeluMLP(nn.Module):
    """经典 GELU-MLP（GPT-2 用）：w2( GELU(w1(x)) )。两个矩阵。

    GELU 手写精确公式：0.5·x·(1 + erf(x/√2))。
    和 ReLU 的区别：ReLU 在负数区直接砍成 0，GELU 是光滑曲线，
    小负值也能通过一点，实践中 Transformer 用它效果更好。
    """

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        h = self.w1(x)
        h = 0.5 * h * (1.0 + torch.erf(h / math.sqrt(2.0)))   # 手写 GELU
        return self.w2(h)


# ---------- Transformer Block（pre-norm + 残差） ----------

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 注意：两个 norm 是独立的实例，各有各的可学习参数，不能共享
        self.attn_norm = make_norm(config.norm_type, config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = make_norm(config.norm_type, config.d_model)
        self.ffn = SwiGLU(config) if config.ffn_type == "swiglu" else GeluMLP(config)

    def forward(self, x, positions=None):
        x = x + self.attn(self.attn_norm(x), positions=positions)   # 残差 1
        x = x + self.ffn(self.ffn_norm(x))                          # 残差 2
        return x


# ---------- 完整语言模型 ----------

class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # learned 位置编码才需要位置嵌入表；rope/none 不需要
        if config.pos_type == "learned":
            self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_f = make_norm(config.norm_type, config.d_model)   # 收尾归一化
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 【权重初始化】PyTorch 默认初始化（Embedding 甚至是 std=1 的正态）
        # 对这个尺度的模型偏大，训练初期 logits 会爆炸。
        # CS336 推荐：截断正态，Linear 的 std = √(2/(in+out))。
        self.apply(self._init_weights)

        # 权重绑定：lm_head 和 token embedding 共用同一张表。
        # 直觉：embedding 学的是"每个 token 长什么样"，lm_head 问的是
        # "输出像哪个 token"——同一张表两边用，省 3M 参数且效果略好。
        # 注意必须在 _init_weights 之后绑定，否则会被初始化两遍。
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            # 截断到 ±3σ：防止极端初始值
            std = math.sqrt(2.0 / (module.weight.shape[0] + module.weight.shape[1]))
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        elif isinstance(module, nn.Embedding):
            # 【踩坑记录】权重绑定时，embedding 表同时充当 lm_head：
            # logits = x @ E^T，初始 logits 的 std ≈ sqrt(d_model) × std_E。
            # std_E=1.0（CS336 非绑定方案的取值）会让初始 logits std≈20，
            # 初始 loss 高达几百（健康值 ≈ ln(vocab_size)），训练前期全在
            # "收拾残局"。绑定方案必须用 GPT-2 式的 std=0.02：
            # 初始 logits std ≈ sqrt(384)×0.02 ≈ 0.4，初始 loss ≈ ln(8192) ≈ 9。
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def forward(self, idx, positions=None):
        # idx: (B, T) 的 token id
        B, T = idx.shape
        assert T <= self.config.context_length, f"序列长度 {T} 超过 context_length"

        x = self.token_embedding(idx)                    # (B, T, d_model)

        if self.config.pos_type == "learned":
            if positions is None:
                positions = torch.arange(T, device=idx.device)
            x = x + self.position_embedding(positions)   # 加上可学习位置向量
        # pos_type == "rope"：位置信息在注意力内部施加，这里什么都不加
        # pos_type == "none"：完全不给位置信息（消融对照组）

        for block in self.blocks:
            x = block(x, positions=positions)
        x = self.norm_f(x)
        logits = self.lm_head(x)                         # (B, T, vocab_size)
        # 注意：这里【不做】 softmax——交叉熵损失内部会做（数值更稳定），
        # 采样时也只需要 logits。forward 直接返回原始分数是标准做法。
        return logits
