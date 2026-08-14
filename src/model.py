"""
模型：LayerNorm / GELU / MLP / 因果注意力 / Block / GPT

==================== 给初学者的整体说明 ====================

【模型在学什么？】
语言模型只做一件事：给定前面 T 个 token，预测第 T+1 个 token 是什么。
输入是 id 序列 [5, 23, 998, ...]，输出是每个位置上"下一个 token 是谁"
的概率分布（vocab_size 个分数，叫 logits）。训练就是让正确 token 的
概率越来越高。

【数据在模型里的形状变化】（B=batch 大小, T=序列长度, C= embedding 维度）
  ids:        (B, T)            整数 id
    │  token embedding + position embedding
    ▼
  x:          (B, T, C)         每个 token 变成一个 C 维向量
    │  × N 个 Block（Transformer 的基本单元，堆叠 N 层）
    ▼
  x:          (B, T, C)         形状不变，但每个向量已"看过"左边的上下文
    │  最后的 LayerNorm + 线性层（投影回词表）
    ▼
  logits:     (B, T, vocab_size) 每个位置对每个候选 token 的打分

【一个 Block 里有什么？】（Pre-LN 结构，现代 GPT 的标准做法）
  x = x + Attention(LayerNorm(x))    # 子层 1：token 之间交换信息
  x = x + MLP(LayerNorm(x))          # 子层 2：每个 token 独立地"思考"
  两个子层都套着残差连接（x = x + ...）：梯度可以沿"高速公路"直传到底层，
  是几十层网络能训练起来的关键。

【因果注意力（Causal Attention）——GPT 的灵魂】
注意力让每个位置去"参考"其他位置的向量。但 GPT 是生成模型，
第 t 个位置预测第 t+1 个 token 时，绝对不许偷看 t+1 及之后的内容
（否则就是开卷考试，学不到东西）。做法：给注意力分数矩阵加一个
上三角为 -inf 的掩码（mask），softmax 后未来位置的权重变成 0。
这就是 "causal"（因果）/ "decoder-only" 的含义。

多头注意力（Multi-Head）：把 C 维切成 h 份，并行做 h 个小注意力再拼回去。
不同的头可以学不同的关注模式（有的看语法、有的看指代……）。

【LayerNorm 是干什么的？】
把每个 token 的 C 维向量归一化成均值 0、方差 1，再学一个缩放和平移：
  y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
作用：稳住每一层的数值范围，训练不发散。eps 是防止除以 0 的小常数。

【为什么是 GELU 而不是 ReLU？】
MLP 里的激活函数。ReLU 在负数区直接砍成 0（梯度也为 0，神经元可能"死掉"），
GELU 是一条光滑曲线，负数区也保留一点坡度和信息，实践中 Transformer 用
GELU 效果更好。

【本文件的实现顺序】
  1. LayerNorm       归一化（手写，不用 nn.LayerNorm）
  2. MLP             Linear -> GELU -> Linear（维度先放大 4 倍再缩回）
  3. CausalAttention 多头因果注意力（含 -inf 掩码）
  4. Block           LN -> Attn -> 残差 -> LN -> MLP -> 残差
  5. GPT             embedding + N×Block + 输出头 + forward/loss
注意：输出头和 token embedding 共享权重（weight tying），
省参数且效果略好，是 GPT-2 的做法。

依赖：torch
"""

# TODO: 按上面的顺序实现
