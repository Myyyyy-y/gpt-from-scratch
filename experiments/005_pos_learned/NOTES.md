# 位置编码消融（003 vs 005 vs 006）— rope / learned / none

## Goal
量化"显式位置信息"的价值，并验证 RoPE（2023 主流）相对 learned（GPT-2 经典）的改进幅度。

## Setup
- 模型：29M（8 层 / 512 维 / 8 头），lr 1e-3（LR sweep 冠军配置），1 epoch，bs 64×256
- 唯一变量 pos_type：rope（003，复用）/ learned（005）/ none（006）
- 曲线图：assets/pos_ablation.png

## Results
| pos_type | best val loss |
|---|---|
| **rope** | **1.3832** |
| learned | 1.4071 |
| none | 1.4259 |

## Conclusions
1. RoPE 最优，复现教科书预期（相对位置编码的真实收益 -0.04）
2. **learned ≈ none（差距仅 0.019）**——在 256 短上下文 + TinyStories 短故事设置下，
   可学习绝对位置编码几乎没有提供价值，与完全无位置信息打平
3. none 组没有任何位置信息仍达 1.43：因果掩码本身隐式泄露了顺序线索
   （注意力矩阵的下三角结构让不同位置的计算图天然不同）
4. 注意：本结论限于短上下文设置；长上下文/长程依赖任务上三者差距预计会拉大

## Limitations
- 单 seed；rope 与 learned 差距（0.024）大于估计噪声（sem±0.005），
  learned 与 none 差距（0.019）处于边界
- 未测长上下文（如 1024）下三者差距是否扩大

## Next steps
- 选做：长上下文复测；其余消融（归一化 / FFN / QK-norm / Muon 见 LOCAL_PLAN 阶段 6）
