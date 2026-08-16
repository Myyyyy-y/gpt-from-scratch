# 005_pos_learned — 位置编码消融：learned

## Goal
量化显式位置信息的价值，并验证 RoPE（2023 主流）相对 learned（GPT-2 经典）
的改进幅度。本组为 learned 组（唯一变量 pos_type）。

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
1. RoPE 最优，与既有研究结论一致（相对位置编码的收益为 -0.04）
2. learned 与 none 差距仅 0.019：在 256 token 短上下文与 TinyStories 短故事
   设置下，可学习绝对位置编码几乎未提供有效增益
3. none 组在无位置信息下仍达 1.4259，说明因果掩码本身携带一定的顺序线索
   （注意力矩阵的下三角结构使不同位置的计算图天然不同）
4. 结论限于短上下文设置；长上下文 / 长程依赖任务上三者差距预计扩大

## Limitations
- 单 seed；rope 与 learned 差距（0.024）大于估计噪声（sem ±0.005），
  learned 与 none 差距（0.019）处于边界
- 未测试长上下文（如 1024）下三者差距的变化

## Next steps
- 可选：长上下文复测；其余消融（归一化 / FFN / QK-Norm / Muon）
