# 011_ffn_relu2 — FFN 激活：ReLU²

## Goal
验证 speedrun 社区的 FFN 简化主张：`relu(x)²` 替代 SwiGLU，
效果相当、计算更省（SwiGLU 3 个线性层 → ReLU² 2 个）。

## Setup
- 29M 同规模配置（8/512/8，ffn_type=relu2），28500 步，lr 1e-3，bf16
- 实际参数 23.6M（FFN 少一个线性层，比 SwiGLU 版少约 5.5M）

## Results
- best val loss = **1.4083**（@27750），final 1.4147
- 对照：SwiGLU 冠军组（003）1.3832

## Conclusions
全程贴合 SwiGLU 曲线，最终差距 0.025（SwiGLU 小胜）；
以约 1/3 的 FFN 参数与计算量拿到 98% 效果，性价比取胜。

## Next steps
- 技术组合实验（ReLU² 与 QK-norm/Muon/untied/zero-init 的协同）
