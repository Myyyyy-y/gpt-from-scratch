# 002_29m_lr_3e4 — LR sweep：lr=3e-4

## Goal
29M 上 LR sweep 四组之一（唯一变量 max_lr），考察低学习率端的表现。

## Setup
- 29M（8/512/8，RMSNorm + SwiGLU + RoPE），28500 步（1 epoch）
- lr 3e-4，min_lr 3e-5，warmup 200 + cosine，bf16

## Results
- best val loss = **1.4467**（@27750），final 1.4532
- 对照：1e-3 → 1.3832；3e-3 → 发散

## Conclusions
lr 3e-4 偏保守：收敛稳定但欠拟合，比 1e-3 差 0.063。
最优 lr 位于 1e-3 附近的平坦区间。

## Next steps
- 采用冠军配置 003（lr 1e-3）作为后续实验基座
