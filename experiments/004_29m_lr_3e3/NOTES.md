# 004_29m_lr_3e3 — LR sweep：lr=3e-3（发散组）

## Goal
29M 上 LR sweep 四组之一（唯一变量 max_lr），验证高 lr 上限。

## Setup
- 29M（8/512/8），28500 步，lr 3e-3，min_lr 3e-5，bf16

## Results
- best val loss = **2.2837**（@1750，过早）
- val 中段反弹至 ~3.9，gnorm 飙至 ~50，final 2.5847

## Conclusions
lr 3e-3 训练发散、不可用；完整留档作为 QK-Norm 拯救实验（008）的对照。

## Next steps
- 008：QK-norm 拯救高 lr 发散
