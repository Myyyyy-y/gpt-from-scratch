# 013_champion_seed1 — 最优组复跑（seed=1）

## Goal
多 seed 显著性检验的一部分：同一最优配置（29M / lr 1e-3 / 28500 步）更换随机
种子重跑，与 seed=0（003）及其他 seed 共同报告均值 ± std，检验结论的稳定性。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16，seed=1

## Results
- best val loss = **1.3668**（@27500），final valid 1.3985 / train 1.3980

## Conclusions
均值 ± std 汇总见 docs/训练技术验证报告.md 的多 seed 小节。
