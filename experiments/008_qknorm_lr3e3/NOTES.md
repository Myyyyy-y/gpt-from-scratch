# 008_qknorm_lr3e3 — QK-Norm 对高 lr 发散的修复

## Goal
验证 QK-Norm（2025 标准组件）能否修复 LR sweep 中发散的 lr=3e-3 组（004）。

## Setup
- 29M（8/512/8），lr 3e-3，min_lr 3e-4，1 epoch，与 004 唯一差异为 qk_norm=True
- 对照：004（发散，best 2.2837）、003 冠军（lr 1e-3，1.3832）

## Results
- best val loss = **1.4020**，全程稳定（gnorm ≤ 0.4，对照组曾达约 50）
- 中期领先冠军（step 11750：1.635 vs 1.684），末端被反超

## Conclusions
1. QK-Norm 完全消除高 lr 发散
2. 最终成绩未超冠军，但存在混淆变量（本组 min_lr=3e-4 vs 冠军 3e-5，退火深度
   差 10 倍），不能将该差异干净归因于 QK-Norm
3. 中期领先表明大 lr + QK-Norm 的收敛速度优势真实存在

## Next steps
- QK-Norm + 冠军配置（lr 1e-3, min_lr 3e-5）的干净归因（012_qknorm_clean）
- QK-Norm + 3e-3 + min_lr 3e-5 探测该技术的性能上限
