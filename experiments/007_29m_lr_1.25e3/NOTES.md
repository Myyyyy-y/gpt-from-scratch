# 007_29m_lr_1.25e3 — LR sweep：lr=1.25e-3

## Goal
验证参考项目 H 实测的最优 lr（1.25e-3），与 1e-3 互证。

## Setup
- 29M（8/512/8），28500 步，lr 1.25e-3，min_lr 1.25e-4（max/10），bf16

## Results
- best val loss = **1.3964**（@27750），final 1.4016

## Conclusions
与 1e-3（1.3832）同处盆地底部，略差 0.013，方向互证参考项目 H。
注意：本组 min_lr=1.25e-4 与 003 的 3e-5 不同，属小混淆变量。

## Next steps
- 无（冠军仍为 003）
