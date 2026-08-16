# 009_muon_wd — Muon v2：+decoupled weight decay

## Goal
修复 v1 发散：补充 decoupled weight decay，Muon 侧同样采用 warmup/cosine 调度。

## Setup
- 29M（8/512/8），28500 步，lr 1e-3（Hybrid 分工同 v1）
- muon_lr=0.01，weight_decay=0.1

## Results
- best val loss = **1.3716**（@28250），final 1.3756
- 对照：AdamW 最优组（003）1.3832

## Conclusions
修复有效：不仅消除发散，还小幅优于手写 AdamW（-0.0116），验证了
"Muon + weight decay"在现代小模型上的竞争力。

## Next steps
- 技术组合实验（QK-Norm + Muon + ReLU² + untied + zero-init）
