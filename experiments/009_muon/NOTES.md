# 009_muon — Muon v1：无 weight decay（发散）

## Goal
按原版 Muon（无 decoupled weight decay）实现并复现其已知问题，
为 v2 修复提供对照。

## Setup
- 29M（8/512/8），28500 步，lr 1e-3（Hybrid：Muon 管隐藏矩阵，AdamW 管词表/norm）
- muon_lr=0.02，无 weight decay

## Results
- best val loss = **2.0638**（@1750，过早），final 3.0322
- ~5000 步后 gnorm 飙至 ~194：权重无约束增长导致发散

## Conclusions
亲手复现了 Moonlight 报告"Muon 需要 weight decay"的原始现象。

## Next steps
- v2（009_muon_wd）：补 decoupled weight decay + 双侧 lr 调度
