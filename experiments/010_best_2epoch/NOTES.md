# 010_best_2epoch — 2-epoch 重训

## Goal
最优配置（29M + lr 1e-3）跑 2 epochs（57000 步），检验数据重复利用的
边际收益，目标冲击 1.3x。

## Setup
- 29M（8/512/8，AdamW），57000 步 = 2 epochs，lr 1e-3，min_lr 3e-5，bf16

## Results
- best val loss = **1.3212**（@56000，较 1-epoch 冠军 1.3832 提升 0.062）
- final val loss = 1.3631（回摆 0.042，第二圈末期过拟合）

## Conclusions
数据翻倍带来约 0.06 的实打实提升（全项目最低 loss）；
best 在 56000 步而非终点，早停可锁定最优。

## Next steps
- 早停版 2-epoch（56000 步截断）
- 技术组合实验
