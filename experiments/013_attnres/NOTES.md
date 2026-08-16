# 013_attnres — Attention Residuals 深度残差路由训练

## Goal
训练 AttnRes（Kimi 2024 深度残差路由，对标参考项目 L 的 AttnRes-lite），
检验深层幅值控制在小模型上是否带来训练收益；配合幅值分析图评估内部表示。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：--attn_res（每层可学习 query 零初始化 + 跨层 softmax 路由）
- 注意：attn_res 模式不支持 KV cache（decode 阶段历史层输出无法增量缓存）

## Results
- best val loss = **1.3793**（@27750），final valid 1.3859 / train 1.3638
- 对照：冠军组 1.3832

## Conclusions
与冠军组（1.3832）基本持平（best 1.3793，差 -0.0039，噪声范围内）。
深层幅值受控未带来显著 loss 收益，也未付出代价；
幅值对比图见 assets/magnitude_comparison.png。
