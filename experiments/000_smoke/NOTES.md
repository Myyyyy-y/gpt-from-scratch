# 000_smoke — 全链路点火测试

## Goal
验证 BPE → memmap 数据 → 手写模型 → 手写训练栈的整条链路能跑通：
loss 能从初始值正常下降、无 NaN、checkpoint/日志落盘正常。

## Setup
- 模型：16M（6 层 / d_model 384 / 6 头，RMSNorm + SwiGLU + RoPE，权重绑定）
- 训练：500 步，bs 64×256，lr 3e-4（warmup 200），bf16
- 小步数设计：只验链路，不追求收敛

## Results
- 初始 loss ≈ ln(8192) ≈ 9，最终 train 3.18 / best val 3.1734
- 无 NaN，日志与 checkpoint 正常

## Conclusions
链路可用，可以进入全量训练（001）。

## Next steps
- 全量 28500 步基线（001_baseline）
