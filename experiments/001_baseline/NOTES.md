# 001_baseline — 16M 基线训练

## Goal
验证全链路（BPE → memmap 数据 → 手写模型 → 手写训练栈）在真实数据上端到端
可用，并建立后续所有实验的对照锚点。

## Setup
- 模型：16M（6 层 / d_model 384 / 6 头 / d_ff 1344，RMSNorm + SwiGLU + RoPE，权重绑定）
- 数据：TinyStories 全量 train（4.67 亿 token，自训 BPE vocab 8192）
- 训练：28500 步（1 epoch），bs 64×256，lr 3e-4（warmup 200 + cosine 至 3e-5），bf16
- 硬件：单卡 RTX 4090，37 分钟

## Results
- best val loss = **1.5317**（step 27750；训练内 bf16/20-batch 估计）
- 精确复测（eval.py，fp32/100-batch）：**1.5468 ± 0.0045**
- 吞吐：约 207k tokens/s
- train/val 差距仅 0.04，无过拟合迹象

## Limitations
- lr 3e-4 偏保守（后续 LR sweep 表明 1e-3 更优）
- 曲线末端仍在下降，1 epoch 未充分收敛
- 16M 容量相对 4.67 亿 token 数据量偏小（29 token/参数 > Chinchilla 20）

## Next steps
- 提高 lr 并放大模型，即 002/003/004/007 的 29M LR sweep
