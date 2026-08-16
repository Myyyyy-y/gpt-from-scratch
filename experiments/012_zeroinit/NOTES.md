# 012_zeroinit — 输出投影零初始化（zero-init）

## Goal
把每个 Block 的 out_proj / w2 初始化为 0，训练开局每层 = 恒等映射，
检验"从直通状态起步"在高 lr / 深层下是否有额外收益（对照 003 冠军）。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：zero_init_proj=True

## Results
- best val loss = **1.3933**（@27750），final train 1.3715
- 对照：冠军组（003）1.3832

## Conclusions
零初始化 **无增益**（差于冠军 0.0101）。warmup + RMSNorm 下训练本就稳定，
"恒等起步"没有额外价值，且前期学习更慢（低层从零投影出发）。

## Next steps
- 结论已收敛：29M / 28500 步设置下不建议默认开启
