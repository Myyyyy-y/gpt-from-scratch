# 013_layernorm — 归一化消融：LayerNorm vs RMSNorm

## Goal
补"选做消融"表里归一化一行：同配置下 LayerNorm vs RMSNorm 冠军组，
检验 pre-norm 结构下归一化选型的影响。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：norm_type=layernorm

## Results
- best val loss = **1.3817**（@27750），final valid 1.3879 / train 1.3609
- 对照：RMSNorm 冠军组 1.3832

## Conclusions
最终 LayerNorm（1.3817）与 RMSNorm（1.3832）基本持平（差 ~0.002，噪声范围内），
推翻了早前基于 step 5250 中间快照的"明显落后"初判——中间态不具外推性。
pre-norm 结构下归一化选型对 29M 规模影响可忽略；RMSNorm 计算更省，仍为默认选择。
