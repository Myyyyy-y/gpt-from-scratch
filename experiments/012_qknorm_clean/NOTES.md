# 012_qknorm_clean — QK-Norm 干净归因（最优配置）

## Goal
消除文档遗留的混淆变量：008 的 QK 组使用 min_lr=3e-4，最优组为 3e-5，
"QK 未超最优组"无法干净归因。本组使用与最优组完全相同的配置（lr 1e-3 / min_lr 3e-5），
唯一差异为 qk_norm=True，回答"QK-Norm 在最优 lr 下是否有收益"。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，min_lr 3e-5，bf16
- 与最优组唯一差异：qk_norm=True

## Results
- best val loss = **1.3746**（@27750），final train 1.3489
- 对照：最优组 1.3832、008 的 QK 组（min_lr 3e-4）1.4020、Muon v2 1.3716

## Conclusions
**QK-Norm 在最优配置下确有真实收益**：1.3746 < 1.3832（-0.0086），且训练全程
稳定（gnorm ≤ 0.4）。文档中"QK 未超最优组"的旧结论系 min_lr 混淆所致，干净归因
后确认 QK-Norm 稳定且有效。仍略逊 Muon v2（1.3716），但两者机制正交，可分别使用。

## Next steps
- 与 Muon 的叠加已由 012_combo 测定（组合收益不叠加，见该 NOTES）
