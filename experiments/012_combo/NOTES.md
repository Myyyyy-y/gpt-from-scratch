# 012_combo — 技术组合：QK-Norm + Muon + ReLU² + untied + zero-init

## Goal
验证分项技术全叠加的协同效应：QK-Norm + Muon + ReLU² + untied + zero-init。
本组为组合实验，不做单一变量归因，仅回答"全叠加是否带来额外收益"。

## Setup
- 29M 规模，28500 步，max_lr 1e-3 / min_lr 3e-5，Muon（muon_lr 1e-2），bf16
- ffn_type=relu2（少一个线性层）、qk_norm=True、zero_init_proj=True、tie_weights=False
- 实际参数 27.8M（ReLU² 节省约 5.5M，untied 增加约 3.1M）

## Results
- best val loss = **1.3900**（@27750），final train 1.3661
- 对照：最优组 1.3832、Muon v2 1.3716、QK-Norm 干净归因 1.3746

## Conclusions
全叠加未产生协同增益（1.3900 差于最优组 0.007，更差于 QK-Norm 单项 1.3746）。
单项收益方向（QK-Norm / Muon 各自有效）与组合结果不一致，提示：组合中各技术
的 lr 量纲需重新调整（Muon 侧与 AdamW 侧不一致），各项技术应独立评估其最优超参。

## Next steps
- 若继续组合方向：对 combo 的 lr/muon_lr 做轻量 sweep 后再下结论
