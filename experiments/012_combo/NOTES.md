# 012_combo — 技术组合：QK-Norm + Muon + ReLU² + untied + zero-init

## Goal
验证分项技术全叠加的协同效应：QK-norm + Muon + ReLU² + untied + zero-init
（组合实验，不做"唯一变量"归因，只回答"全上有没有更多收益"）。

## Setup
- 29M 规模，28500 步，max_lr 1e-3 / min_lr 3e-5，Muon（muon_lr 1e-2），bf16
- ffn_type=relu2（少一个线性层）、qk_norm=True、zero_init_proj=True、tie_weights=False
- 实际参数 27.8M（ReLU² 省 ~5.5M，untied 增 ~3.1M）

## Results
- best val loss = **1.3900**（@27750），final train 1.3661
- 对照：冠军组 1.3832、Muon v2 1.3716、QK-norm 干净归因 1.3746

## Conclusions
全叠加 **未产生协同增益**（1.3900 差于冠军 0.007，更差于 QK-norm 单项 1.3746）。
单项收益方向（QK-norm / Muon 各自有效）与组合结果不一致，提示：
组合中各技术的 lr 量纲需要重新调（Muon 侧 lr 与 AdamW 侧不同），
"无脑叠加"不是捷径，每项技术应独立评估其最优超参。

## Next steps
- 若继续组合方向：对 combo 做 lr/muon_lr 的轻量 sweep 再下结论
