# 013_data_3m — 数据量消融（3M token）

## Goal
只用训练集前 300 万 token 训练（全量约 5 亿），量化"数据量"这一变量的影响，
补 README 选做消融表的数据量行。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：--train_limit 3000000（数据管线 memmap 零拷贝截断）

## Results
- best val loss = **2.3131**（@750），final valid 4.7169 / train 0.0770
- 对照：全量冠军组 1.3832

## Conclusions
数据量从约 5 亿 token 缩减到 300 万（约 1/170）后严重过拟合：
val loss 在 step 750 触底 2.31 后持续恶化到 final valid 4.72，
而 train loss 降到 0.08。
说明 29M 模型在该任务上仍需全量数据，数据量是当前配置的硬瓶颈
（对照全量冠军组 1.3832）。
