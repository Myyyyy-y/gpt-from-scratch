# 012_untied — 解开 embedding / lm_head 权重绑定

## Goal
默认 lm_head.weight = token_embedding.weight（GPT-2 式共享，省 ~3.1M 参数）；
--untie 后两者独立（LLaMA 式），检验输出头独立容量在小模型上是否值得。

## Setup
- 29M 同规模（8/512/8，swiglu），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：tie_weights=False（untied）
- 实际参数 33.3M（+4.2M 相对绑定版）

## Results
- best val loss = **1.3823**（@27750），final train 1.3582
- 对照：冠军组（003）1.3832

## Conclusions
解开绑定 **基本持平**（1.3823 vs 1.3832，差距 0.001 在噪声内），
但多 4.2M 参数。29M 小模型上共享权重是更优选择（省参数且不损效果）。

## Next steps
- 结论已收敛：默认保持 tie_weights=True
