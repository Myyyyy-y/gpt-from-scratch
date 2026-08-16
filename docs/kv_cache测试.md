# KV cache 测试报告

> 对标参考项目 LitzGymrat 的《kv cache测试.md》，记录本项目的实测数据与分析。

## 测试设置

- 模型：29M（8 层 / d_model 512 / 8 头），checkpoint = LR sweep 最优组 best.pt
- 同一 prompt（"Once upon a time"）、同一生成长度（200 tokens）、同一随机种子
- 两版实现：无 cache（每步整段重算）vs 有 cache（prefill + 增量 decode）
- warmup 各跑一遍不计时；CUDA 计时带 `torch.cuda.synchronize()`

## 正确性验证

贪心解码（temperature=0）下，有/无 cache 输出逐 token 完全一致
（tests/test_sample.py::test_generate_cache_matches_nocache）。
cache 仅消除重复计算，不改变结果。

## 实测数据

### 16M 模型（6 层 / 384 维）

| 环境 | 无 cache | 有 cache | 加速比 |
|---|---|---|---|
| RTX 4090 / bf16 | 78.0 tok/s | 81.5 tok/s | 1.04× |
| CPU / fp32 | 70.5 tok/s | 115.0 tok/s | **1.63×** |

### 29M 模型（8 层 / 512 维）

| 环境 | 无 cache | 有 cache | 加速比 |
|---|---|---|---|
| CPU / fp32 | 36.3 tok/s | 63.9 tok/s | **1.76×** |
| RTX 4090 / bf16 | 52.29 tok/s | 54.79 tok/s | **1.05×**（140 tokens）|

## 分析：加速比为何低于参考项目的 2.49×

1. **GPU 上加速有限的原因**：16M 模型 256 token 的前向计算仅约 0.1ms，而每步
   数百次 CUDA kernel 启动的开销约 12ms——耗时由启动开销主导，cache 节省的
   计算占比过小。
2. **CPU 上加速明显**：CPU 不存在 kernel 启动开销，计算为真正瓶颈，cache 消除
   的重复计算直接转化为加速（1.63×）。
3. **加速比随模型规模上升**：16M（1.63×）→ 29M（1.76×）。模型越大，计算占比
   越高，cache 收益越大，与理论预期一致，也解释了参考项目（模型更大）测得
   2.49× 的原因。

## 结论

KV cache 实现正确、方向有效。在本项目模型规模（16M~29M）与序列长度（≤256）下，
收益被固定开销稀释；该优化对大模型 / 长序列为关键优化（计算量由 O(N²) 降为
O(N)）。本实验定量展示了"优化收益取决于瓶颈位置"这一系统原理。
