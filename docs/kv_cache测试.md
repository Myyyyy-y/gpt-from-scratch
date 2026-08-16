# KV cache 测试报告

> 对标参考项目 LitzGymrat 的《kv cache测试.md》，记录本项目的实测数据与分析。

## 测试设置

- 模型：29M（8 层 / d_model 512 / 8 头），checkpoint = LR sweep 冠军组 best.pt
- 同一 prompt（"Once upon a time"）、同一生成长度（200 tokens）、同一随机种子
- 两版实现：无 cache（每步整段重算）vs 有 cache（prefill + 增量 decode）
- warmup 各跑一遍不计时；CUDA 计时带 `torch.cuda.synchronize()`

## 正确性验证（先于性能）

贪心解码（temperature=0）下，有/无 cache 输出**逐 token 完全一致**
（tests/test_sample.py::test_generate_cache_matches_nocache）。
cache 只省计算，不改变结果。

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

## 分析：为什么没有达到参考项目的 2.49×

1. **GPU 上几乎无加速的原因**：16M 模型 256 token 的前向计算仅 ~0.1ms，
   而每步几百次 CUDA kernel 启动的开销约 ~12ms——耗时被启动开销主导，
   cache 省掉的计算在总耗时中占比过小。
2. **CPU 上加速明显**：CPU 无 kernel 启动开销问题，计算是真瓶颈，
   cache 消除的重复计算直接体现为加速。
3. **加速比随模型规模上升**：16M（1.63×）→ 29M（1.76×）。
   模型越大，计算占比越高，cache 收益越大——与理论预期一致，
   也是参考项目（更大模型）测得 2.49× 的原因。

## 结论

KV cache 实现正确且方向有效；在本项目的模型规模（16M~29M）和序列长度
（≤256）下，收益被固定开销稀释。该优化对大模型/长序列是刚需（计算量
从 O(N²) 降为 O(N)），本项目实测数据恰好定量展示了"优化收益取决于
瓶颈位置"这一系统原理。
