"""
评估：val loss / KV cache 测速

==================== 给初学者的整体说明 ====================

【这个文件回答两个问题】
  1. 模型学得到底好不好？  —— 用验证集 loss 客观打分
  2. KV cache 到底快多少？ —— 用同一 prompt 两种生成方式计时对比

【为什么不能用"肉眼看生成文本"当评估？】
生成文本看着通顺，可能只是把训练语料背下来了（过拟合）。
真正的标准是：在模型【从来没见过】的 val 数据上，
预测下一个 token 准不准。这就是 val loss 存在的意义。

【val loss 怎么算？】
和训练时算 loss 一模一样（cross_entropy），但有三点不同：
  1. 数据来自 val.bin（训练从没碰过的那 10%）
  2. 包在 torch.no_grad() 里：不算梯度，省显存也更快
  3. model.eval()：关掉 dropout 等"训练专用"行为，
     保证评估结果确定、可复现
为了数值稳定，多抽几个 batch 取平均（比如 50 个），
单次 batch 的 loss 波动很大，平均后才有比较意义。

【val loss 和 perplexity（困惑度）】
论文里常报 perplexity = exp(loss)。直觉理解：
loss=2.0 -> ppl≈7.4，相当于模型预测下一个词时，
平均在 7~8 个候选里"纠结"。越低越好。
自己训练时看 loss 就够，ppl 只是换了种说法。

【KV cache 测速怎么做才公平？】
  1. 同一个 checkpoint、同一个 prompt、同样的生成长度
  2. 固定随机种子（或干脆断言两种方式的输出逐 token 一致，
     顺带验证了 cache 实现的正确性）
  3. 先各跑一遍"热身"（GPU 第一次调用有初始化开销，不计时）
  4. 用 time.perf_counter() 计时，多轮取平均
典型结果：生成长度越长，cache 版优势越大（朴素版 O(L²) vs 缓存版 O(L)）。

【本文件的实现顺序】
  1. estimate_val_loss(model, n_batches)：val 集上的平均 loss
  2. bench_kv_cache(model, prompt, n_tokens)：开/关 cache 各计时，
     打印加速比，并断言输出一致

依赖：torch、time、data.py 的 get_batch、sample.py 的 generate
"""

# TODO: 实现 estimate_val_loss / bench_kv_cache
