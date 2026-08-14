"""
采样：top-k / top-p / temperature，支持 KV cache 增量 decode

==================== 给初学者的整体说明 ====================

【采样在干什么？】
训练好的模型每次前向输出的是"下一个 token 的概率分布"。
采样（sampling/decoding）就是反复做三件事：
  1. 算出下一个 token 的概率分布
  2. 从中挑一个 token（接在序列末尾）
  3. 回到第 1 步，直到生成够长度或遇到 <|endoftext|>
生成是一个字一个字"自回归"滚出来的，不是一口气吐出来的。

【为什么不直接挑概率最大的（贪心）？】
贪心（argmax）每条路都只走概率最高的分叉，生成结果死板、爱复读
（"很好很好很好……"）。引入一点随机性，文本才像人写的。
下面三个旋钮控制"随机多少"：

【三个关键旋钮】

1. temperature（温度）
   采样前把 logits 除以 T 再 softmax：
   - T < 1（如 0.7）：分布变尖锐，高概率 token 更突出 -> 保守、稳定
   - T = 1：原样采样
   - T > 1（如 1.3）：分布变平坦，低概率 token 也有机会 -> 放飞、易胡言
   T -> 0 时退化为贪心。

2. top-k
   只保留概率最高的 k 个候选，其余概率清零后再归一化采样。
   砍掉长尾里的荒谬选项（几万个 token 里大部分都是离谱的）。
   k=50 是常用值；k=1 等价于贪心。

3. top-p（nucleus sampling）
   按概率从高到低累加，砍到累计概率刚好超过 p（如 0.9）为止，
   只在这"核心集合"里采样。比 top-k 灵活：分布尖锐时集合自动变小
   （只有 2~3 个候选），分布平坦时自动变大。
   top-k 和 top-p 可以同时用（先 k 后 p）。

【KV cache：让生成快 N 倍的关键优化】
朴素做法：每生成一个新 token，都把整个序列重新过一遍模型，
序列越长越慢，总复杂度 O(L²)。
但注意力的因果掩码保证了：旧 token 的 Key/Value 不会因为
新 token 的加入而改变。所以可以把每层的 K、V 缓存下来，
新 token 只算自己的 Q 去和缓存的 K、V 做注意力，
每步只处理 1 个 token 而不是 L 个，总复杂度降到 O(L)。
实现上需要模型 forward 支持传入 past_key_values 参数。
本文件会用"开/关 KV cache 各生成一遍、对比耗时"来直观展示收益，
最终输出必须逐 token 完全一致（cache 只是省计算，不改变结果）。

【本文件的实现顺序】
  1. sample_next(logits, temperature, top_k, top_p)：分布 -> 一个 id
  2. generate(prompt, max_new_tokens, ...)：自回归主循环
     - prompt 先 encode，逐 token 生成，遇 <|endoftext|> 提前停
  3. KV cache 版本 + 与朴素版本的耗时/一致性对比

依赖：torch、tokenizer.py 训练好的分词器、checkpoints/ 里的模型权重
"""

# TODO: 实现 generate
