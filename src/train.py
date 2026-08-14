"""
训练循环：AdamW / warmup / cosine / grad clip / bf16 混合精度

==================== 给初学者的整体说明 ====================

【训练循环在干什么？】
一句话：重复几万次同一个动作——
  取一个 batch -> 前向算 loss -> 反向算梯度 -> 优化器更新参数
每一步模型就变好一点点。这个文件负责把"变好一点点"的过程
组织得又快又稳。

【一个标准训练 step 长什么样】
  x, y = get_batch("train")
  logits = model(x)                       # 前向
  loss = cross_entropy(logits, y)         # 和正确答案比，差多少
  loss.backward()                         # 反向传播：每个参数该往哪改
  optimizer.step()                        # 真的去改参数
  optimizer.zero_grad()                   # 清梯度（PyTorch 默认累加！）

【本文件要解决的 5 个工程问题】

1. AdamW 优化器
   SGD 的豪华版：给每个参数单独维护"学习率"，根据历史梯度自动调节。
   AdamW 和 Adam 的唯一区别：权重衰减（weight decay，防止过拟合的
   惩罚项）从梯度里剥离出来单独施加，实践证明这样效果更好。
   注意：LayerNorm 的 gamma/beta 和 bias 不该做 weight decay。

2. 学习率调度：warmup + cosine
   学习率（lr）是"每步走多大"的步长，是最敏感的超参数。
   - warmup：刚开始参数是随机初始化的，梯度又乱又大，
     前几百步让 lr 从 0 线性爬升到峰值，防止一脚踩飞。
   - cosine decay：之后让 lr 按余弦曲线缓慢降到接近 0，
     后期小碎步微调，收敛更精细。

3. 梯度裁剪（grad clip）
   偶尔某个 batch 会产生异常大的梯度（"梯度爆炸"），一步把模型带崩。
   做法：更新前把梯度的总范数（norm）截断到阈值（常用 1.0），
   方向不变、只缩小步长，相当于给训练装了保险丝。

4. bf16 混合精度（AMP）
   默认 float32 训练又慢又占显存。bf16 用一半显存、快约一倍，
   且数值范围大不容易溢出（比 fp16 省心，不需要 loss scaling）。
   用 torch.autocast 包住前向即可，框架自动决定哪些算子用 bf16。

5. 定期存档与验证
   - 每 N step 在 val 集上算一次 loss：train loss 降而 val loss 升
     = 过拟合，该停手或加正则了。
   - 定期把模型权重存到 checkpoints/：训练几小时崩了不至于白跑，
     存"当前最好"的那一份，别存最后一份。

【训练时盯着什么看？】
  loss 曲线：前期应快速下降，后期缓慢下降。不下降 -> 查 lr / 数据；
  剧烈震荡 -> lr 太大；train 降 val 不降 -> 过拟合。

【本文件的实现顺序】
  1. 超参数配置（lr、batch_size、总 step 数等，集中放文件开头）
  2. 学习率调度函数 get_lr(step)
  3. 主循环：get_batch -> forward -> backward -> clip -> step
  4. 定期：估 val loss + 存 checkpoint + 打印进度（tqdm）

依赖：torch、tqdm
"""

# TODO: 实现训练主循环
