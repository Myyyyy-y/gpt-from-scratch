"""
GPT from Scratch：从零实现的 decoder-only 小语言模型

==================== 给初学者的整体说明 ====================

【这个项目要干什么？】
不调用任何现成大模型库，只用 PyTorch 的基础算子，从零写一个 GPT：
训练它读一堆文本，然后能自己"接龙"生成新文本。

【一条数据的完整旅程】
  原始文本
    │  tokenizer.py   文字 <-> 数字 id（BPE 分词）
    ▼
  id 序列（存成二进制文件）
    │  data.py        随机切出固定长度的 chunk，组装成 batch
    ▼
  (B, T) 的整数张量
    │  model.py       GPT 模型：每个位置预测下一个 token
    ▼
  logits (B, T, vocab_size)
    │  train.py       算 loss、反向传播、更新参数，循环 N 轮
    ▼
  checkpoints/ 里的模型权重
    │  sample.py      给个开头，让模型一个字一个字往下生成
    ▼
  生成的文本
    │  eval.py        用 val loss 客观衡量模型学得好不好

【各模块职责一览】
  tokenizer.py   字节级 BPE 分词器（已完成，含测试）
  model.py       GPT 模型本体：LayerNorm / 注意力 / MLP / Block
  data.py        数据管线：随机 chunk 采样 + batch 组装
  train.py       训练循环：AdamW / warmup / cosine / 混合精度
  sample.py      文本生成：temperature / top-k / top-p / KV cache
  eval.py        评估：验证集 loss、生成速度对比

【阅读顺序建议】
tokenizer.py -> model.py -> data.py -> train.py -> sample.py -> eval.py
每个文件开头都有本文件的整体说明，代码里的关键步骤都有逐行注释。

依赖：torch、numpy、regex、tqdm（见 requirements.txt）
"""
