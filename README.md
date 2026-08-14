# GPT from Scratch

从零实现的一个 decoder-only 小语言模型（LLaMA 式架构：RMSNorm + SwiGLU）：自己训练 **BPE 分词器**、自己写**数据管线**、自己实现**多头因果注意力 + 训练循环**，在**全量 TinyStories**（~5 亿 token）上训练到验证集 loss ≈ 1.4x（16M 参数）。

**不依赖** `nn.Transformer`、`tiktoken`、`transformers.Trainer` —— 模型、分词器、训练全部从零手写。

> 状态：数据管线已完成，模型/训练实现中，结果表将在跑完后更新。

## 亮点

- **全链路从零实现**：BPE 分词器 → 数据管线 → 模型 → 训练 → 采样，无黑盒组件
- **组件全可切换**：归一化（RMSNorm / LayerNorm）、FFN（SwiGLU / GELU）、位置编码（RoPE / learned / none）都是配置开关，为消融实验设计
- **多组对比消融**：位置编码 ×3、学习率 ×3，选做归一化 / FFN / 数据量消融
- **KV cache 推理加速**：增量 decode，测速对比见下方表格
- **工程完整**：pytest 单元测试、jsonl 实验日志、checkpoint 断点续训、bf16 混合精度
- **实验纪律**：每个实验独立目录（config + 日志 + 结论），硬件与预处理耗时全部记录

## 模型配置

| 参数 | 值 |
|---|---|
| 架构 | LLaMA 式：RMSNorm + SwiGLU + RoPE（pre-norm，无 bias）|
| vocab_size | 8192（BPE 在 TinyStories 训练集上训练）|
| n_layer / n_head / n_embd | 6 / 6 / 384 |
| d_ff（SwiGLU 隐层） | 1344 |
| block_size | 256 |
| 参数量 | ~16M（token embedding 与输出头权重共享）|
| 训练数据 | 全量 TinyStories train（~5 亿 token，Chinchilla 比例充足）|
| optimizer | AdamW：lr 3e-4，warmup 200，cosine 退火到 3e-5 |
| weight_decay / grad_clip | 0.1 / 1.0 |
| dtype | bf16 |

## 结果

| 指标 | 数值 |
|---|---|
| 验证集 loss | 待填（目标 ≤ 1.50）|
| 训练吞吐（token/s） | 待填 |
| 采样（无 KV cache，token/s） | 待填 |
| 采样（KV cache，token/s） | 待填（目标 ~2.5×）|

训练曲线：`assets/loss_curve.png`

### 位置编码消融

| 位置编码 | 验证集 loss |
|---|---|
| RoPE（baseline）| 待填 |
| learned | 待填 |
| none | 待填 |

### LR 消融

| lr | 验证集 loss |
|---|---|
| 1e-4 | 待填 |
| 3e-4 | 待填 |
| 1e-3 | 待填 |

### 选做消融

| 变量 | 对照组 | 验证集 loss |
|---|---|---|
| 归一化 | RMSNorm / LayerNorm | 待填 |
| FFN | SwiGLU / GELU | 待填 |
| 数据量 | 全量 ~5 亿 / 300 万 token | 待填 |

## 采样示例

（训练完成后填入 2–3 个生成的小故事）

## 复现步骤

```bash
conda activate <你的环境>        # 需 torch>=2.1, numpy, datasets, matplotlib, pytest, regex
pip install -r requirements.txt

# 1. 数据：下载全量 TinyStories → 训练 BPE → tokenize → 缓存 .bin
#    约需 2GB 磁盘 + 1~2 小时编码（多进程），可用 --hf_cache 指定缓存位置
python scripts/prepare_data.py --vocab_size 8192 --train_token_budget 600000000 --workers 32

# 2. 训练（可用 CUDA_VISIBLE_DEVICES 指定显卡）
bash scripts/run_train.sh

# 3. 采样
python -m src.sample --ckpt checkpoints/best.pt

# 4. 单元测试
pytest tests/ -v
```

## 目录结构

```
gpt-from-scratch/
├── src/
│   ├── tokenizer.py    # BPE：train / encode / decode
│   ├── data.py         # 随机 chunk 采样 + batch 组装
│   ├── model.py        # LayerNorm / GELU / MLP / 因果注意力 / Block / GPT
│   ├── train.py        # 训练循环（AdamW / warmup / cosine / clip / bf16）
│   ├── sample.py       # top-k / top-p / temperature 采样
│   └── eval.py         # val loss / KV cache 测速
├── scripts/            # prepare_data.py / run_train.sh / smoke_test.py
├── tests/              # test_tokenizer.py / test_model.py
├── experiments/        # 001_baseline/ 002_rope/ 003_lr_sweep/...
├── data/  checkpoints/ # 数据与权重（git 忽略）
└── assets/             # loss_curve.png 等
```

## 参考与致谢

- Karpathy: [nanoGPT](https://github.com/karpathy/nanoGPT) / [minbpe](https://github.com/karpathy/minbpe)
- Stanford [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [动手学深度学习 d2l.ai](https://zh.d2l.ai/)
- 参考项目：[LitzGymrat/CS336-LLM_from_scratch-assginment1](https://github.com/LitzGymrat/CS336-LLM_from_scratch-assginment1)、[Hurricane0698/TransformerLM-from-scratch](https://github.com/Hurricane0698/TransformerLM-from-scratch)
