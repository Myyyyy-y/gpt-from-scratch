# GPT from Scratch

从零实现的一个 decoder-only 小语言模型（LLaMA 式架构：RMSNorm + SwiGLU）：自己训练 **BPE 分词器**、自己写**数据管线**、自己实现**多头因果注意力 + 训练循环**，在**全量 TinyStories**（~5 亿 token）上训练到验证集 loss ≈ 1.4x（16M 参数）。

**不依赖** `nn.Transformer`、`tiktoken`、`transformers.Trainer` —— 模型、分词器、训练全部从零手写。

> 状态：训练技术验证进行中——QK-Norm / Muon v2 / ReLU² / 2-epoch /
> 技术组合与单项消融已完成；归一化 / 数据量 / AttnRes / 多 seed 在 GPU
> 队列中（详见 `docs/训练技术验证报告.md`）。

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
| n_layer / n_head / n_embd | 6 / 6 / 384（baseline）；**8 / 8 / 512（29M 主力，消融后选定）** |
| d_ff（SwiGLU 隐层） | 1344 |
| block_size | 256 |
| 参数量 | ~16M（baseline）/ ~29M（主力）（token embedding 与输出头权重共享）|
| 训练数据 | 全量 TinyStories train（~5 亿 token，Chinchilla 比例充足）|
| optimizer | AdamW：lr 3e-4，warmup 200，cosine 退火到 3e-5 |
| weight_decay / grad_clip | 0.1 / 1.0 |
| dtype | bf16 |

## 结果

| 指标 | 数值 |
|---|---|
| 验证集 loss | 16M baseline：1.5317；**29M + 最优 lr：1.3832**（精确复测 1.3979±0.005）|
| 训练吞吐（token/s） | 16M：~207,000；29M：~125,000（单卡 RTX 4090，bf16）|
| 采样（无 KV cache，token/s） | 16M：78.0 GPU / 70.5 CPU；29M：118.1 GPU / 36.3 CPU |
| 采样（KV cache，token/s） | 16M：81.5 GPU（1.04×）/ 115.0 CPU（1.63×）；29M：125.7 GPU（1.06×）/ 63.9 CPU（1.76×）|

> KV cache 加速比随模型规模上升（16M→29M：GPU 1.04→1.06×，CPU 1.63→1.76×），
> 验证了我们的分析：模型越大，计算在总耗时中占比越高，cache 收益越大。
> 未达 2.5× 的原因是本实验模型规模偏小（参考实现 2.49× 来自更大模型）。
> GPU 上的加速比受 kernel 启动开销主导，小模型推理的瓶颈不在计算。
> 正确性由逐 token 一致性测试保证。

**示例 3**（29M 最优模型，lr 1e-3 组 best.pt）：

> Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toys, but he didn't like to clean them up. His mommy would always tell him to do it, but he never listened... He learned that sometimes it's important to do things that are hard to do, even if you don't like it. From that day on, Timmy promised to always listen to his mommy and do what she asked. **<|endoftext|>**（完整收尾，16M 版故事会中途截断）

训练曲线与实验图（`assets/`）：

| 图 | 内容 |
|---|---|
| ![16M baseline](assets/loss_curve.png) | 16M baseline 训练全程 |
| ![LR sweep](assets/lr_sweep.png) | LR 四组对比（含 3e-3 发散曲线） |
| ![gnorm](assets/gnorm_divergence.png) | 发散的梯度范数尖峰（最高 ~22000）vs QK-Norm 组平稳直线 |
| ![位置编码](assets/pos_ablation.png) | 位置编码三组对比 |
| ![规模对比](assets/scale_16m_vs_29m.png) | 16M vs 29M（29M 全程领先） |
| ![冠军组](assets/champion_full.png) | 冠军组 train/val 双曲线 |
| ![lr 调度](assets/lr_schedule.png) | warmup + cosine 调度真实形状 |
| ![ReLU² vs SwiGLU](assets/ffn_relu2.png) | FFN 激活对比：ReLU² vs SwiGLU 冠军 |
| ![Muon vs AdamW](assets/muon_vs_adamw.png) | Muon v2 vs AdamW 冠军对比 |
| ![2-epoch vs 1-epoch](assets/2epoch_vs_1epoch.png) | 2-epoch 重训 vs 冠军 1-epoch |

### 位置编码消融（29M + lr 1e-3，唯一变量 pos_type）

| 位置编码 | 验证集 loss |
|---|---|
| **RoPE（baseline）** | **1.3832** |
| learned | 1.4071 |
| none | 1.4259 |

> 结论：RoPE 最优，复现教科书预期；但 learned ≈ none（差距仅 0.019）——
> 在 256 短上下文 + 短故事的设置下，可学习绝对位置几乎没提供价值，
> 而 RoPE 的相对位置编码带来稳定收益（-0.04）。
> none 组完全无位置信息仍达 1.43：因果掩码本身隐式泄露了顺序线索。
> 曲线对比见 `assets/pos_ablation.png`。

### LR 消融（29M，其他条件全同）

| lr | 验证集 loss | 备注 |
|---|---|---|
| 3e-4 | 1.4467 | 偏保守，欠拟合 |
| **1e-3** | **1.3832**（精确复测 1.3979±0.005）| **最优** |
| 1.25e-3（参考项目最优值） | 1.3964 | 与 1e-3 同处"盆地"底部 |
| 3e-3 | 发散（best 2.28 → 反弹 3.3，gnorm 飙至 ~50） | 过高不可用 |

> 注：训练日志的 val loss 是 bf16 + 20 batch 的快速估计；括号内为
> fp32 + 100 batch 的精确复测（eval.py）。两个参考项目的最优 lr
> 分别为 1.25e-3（H，sweep 实测）和 1e-3（L），本实验结果与之互证。

### 选做消融

| 变量 | 对照组 | 验证集 loss |
|---|---|---|
| 归一化 | RMSNorm / LayerNorm | LayerNorm best **1.3817**（@27750），与 RMSNorm 1.3832 基本持平 |
| FFN | SwiGLU / ReLU²（GELU 未做） | ReLU² 1.4083，见 `docs/训练技术验证报告.md` |
| 数据量 | 全量 ~5 亿 / 300 万 token | 3M token 严重过拟合：best **2.31**@750 → final valid 4.72 |

### 训练技术验证（详见 `docs/训练技术验证报告.md`）

| 技术 | 实验 | 结果 |
|---|---|---|
| **QK-Norm** | lr 3e-3 发散 → +QK-Norm | **发散治愈：2.28(反弹3.9) → 1.4020** |
| **QK-Norm 干净归因** | 冠军配置（lr 1e-3 / min_lr 3e-5）± QK-Norm | **1.3746 < 冠军 1.3832**（-0.0086，真实收益） |
| Muon（vs 手写 AdamW） | v2 补 decoupled weight decay | v1 复现发散（gnorm~194）；v2 修复后 **1.3716 < 冠军 1.3832** |
| ReLU² | vs SwiGLU | 全程贴合，最终 1.4083（落后 0.025，省约 1/3 FFN 计算） |
| 输出投影零初始化 | 冠军配置 + zero_init_proj | 1.3933（无增益，不建议默认开） |
| 解开权重绑定（untie） | 冠军配置 − tie | 1.3823 ≈ 冠军（+4.2M 参数，无收益） |
| 技术组合（全叠加） | QK+Muon+ReLU²+untied+zero-init | 1.3900（无协同增益，各技术需独立调优） |
| 2-epoch 重训 | 最优配置 ×2 | best **1.3212**（+0.062）；final 1.3631 回摆 |
| 多 seed 显著性 | 冠军组 seed 1/2/3 | GPU 队列中（seed1/2 已完成：1.3668 / 1.3761） |

## 采样示例

**示例 1**（prompt: "Once upon a time"，temperature=0.8, top_k=50, top_p=0.95）：

> Once upon a time there was a little girl named Alice. She was only three years old but she was very curious. One day Alice decided to explore her garden. She took a stick and started to poke around the plants... Finally, they found a little patch of colorful flowers that looked like butterflies. Alice was delighted and said, "This is the best garden ever!"

**示例 2**（同参数）：

> Later that day, Timmy's dad came home from work and asked him to help deliver a package to his grandma... Timmy learned that it's important to be careful and not put [it in his mouth]

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
python -m src.sample --ckpt experiments/003_29m_lr_1e3/best.pt --benchmark

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
