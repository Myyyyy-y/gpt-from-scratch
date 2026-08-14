"""
训练栈测试：AdamW / 交叉熵 / 梯度裁剪 / LR 调度 / 小语料过拟合

==================== 给初学者的整体说明 ====================

【本文件的验证策略】
1. 数值对照：手写的 AdamW / 交叉熵 和 PyTorch 官方实现逐一比对——
   手写代码是"答卷"，PyTorch 是"阅卷老师"。
2. 性质验证：LR 曲线的关键点数值、梯度裁剪的缩放比例。
3. 端到端冒烟（最重要的一个）：test_overfit_tiny_corpus 让模型在
   100 句重复语料上训练 120 步。如果 分词器→数据→模型→优化器→调度
   这条链路上【任何一环】有 bug，loss 都降不下去。它能降到很低，
   就说明整条链路是通的——这是训练大模型前必须的"点火测试"。
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare_data as pd
from src.tokenizer import BPE
from src.train import AdamW, TrainConfig, clip_grad_norm, cross_entropy, get_lr, train

# 100 句高度重复的文本：模式极少，一个健康的小模型几百步内就该能"背下来"
TEXTS = [f"sentence number {i} with some words to memorize" for i in range(20)] * 5


# ---------- 数值对照：手写 vs PyTorch 官方 ----------

def test_adamw_matches_torch():
    """手写 AdamW 与 torch.optim.AdamW 在相同输入下走 5 步，参数必须一致。

    两个一样的线性层、一样的假数据、一样的超参，各自更新 5 步后
    逐参数比较——手写版哪怕偏差修正或权重衰减写错一点，都会对不上。
    """
    torch.manual_seed(0)
    m1 = torch.nn.Linear(8, 4, bias=False)
    m2 = torch.nn.Linear(8, 4, bias=False)
    m2.load_state_dict(m1.state_dict())          # 保证两个模型初始权重完全相同

    o1 = AdamW(m1.named_parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    x = torch.randn(16, 8)
    y = torch.randn(16, 4)
    for _ in range(5):
        l1 = ((m1(x) - y) ** 2).mean()
        l1.backward()
        o1.step()
        o1.zero_grad()

        l2 = ((m2(x) - y) ** 2).mean()
        l2.backward()
        o2.step()
        o2.zero_grad()

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-5)


def test_cross_entropy_matches_torch():
    """手写交叉熵 vs F.cross_entropy。"""
    torch.manual_seed(0)
    logits = torch.randn(2, 16, 100) * 3         # 乘 3 拉大数值，更考验数值稳定性
    targets = torch.randint(0, 100, (2, 16))
    ref = F.cross_entropy(logits.reshape(-1, 100), targets.reshape(-1))
    assert torch.allclose(cross_entropy(logits, targets), ref, atol=1e-5)


# ---------- 性质验证 ----------

def test_clip_grad_norm():
    """梯度裁剪：超过阈值后总范数应恰好被压到阈值，方向不变。"""
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([3.0, 4.0])            # 范数 = 5.0
    norm = clip_grad_norm([p], max_norm=1.0)
    assert abs(norm - 5.0) < 1e-5                # 返回的是裁剪前的范数
    assert torch.allclose(p.grad, torch.tensor([0.6, 0.8]), atol=1e-5)   # 等比缩到范数 1


def test_clip_grad_norm_noop_when_small():
    """范数没超阈值时，梯度必须原封不动。"""
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([0.3, 0.4])            # 范数 = 0.5 < 1.0
    clip_grad_norm([p], max_norm=1.0)
    assert torch.allclose(p.grad, torch.tensor([0.3, 0.4]))


def test_lr_schedule():
    """LR 曲线的四个关键点：warmup 起点/终点、cosine 起点、退火终点。"""
    args = (5000, 100, 1e-3, 1e-4)               # max_steps, warmup, max_lr, min_lr
    assert get_lr(0, *args) == 1e-5              # warmup 从接近 0 开始爬
    assert get_lr(99, *args) == 1e-3             # warmup 结束恰好到峰值
    assert get_lr(100, *args) == 1e-3            # cosine 起点还是峰值
    assert get_lr(5000, *args) == 1e-4           # 退火终点 = min_lr
    # 中间段单调性：warmup 段递增，cosine 段递减
    assert get_lr(50, *args) < get_lr(99, *args)
    assert get_lr(500, *args) < get_lr(100, *args)
    assert get_lr(4000, *args) < get_lr(500, *args)
    assert 1e-4 < get_lr(2000, *args) < 1e-3


# ---------- 端到端冒烟：整条链路能学习 ----------

def test_overfit_tiny_corpus(tmp_path):
    """小语料过拟合测试：证明 数据->模型->损失->优化器->调度 全链路无 bug。

    【原理】100 句重复文本的模式极少，一个健康的小模型在 120 步内
    就该把 loss 压到很低（接近"背下来"）。如果 loss 降不动，
    说明链路上有 bug——这个测试是训练真实数据前的点火检查。

    【易错点】BPE 词表最小也有 257（256 字节 + 1 特殊 token），
    vocab_size 参数传 100 也拦不住；模型 Embedding 必须按
    【真实词表大小】len(bpe.vocab) 建，否则 id 越界直接崩。
    """
    torch.manual_seed(0)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(TEXTS), encoding="utf-8")
    bpe = BPE.train(str(corpus), 280, ["<|endoftext|>"])
    eot = bpe.encode("<|endoftext|>")[0]
    pd.encode_corpus_to_bin(str(corpus), str(tmp_path / "train.bin"), bpe, eot, workers=2)

    cfg = TrainConfig(
        data_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        vocab_size=len(bpe.vocab),               # ← 用真实词表大小，别拍脑袋
        n_layers=1, d_model=32, n_heads=4, d_ff=64, context_length=16,
        batch_size=8, max_steps=120, max_lr=1e-2, min_lr=1e-3, warmup_steps=10,
        weight_decay=0.0, eval_interval=10**9, save_interval=10**9,
        device="cpu", dtype="fp32",
    )
    train(cfg)

    # cfg.out_dir 是 str，用 Path 包一层才能用 / 拼路径
    log_path = Path(cfg.out_dir) / "log.jsonl"
    losses = [json.loads(line)["loss"] for line in
              log_path.read_text().splitlines()
              if json.loads(line)["split"] == "train"]
    assert len(losses) >= 2
    assert losses[0] > 3.0          # 初始 loss 应接近 ln(vocab)≈5.6（随机猜的水平）
    assert losses[-1] < 1.5         # 小语料能被"背下来" -> 训练栈没问题
    assert losses[-1] < losses[0]
