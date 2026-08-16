"""Training-stack tests: AdamW/CE numerics, grad clip, LR schedule, tiny overfit."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare_data as pd
from src.tokenizer import BPE
from src.train import (AdamW, Muon, TrainConfig, clip_grad_norm, cross_entropy,
                       get_lr, train, zeropower_via_newtonschulz5)

TEXTS = [f"sentence number {i} with some words to memorize" for i in range(20)] * 5


# ---------- numerics vs PyTorch reference ----------

def test_adamw_matches_torch():
    torch.manual_seed(0)
    m1 = torch.nn.Linear(8, 4, bias=False)
    m2 = torch.nn.Linear(8, 4, bias=False)
    m2.load_state_dict(m1.state_dict())

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
    torch.manual_seed(0)
    logits = torch.randn(2, 16, 100) * 3
    targets = torch.randint(0, 100, (2, 16))
    ref = F.cross_entropy(logits.reshape(-1, 100), targets.reshape(-1))
    assert torch.allclose(cross_entropy(logits, targets), ref, atol=1e-5)


# ---------- properties ----------

def test_clip_grad_norm():
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([3.0, 4.0])            # norm = 5.0
    norm = clip_grad_norm([p], max_norm=1.0)
    assert abs(norm - 5.0) < 1e-5
    assert torch.allclose(p.grad, torch.tensor([0.6, 0.8]), atol=1e-5)


def test_clip_grad_norm_noop_when_small():
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([0.3, 0.4])            # norm = 0.5 < 1.0
    clip_grad_norm([p], max_norm=1.0)
    assert torch.allclose(p.grad, torch.tensor([0.3, 0.4]))


def test_lr_schedule():
    args = (5000, 100, 1e-3, 1e-4)
    assert get_lr(0, *args) == 1e-5
    assert get_lr(99, *args) == 1e-3
    assert get_lr(100, *args) == 1e-3
    assert get_lr(5000, *args) == 1e-4
    assert get_lr(50, *args) < get_lr(99, *args)
    assert get_lr(500, *args) < get_lr(100, *args)
    assert get_lr(4000, *args) < get_lr(500, *args)
    assert 1e-4 < get_lr(2000, *args) < 1e-3


# ---------- Muon ----------

def test_newtonschulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(64, 32) * torch.logspace(0, 2, 32)
    s_in = torch.linalg.svdvals(G)
    s_out = torch.linalg.svdvals(zeropower_via_newtonschulz5(G))
    assert s_in.max() / s_in.min() > 10
    assert 0.5 < s_out.min() and s_out.max() < 1.25


def test_muon_learns_tiny_problem():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 4, bias=False)
    opt = Muon(lin.parameters(), lr=0.05)
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    losses = []
    for _ in range(30):
        loss = ((lin(x) - y) ** 2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert losses[-1] < losses[0] * 0.7, (losses[0], losses[-1])


# ---------- end-to-end overfit smoke tests ----------

def test_overfit_tiny_corpus(tmp_path):
    """A healthy pipeline must memorize a tiny repeated corpus within 120 steps."""
    torch.manual_seed(0)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(TEXTS), encoding="utf-8")
    bpe = BPE.train(str(corpus), 280, ["<|endoftext|>"])
    eot = bpe.encode("<|endoftext|>")[0]
    pd.encode_corpus_to_bin(str(corpus), str(tmp_path / "train.bin"), bpe, eot, workers=2)

    cfg = TrainConfig(
        data_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        vocab_size=len(bpe.vocab),
        n_layers=1, d_model=32, n_heads=4, d_ff=64, context_length=16,
        batch_size=8, max_steps=120, max_lr=1e-2, min_lr=1e-3, warmup_steps=10,
        weight_decay=0.0, eval_interval=10**9, save_interval=10**9,
        device="cpu", dtype="fp32",
    )
    train(cfg)

    log_path = Path(cfg.out_dir) / "log.jsonl"
    losses = [json.loads(line)["loss"] for line in
              log_path.read_text().splitlines()
              if json.loads(line)["split"] == "train"]
    assert len(losses) >= 2
    assert losses[0] > 3.0
    assert losses[-1] < 1.5
    assert losses[-1] < losses[0]


def test_attn_res_overfit_tiny_corpus(tmp_path):
    """Attention Residuals must not break end-to-end learning."""
    torch.manual_seed(0)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(TEXTS), encoding="utf-8")
    bpe = BPE.train(str(corpus), 280, ["<|endoftext|>"])
    eot = bpe.encode("<|endoftext|>")[0]
    pd.encode_corpus_to_bin(str(corpus), str(tmp_path / "train.bin"), bpe, eot, workers=2)

    cfg = TrainConfig(
        data_dir=str(tmp_path),
        out_dir=str(tmp_path / "out"),
        vocab_size=len(bpe.vocab),
        n_layers=2, d_model=32, n_heads=4, d_ff=64, context_length=16,
        batch_size=8, max_steps=60, max_lr=1e-2, min_lr=1e-3, warmup_steps=10,
        weight_decay=0.0, eval_interval=10**9, save_interval=10**9,
        attn_res=True, device="cpu", dtype="fp32",
    )
    train(cfg)

    log_path = Path(cfg.out_dir) / "log.jsonl"
    losses = [json.loads(line)["loss"] for line in
              log_path.read_text().splitlines()
              if json.loads(line)["split"] == "train"]
    assert losses[0] > 3.0
    assert losses[-1] < losses[0]
