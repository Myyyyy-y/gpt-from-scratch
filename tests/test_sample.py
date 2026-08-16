"""Sampling + KV-cache correctness tests."""

import torch

from src.model import ModelConfig, TransformerLM
from src.sample import generate, sample_next
from src.tokenizer import BPE


def _tiny_bpe(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(["once upon a time there was a little girl named lily"] * 30),
                      encoding="utf-8")
    return BPE.train(str(corpus), 300, ["<|endoftext|>"])


def _tiny_model(vocab_size):
    torch.manual_seed(0)
    return TransformerLM(ModelConfig(vocab_size=vocab_size, n_layers=2, d_model=32,
                                     n_heads=4, d_ff=64, context_length=64)).eval()


# ---------- sampling ----------

def test_sample_greedy():
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    assert sample_next(logits, temperature=0) == 1


def test_sample_topk():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, 8.0, 0.0]])
    got = {sample_next(logits, temperature=1.0, top_k=2) for _ in range(50)}
    assert got <= {0, 1}


def test_sample_topp():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, 0.0, 0.0]])
    got = {sample_next(logits, temperature=1.0, top_p=0.9) for _ in range(50)}
    assert got <= {0, 1}


# ---------- KV cache ----------

def test_kv_cache_equivalent():
    """prefill + single-step decode must match one full forward pass.

    Covers the three easy-to-break parts: RoPE absolute positions,
    mask diagonal offset, and K/V concat order.
    """
    model = _tiny_model(vocab_size=100)
    seq = torch.randint(0, 100, (1, 20))
    with torch.no_grad():
        logits_full = model(seq)
        logits_pre, kvs = model(seq[:, :-1], use_cache=True)
        logits_new, _ = model(seq[:, -1:], past_kvs=kvs, use_cache=True)
    assert torch.allclose(logits_full[:, -1], logits_new[:, -1], atol=1e-5)
    assert torch.allclose(logits_pre[:, -1], logits_full[:, -2], atol=1e-5)


def test_generate_cache_matches_nocache(tmp_path):
    """Greedy generation must be token-identical with and without KV cache."""
    bpe = _tiny_bpe(tmp_path)
    model = _tiny_model(vocab_size=len(bpe.vocab))
    prompt = "once upon a time"
    ids_c = generate(model, bpe, prompt, max_new_tokens=30, temperature=0.0,
                     top_k=0, top_p=1.0, device="cpu", use_cache=True)
    ids_n = generate(model, bpe, prompt, max_new_tokens=30, temperature=0.0,
                     top_k=0, top_p=1.0, device="cpu", use_cache=False)
    assert ids_c == ids_n
