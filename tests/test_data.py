"""Data pipeline tests: encode -> .bin -> read-back roundtrip + batch shift."""

import numpy as np
import torch

import prepare_data as pd
from src.data import TokenDataset
from src.tokenizer import BPE

TEXTS = [
    "Once upon a time there was a little girl named Lily.",
    "She loved to play in the park with her dog.",
    "One day the dog ran away and Lily was sad.",
    "The end was happy and everyone smiled.",
]


def _make_bpe(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(TEXTS), encoding="utf-8")
    bpe = BPE.train(str(corpus), 300, ["<|endoftext|>"])
    eot = bpe.encode("<|endoftext|>")[0]
    return corpus, bpe, eot


def test_encode_bin_roundtrip(tmp_path):
    corpus, bpe, eot = _make_bpe(tmp_path)
    bin_path = tmp_path / "train.bin"
    pd.encode_corpus_to_bin(str(corpus), str(bin_path), bpe, eot, workers=2)

    arr = np.fromfile(bin_path, dtype=np.uint16)
    text = bpe.decode(arr.tolist())

    assert "<|endoftext|>" in text
    for t in TEXTS:
        assert t in text
    assert eot not in bpe.encode("Once upon a time")


def test_token_dataset_batch(tmp_path):
    corpus, bpe, eot = _make_bpe(tmp_path)
    bin_path = tmp_path / "train.bin"
    pd.encode_corpus_to_bin(str(corpus), str(bin_path), bpe, eot, workers=2)

    ds = TokenDataset(str(bin_path))
    x, y = ds.get_batch(4, 16, "cpu")

    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # y is x shifted right by one: each position predicts the next token
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_token_dataset_max_tokens_limit(tmp_path):
    bin_path = tmp_path / "t.bin"
    np.arange(1000, dtype=np.uint16).tofile(bin_path)
    ds = TokenDataset(str(bin_path), max_tokens=100)
    assert ds.n_tokens == 100
    full = TokenDataset(str(bin_path))
    assert full.n_tokens == 1000
