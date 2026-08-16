"""Tokenizer tests: roundtrip, special tokens, no-OOV, save/load, naive parity."""

import regex as re
from collections import Counter

from src.tokenizer import BPE, GPT2_SPLIT_PATTERN, train_bpe

# CS336 reference corpus: word frequencies are designed so merge order is predictable
SAMPLE = (
    "low low low low low lower lower widest widest widest "
    "newest newest newest newest newest newest <|endoftext|>."
)


def _make_corpus(tmp_path, text):
    p = tmp_path / "corpus.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


def _naive_train_bpe(path, vocab_size, special_tokens=None):
    """Slow, obvious BPE training used as the reference for the optimized version."""
    special_tokens = special_tokens or []
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        pat = "(" + "|".join(re.escape(s) for s in special_tokens) + ")"
        chunks = re.split(pat, text)
    else:
        chunks = [text]

    words = Counter()
    for chunk in chunks:
        if not chunk or chunk in special_tokens:
            continue
        for m in re.finditer(GPT2_SPLIT_PATTERN, chunk):
            words[tuple(bytes([b]) for b in m.group().encode("utf-8"))] += 1

    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    merges = []
    while len(vocab) < vocab_size and words:
        pair_counts = Counter()
        for w, cnt in words.items():
            for i in range(len(w) - 1):
                pair_counts[(w[i], w[i + 1])] += cnt
        if not pair_counts:
            break

        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        new_words = Counter()
        for w, cnt in words.items():
            nw = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                    nw.append(best[0] + best[1])
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            new_words[tuple(nw)] += cnt
        words = new_words
    return vocab, merges


def test_train_bpe_expected_layout(tmp_path):
    path = _make_corpus(tmp_path, SAMPLE)
    special = ["<|endoftext|>"]
    vocab, merges = train_bpe(path, 263, special)
    assert len(vocab) == 263
    assert vocab[256] == "<|endoftext|>".encode("utf-8")
    assert all(len(vocab[i]) >= 2 for i in range(257, 263))


def test_roundtrip_unicode_emoji(tmp_path):
    """encode -> decode must roundtrip chars never seen in the training corpus."""
    path = _make_corpus(tmp_path, SAMPLE)
    bpe = BPE.train(path, 263, ["<|endoftext|>"])
    text = "Hello 世界 🎉! <|endoftext|> Nice to meet you."
    assert bpe.decode(bpe.encode(text)) == text


def test_no_oov(tmp_path):
    path = _make_corpus(tmp_path, SAMPLE)
    bpe = BPE.train(path, 263, ["<|endoftext|>"])
    for ch in "abcXYZ0123!@# 中文汉字🎉":
        for tid in bpe.encode(ch):
            assert 0 <= tid < 263


def test_special_token_preserved(tmp_path):
    path = _make_corpus(tmp_path, SAMPLE)
    bpe = BPE.train(path, 263, ["<|endoftext|>"])
    assert bpe.encode("<|endoftext|>") == [256]
    assert bpe.decode(bpe.encode("hi <|endoftext|> hi")) == "hi <|endoftext|> hi"


def test_save_load(tmp_path):
    path = _make_corpus(tmp_path, SAMPLE)
    bpe = BPE.train(path, 263, ["<|endoftext|>"])
    save_path = tmp_path / "bpe.json"
    bpe.save(str(save_path))
    bpe2 = BPE.load(str(save_path))
    assert bpe2.vocab == bpe.vocab
    assert bpe2.merges == bpe.merges
    text = "low low lower newest <|endoftext|>"
    assert bpe2.decode(bpe2.encode(text)) == text


def test_optimized_matches_naive(tmp_path):
    """The incremental trainer must produce the exact vocab and merge order of the naive one."""
    path = _make_corpus(tmp_path, SAMPLE)
    v1, m1 = train_bpe(path, 263, ["<|endoftext|>"])
    v2, m2 = _naive_train_bpe(path, 263, ["<|endoftext|>"])
    assert v1 == v2
    assert m1 == m2
