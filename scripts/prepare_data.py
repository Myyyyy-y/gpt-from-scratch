"""Data pipeline: download TinyStories -> train/load BPE -> encode to uint16 .bin.

Offline tokenization means training reads raw integer ids via memmap with zero
tokenization overhead. Outputs: corpus.txt, bpe.json, train.bin, valid.bin, meta.json.
"""

# HF_DATASETS_CACHE must be set before `import datasets` (the library pins the
# cache path at import time), so parse --hf_cache with a minimal pre-parser.
import argparse
import os

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--hf_cache",
                  default=os.environ.get("HF_DATASETS_CACHE",
                                         os.path.expanduser("~/hmy/hf_cache")))
_pre_args, _ = _pre.parse_known_args()
os.environ["HF_DATASETS_CACHE"] = _pre_args.hf_cache

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from datasets import load_dataset

# make `from src.xxx import` work when run directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tokenizer import BPE                    # noqa: E402

SPECIAL_TOKENS = ["<|endoftext|>"]              # story separator gets its own id

# Pool children are separate processes, so the tokenizer is per-worker state
_WORKER_BPE = None
_WORKER_EOT = None


def _init_worker(bpe, eot_id):
    global _WORKER_BPE, _WORKER_EOT
    _WORKER_BPE = bpe
    _WORKER_EOT = eot_id


def _encode_text(text):
    # explicit end-of-story signal
    ids = _WORKER_BPE.encode(text)
    return ids + [_WORKER_EOT]


def _write_encoded(pool, texts, f):
    n = 0
    for ids in pool.map(_encode_text, texts):
        f.write(np.asarray(ids, dtype=np.uint16).tobytes())
        n += len(ids)
    return n


def _iter_stories(split, max_examples=None):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for i, story in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        yield story["text"]


def collect_corpus(out_dir, n_stories):
    """Write the first n_stories train stories to corpus.txt (BPE training text)."""
    corpus_path = out_dir / "corpus.txt"
    # reuse only if non-empty: a crashed run can leave a 0-byte file behind
    if corpus_path.exists() and corpus_path.stat().st_size > 0:
        print(f"[*] reusing existing corpus: {corpus_path}")
        return corpus_path
    print(f"[*] downloading TinyStories train, collecting {n_stories} stories as BPE corpus...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(_iter_stories("train", n_stories)):
            f.write(text + "\n")
            if (i + 1) % 5000 == 0:
                print(f"    ... {i + 1} stories")
    return corpus_path


def encode_corpus_to_bin(text_path, bin_path, bpe, eot_id, workers=8):
    """Local txt (one story per line) -> .bin. For tests / small corpora."""
    with open(text_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n_tokens = 0
    with Pool(processes=workers, initializer=_init_worker, initargs=(bpe, eot_id)) as pool:
        with open(bin_path, "wb") as fout:
            n_tokens += _write_encoded(pool, lines, fout)
    return n_tokens


def build_split(split, out_path, token_budget, bpe, eot_id, max_examples=None, workers=8):
    """Encode a dataset split until the token budget is reached; idempotent."""
    if out_path.exists() and out_path.stat().st_size > 0:
        n = np.fromfile(out_path, dtype=np.uint16).size
        print(f"[*] reusing existing {out_path.name} ({n} tokens)")
        return n
    print(f"[*] encoding {split} -> {out_path.name} (budget {token_budget} tokens)")
    n_tokens = 0
    texts = []
    with Pool(processes=workers, initializer=_init_worker, initargs=(bpe, eot_id)) as pool:
        with open(out_path, "wb") as f:
            for i, text in enumerate(_iter_stories(split, max_examples)):
                texts.append(text)
                if len(texts) >= 2000:
                    n_tokens += _write_encoded(pool, texts, f)
                    texts = []
                    print(f"    {split}: {n_tokens} tokens")
                    if n_tokens >= token_budget:
                        break
            if texts:
                n_tokens += _write_encoded(pool, texts, f)
    print(f"[✓] {split}: {n_tokens} tokens")
    return n_tokens


def main():
    ap = argparse.ArgumentParser(parents=[_pre])
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--tokenizer_path", default=None, help="existing bpe.json; train a new one if omitted")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--bpe_sample_stories", type=int, default=20000)
    ap.add_argument("--train_token_budget", type=int, default=3_000_000)
    ap.add_argument("--valid_token_budget", type=int, default=150_000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # uint16 ids overflow silently past 65535, so reject it up front
    assert args.vocab_size <= 65535, "vocab_size exceeds uint16 max 65535"

    # --- step 1: tokenizer (reuse existing or train a new one) ---
    if args.tokenizer_path and Path(args.tokenizer_path).exists():
        print(f"[*] reusing tokenizer: {args.tokenizer_path}")
        bpe = BPE.load(args.tokenizer_path)
    else:
        corpus_path = collect_corpus(out_dir, args.bpe_sample_stories)
        print(f"[*] training BPE (vocab_size={args.vocab_size})...")
        bpe = BPE.train(str(corpus_path), args.vocab_size, SPECIAL_TOKENS)
        tok_path = out_dir / "bpe.json"
        bpe.save(str(tok_path))
        print(f"[saved] tokenizer: {tok_path}")

    eot_id = bpe.encode(SPECIAL_TOKENS[0])[0]
    print(f"    vocab_size={len(bpe.vocab)}  eot_id={eot_id}")

    # --- step 2: encode train / validation splits in parallel ---
    build_split("train", out_dir / "train.bin", args.train_token_budget,
                bpe, eot_id, workers=args.workers)
    build_split("validation", out_dir / "valid.bin", args.valid_token_budget,
                bpe, eot_id, workers=args.workers)

    # --- step 3: meta info + sanity check ---
    meta = {
        "vocab_size": len(bpe.vocab),
        "eot_id": eot_id,
        "train_tokens": np.fromfile(out_dir / "train.bin", dtype=np.uint16).size,
        "valid_tokens": np.fromfile(out_dir / "valid.bin", dtype=np.uint16).size,
        "special_tokens": SPECIAL_TOKENS,
        "source": "roneneldan/TinyStories",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # decode the first 200 valid tokens to catch e.g. train/valid vocab mismatch
    arr = np.fromfile(out_dir / "valid.bin", dtype=np.uint16)
    print("\n[sanity] first 200 valid tokens decoded:")
    print(bpe.decode(arr[:200].tolist()))
    print("\ndone! training reads data/meta.json for eot_id etc.")


if __name__ == "__main__":
    main()
