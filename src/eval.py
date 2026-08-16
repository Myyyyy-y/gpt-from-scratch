"""Checkpoint evaluation: precise val loss + KV-cache speed report."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import TokenDataset
from src.model import ModelConfig, TransformerLM
from src.sample import benchmark
from src.tokenizer import BPE
from src.train import cross_entropy


def evaluate_checkpoint(ckpt_path, data_dir, n_batches=100, batch_size=64,
                        context_length=256, device="cpu", dtype=torch.float32):
    """Load a checkpoint and report mean val loss over n_batches, plus its SEM."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TransformerLM(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    valid = TokenDataset(str(Path(data_dir) / "valid.bin"))
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = valid.get_batch(batch_size, context_length, device)
            with torch.autocast(device_type="cuda", dtype=dtype,
                                enabled=(dtype != torch.float32 and device.startswith("cuda"))):
                losses.append(cross_entropy(model(x), y).item())
    losses = np.array(losses)
    return {"val_loss": round(float(losses.mean()), 4),
            "sem": round(float(losses.std() / np.sqrt(len(losses))), 4),
            "n_batches": n_batches,
            "ckpt_step": ckpt.get("step")}


def main():
    ap = argparse.ArgumentParser(description="checkpoint evaluation: precise val loss + KV-cache speed")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n_batches", type=int, default=100)
    ap.add_argument("--device", default="")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--bench_tokens", type=int, default=0,
                    help=">0 also benchmarks KV-cache decoding (times generation of this many tokens)")
    ap.add_argument("--prompt", default="Once upon a time")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    res = evaluate_checkpoint(args.ckpt, args.data_dir, n_batches=args.n_batches,
                              device=device, dtype=dtype)
    print(f"[eval] {args.ckpt}")
    print(f"  val_loss = {res['val_loss']} ± {res['sem']} "
          f"({res['n_batches']} batches, checkpoint from step {res['ckpt_step']})")

    if args.bench_tokens > 0:
        ckpt = torch.load(args.ckpt, map_location=device)
        model = TransformerLM(ModelConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["model"])
        if dtype == torch.bfloat16 and device.startswith("cuda"):
            model = model.to(dtype)
        model.eval()
        bpe = BPE.load(str(Path(args.data_dir) / "bpe.json"))
        eot = json.loads((Path(args.data_dir) / "meta.json").read_text())["eot_id"]
        r = benchmark(model, bpe, args.prompt, max_new_tokens=args.bench_tokens,
                      eot_id=eot, device=device)
        print(f"  KV-cache speed (generate {r['new_tokens']} tokens):")
        print(f"    no cache: {r['baseline_tokens_per_sec']} tok/s")
        print(f"    cache   : {r['kv_cache_tokens_per_sec']} tok/s")
        print(f"    speedup : {r['speedup']}x")


if __name__ == "__main__":
    main()
