"""Per-block output-magnitude analysis (mean L2 norm of per-layer hidden states).

Mirrors reference project L and Kimi's Attention Residuals Figure 5(b):
residual-stream magnitude should grow with depth but stay controlled under AttnRes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import TokenDataset
from src.model import ModelConfig, TransformerLM


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TransformerLM(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def block_magnitudes(model, tokens, device, batch_size=8, context_length=128):
    """One forward pass on tokens; mean L2 norm per layer output (incl. embedding)."""
    starts = np.random.randint(0, max(1, len(tokens) - context_length - 1), size=batch_size)
    idx = torch.from_numpy(
        tokens[starts[:, None] + np.arange(context_length)[None, :]].astype(np.int64)).to(device)
    with torch.no_grad():
        _, hidden_states = model(idx, return_hidden_states=True)
    return [float(torch.norm(h, p=2, dim=-1).mean().item()) for h in hidden_states]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="may be given multiple times")
    ap.add_argument("--label", action="append", required=True, help="one per --ckpt, same order")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="assets/magnitude_comparison.png")
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert len(args.ckpt) == len(args.label), "--ckpt and --label counts must match"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokens = TokenDataset(str(Path(args.data_dir) / "valid.bin")).tokens

    plt.figure(figsize=(8, 5))
    for ckpt_path, label in zip(args.ckpt, args.label):
        model, ckpt = load_model(ckpt_path, device)
        mags = block_magnitudes(model, tokens, device)
        layers = np.arange(len(mags))
        plt.plot(layers, mags, marker="o", linewidth=2, label=f"{label} ({ckpt['step']} steps)")
        print(f"[{label}] {ckpt_path}")
        print(f"  per-block L2: {', '.join(f'{m:.2f}' for m in mags)}")

    plt.xlabel("Block index (0 = embedding output)")
    plt.ylabel("Output magnitude (mean L2 norm)")
    plt.title("Residual stream magnitude across blocks")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
