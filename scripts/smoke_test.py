"""End-to-end smoke test: a short training run to verify the full pipeline.

Run after any change to model.py / train.py / data.py, before real training.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick_gpu", action="store_true",
                    help="smoke with the real 16M config on GPU (default: CPU mini-model)")
    args = ap.parse_args()

    if args.quick_gpu:
        cfg = TrainConfig(out_dir="experiments/000_smoke", max_steps=500,
                          eval_interval=250, dtype="bf16", device="cuda")
        expect_below = 4.0
    else:
        # CPU mini-model: fast correctness check, not a performance check
        cfg = TrainConfig(out_dir="experiments/000_smoke_cpu",
                          n_layers=2, d_model=64, n_heads=4, d_ff=128,
                          context_length=64, batch_size=8, max_steps=100,
                          warmup_steps=10, eval_interval=10**9, save_interval=10**9,
                          device="cpu", dtype="fp32")
        expect_below = 6.5

    train(cfg)

    losses = [json.loads(l)["loss"] for l in
              (Path(cfg.out_dir) / "log.jsonl").read_text().splitlines()
              if json.loads(l)["split"] == "train"]
    first, last = losses[0], sum(losses[-5:]) / 5
    print(f"\n[smoke] initial loss {first:.2f} (expect ≈ ln(vocab) ≈ 9.0)")
    print(f"[smoke] final loss {last:.2f} (require < {expect_below})")
    assert 7.0 < first < 12.0, "initial loss out of range: check init (see README pitfall notes)"
    assert last < expect_below, "loss not decreasing: check optimizer/data/backward"
    assert last < first, "loss went up!"
    print("[smoke] ✓ pipeline OK, ready for a real run")


if __name__ == "__main__":
    main()
