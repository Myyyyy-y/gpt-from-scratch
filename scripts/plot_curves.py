"""
画 loss 曲线：读实验目录的 log.jsonl，输出 PNG 到 assets/

用法：
  # 单实验 train/val 双曲线
  python scripts/plot_curves.py experiments/001_baseline

  # 多实验 val loss 对比（消融用）
  python scripts/plot_curves.py experiments/002_29m_lr_3e4 experiments/003_29m_lr_1e3 \
      experiments/004_29m_lr_3e3 --metric valid --out assets/lr_sweep.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 无显示环境（服务器）也能出图
import matplotlib.pyplot as plt


def load_log(exp_dir):
    """读 log.jsonl，返回 {split: (steps, losses)}；train 记录还带 lr / grad_norm。"""
    series = {"train": ([], []), "valid": ([], [])}
    extras = {"lr": ([], []), "grad_norm": ([], [])}
    for line in (Path(exp_dir) / "log.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec["split"] in series:
            series[rec["split"]][0].append(rec["step"])
            series[rec["split"]][1].append(rec["loss"])
        for k in extras:
            if k in rec:
                extras[k][0].append(rec["step"])
                extras[k][1].append(rec[k])
    series.update(extras)
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--metric", choices=["train", "valid", "both", "lr", "grad_norm"],
                    default="both")
    ap.add_argument("--out", default=None, help="输出路径（默认 assets/<名字>.png）")
    args = ap.parse_args()

    Path("assets").mkdir(exist_ok=True)
    out = args.out or f"assets/{'_vs_'.join(Path(e).name for e in args.experiments)}.png"

    plt.figure(figsize=(8, 5))
    for exp in args.experiments:
        series = load_log(exp)
        name = Path(exp).name
        if args.metric in ("train", "both", "valid"):
            if args.metric in ("train", "both"):
                s, l = series["train"]
                plt.plot(s, l, alpha=0.4, label=f"{name} train")
            if args.metric in ("valid", "both"):
                s, l = series["valid"]
                plt.plot(s, l, linewidth=2, marker="o", markersize=3, label=f"{name} val")
        else:
            s, v = series[args.metric]
            plt.plot(s, v, linewidth=1.5, label=name)

    plt.xlabel("step")
    plt.ylabel("loss" if args.metric in ("train", "valid", "both") else args.metric)
    plt.title(" / ".join(Path(e).name for e in args.experiments))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[✓] 图已保存: {out}")


if __name__ == "__main__":
    main()
