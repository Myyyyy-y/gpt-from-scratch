"""
评估：checkpoint 精确 val loss / KV cache 测速报告

==================== 给初学者的整体说明 ====================

【这个文件回答两个问题】
  1. 模型学得到底好不好？——用验证集 loss 客观打分
  2. KV cache 到底快多少？——同一 prompt 两种生成方式计时对比

【为什么训练里已经算了 val loss，还要这个文件？】
训练循环里的评估只抽 20 个 batch（eval_batches=20），是为了【快】——
每 250 步插一次，不能让评估拖慢训练。代价是噪声大：单次 val loss
有 ±0.01 级的抖动。
而"写进 README 的最终成绩"需要更准的数字：本文件用 100+ 个 batch
独立评估，把噪声压小一个量级。这就是"训练中的随堂测"和"交卷前
的正式阅卷"的区别。

【与规模无关】模型配置存在 checkpoint 里（model_config），
本文件直接还原——16M、29M 或任何消融变体都通用。

用法：
  python -m src.eval --ckpt experiments/003_29m_lr_1e3/best.pt
  python -m src.eval --ckpt ... --bench_tokens 256   # 顺带测速
"""
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
    """加载 checkpoint，在 valid.bin 上抽 n_batches 个 batch 算平均 loss。

    三个关键装饰（与训练内评估相同）：
      model.eval()      关闭训练专用行为，保证结果确定可复现
      torch.no_grad()   不算梯度，省显存更快
      多 batch 取平均   单 batch 噪声大，100 个 batch 后均值才稳
    """
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
    # 除均值外还报标准误（mean 的不确定度）：sem = std / sqrt(n)
    # 两次评估均值差小于 2 倍 sem 时，差异不可信——这是实验纪律的一部分
    return {"val_loss": round(float(losses.mean()), 4),
            "sem": round(float(losses.std() / np.sqrt(len(losses))), 4),
            "n_batches": n_batches,
            "ckpt_step": ckpt.get("step")}


def main():
    ap = argparse.ArgumentParser(description="checkpoint 精确评估 + KV cache 测速")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--n_batches", type=int, default=100)
    ap.add_argument("--device", default="")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--bench_tokens", type=int, default=0,
                    help=">0 则顺带做 KV cache 测速（生成该长度的文本计时）")
    ap.add_argument("--prompt", default="Once upon a time")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    res = evaluate_checkpoint(args.ckpt, args.data_dir, n_batches=args.n_batches,
                              device=device, dtype=dtype)
    print(f"[eval] {args.ckpt}")
    print(f"  val_loss = {res['val_loss']} ± {res['sem']} "
          f"（{res['n_batches']} batches，训练自 step {res['ckpt_step']}）")

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
        print(f"  KV cache 测速（生成 {r['new_tokens']} tokens）：")
        print(f"    无 cache: {r['baseline_tokens_per_sec']} tok/s")
        print(f"    有 cache: {r['kv_cache_tokens_per_sec']} tok/s")
        print(f"    加速比  : {r['speedup']}x")


if __name__ == "__main__":
    main()
