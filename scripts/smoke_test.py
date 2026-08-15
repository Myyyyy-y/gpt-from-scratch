"""
端到端冒烟测试：500 步小跑，验证全链路在真实数据上工作。

==================== 什么时候用这个 ====================

改了 model.py / train.py / data.py 的任何代码之后、正式训练之前。
单元测试（pytest）验证"零件合格"，本脚本验证"整车能开"。

【判断标准】
  - 第 1 步 loss ≈ ln(vocab_size) ≈ 9.0（随机猜测水平）
  - loss 持续下降，无 nan/inf
  - 500 步后 train loss 应降到 4 以下（16M 默认配置实测约 3.1）

用法：
  python scripts/smoke_test.py                # CPU 小模型，30 秒
  python scripts/smoke_test.py --quick_gpu    # GPU 上 16M 配置，约 3 分钟
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
                    help="用真实 16M 配置在 GPU 上冒烟（默认是 CPU 迷你模型）")
    args = ap.parse_args()

    if args.quick_gpu:
        # 真实配置 + GPU：最接近正式训练的冒烟
        cfg = TrainConfig(out_dir="experiments/000_smoke", max_steps=500,
                          eval_interval=250, dtype="bf16", device="cuda")
        expect_below = 4.0
    else:
        # CPU 迷你模型：30 秒快速验证链路（不测性能，只测正确性）
        cfg = TrainConfig(out_dir="experiments/000_smoke_cpu",
                          n_layers=2, d_model=64, n_heads=4, d_ff=128,
                          context_length=64, batch_size=8, max_steps=100,
                          warmup_steps=10, eval_interval=10**9, save_interval=10**9,
                          device="cpu", dtype="fp32")
        expect_below = 6.5

    train(cfg)

    # 读日志验证：初始 loss 接近 ln(vocab)，且末段显著低于初段
    losses = [json.loads(l)["loss"] for l in
              (Path(cfg.out_dir) / "log.jsonl").read_text().splitlines()
              if json.loads(l)["split"] == "train"]
    first, last = losses[0], sum(losses[-5:]) / 5
    print(f"\n[smoke] 初始 loss {first:.2f}（期望 ≈ ln(vocab)≈9.0）")
    print(f"[smoke] 末段 loss {last:.2f}（要求 < {expect_below}）")
    assert 7.0 < first < 12.0, "初始 loss 异常：查初始化（参考 README 踩坑记录）"
    assert last < expect_below, "loss 降不下去：查优化器/数据/反向传播"
    assert last < first, "loss 没有下降！"
    print("[smoke] ✓ 全链路正常，可以上正式训练")


if __name__ == "__main__":
    main()
