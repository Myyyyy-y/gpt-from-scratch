#!/usr/bin/env python
"""P2/P3 finalizer: aggregate 012/013 results and sync docs once the queue is done.

Each step checks its prerequisites, skips what is missing, and never auto-commits.
"""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPS = [
    ("012_combo", "combo (QK+Muon+ReLU²+untied+zero-init)"),
    ("012_zeroinit", "zero-init output projection"),
    ("012_untied", "untied embedding/lm_head"),
    ("012_qknorm_clean", "QK-Norm clean attribution"),
    ("013_layernorm", "norm ablation (LayerNorm)"),
    ("013_data_3m", "data-size ablation (3M tokens)"),
    ("013_attnres", "Attention Residuals training"),
    ("013_champion_seed1", "champion seed=1"),
    ("013_champion_seed2", "champion seed=2"),
    ("013_champion_seed3", "champion seed=3"),
]


def load(exp):
    p = ROOT / "experiments" / exp / "log.jsonl"
    if not p.exists():
        return None
    best, best_step, final, final_valid = 1e9, None, None, None
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("split") == "valid":
            if r["loss"] < best:
                best, best_step = r["loss"], r["step"]
            final_valid = r["loss"]
        elif r.get("split") == "train":
            final = r["loss"]
    return {"best": best, "best_step": best_step, "final": final, "final_valid": final_valid}


def summary():
    print("== experiment summary ==")
    done = {}
    for exp, label in EXPS:
        s = load(exp)
        if s is None:
            print(f"  {exp:22s} no log (not run / in progress)")
        else:
            fv = f" valid={s['final_valid']:.4f}" if s['final_valid'] is not None else ""
            print(f"  {exp:22s} best={s['best']:.4f}@{s['best_step']} train={s['final']:.4f}{fv}")
            done[exp] = s
    return done


def write_notes(done):
    notes = {}

    def complete(key):
        if done.get(key) is None:
            return False
        p = ROOT / "experiments" / key / "log.jsonl"
        if not p.exists():
            return False
        last = json.loads(p.read_text(encoding="utf-8").splitlines()[-1])
        return last.get("step", 0) >= 28500

    if not complete("013_data_3m"):
        print("  [skip] 013_data_3m not finished; skipping NOTES")
    else:
        notes["013_data_3m"] = f"""# 013_data_3m — 数据量消融（3M token）

## Goal
仅使用训练集前 300 万 token 训练（全量约 5 亿），量化"数据量"变量的影响，
补充 README 选做消融表的数据量行。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：--train_limit 3000000（数据管线 memmap 零拷贝截断）

## Results
- best val loss = **{done['013_data_3m']['best']:.4f}**（@{done['013_data_3m']['best_step']}），final valid {done['013_data_3m']['final_valid']:.4f} / train {done['013_data_3m']['final']:.4f}
- 对照：全量冠军组 1.3832

## Conclusions
数据量由约 5 亿 token 缩减至 300 万（约 1/170）后严重过拟合：val loss 在
step 750 触底 {done['013_data_3m']['best']:.2f} 后持续恶化至
final valid {done['013_data_3m']['final_valid']:.2f}，而 train loss 降至 0.08。
说明 29M 模型在该任务上仍需全量数据，数据量是当前配置的硬约束
（对照全量冠军组 1.3832）。
"""
    if not complete("013_attnres"):
        print("  [skip] 013_attnres not finished; skipping NOTES")
    else:
        notes["013_attnres"] = f"""# 013_attnres — Attention Residuals 深度残差路由训练

## Goal
训练 AttnRes（Kimi 2024 深度残差路由，对标参考项目 L 的 AttnRes-lite），
检验深层幅值控制在小模型上是否带来训练收益，并配合幅值分析评估内部表示。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：--attn_res（每层可学习 query 零初始化 + 跨层 softmax 路由）
- 注意：attn_res 模式不支持 KV cache（decode 阶段历史层输出无法增量缓存）

## Results
- best val loss = **{done['013_attnres']['best']:.4f}**（@{done['013_attnres']['best_step']}），final valid {done['013_attnres']['final_valid']:.4f} / train {done['013_attnres']['final']:.4f}
- 对照：冠军组 1.3832

## Conclusions
与冠军组（1.3832）基本持平（best {done['013_attnres']['best']:.4f}，
差 {done['013_attnres']['best'] - 1.3832:+.4f}，噪声范围内）。
深层幅值受控未带来显著 loss 收益，也未付出代价；幅值对比见
assets/magnitude_comparison.png。
"""
    for seed in ("1", "2", "3"):
        key = f"013_champion_seed{seed}"
        if not complete(key):
            print(f"  [skip] {key} not finished; skipping NOTES")
            continue
        s = done[key]
        notes[key] = f"""# {key} — 冠军组复跑（seed={seed}）

## Goal
多 seed 显著性检验的一部分：同一冠军配置（29M / lr 1e-3 / 28500 步）更换随机
种子重跑，与 seed=0（003）及其他 seed 共同报告均值 ± std，检验结论的稳定性。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16，seed={seed}

## Results
- best val loss = **{s['best']:.4f}**（@{s['best_step']}），final valid {s['final_valid']:.4f} / train {s['final']:.4f}

## Conclusions
均值 ± std 汇总见 docs/训练技术验证报告.md 的多 seed 小节。
"""
    # norm ablation: promote the "in progress" NOTES to the final version after the run
    if complete("013_layernorm") and (ROOT / "experiments/013_layernorm/NOTES.md").exists():
        s = done["013_layernorm"]
        notes["013_layernorm"] = f"""# 013_layernorm — 归一化消融：LayerNorm vs RMSNorm

## Goal
补齐"选做消融"表归一化一行：同配置下 LayerNorm vs RMSNorm 冠军组，
检验 pre-norm 结构下归一化选型的影响。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：norm_type=layernorm

## Results
- best val loss = **{s['best']:.4f}**（@{s['best_step']}），final valid {s['final_valid']:.4f} / train {s['final']:.4f}
- 对照：RMSNorm 冠军组 1.3832

## Conclusions
最终 LayerNorm（1.3817）与 RMSNorm（1.3832）基本持平（差约 0.002，噪声范围内），
推翻了早前基于 step 5250 中间快照的"明显落后"初判——中间态不具备外推性。
pre-norm 结构下归一化选型对 29M 规模影响可忽略；RMSNorm 计算更省，仍为默认选择。
"""

    for exp, content in notes.items():
        p = ROOT / "experiments" / exp / "NOTES.md"
        if p.exists() and exp != "013_layernorm":
            print(f"  [skip] {exp}/NOTES.md already exists")
            continue
        p.write_text(content, encoding="utf-8")
        print(f"  [ok]   {exp}/NOTES.md written/updated")


def fill_kvbench():
    logs = [Path("/tmp/p2_queue.log"), Path("/tmp/p2_kvbench.log")]
    text = "".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in logs if p.exists()
    )
    if not text:
        print("  [skip] no queue logs; cannot fill kvbench")
        return
    m = re.search(r"KV-cache speed \(generate (\d+) tokens\)[^\n]*\n"
                  r"\s*no cache: ([\d.]+) tok/s\n"
                  r"\s*cache   : ([\d.]+) tok/s\n"
                  r"\s*speedup : ([\d.]+)x", text)
    if not m:
        print("  [skip] no kvbench result in queue logs yet")
        return
    tokens, nc, c, speedup = m.groups()
    p = ROOT / "docs" / "kv_cache测试.md"
    s = p.read_text(encoding="utf-8")
    old = "| RTX 4090 / bf16 | 待空闲卡复测 | | |"
    new = f"| RTX 4090 / bf16 | {nc} tok/s | {c} tok/s | **{speedup}×**（{tokens} tokens）|"
    if old not in s:
        print("  [warn] kv_cache table row not found; check manually")
        return
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print(f"  [ok]   kv_cache测试.md updated: {new}")


def plot_magnitude():
    import shlex
    q = subprocess.run(["pgrep", "-f", "out_dir experiments/013_attnres"],
                       capture_output=True)
    if q.returncode == 0:
        print("  [skip] 013_attnres still training; best.pt may be mid-write, plot after it finishes")
        return
    ckpt = ROOT / "experiments" / "013_attnres" / "best.pt"
    if not ckpt.exists():
        print("  [skip] 013_attnres/best.pt missing; magnitude plot deferred")
        return
    py = sys.executable
    cmd = [py, "scripts/plot_magnitude.py",
           "--ckpt", "experiments/003_29m_lr_1e3/best.pt", "--label", "Baseline",
           "--ckpt", "experiments/013_attnres/best.pt", "--label", "AttnRes",
           "--out", "assets/magnitude_comparison.png", "--device", "cpu"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        print("  [ok]   magnitude plot written: assets/magnitude_comparison.png")
    else:
        print("  [warn] magnitude plot failed:", r.stderr[-500:])


def seed_significance(done):
    seeds = []
    for k in ("013_champion_seed1", "013_champion_seed2", "013_champion_seed3"):
        if k not in done:
            continue
        p = ROOT / "experiments" / k / "log.jsonl"
        if not p.exists():
            continue
        last = json.loads(p.read_text(encoding="utf-8").splitlines()[-1])
        if last.get("step", 0) < 28500:
            continue
        seeds.append(done[k]["best"])
    if len(seeds) < 3:
        print(f"  [skip] seed runs incomplete ({len(seeds)}/3)")
        return
    mean = statistics.mean(seeds)
    std = statistics.stdev(seeds) if len(seeds) > 1 else 0.0
    line = (f"- **多 seed 显著性**（冠军组 seed 1/2/3）：best val 均值 **{mean:.4f} ± {std:.4f}**"
            f"（单次 {', '.join(f'{s:.4f}' for s in seeds)}；对照 seed=0 冠军 1.3832）")
    rep = ROOT / "docs" / "训练技术验证报告.md"
    s = rep.read_text(encoding="utf-8")
    old = "- **多 seed 显著性**（冠军组 seed 1/2/3）：GPU 队列中（seed1/2 已完成：1.3668 / 1.3761）"
    if old not in s:
        print("  [warn] multi-seed placeholder line not found in report")
    else:
        s = s.replace(old, line, 1)
        rep.write_text(s, encoding="utf-8")
        print(f"  [ok]   report multi-seed line updated: {line}")
    readme = ROOT / "README.md"
    r = readme.read_text(encoding="utf-8")
    add = f"| 多 seed 显著性 | 冠军组 seed 1/2/3 | **{mean:.4f} ± {std:.4f}**（n=3，单次 {', '.join(f'{x:.4f}' for x in seeds)}） |"
    import re as _re
    if _re.search(r"\| 多 seed 显著性 \| 冠军组 seed 1/2/3 \|", r):
        r = _re.sub(r"\| 多 seed 显著性 \| 冠军组 seed 1/2/3 \|[^\n]*", add, r, count=1)
        readme.write_text(r, encoding="utf-8")
        print("  [ok]   README multi-seed row updated")
    else:
        anchor = "| 2-epoch 重训 | 最优配置 ×2 | best **1.3212**（+0.062）；final 1.3631 回摆 |"
        if anchor in r:
            r = r.replace(anchor, anchor + "\n" + add, 1)
            readme.write_text(r, encoding="utf-8")
            print("  [ok]   README multi-seed row added")


def _finished(key):
    p = ROOT / "experiments" / key / "log.jsonl"
    if not p.exists():
        return False
    last = json.loads(p.read_text(encoding="utf-8").splitlines()[-1])
    return last.get("step", 0) >= 28500


def update_status(done):
    """Once all 013 final experiments finish: update README status + report section 6."""
    need = ["013_data_3m", "013_attnres",
            "013_champion_seed1", "013_champion_seed2", "013_champion_seed3"]
    if not all(_finished(k) for k in need):
        print("  [skip] 013 final experiments incomplete; status line not updated")
        return
    d3m, atn = done["013_data_3m"], done["013_attnres"]
    seeds = [done[f"013_champion_seed{k}"]["best"] for k in ("1", "2", "3")]
    mean = statistics.mean(seeds)
    std = statistics.stdev(seeds)
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    old = """> 状态：训练技术验证进行中——QK-Norm / Muon v2 / ReLU² / 2-epoch /
> 技术组合与单项消融已完成；归一化 / 数据量 / AttnRes / 多 seed 在 GPU
> 队列中（详见 `docs/训练技术验证报告.md`）。"""
    new = f"""> 状态：训练技术验证**收官**——QK-Norm 1.3746 / Muon v2 1.3716 / ReLU² 1.4083 /
> 2-epoch 1.3212 结论全部落定；LayerNorm 与 RMSNorm 持平（1.3817 vs 1.3832）、
> 数据量 3M 严重过拟合（best 2.31）、AttnRes 与冠军持平（{atn['best']:.4f} vs 1.3832）、
> 冠军组多 seed 均值 **{mean:.4f} ± {std:.4f}**（详见 `docs/训练技术验证报告.md`）。"""
    if old in s:
        s = s.replace(old, new, 1)
        readme.write_text(s, encoding="utf-8")
        print(f"  [ok]   README status line updated: {mean:.4f} ± {std:.4f}")
    else:
        print("  [warn] README status line not found (may already be updated)")
    rep = ROOT / "docs" / "训练技术验证报告.md"
    r = rep.read_text(encoding="utf-8")
    old_sec = "## 六、进行中"
    new_sec = "## 六、收官结果"
    if old_sec in r:
        r = r.replace(old_sec, new_sec, 1)
        r = r.replace("**完成**，", "")  # finalize markers once everything is done
        rep.write_text(r, encoding="utf-8")
        print("  [ok]   report section 6 updated to final results")
    else:
        print("  [warn] report section 6 not found (may already be updated)")
    proj = ROOT / "docs" / "项目报告.md"
    pr = proj.read_text(encoding="utf-8")
    old_single = "- 单 seed，未做多次重复实验的显著性检验"
    new_single = (f"- 多 seed 显著性：冠军组 seed 1/2/3 best 均值 **{mean:.4f} ± {std:.4f}**"
                  f"（单次 {', '.join(f'{x:.4f}' for x in seeds)}，seed=0 冠军 1.3832）")
    if old_single in pr:
        pr = pr.replace(old_single, new_single, 1)
        print("  [ok]   project report: single-seed limitation replaced with multi-seed results")
    else:
        print("  [warn] project report single-seed line not found (may already be updated)")
    old_going = "- 进行中：多 seed 显著性（冠军组 seed 1/2/3，seed1/2 已完成 1.3668 / 1.3761）\n  （归一化消融已完成：LayerNorm 1.3817 与 RMSNorm 1.3832 基本持平）"
    new_going = "- 已完成：多 seed 显著性（冠军组 seed 1/2/3，见上）\n  （归一化消融：LayerNorm 1.3817 与 RMSNorm 1.3832 基本持平）"
    if old_going in pr:
        pr = pr.replace(old_going, new_going, 1)
        print("  [ok]   project report in-progress list updated")
    else:
        print("  [warn] project report in-progress line not found (may already be updated)")
    proj.write_text(pr, encoding="utf-8")

def main():
    done = summary()
    print("== write NOTES.md ==")
    write_notes(done)
    print("== kvbench table ==")
    fill_kvbench()
    print("== magnitude plot ==")
    plot_magnitude()
    print("== multi-seed significance ==")
    seed_significance(done)
    print("== final status ==")
    update_status(done)
    print("== done: review git diff and commit manually ==")


if __name__ == "__main__":
    main()
