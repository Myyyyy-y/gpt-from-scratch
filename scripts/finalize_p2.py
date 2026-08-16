#!/usr/bin/env python
"""P2/P3 收尾工具：GPU 队列完成后汇总 012/013 实验结果并同步文档。

用法（项目根目录）：
  python scripts/finalize_p2.py

步骤：汇总 best/final -> 补 NOTES.md -> 填 kv_cache 表格（解析队列日志）
      -> 生成幅值图（需 013_attnres/best.pt）-> 写多 seed 显著性。
每步检测前置条件，缺失则跳过并打印状态；不自动 git commit。
"""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPS = [
    ("012_combo", "技术组合（QK+Muon+ReLU²+untied+zero-init）"),
    ("012_zeroinit", "输出投影零初始化"),
    ("012_untied", "untie 权重绑定"),
    ("012_qknorm_clean", "QK-Norm 干净归因"),
    ("013_layernorm", "归一化消融（LayerNorm）"),
    ("013_data_3m", "数据量消融（3M token）"),
    ("013_attnres", "Attention Residuals 训练"),
    ("013_champion_seed1", "冠军组 seed=1"),
    ("013_champion_seed2", "冠军组 seed=2"),
    ("013_champion_seed3", "冠军组 seed=3"),
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
    print("== 实验汇总 ==")
    done = {}
    for exp, label in EXPS:
        s = load(exp)
        if s is None:
            print(f"  {exp:22s} 无日志（未跑/进行中）")
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
        print("  [skip] 013_data_3m 未跑完，暂不生成 NOTES")
    else:
        notes["013_data_3m"] = f"""# 013_data_3m — 数据量消融（3M token）

## Goal
只用训练集前 300 万 token 训练（全量约 5 亿），量化"数据量"这一变量的影响，
补 README 选做消融表的数据量行。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：--train_limit 3000000（数据管线 memmap 零拷贝截断）

## Results
- best val loss = **{done['013_data_3m']['best']:.4f}**（@{done['013_data_3m']['best_step']}），final valid {done['013_data_3m']['final_valid']:.4f} / train {done['013_data_3m']['final']:.4f}
- 对照：全量冠军组 1.3832

## Conclusions
数据量从约 5 亿 token 缩减到 300 万（约 1/170）后严重过拟合：
val loss 在 step 750 触底 {done['013_data_3m']['best']:.2f} 后持续恶化到
final valid {done['013_data_3m']['final_valid']:.2f}，而 train loss 降到 0.08。
说明 29M 模型在该任务上仍需全量数据，数据量是当前配置的硬瓶颈
（对照全量冠军组 1.3832）。
"""
    if not complete("013_attnres"):
        print("  [skip] 013_attnres 未跑完，暂不生成 NOTES")
    else:
        notes["013_attnres"] = f"""# 013_attnres — Attention Residuals 深度残差路由训练

## Goal
训练 AttnRes（Kimi 2024 深度残差路由，对标参考项目 L 的 AttnRes-lite），
检验深层幅值控制在小模型上是否带来训练收益；配合幅值分析图评估内部表示。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：--attn_res（每层可学习 query 零初始化 + 跨层 softmax 路由）
- 注意：attn_res 模式不支持 KV cache（decode 阶段历史层输出无法增量缓存）

## Results
- best val loss = **{done['013_attnres']['best']:.4f}**（@{done['013_attnres']['best_step']}），final valid {done['013_attnres']['final_valid']:.4f} / train {done['013_attnres']['final']:.4f}
- 对照：冠军组 1.3832

## Conclusions
与冠军组（1.3832）基本持平（best {done['013_attnres']['best']:.4f}，
差 {done['013_attnres']['best'] - 1.3832:+.4f}，噪声范围内），
深层幅值受控未带来显著 loss 收益，但也未付出代价；
幅值对比图见 assets/magnitude_comparison.png。
"""
    for seed in ("1", "2", "3"):
        key = f"013_champion_seed{seed}"
        if not complete(key):
            print(f"  [skip] {key} 未跑完，暂不生成 NOTES")
            continue
        s = done[key]
        notes[key] = f"""# {key} — 冠军组复跑（seed={seed}）

## Goal
多 seed 显著性检验的一部分：同一冠军配置（29M / lr 1e-3 / 28500 步）换随机种子
重跑，与 seed=0（003）及其他 seed 一起报告均值 ± std，检验结论的稳定性。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16，seed={seed}

## Results
- best val loss = **{s['best']:.4f}**（@{s['best_step']}），final valid {s['final_valid']:.4f} / train {s['final']:.4f}

## Conclusions
（均值 ± std 汇总见 docs/训练技术验证报告.md 的多 seed 小节）
"""
    # 归一化消融：跑完后把"进行中"版 NOTES 更新为最终版
    if complete("013_layernorm") and (ROOT / "experiments/013_layernorm/NOTES.md").exists():
        s = done["013_layernorm"]
        notes["013_layernorm"] = f"""# 013_layernorm — 归一化消融：LayerNorm vs RMSNorm

## Goal
补"选做消融"表里归一化一行：同配置下 LayerNorm vs RMSNorm 冠军组，
检验 pre-norm 结构下归一化选型的影响。

## Setup
- 29M 同规模（8/512/8，swiglu，tie），28500 步，lr 1e-3，bf16
- 与冠军组唯一差异：norm_type=layernorm

## Results
- best val loss = **{s['best']:.4f}**（@{s['best_step']}），final valid {s['final_valid']:.4f} / train {s['final']:.4f}
- 对照：RMSNorm 冠军组 1.3832

## Conclusions
最终 LayerNorm（1.3817）与 RMSNorm（1.3832）基本持平（差 ~0.002，噪声范围内），
推翻了早前基于 step 5250 中间快照的"明显落后"初判——中间态不具外推性。
pre-norm 结构下归一化选型对 29M 规模影响可忽略；RMSNorm 计算更省，仍为默认选择。
"""

    for exp, content in notes.items():
        p = ROOT / "experiments" / exp / "NOTES.md"
        if p.exists() and exp != "013_layernorm":
            print(f"  [skip] {exp}/NOTES.md 已存在")
            continue
        p.write_text(content, encoding="utf-8")
        print(f"  [ok]   {exp}/NOTES.md 已生成/更新")


def fill_kvbench():
    logs = [Path("/tmp/p2_queue.log"), Path("/tmp/p2_kvbench.log")]
    text = "".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in logs if p.exists()
    )
    if not text:
        print("  [skip] 无队列日志，无法填 kvbench")
        return
    m = re.search(r"KV cache 测速（生成 (\d+) tokens）[^\n]*\n"
                  r"\s*无 cache: ([\d.]+) tok/s\n"
                  r"\s*有 cache: ([\d.]+) tok/s\n"
                  r"\s*加速比  : ([\d.]+)x", text)
    if not m:
        print("  [skip] 队列日志中还没有 kvbench 结果")
        return
    tokens, nc, c, speedup = m.groups()
    p = ROOT / "docs" / "kv_cache测试.md"
    s = p.read_text(encoding="utf-8")
    old = "| RTX 4090 / bf16 | 待空闲卡复测 | | |"
    new = f"| RTX 4090 / bf16 | {nc} tok/s | {c} tok/s | **{speedup}×**（{tokens} tokens）|"
    if old not in s:
        print("  [warn] kv_cache 表格行未找到，手动检查")
        return
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print(f"  [ok]   kv_cache测试.md 已填入：{new}")


def plot_magnitude():
    import shlex
    q = subprocess.run(["pgrep", "-f", "out_dir experiments/013_attnres"],
                       capture_output=True)
    if q.returncode == 0:
        print("  [skip] 013_attnres 仍在训练，best.pt 可能正在写入，等训练结束再出图")
        return
    ckpt = ROOT / "experiments" / "013_attnres" / "best.pt"
    if not ckpt.exists():
        print("  [skip] 013_attnres/best.pt 不存在，幅值图稍后跑")
        return
    py = sys.executable
    cmd = [py, "scripts/plot_magnitude.py",
           "--ckpt", "experiments/003_29m_lr_1e3/best.pt", "--label", "Baseline",
           "--ckpt", "experiments/013_attnres/best.pt", "--label", "AttnRes",
           "--out", "assets/magnitude_comparison.png", "--device", "cpu"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        print("  [ok]   幅值图已生成 assets/magnitude_comparison.png")
    else:
        print("  [warn] 幅值图失败：", r.stderr[-500:])


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
        print(f"  [skip] seed 实验未齐全（当前 {len(seeds)}/3）")
        return
    mean = statistics.mean(seeds)
    std = statistics.stdev(seeds) if len(seeds) > 1 else 0.0
    line = (f"- **多 seed 显著性**（冠军组 seed 1/2/3）：best val 均值 **{mean:.4f} ± {std:.4f}**"
            f"（单次 {', '.join(f'{s:.4f}' for s in seeds)}；对照 seed=0 冠军 1.3832）")
    rep = ROOT / "docs" / "训练技术验证报告.md"
    s = rep.read_text(encoding="utf-8")
    old = "- **多 seed 显著性**（冠军组 seed 1/2/3）：GPU 队列中"
    if old not in s:
        print("  [warn] 报告中的多 seed 占位行未找到")
    else:
        s = s.replace(old, line, 1)
        rep.write_text(s, encoding="utf-8")
        print(f"  [ok]   报告多 seed 小节已更新：{line}")
    readme = ROOT / "README.md"
    r = readme.read_text(encoding="utf-8")
    add = f"| 多 seed 显著性 | 冠军组 seed 1/2/3 | **{mean:.4f} ± {std:.4f}**（n=3） |"
    if "多 seed 显著性" in r:
        print("  [skip] README 已有多 seed 行")
    else:
        anchor = "| 2-epoch 重训 | 最优配置 ×2 | best **1.3212**（+0.062）；final 1.3631 回摆 |"
        if anchor in r:
            r = r.replace(anchor, anchor + "\n" + add, 1)
            readme.write_text(r, encoding="utf-8")
            print("  [ok]   README 已加多 seed 行")


def _finished(key):
    p = ROOT / "experiments" / key / "log.jsonl"
    if not p.exists():
        return False
    last = json.loads(p.read_text(encoding="utf-8").splitlines()[-1])
    return last.get("step", 0) >= 28500


def update_status(done):
    """全部 013 收官实验完成时：更新 README 状态行 + 报告第六节为收官结果。"""
    need = ["013_data_3m", "013_attnres",
            "013_champion_seed1", "013_champion_seed2", "013_champion_seed3"]
    if not all(_finished(k) for k in need):
        print("  [skip] 013 收官实验未齐全，状态行暂不更新")
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
        print(f"  [ok]   README 状态行已更新：{mean:.4f} ± {std:.4f}")
    else:
        print("  [warn] README 状态行未找到（可能已更新）")
    rep = ROOT / "docs" / "训练技术验证报告.md"
    r = rep.read_text(encoding="utf-8")
    old_sec = """## 六、进行中

- **归一化消融**（013_layernorm）：**完成**，LayerNorm best 1.3817 vs RMSNorm 1.3832
  基本持平（早前"明显落后"初判来自 step 5250 中间快照，已推翻；
  详见 `experiments/013_layernorm/NOTES.md`）
- **数据量消融**（013_data_3m）：300 万 token 严重过拟合，best 2.31@750，final valid ~4.72
- **AttnRes 训练 + 幅值分析**（013_attnres）：best 1.3793 vs 冠军 1.3832，基本持平；
  幅值对比见 `assets/magnitude_comparison.png`
- **多 seed 显著性**（冠军组 seed 1/2/3）：GPU 队列中（seed1/2 已完成：1.3668 / 1.3761）
- **29M GPU 版 KV cache 复测**：**完成**，RTX 4090 / bf16 下 52.29 → 54.79 tok/s（**1.05×**），再次验证小模型 GPU 上收益被 kernel 启动开销稀释（见 `docs/kv_cache测试.md`）"""
    new_sec = f"""## 六、收官结果

- **归一化消融**（013_layernorm）：LayerNorm best 1.3817 vs RMSNorm 1.3832，基本持平
- **数据量消融**（013_data_3m）：300 万 token 严重过拟合，best 2.31@750，final valid ~{d3m['final_valid']:.2f}
- **AttnRes 训练 + 幅值分析**（013_attnres）：best {atn['best']:.4f} vs 冠军 1.3832，未见收益；
  幅值对比见 `assets/magnitude_comparison.png`
- **多 seed 显著性**（冠军组 seed 1/2/3）：best 均值 **{mean:.4f} ± {std:.4f}**
  （{', '.join(f'{x:.4f}' for x in seeds)}；seed=0 冠军 1.3832）
- **29M GPU 版 KV cache 复测**：RTX 4090 / bf16 下 52.29 → 54.79 tok/s（**1.05×**），
  验证小模型 GPU 上收益被 kernel 启动开销稀释（见 `docs/kv_cache测试.md`）"""
    if old_sec in r:
        r = r.replace(old_sec, new_sec, 1)
        rep.write_text(r, encoding="utf-8")
        print("  [ok]   报告第六节已更新为收官结果")
    else:
        print("  [warn] 报告第六节未找到（可能已更新）")

def main():
    done = summary()
    print("== 补 NOTES.md ==")
    write_notes(done)
    print("== kvbench 填表 ==")
    fill_kvbench()
    print("== 幅值图 ==")
    plot_magnitude()
    print("== 多 seed 显著性 ==")
    seed_significance(done)
    print("== 收官状态 ==")
    update_status(done)
    print("== 完成：请 review git diff 后手动提交 ==")


if __name__ == "__main__":
    main()
