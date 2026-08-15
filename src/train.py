"""
训练模块：手写 AdamW + warmup/cosine 学习率 + 梯度裁剪 + 交叉熵 + bf16 + checkpoint

==================== 给初学者的整体说明 ====================

【训练循环在干什么？】
重复几千次同一个四步动作：
  1. 取一个 batch 的 (x, y)          —— y 是 x 右移一位的"标准答案"
  2. 前向：logits = model(x)，和 y 算交叉熵损失 loss（"预测有多离谱"）
  3. 反向：loss.backward() 算出每个参数的梯度（"每个参数该往哪边调"）
  4. optimizer.step() 按梯度更新参数
每一步 loss 降一点点，几万步后模型就"学会"了语料的统计规律。

【本文件手写了哪四样东西？为什么不用 PyTorch 现成的？】
  1. AdamW       —— 优化器（PyTorch 有 torch.optim.AdamW）
  2. 学习率调度   —— warmup + cosine（PyTorch 有 lr_scheduler）
  3. 梯度裁剪     ——（PyTorch 有 clip_grad_norm_）
  4. 交叉熵损失   ——（PyTorch 有 F.cross_entropy）
"从零实现"的价值在于：这些是训练出问题的三大重灾区（lr、梯度爆炸、损失
数值不稳定），亲手写过一遍，以后调不收敛的模型时才知道去哪查。
测试里会拿 PyTorch 官方实现当"阅卷老师"逐一比对，保证手写版正确。

【工程部分】bf16 混合精度、JSONL 日志（逐行 flush）、checkpoint 断点续训、
按验证集 loss 保存最优模型、实验目录规范（config.json + log.jsonl）。
"""

import argparse
import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.data import TokenDataset
from src.model import ModelConfig, TransformerLM


# ---------- 手写 AdamW ----------

class AdamW:
    """从零实现的 AdamW 优化器。

    【直觉】SGD 是"所有参数用同一个步长"，Adam 给每个参数单独调步长：
    历史梯度大的参数走小步，历史梯度小的参数走大步。为此为每个参数
    维护两个滑动平均：
      m = 梯度的一阶矩（动量："梯度最近在往哪指"）
      v = 梯度的二阶矩（"梯度最近有多大"）
    更新量 ≈ lr * m / (sqrt(v) + eps)。

    【AdamW 和 Adam 的唯一区别】权重衰减（weight decay，防止过拟合的
    "参数别太大"惩罚）不混进梯度，而是直接对参数本身缩小：
      p = p * (1 - lr * wd)
    这叫"解耦权重衰减"。混进梯度会被自适应步长扭曲，解耦后更干净，
    实测效果更好——这就是 AdamW 取代 Adam 成为标配的原因。

    【两个细节】
    - 偏差修正（bias correction）：m 和 v 从 0 开始累积，前几步严重
      偏小（偏向 0），要除以 (1 - beta^t) 修正，否则模型刚起步就"瘸腿"。
    - norm / bias 参数不做权重衰减：归一化的缩放参数衰减了会损害
      表达能力，这是 LLaMA/CS336 的惯例。
    """

    def __init__(self, named_params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 1                      # 更新步数（从 1 开始，偏差修正要用）

        self.groups = []                # [(param, m, v, wd), ...]
        seen = set()
        for name, p in named_params:
            # 权重绑定时 lm_head.weight 和 embedding.weight 是同一个张量，
            # 用 id() 去重，防止它被注册两次、更新两遍
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            wd = 0.0 if ("norm" in name.lower() or name.endswith("bias")) else weight_decay
            self.groups.append((p, torch.zeros_like(p), torch.zeros_like(p), wd))

    def step(self):
        """对所有参数执行一步 AdamW 更新（梯度需已由 backward() 算好）。"""
        for p, m, v, wd in self.groups:
            if p.grad is None:
                continue
            grad = p.grad
            if wd != 0:
                p.data.add_(p.data, alpha=-self.lr * wd)      # 解耦权重衰减
            # 滑动平均更新（mul_/add_ 结尾带下划线 = 原地修改，省内存）
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            # 偏差修正：bc1/bc2 随 t 增大趋近 1，前几步修正力度最大
            bc1 = 1 - self.beta1 ** self.t
            bc2 = 1 - self.beta2 ** self.t
            denom = v.sqrt().div_(math.sqrt(bc2)).add_(self.eps)
            step_size = self.lr / bc1
            p.data.addcdiv_(m, denom, value=-step_size)       # p -= step_size * m/denom
        self.t += 1

    def zero_grad(self):
        """清空梯度。PyTorch 默认梯度是【累加】的，每步开始前必须清零。"""
        for p, _, _, _ in self.groups:
            if p.grad is not None:
                p.grad = None

    def state_dict(self):
        """打包优化器状态（断点续训需要：m/v 和步数 t 都得存）。"""
        return {
            "t": self.t, "lr": self.lr,
            "beta1": self.beta1, "beta2": self.beta2,
            "eps": self.eps, "weight_decay": self.weight_decay,
            "m": [m.clone() for _, m, _, _ in self.groups],
            "v": [v.clone() for _, _, v, _ in self.groups],
        }

    def load_state_dict(self, sd):
        self.t = sd["t"]
        self.lr = sd["lr"]
        self.beta1, self.beta2 = sd["beta1"], sd["beta2"]
        self.eps = sd["eps"]
        self.weight_decay = sd["weight_decay"]
        for i, (p, m, v, _) in enumerate(self.groups):
            m.copy_(sd["m"][i])
            v.copy_(sd["v"][i])


# ---------- 学习率调度：warmup + cosine ----------

def get_lr(step, max_steps, warmup_steps, max_lr, min_lr):
    """学习率随步数变化的曲线：线性 warmup -> cosine 退火 -> 恒定 min_lr。

    【为什么 warmup？】训练初期参数是随机初始化的，梯度又大又乱，
    一上来就用最大学习率容易"一脚踩飞"。前 warmup_steps 步让 lr
    从 0 线性爬到峰值，相当于先小碎步热身。

    【为什么 cosine 退火？】训练后期用大 lr 会在最优点附近来回跳动，
    按余弦曲线缓慢降到 min_lr，后期小碎步微调，收敛更精细。
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps            # 线性爬升
    if step > max_steps:
        return min_lr                                        # 退火结束后恒定
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))       # 从 1 平滑降到 0
    return min_lr + coeff * (max_lr - min_lr)


# ---------- 手写梯度裁剪 ----------

def clip_grad_norm(parameters, max_norm, eps=1e-6):
    """把所有参数的梯度【作为一个整体】计算 L2 范数，超过 max_norm 就等比缩小。

    【防什么？】偶尔某个 batch 会产生异常大的梯度（"梯度爆炸"），
    一步把模型带崩、loss 飙成 nan。裁剪相当于给训练装保险丝。

    【要点】是按"所有参数梯度拼起来的总范数"统一缩放，而不是逐个参数
    各裁各的——只缩步长、不改各参数间的相对方向。
    公式：若 norm > max_norm，则 grad *= max_norm / (norm + eps)
    """
    total_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_sq += p.grad.pow(2).sum().item()
    norm = math.sqrt(total_sq)
    if norm > max_norm:
        scale = max_norm / (norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)
    return norm                          # 返回原始范数，方便记日志观察


# ---------- 手写交叉熵损失 ----------

def cross_entropy(logits, targets):
    """手写交叉熵：logits (B, T, V) 与 targets (B, T) 的平均负对数似然。

    数学：对每个位置，loss = -log( softmax(logits)[正确token] )
    直接用 logsumexp 技巧算，避免真的构造 softmax 概率（数值更稳）：
        log_softmax(z)[c] = z[c] - (max(z) + log( Σ exp(z - max) ))
    减 max 是防 exp 溢出的标准操作（softmax 平移不变性）。

    【为什么先 .float()？】bf16 下 V 个数（几千）的 exp 求和会损失精度，
    交叉熵是损失的"最后一公里"，升到 fp32 算几乎是免费的保险。
    """
    B, T, V = logits.shape
    z = logits.reshape(B * T, V).float()
    z_max = z.max(dim=-1, keepdim=True).values
    logsumexp = z_max + torch.log(torch.exp(z - z_max).sum(dim=-1, keepdim=True))
    log_probs = z - logsumexp                              # log_softmax 的等价物
    target_logp = log_probs.gather(-1, targets.reshape(-1, 1))
    return -target_logp.mean()


# ---------- 配置 ----------

@dataclass
class TrainConfig:
    """训练的全部超参数。消融实验只改对应字段（如 pos_type / max_lr）。"""
    data_dir: str = "data"
    out_dir: str = "experiments/001_baseline"   # 实验纪律：每次实验独立目录
    # --- 模型结构（默认值与 README baseline 对齐）---
    vocab_size: int = 8192        # 会被 data/meta.json 覆盖，这里只是兜底
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1344
    context_length: int = 256
    # --- 消融开关（透传给 ModelConfig）---
    norm_type: str = "rmsnorm"    # "rmsnorm" | "layernorm"
    ffn_type: str = "swiglu"      # "swiglu" | "gelu"
    pos_type: str = "rope"        # "rope" | "learned" | "none"
    # --- 训练超参数 ---
    batch_size: int = 64          # 每步 token 数 = 64 × 256 = 16384
    max_steps: int = 5000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95           # LLM 惯例用 0.95 而不是默认 0.999（梯度噪声大）
    grad_clip: float = 1.0
    # --- 日志 / 评估 / 存档 ---
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20        # 每次评估抽 20 个 batch 取平均（单 batch 噪声大）
    save_interval: int = 1000
    seed: int = 0
    dtype: str = "bf16"           # bf16 / fp32
    device: str = ""              # 空则自动选 cuda/cpu
    resume: str = ""              # checkpoint 路径；非空则续训


# ---------- checkpoint ----------

def save_checkpoint(path, model, optimizer, cfg, step, val_loss):
    """存档 = 模型权重 + 优化器状态 + 训练进度 + 全部配置。

    优化器状态（m/v/t）必须存：只存模型权重续训，AdamW 的动量丢了，
    等于让模型"带着新习惯接着旧训练"，loss 会明显地跳一下。
    """
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "train_config": asdict(cfg),
        "model_config": asdict(model.config),
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"], ckpt.get("val_loss", float("inf"))


# ---------- 训练 ----------

def set_seed(seed):
    """固定所有随机源：同样的配置能复现同样的结果（消融实验的前提）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, valid_ds, cfg, device, dtype, use_amp):
    """在验证集上抽 eval_batches 个 batch 算平均 loss。

    三个关键装饰：
      model.eval()      切到评估模式（关掉 dropout 等训练专用行为）
      torch.no_grad()   不算梯度，省显存也更快
      多 batch 取平均   单 batch 的 val loss 噪声很大，平均后才有比较意义
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = valid_ds.get_batch(cfg.batch_size, cfg.context_length, device)
            with _autocast_ctx(device, dtype, use_amp):
                logits = model(x)
                losses.append(cross_entropy(logits, y).item())
    model.train()
    return float(np.mean(losses))


def _autocast_ctx(device, dtype, use_amp):
    """混合精度上下文。不开 AMP 时返回"什么都不做"的空上下文。

    注意坑：torch.autocast(enabled=False, dtype=torch.float32) 也会
    校验 dtype——CPU autocast 只接受 bf16，传 fp32 直接报错。
    所以不开 AMP 时必须完全绕开 autocast。
    """
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def train(cfg):
    set_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据 + meta（vocab_size 以 meta.json 为准，防止和数据管线的词表不一致）
    data_dir = Path(cfg.data_dir)
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg.vocab_size = meta["vocab_size"]
    train_ds = TokenDataset(str(data_dir / "train.bin"))
    valid_path = data_dir / "valid.bin"
    valid_ds = TokenDataset(str(valid_path)) if valid_path.exists() else None

    model_cfg = ModelConfig(
        vocab_size=cfg.vocab_size, n_layers=cfg.n_layers, d_model=cfg.d_model,
        n_heads=cfg.n_heads, d_ff=cfg.d_ff, context_length=cfg.context_length,
        norm_type=cfg.norm_type, ffn_type=cfg.ffn_type, pos_type=cfg.pos_type,
    )
    model = TransformerLM(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(model.named_parameters(), lr=cfg.max_lr,
                      betas=(cfg.beta1, cfg.beta2), eps=1e-8,
                      weight_decay=cfg.weight_decay)

    step = 0
    best_val = float("inf")
    if cfg.resume:
        step, best_val = load_checkpoint(cfg.resume, model, optimizer, device)
        print(f"[*] 从 step {step} 续训（best_val={best_val:.4f}）")

    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32
    use_amp = (dtype == torch.bfloat16) and device.startswith("cuda")

    # 【实验纪律】训练开始前先把完整配置落盘——三个月后能精确复现这次实验
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"train": asdict(cfg), "model": asdict(model_cfg),
                   "n_params": n_params}, f, ensure_ascii=False, indent=2)
    print(f"[*] 参数量: {n_params/1e6:.1f}M  设备: {device}  精度: {cfg.dtype}")

    log_file = open(out_dir / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    while step < cfg.max_steps:
        step += 1
        lr = get_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.max_lr, cfg.min_lr)
        optimizer.lr = lr                               # 手动把调度后的 lr 注入优化器

        x, y = train_ds.get_batch(cfg.batch_size, cfg.context_length, device)
        optimizer.zero_grad()
        # bf16 混合精度：autocast 内的矩阵乘自动用 bf16（快约一倍、省一半显存），
        # 归一化/损失等敏感算子框架自动保持 fp32。bf16 数值范围大，不需要
        # fp16 那套麻烦的 loss scaling。
        with _autocast_ctx(device, dtype, use_amp):
            logits = model(x)
            loss = cross_entropy(logits, y)
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0 or step == 1:
            tps = cfg.batch_size * cfg.context_length * step / max(1e-9, time.time() - t0)
            msg = {"step": step, "split": "train", "loss": round(loss.item(), 4),
                   "lr": lr, "grad_norm": round(grad_norm, 3),
                   "tokens_per_sec": round(tps)}
            log_file.write(json.dumps(msg) + "\n")
            log_file.flush()                            # 逐行 flush：中途崩溃不丢已跑数据
            print(f"step {step:>6} loss {loss.item():.4f} lr {lr:.2e} "
                  f"gnorm {grad_norm:.2f} {tps:,.0f} tok/s")

        if valid_ds is not None and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
            val_loss = evaluate(model, valid_ds, cfg, device, dtype, use_amp)
            log_file.write(json.dumps({"step": step, "split": "valid",
                                       "loss": round(val_loss, 4)}) + "\n")
            log_file.flush()
            if val_loss < best_val:
                best_val = val_loss
                # 只在变优时覆盖 best.pt：存的是"验证集上最好"的模型，
                # 不是"最后"的模型（后期可能过拟合，最后≠最好）
                save_checkpoint(out_dir / "best.pt", model, optimizer, cfg, step, best_val)
            print(f"  [eval] step {step} val_loss {val_loss:.4f} (best {best_val:.4f})")

        if step % cfg.save_interval == 0:
            save_checkpoint(out_dir / f"ckpt_{step}.pt", model, optimizer, cfg, step, best_val)

    log_file.close()
    save_checkpoint(out_dir / "final.pt", model, optimizer, cfg, step, best_val)
    print(f"训练完成：{out_dir / 'final.pt'}，best_val={best_val:.4f}")


def main():
    ap = argparse.ArgumentParser(description="训练 decoder-only Transformer（手写训练栈）")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="experiments/001_baseline")
    # 模型规模（默认 = 16M baseline；29M 用 --n_layers 8 --d_model 512 --n_heads 8）
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=1344)
    # 消融开关
    ap.add_argument("--norm_type", choices=["rmsnorm", "layernorm"], default="rmsnorm")
    ap.add_argument("--ffn_type", choices=["swiglu", "gelu"], default="swiglu")
    ap.add_argument("--pos_type", choices=["rope", "learned", "none"], default="rope")
    # 常用训练超参数
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--context_length", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--max_lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    train(TrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
