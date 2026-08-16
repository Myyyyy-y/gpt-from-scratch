"""Training stack from scratch: AdamW/Muon, warmup+cosine LR, gradient clipping,
cross-entropy, bf16 AMP, JSONL logging, and checkpointing with best-val saving."""

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


# ---------- AdamW ----------

class AdamW:
    """From-scratch AdamW with decoupled weight decay; no decay on norms/biases."""

    def __init__(self, named_params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 1

        self.groups = []                # [(param, m, v, wd), ...]
        seen = set()
        for name, p in named_params:
            # dedupe tied weights by tensor id to avoid double updates
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            wd = 0.0 if ("norm" in name.lower() or name.endswith("bias")) else weight_decay
            self.groups.append((p, torch.zeros_like(p), torch.zeros_like(p), wd))

    def step(self):
        for p, m, v, wd in self.groups:
            if p.grad is None:
                continue
            grad = p.grad
            if wd != 0:
                p.data.add_(p.data, alpha=-self.lr * wd)
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            bc1 = 1 - self.beta1 ** self.t
            bc2 = 1 - self.beta2 ** self.t
            denom = v.sqrt().div_(math.sqrt(bc2)).add_(self.eps)
            step_size = self.lr / bc1
            p.data.addcdiv_(m, denom, value=-step_size)
        self.t += 1

    def zero_grad(self):
        for p, _, _, _ in self.groups:
            if p.grad is not None:
                p.grad = None

    def state_dict(self):
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


# ---------- Muon ----------

def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration: drives singular values of G toward 1."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon:
    """Momentum + Newton-Schulz-orthogonalized update for 2D weight matrices."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.1):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.groups = [(p, torch.zeros_like(p)) for p in params]   # (param, momentum buffer)

    def step(self):
        for p, buf in self.groups:
            if p.grad is None:
                continue
            if self.weight_decay != 0:
                p.data.add_(p.data, alpha=-self.lr * self.weight_decay)
            g = p.grad
            buf.lerp_(g, 1 - self.momentum)
            d = g.lerp(buf, self.momentum) if self.nesterov else buf
            o = zeropower_via_newtonschulz5(d)
            # scale update by matrix aspect ratio so wide/tall matrices compare
            scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
            p.data.add_(o, alpha=-self.lr * scale)

    def zero_grad(self):
        for p, _ in self.groups:
            if p.grad is not None:
                p.grad = None

    def state_dict(self):
        return {"lr": self.lr, "momentum": self.momentum,
                "buf": [b.clone() for _, b in self.groups]}

    def load_state_dict(self, sd):
        self.lr = sd["lr"]
        self.momentum = sd["momentum"]
        for i, (p, b) in enumerate(self.groups):
            b.copy_(sd["buf"][i])


class HybridOptimizer:
    """Muon for hidden matrices + AdamW for embeddings/lm_head/norms/vectors."""

    def __init__(self, named_params, muon_lr=0.02, adam_lr=3e-4,
                 betas=(0.9, 0.95), weight_decay=0.1):
        muon_params, adam_named = [], []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            if p.ndim == 2 and "embedding" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adam_named.append((name, p))
        self.muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
        self.adam = AdamW(adam_named, lr=adam_lr, betas=betas, weight_decay=weight_decay)
        self._adam_max = adam_lr
        self._muon_max = muon_lr

    @property
    def lr(self):
        return self.adam.lr

    @lr.setter
    def lr(self, v):
        # scale both optimizers by the AdamW-side schedule ratio
        ratio = v / self._adam_max if self._adam_max else 1.0
        self.adam.lr = v
        self.muon.lr = self._muon_max * ratio

    def step(self):
        self.muon.step()
        self.adam.step()

    def zero_grad(self):
        self.muon.zero_grad()
        self.adam.zero_grad()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, sd):
        self.muon.load_state_dict(sd["muon"])
        self.adam.load_state_dict(sd["adam"])


# ---------- LR schedule ----------

def get_lr(step, max_steps, warmup_steps, max_lr, min_lr):
    """Linear warmup -> cosine decay -> constant min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ---------- gradient clipping ----------

def clip_grad_norm(parameters, max_norm, eps=1e-6):
    """Scale all gradients together if their global L2 norm exceeds max_norm."""
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
    return norm


# ---------- cross entropy ----------

def cross_entropy(logits, targets):
    """Mean negative log-likelihood via logsumexp, computed in fp32 for stability."""
    B, T, V = logits.shape
    z = logits.reshape(B * T, V).float()
    z_max = z.max(dim=-1, keepdim=True).values
    logsumexp = z_max + torch.log(torch.exp(z - z_max).sum(dim=-1, keepdim=True))
    log_probs = z - logsumexp
    target_logp = log_probs.gather(-1, targets.reshape(-1, 1))
    return -target_logp.mean()


# ---------- config ----------

@dataclass
class TrainConfig:
    """All training hyperparameters; ablations toggle one field at a time."""
    data_dir: str = "data"
    out_dir: str = "experiments/001_baseline"
    # --- model (defaults match the README baseline) ---
    vocab_size: int = 8192        # overridden by data/meta.json when present
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1344
    context_length: int = 256
    # --- ablation switches (forwarded to ModelConfig) ---
    norm_type: str = "rmsnorm"
    ffn_type: str = "swiglu"
    pos_type: str = "rope"
    qk_norm: bool = False
    zero_init_proj: bool = False
    attn_res: bool = False
    tie_weights: bool = True
    # --- training ---
    batch_size: int = 64
    max_steps: int = 5000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # --- logging / eval / checkpoint ---
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20
    save_interval: int = 1000
    seed: int = 0
    train_limit: int = 0          # >0: train on the first N tokens only
    dtype: str = "bf16"
    device: str = ""
    wandb_project: str = ""
    resume: str = ""
    # --- training techniques ---
    optimizer: str = "adamw"
    muon_lr: float = 0.02


# ---------- checkpoint ----------

def save_checkpoint(path, model, optimizer, cfg, step, val_loss):
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


# ---------- training ----------

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, valid_ds, cfg, device, dtype, use_amp):
    """Mean val loss over eval_batches batches (eval mode, no grad)."""
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
    # CPU autocast only accepts bf16, so skip autocast entirely when AMP is off
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def train(cfg):
    set_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg.data_dir)
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg.vocab_size = meta["vocab_size"]
    train_ds = TokenDataset(str(data_dir / "train.bin"), max_tokens=cfg.train_limit)
    valid_path = data_dir / "valid.bin"
    valid_ds = TokenDataset(str(valid_path)) if valid_path.exists() else None

    model_cfg = ModelConfig(
        vocab_size=cfg.vocab_size, n_layers=cfg.n_layers, d_model=cfg.d_model,
        n_heads=cfg.n_heads, d_ff=cfg.d_ff, context_length=cfg.context_length,
        norm_type=cfg.norm_type, ffn_type=cfg.ffn_type, pos_type=cfg.pos_type,
        qk_norm=cfg.qk_norm, zero_init_proj=cfg.zero_init_proj,
        attn_res=cfg.attn_res, tie_weights=cfg.tie_weights,
    )
    model = TransformerLM(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if cfg.optimizer == "muon":
        optimizer = HybridOptimizer(model.named_parameters(),
                                    muon_lr=cfg.muon_lr, adam_lr=cfg.max_lr,
                                    betas=(cfg.beta1, cfg.beta2),
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = AdamW(model.named_parameters(), lr=cfg.max_lr,
                          betas=(cfg.beta1, cfg.beta2), eps=1e-8,
                          weight_decay=cfg.weight_decay)

    step = 0
    best_val = float("inf")
    if cfg.resume:
        step, best_val = load_checkpoint(cfg.resume, model, optimizer, device)
        print(f"[*] resuming from step {step} (best_val={best_val:.4f})")

    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32
    use_amp = (dtype == torch.bfloat16) and device.startswith("cuda")

    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, name=out_dir.name,
                                   config=asdict(cfg), reinit=True)
            print(f"[*] W&B tracking enabled: {cfg.wandb_project}/{out_dir.name}")
        except Exception as e:
            print(f"[!] W&B unavailable, skipping: {e}")

    # dump the full config before training so experiments are reproducible later
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"train": asdict(cfg), "model": asdict(model_cfg),
                   "n_params": n_params}, f, ensure_ascii=False, indent=2)
    print(f"[*] params: {n_params/1e6:.1f}M  device: {device}  dtype: {cfg.dtype}")

    log_file = open(out_dir / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    while step < cfg.max_steps:
        step += 1
        lr = get_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.max_lr, cfg.min_lr)
        optimizer.lr = lr

        x, y = train_ds.get_batch(cfg.batch_size, cfg.context_length, device)
        optimizer.zero_grad()
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
            log_file.flush()
            if wandb_run is not None:
                wandb_run.log({"step": step, "train/loss": loss.item(),
                               "lr": lr, "train/grad_norm": grad_norm})
            print(f"step {step:>6} loss {loss.item():.4f} lr {lr:.2e} "
                  f"gnorm {grad_norm:.2f} {tps:,.0f} tok/s")

        if valid_ds is not None and (step % cfg.eval_interval == 0 or step == cfg.max_steps):
            val_loss = evaluate(model, valid_ds, cfg, device, dtype, use_amp)
            log_file.write(json.dumps({"step": step, "split": "valid",
                                       "loss": round(val_loss, 4)}) + "\n")
            log_file.flush()
            if wandb_run is not None:
                wandb_run.log({"step": step, "val/loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out_dir / "best.pt", model, optimizer, cfg, step, best_val)
            print(f"  [eval] step {step} val_loss {val_loss:.4f} (best {best_val:.4f})")

        if step % cfg.save_interval == 0:
            save_checkpoint(out_dir / f"ckpt_{step}.pt", model, optimizer, cfg, step, best_val)

    log_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    save_checkpoint(out_dir / "final.pt", model, optimizer, cfg, step, best_val)
    print(f"training finished: {out_dir / 'final.pt'}, best_val={best_val:.4f}")


def main():
    ap = argparse.ArgumentParser(description="train a decoder-only Transformer (from-scratch training stack)")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="experiments/001_baseline")
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=1344)
    # ablation switches
    ap.add_argument("--norm_type", choices=["rmsnorm", "layernorm"], default="rmsnorm")
    ap.add_argument("--ffn_type", choices=["swiglu", "gelu", "relu2"], default="swiglu")
    ap.add_argument("--pos_type", choices=["rope", "learned", "none"], default="rope")
    ap.add_argument("--qk_norm", action="store_true", help="QK-Norm (normalize Q/K before attention scores)")
    ap.add_argument("--zero_init_proj", action="store_true", help="zero-init output projections")
    ap.add_argument("--attn_res", action="store_true",
                    help="Attention Residuals deep residual routing (Kimi 2024)")
    ap.add_argument("--untie", action="store_true", help="untie embedding/lm_head weight sharing")
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--muon_lr", type=float, default=0.02)
    # training hyperparameters
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
    ap.add_argument("--train_limit", type=int, default=0,
                    help="use only the first N training tokens (data-size ablation)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="")
    ap.add_argument("--wandb_project", default="", help="non-empty enables W&B tracking")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    kw = vars(args)
    kw["tie_weights"] = not kw.pop("untie")
    train(TrainConfig(**kw))


if __name__ == "__main__":
    main()
