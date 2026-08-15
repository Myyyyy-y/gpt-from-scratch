"""
文本生成（采样）+ KV cache 测速对比

==================== 给初学者的整体说明 ====================

【这个文件解决什么问题？】
训练好的模型只会一件事：输入一串 token，输出"下一个 token 的概率分布"。
本文件把这个能力包装成"讲故事"：
  1. 采样：从概率分布里挑下一个 token（temperature / top-k / top-p）
  2. 自回归循环：挑一个 → 接在末尾 → 再问模型，直到写完或遇到 <|endoftext|>
  3. KV cache：生成加速（每步只算新 token，不重复算历史）
  4. benchmark：同条件对比"有/无 cache"的速度（README 的 2.5× 目标）

【为什么不直接挑概率最大的 token（贪心）？】
贪心 = 永远选概率最高的。结果是文本死板、爱复读（"很好很好很好"），
而且同一 prompt 生成一万次都一模一样。采样引入可控的随机性：
概率大的更可能被选中，但小概率选项也有机会——文本才像人写的。

用法（在项目根目录）：
  python -m src.sample --ckpt experiments/001_baseline/best.pt
  python -m src.sample --ckpt experiments/001_baseline/best.pt --benchmark
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.model import TransformerLM, ModelConfig
from src.tokenizer import BPE


def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0):
    """从 logits 分布中挑一个 token id。logits: (1, vocab_size)。

    三个旋钮按顺序依次施加：
      temperature: 温度。logits / T 后再 softmax——
                   T<1 分布变尖锐（保守），T>1 变平坦（放飞），T=0 退化为贪心
      top_k:       只保留分数最高的 k 个候选，其余设为 -inf（采样时概率为 0）
      top_p:       nucleus 采样。按概率从高到低累加，恰好超过 p 就截断，
                   只在这个"核心集合"里抽。比 top-k 灵活：模型很确定时集合
                   自动变小，不确定时自动变大
    """
    # temperature=0 的语义约定为贪心（直接 argmax，不做任何随机）
    if temperature == 0:
        return int(logits.argmax())

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.numel())
        # topk 拿到前 k 个的 (值, 下标)，scatter_ 把它们填回 -inf 底板上
        # ——等价于"非前 k 的全部抹掉"
        topk = torch.topk(logits, k)
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, topk.indices, topk.values)

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        remove = cumsum > top_p
        # 掩码右移一位：保留"恰好让累计概率越过 p"的那个 token，
        # 否则可能出现全被移除的空集合
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        # 把排序后的结果映射回原始下标顺序
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    # multinomial：按 probs 定义的分布抽一次签（概率加权随机）
    return int(torch.multinomial(probs, 1).item())


def generate(model, bpe, prompt, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95,
             eot_id=None, device="cpu", use_cache=True):
    """自回归生成，返回完整 token id 列表（含 prompt）。

    use_cache=True：prefill 一次算完 prompt 并缓存 K/V，之后每步只喂 1 个新
                    token（位置由模型内部从缓存长度推断，见 model.py）
    use_cache=False：每步把整段序列重新过一遍模型（慢，作为对照组）
    """
    model.eval()
    ids = bpe.encode(prompt)
    # prompt 太长会顶爆 context_length，只保留末尾（故事开头信息可以丢）
    ids = ids[-model.config.context_length:]

    with torch.no_grad():
        if use_cache:
            logits, past_kvs = model(torch.tensor([ids], dtype=torch.long, device=device),
                                     use_cache=True)              # prefill
        else:
            logits = model(torch.tensor([ids], dtype=torch.long, device=device))
        # logits: (1, T, V)，只关心最后一个位置的分布
        nid = sample_next(logits[:, -1, :], temperature, top_k, top_p)
        generated = ids + [nid]
        if eot_id is not None and nid == eot_id:                   # 第一个就结束也要兜住
            return generated

        for _ in range(1, max_new_tokens):
            if use_cache:
                x = torch.tensor([[nid]], dtype=torch.long, device=device)
                logits, past_kvs = model(x, past_kvs=past_kvs, use_cache=True)
            else:
                # 无 cache：整段重算；超长时滑窗截断到 context_length
                window = generated[-model.config.context_length:]
                logits = model(torch.tensor([window], dtype=torch.long, device=device))
            nid = sample_next(logits[:, -1, :], temperature, top_k, top_p)
            generated.append(nid)
            if eot_id is not None and nid == eot_id:
                break
    return generated


def benchmark(model, bpe, prompt, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95,
              eot_id=None, device="cpu", seed=0):
    """同条件对比"无 cache"和"有 cache"的生成速度。

    公平性保障：
      - 同一 checkpoint、同一 prompt、同一生成长度上限
      - torch.manual_seed(seed) 固定随机源：两边生成完全相同的序列
      - 先各跑一遍 warmup（首次调用有 CUDA 初始化开销，不计时）
      - cuda 计时必须 torch.cuda.synchronize()：GPU 是异步执行的，
        不同步的话 time.time() 测到的只是"把任务丢给 GPU"的时间，不是算完的时间
    """
    def run(cached):
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = generate(model, bpe, prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                       top_k=top_k, top_p=top_p, eot_id=eot_id, device=device, use_cache=cached)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return len(out) - len(bpe.encode(prompt)), dt

    run(False); run(True)                     # warmup
    n_nc, t_nc = run(False)
    n_c, t_c = run(True)
    return {
        "new_tokens": n_c,
        "baseline_tokens_per_sec": round(n_nc / t_nc, 2),
        "kv_cache_tokens_per_sec": round(n_c / t_c, 2),
        "speedup": round((n_c / t_c) / (n_nc / t_nc), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/001_baseline/best.pt")
    ap.add_argument("--tokenizer", default="data/bpe.json")
    ap.add_argument("--meta", default="data/meta.json")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--device", default="")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    bpe = BPE.load(args.tokenizer)
    eot_id = json.loads(Path(args.meta).read_text(encoding="utf-8")).get("eot_id")

    # 从 checkpoint 恢复：model_config 存在 ckpt 里，直接还原训练时的配置
    # （三个消融开关也在其中，保证了用什么架构训的就用什么架构生成）
    ckpt = torch.load(args.ckpt, map_location=device)
    model = TransformerLM(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    if args.dtype == "bf16" and device.startswith("cuda"):
        model = model.to(dtype=torch.bfloat16)
    model.eval()

    ids = generate(model, bpe, args.prompt, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                   eot_id=eot_id, device=device)
    print("\n========== 生成结果 ==========")
    print(bpe.decode(ids))
    print("==============================\n")

    if args.benchmark:
        res = benchmark(model, bpe, args.prompt, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                        eot_id=eot_id, device=device, seed=args.seed)
        print(f"Baseline（无缓存）: {res['baseline_tokens_per_sec']} tokens/s")
        print(f"KV cache          : {res['kv_cache_tokens_per_sec']} tokens/s")
        print(f"加速比            : {res['speedup']}x")


if __name__ == "__main__":
    main()
