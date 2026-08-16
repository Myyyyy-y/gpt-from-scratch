"""Text generation (sampling) + KV-cache speed benchmark."""

import argparse
import json
import time
from pathlib import Path

import torch

from src.model import TransformerLM, ModelConfig
from src.tokenizer import BPE


def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Sample one token id from logits (1, vocab_size).

    temperature: scale logits before softmax; 0 = greedy argmax.
    top_k: keep only the k highest-scoring candidates.
    top_p: nucleus sampling over the smallest set with cumulative prob > p.
    """
    if temperature == 0:
        return int(logits.argmax())

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.numel())
        topk = torch.topk(logits, k)
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, topk.indices, topk.values)

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        remove = cumsum > top_p
        # shift the mask right by one so the token crossing p is kept
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate(model, bpe, prompt, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95,
             eot_id=None, device="cpu", use_cache=True):
    """Autoregressive generation; returns the full token id list (prompt included)."""
    model.eval()
    ids = bpe.encode(prompt)
    ids = ids[-model.config.context_length:]

    with torch.no_grad():
        if use_cache:
            logits, past_kvs = model(torch.tensor([ids], dtype=torch.long, device=device),
                                     use_cache=True)              # prefill
        else:
            logits = model(torch.tensor([ids], dtype=torch.long, device=device))
        nid = sample_next(logits[:, -1, :], temperature, top_k, top_p)
        generated = ids + [nid]
        if eot_id is not None and nid == eot_id:
            return generated

        for _ in range(1, max_new_tokens):
            if use_cache:
                x = torch.tensor([[nid]], dtype=torch.long, device=device)
                logits, past_kvs = model(x, past_kvs=past_kvs, use_cache=True)
            else:
                # recompute the whole window; truncate to context_length when long
                window = generated[-model.config.context_length:]
                logits = model(torch.tensor([window], dtype=torch.long, device=device))
            nid = sample_next(logits[:, -1, :], temperature, top_k, top_p)
            generated.append(nid)
            if eot_id is not None and nid == eot_id:
                break
    return generated


def benchmark(model, bpe, prompt, max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.95,
              eot_id=None, device="cpu", seed=0):
    """Compare generation speed with and without KV cache under identical conditions."""
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

    run(False); run(True)                     # warmup (CUDA init must not be timed)
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

    # restore the exact training-time config stored in the checkpoint
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
