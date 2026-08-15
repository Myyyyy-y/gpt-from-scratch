"""
数据管道：下载 TinyStories → 训练/加载 BPE → 编码并缓存成 uint16 .bin

==================== 给初学者的整体说明 ====================

【为什么要这个脚本？】
训练时每一小步都要读数据，如果每次都"读文本 -> 分词"，分词的开销会远超
训练本身（参考项目的实测：编码耗时约是训练耗时的 2 倍）。所以做法是：
  离线一次性完成：文本 -> 分词 -> 整数 id 数组 -> 原样存成二进制文件（.bin）
  训练时：直接按字节偏移随机读 .bin（见 src/data.py 的 memmap），零分词开销。

【uint16 是什么？为什么用它存 id？】
uint16 = 无符号 16 位整数，范围 0~65535，每个 id 只占 2 字节。
我们的 vocab_size 远小于 65535，所以够用——比 int64（8 字节）省 4 倍空间。
（代价：词表绝不能超过 65535，下面有 assert 兜底。）

【产出文件】
  data/corpus.txt    BPE 训练用的原始语料（前 N 篇故事）
  data/bpe.json      训练好的 BPE（vocab + merges）
  data/train.bin     训练集 token id（uint16 二进制）
  data/valid.bin     验证集 token id
  data/meta.json     词表大小 / eot_id / token 数等元信息

用法：
  python scripts/prepare_data.py --out_dir data --vocab_size 8192
  复用已有 tokenizer：python scripts/prepare_data.py --tokenizer_path data/bpe.json
"""

# ---------------------------------------------------------------------------
# 【顺序很重要】HF_DATASETS_CACHE 必须在 import datasets 之前设置！
# datasets 库在 import 时就把缓存路径定死了，之后再改环境变量不生效。
# 所以这里先用一个"迷你解析器"只把 --hf_cache 提前解析出来。
# ---------------------------------------------------------------------------
import argparse
import os

_pre = argparse.ArgumentParser(add_help=False)   # add_help=False：不抢正式解析器的 --help
_pre.add_argument("--hf_cache",
                  default=os.environ.get("HF_DATASETS_CACHE",
                                         os.path.expanduser("~/hmy/hf_cache")))
_pre_args, _ = _pre.parse_known_args()           # 只认 --hf_cache，其余参数原样放行
os.environ["HF_DATASETS_CACHE"] = _pre_args.hf_cache

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from datasets import load_dataset                # HuggingFace 数据集库

# ---------------------------------------------------------------------------
# 【路径修正】直接运行 `python scripts/prepare_data.py` 时，
# Python 只会把 scripts/ 目录加进 sys.path，`from src.xxx import` 会失败。
# 这里手动把【项目根目录】（本文件的父目录的父目录）加进去。
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tokenizer import BPE                    # noqa: E402（import 不在文件顶部，是有意为之）

SPECIAL_TOKENS = ["<|endoftext|>"]              # 故事之间的分隔符，占一个独立 id

# ---------------------------------------------------------------------------
# 多进程编码的"全局状态"
# 为什么需要这个？子进程是 fork/spawn 出来的独立 Python 进程，
# 不能直接访问主进程的变量。Pool 的 initializer 机制会在每个子进程启动时
# 调一次 _init_worker，把分词器"递"给它们，存进各自的全局变量里。
# ---------------------------------------------------------------------------
_WORKER_BPE = None
_WORKER_EOT = None


def _init_worker(bpe, eot_id):
    """每个子进程启动时执行一次：接收主进程 pickle 过来的分词器。"""
    global _WORKER_BPE, _WORKER_EOT
    _WORKER_BPE = bpe
    _WORKER_EOT = eot_id


def _encode_text(text):
    """子进程的任务函数：编码一篇故事，并在末尾追加 eot（故事边界）。

    为什么要在故事之间插 <|endoftext|>？
    拼接成一条长 token 流后，模型需要一个明确的"一个故事讲完了"信号；
    采样时也能用它判断何时停止生成。
    """
    ids = _WORKER_BPE.encode(text)
    return ids + [_WORKER_EOT]


def _write_encoded(pool, texts, f):
    """并行编码一批文本，按顺序追加写入 .bin 文件，返回写入的 token 总数。

    pool.map 会把 texts 分发给各子进程，并【保持顺序】返回结果列表
    （顺序不能乱，否则打乱的只是故事间的排列，不影响训练，但保持一致更好排查）。
    np.asarray(ids, dtype=np.uint16).tobytes() 把 id 列表转成原始字节串，
    f.write 直接落盘——.bin 文件就是这么一个纯字节流，没有任何文件头。
    """
    n = 0
    for ids in pool.map(_encode_text, texts):
        f.write(np.asarray(ids, dtype=np.uint16).tobytes())
        n += len(ids)
    return n


def _iter_stories(split, max_examples=None):
    """流式遍历 TinyStories 的某个 split，逐篇产出故事文本。

    streaming=True：不下载整个数据集，像"在线视频缓冲"一样边下边读，
    几个 GB 的数据集也不会撑爆磁盘和内存。
    """
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for i, story in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        yield story["text"]


def collect_corpus(out_dir, n_stories):
    """收集前 n_stories 篇故事存成 txt，作为 BPE 的训练语料。

    已存在就直接复用（幂等设计：重复运行脚本不会重复下载）。
    注意：BPE 训练语料是 train 集的"前 N 篇"，和 train.bin 内容有重叠——
    这是标准做法（GPT-2 也在训练集上学 BPE），不算数据泄漏。
    """
    corpus_path = out_dir / "corpus.txt"
    # 【踩坑记录】判断"已存在"必须同时检查文件非空：
    # 上次运行若在下载阶段就报错退出，open() 已经创建了 0 字节的空文件，
    # 盲目复用会让 BPE 在空语料上训练——不报错，但学不到任何合并规则。
    if corpus_path.exists() and corpus_path.stat().st_size > 0:
        print(f"[*] 复用已有语料: {corpus_path}")
        return corpus_path
    print(f"[*] 下载 TinyStories train，收集 {n_stories} 篇作为 BPE 训练语料...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(_iter_stories("train", n_stories)):
            f.write(text + "\n")
            if (i + 1) % 5000 == 0:
                print(f"    ... {i + 1} 篇")
    return corpus_path


def encode_corpus_to_bin(text_path, bin_path, bpe, eot_id, workers=8):
    """本地 txt（每行一篇故事）-> .bin。供测试/小语料使用。"""
    with open(text_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n_tokens = 0
    with Pool(processes=workers, initializer=_init_worker, initargs=(bpe, eot_id)) as pool:
        with open(bin_path, "wb") as fout:       # "wb" = 二进制写模式
            n_tokens += _write_encoded(pool, lines, fout)
    return n_tokens


def build_split(split, out_path, token_budget, bpe, eot_id, max_examples=None, workers=8):
    """编码一个数据集 split（train/validation），达到 token 预算就停。

    幂等：输出文件已存在且非空则跳过（上次跑完的成品不重做）。
    注意 HF 数据集的 split 名是 "validation" 而不是 "valid"。
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        n = np.fromfile(out_path, dtype=np.uint16).size
        print(f"[*] 复用已有 {out_path.name}（{n} tokens）")
        return n
    print(f"[*] 编码 {split} -> {out_path.name}（预算 {token_budget} tokens）")
    n_tokens = 0
    texts = []
    with Pool(processes=workers, initializer=_init_worker, initargs=(bpe, eot_id)) as pool:
        with open(out_path, "wb") as f:
            for i, text in enumerate(_iter_stories(split, max_examples)):
                texts.append(text)
                if len(texts) >= 2000:
                    n_tokens += _write_encoded(pool, texts, f)
                    texts = []
                    print(f"    {split}: {n_tokens} tokens")
                    if n_tokens >= token_budget:
                        break
            if texts:                            # 循环结束还剩零头，收尾
                n_tokens += _write_encoded(pool, texts, f)
    print(f"[✓] {split}: {n_tokens} tokens")
    return n_tokens


def main():
    ap = argparse.ArgumentParser(parents=[_pre])  # 继承前面解析过的 --hf_cache
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--tokenizer_path", default=None, help="已有 bpe.json；不填则训练新的")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--bpe_sample_stories", type=int, default=20000)
    ap.add_argument("--train_token_budget", type=int, default=3_000_000)
    ap.add_argument("--valid_token_budget", type=int, default=150_000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # uint16 的硬上限：词表超过 65535 会静默溢出（id 被截断成错误值），
    # 必须在这里拦住。
    assert args.vocab_size <= 65535, "vocab_size 超过 uint16 上限 65535！"

    # ===== 第 1 步：准备分词器（有现成的就加载，没有就训练新的）=====
    if args.tokenizer_path and Path(args.tokenizer_path).exists():
        print(f"[*] 复用 tokenizer: {args.tokenizer_path}")
        bpe = BPE.load(args.tokenizer_path)
    else:
        corpus_path = collect_corpus(out_dir, args.bpe_sample_stories)
        print(f"[*] 训练 BPE（vocab_size={args.vocab_size}）...")
        bpe = BPE.train(str(corpus_path), args.vocab_size, SPECIAL_TOKENS)
        tok_path = out_dir / "bpe.json"
        bpe.save(str(tok_path))
        print(f"[✓] tokenizer 已保存: {tok_path}")

    # 特殊 token 的 id：训练时放在词表第 256 位（见 tokenizer.py）
    eot_id = bpe.encode(SPECIAL_TOKENS[0])[0]
    print(f"    vocab_size={len(bpe.vocab)}  eot_id={eot_id}")

    # ===== 第 2 步：并行编码 train / validation 两个 split =====
    # （HF 上 TinyStories 的验证集叫 "validation"，不是 "valid"）
    build_split("train", out_dir / "train.bin", args.train_token_budget,
                bpe, eot_id, workers=args.workers)
    build_split("validation", out_dir / "valid.bin", args.valid_token_budget,
                bpe, eot_id, workers=args.workers)

    # ===== 第 3 步：写元信息 + 健全性检查 =====
    # meta.json：训练脚本从这里读 vocab_size / eot_id，不用重新加载分词器。
    meta = {
        "vocab_size": len(bpe.vocab),
        "eot_id": eot_id,
        "train_tokens": np.fromfile(out_dir / "train.bin", dtype=np.uint16).size,
        "valid_tokens": np.fromfile(out_dir / "valid.bin", dtype=np.uint16).size,
        "special_tokens": SPECIAL_TOKENS,
        "source": "roneneldan/TinyStories",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # sanity check：把 valid.bin 开头 200 个 id 解码回文字，肉眼确认可读。
    # 【别小看这一步】参考项目靠这类检查发现过"train/valid 用了不同词表"
    # 的隐蔽 bug——那种 bug 不报错，只会让 val loss 莫名其妙地高。
    arr = np.fromfile(out_dir / "valid.bin", dtype=np.uint16)
    print("\n[sanity] valid 前 200 token 解码回文本：")
    print(bpe.decode(arr[:200].tolist()))
    print("\n完成！训练时读 data/meta.json 拿 eot_id 等参数。")


if __name__ == "__main__":
    main()
