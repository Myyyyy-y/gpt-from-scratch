"""
数据管道测试：编码 → 写 .bin → 读回 的一致性 + batch 形状与错位关系。

==================== 给初学者的整体说明 ====================

【这些测试在防什么？】
数据管线的 bug 最阴险：不报错、训练照常跑，但喂给模型的是错数据，
loss 降不下去你还以为是模型的问题。所以这里验证两件最核心的事：
  1. 文本 -> id -> 写文件 -> 读回 -> 文本，这一路必须无损（roundtrip）；
  2. get_batch 产出的 y 必须恰好是 x 右移一位（模型训练的根基假设）。

【pytest 的两个机制】
  - tmp_path：pytest 自动提供的临时目录（每个测试一个，互不污染），
    测试里需要写文件就用它，跑完自动清理。
  - 每个 test_ 开头的函数是一个独立测试，互不影响。
"""
import numpy as np
import torch

import prepare_data as pd
from src.data import TokenDataset
from src.tokenizer import BPE

# 4 篇迷你"故事"，模拟 TinyStories 的风格
TEXTS = [
    "Once upon a time there was a little girl named Lily.",
    "She loved to play in the park with her dog.",
    "One day the dog ran away and Lily was sad.",
    "The end was happy and everyone smiled.",
]


def _make_bpe(tmp_path):
    """测试辅助：用迷你语料训练一个小 BPE，返回 (语料路径, 分词器, eot的id)。"""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(TEXTS), encoding="utf-8")
    bpe = BPE.train(str(corpus), 300, ["<|endoftext|>"])
    eot = bpe.encode("<|endoftext|>")[0]
    return corpus, bpe, eot


def test_encode_bin_roundtrip(tmp_path):
    """文本 -> 编码 -> .bin -> 读回 -> 解码，内容必须完整还原。"""
    corpus, bpe, eot = _make_bpe(tmp_path)
    bin_path = tmp_path / "train.bin"
    pd.encode_corpus_to_bin(str(corpus), str(bin_path), bpe, eot, workers=2)

    # np.fromfile 按 uint16 把二进制文件读成整数数组（memmap 的朴素版，
    # 测试里文件小，直接全读进来）
    arr = np.fromfile(bin_path, dtype=np.uint16)
    text = bpe.decode(arr.tolist())

    assert "<|endoftext|>" in text     # 故事之间确实插入了分隔符
    for t in TEXTS:
        assert t in text                # 每篇故事完整存在，没丢没乱
    # 普通文本编码不应"意外"产出 eot——eot 只能由我们显式追加
    assert eot not in bpe.encode("Once upon a time")


def test_token_dataset_batch(tmp_path):
    """get_batch 的形状和"x/y 错位一位"关系必须成立。"""
    corpus, bpe, eot = _make_bpe(tmp_path)
    bin_path = tmp_path / "train.bin"
    pd.encode_corpus_to_bin(str(corpus), str(bin_path), bpe, eot, workers=2)

    ds = TokenDataset(str(bin_path))
    x, y = ds.get_batch(4, 16, "cpu")

    assert x.shape == (4, 16)           # (batch_size, context_length)
    assert y.shape == (4, 16)
    # y 是 x 右移一位：y 去掉最后一位 == x 去掉第一位。
    # 这保证了"每个位置的预测目标都是下一个 token"——
    # 如果这条断了，模型学到的就是错误的目标，怎么训练都白搭。
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_token_dataset_max_tokens_limit(tmp_path):
    """max_tokens 生效：暴露的 token 数不超过上限，且不影响全量读取。"""
    bin_path = tmp_path / "t.bin"
    np.arange(1000, dtype=np.uint16).tofile(bin_path)
    ds = TokenDataset(str(bin_path), max_tokens=100)
    assert ds.n_tokens == 100
    full = TokenDataset(str(bin_path))
    assert full.n_tokens == 1000
