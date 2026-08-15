"""
采样与 KV cache 测试

==================== 给初学者的整体说明 ====================

【最关键的一个测试：test_generate_cache_matches_nocache】
KV cache 的全部意义是"只省计算、不改结果"。所以这个测试用贪心解码
（temperature=0，无随机性）分别跑有 cache 和无 cache 两条路径，
要求输出的 token 序列【完全相同】——哪怕 cache 里一个位置算错
（RoPE 位置偏移、mask 边界、拼接顺序），序列都会分叉，立刻被抓到。

【另一个核心测试：test_kv_cache_equivalent】
更底层：prefill + 单步 decode 拼出来的 logits，和整段一次性前向的
logits 逐元素对齐——直接验证"缓存版前向 == 朴素版前向"这个数学等式。

【易错点提醒】
BPE 词表最小也有 257（256 字节 + 1 特殊 token），模型的 vocab_size
必须用 len(bpe.vocab)，不能用拍脑袋的小数字（id 越界会崩）。
"""
import torch

from src.model import ModelConfig, TransformerLM
from src.sample import generate, sample_next
from src.tokenizer import BPE


def _tiny_bpe(tmp_path):
    """迷你分词器：语料是一句话重复 30 遍。"""
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(["once upon a time there was a little girl named lily"] * 30),
                      encoding="utf-8")
    return BPE.train(str(corpus), 300, ["<|endoftext|>"])


def _tiny_model(vocab_size):
    """迷你模型：几毫秒内能前向的小配置。"""
    torch.manual_seed(0)
    return TransformerLM(ModelConfig(vocab_size=vocab_size, n_layers=2, d_model=32,
                                     n_heads=4, d_ff=64, context_length=64)).eval()


# ---------- 采样策略 ----------

def test_sample_greedy():
    """temperature=0 必须严格等于 argmax（选分数最高的下标 1）。"""
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    assert sample_next(logits, temperature=0) == 1


def test_sample_topk():
    """top_k=2 时只能采到分数最高的两个（下标 0 和 1），永远抽不到 2、3。"""
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, 8.0, 0.0]])
    got = {sample_next(logits, temperature=1.0, top_k=2) for _ in range(50)}
    assert got <= {0, 1}


def test_sample_topp():
    """top_p=0.9：概率 99%+ 集中在前两个 token，后面的应被核采样截断。"""
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, 0.0, 0.0]])
    got = {sample_next(logits, temperature=1.0, top_p=0.9) for _ in range(50)}
    assert got <= {0, 1}


# ---------- KV cache 正确性 ----------

def test_kv_cache_equivalent():
    """prefill + 单步 decode 的 logits == 整段一次性前向的 logits。

    这直接验证了 cache 实现的三个易错点全对：
    RoPE 绝对位置、掩码对角线偏移、K/V 拼接顺序。
    """
    model = _tiny_model(vocab_size=100)
    seq = torch.randint(0, 100, (1, 20))
    with torch.no_grad():
        logits_full = model(seq)                                   # 整段算
        logits_pre, kvs = model(seq[:, :-1], use_cache=True)       # 先缓存前 19 个
        logits_new, _ = model(seq[:, -1:], past_kvs=kvs, use_cache=True)  # 再喂第 20 个
    # 最后位置（第 20 个 token）的分布，两种算法必须一致
    assert torch.allclose(logits_full[:, -1], logits_new[:, -1], atol=1e-5)
    # prefill 的最后一个位置 == 整段算的倒数第二个位置（位置 18）
    assert torch.allclose(logits_pre[:, -1], logits_full[:, -2], atol=1e-5)


def test_generate_cache_matches_nocache(tmp_path):
    """贪心生成：有/无 cache 的完整输出序列必须逐 token 相同。"""
    bpe = _tiny_bpe(tmp_path)
    model = _tiny_model(vocab_size=len(bpe.vocab))      # 用真实词表大小，防 id 越界
    prompt = "once upon a time"
    ids_c = generate(model, bpe, prompt, max_new_tokens=30, temperature=0.0,
                     top_k=0, top_p=1.0, device="cpu", use_cache=True)
    ids_n = generate(model, bpe, prompt, max_new_tokens=30, temperature=0.0,
                     top_k=0, top_p=1.0, device="cpu", use_cache=False)
    assert ids_c == ids_n
