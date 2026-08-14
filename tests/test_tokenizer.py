"""
tokenizer 测试：roundtrip / 特殊 token / 无 OOV / 保存加载 / 优化版==朴素版

==================== 测试思路说明 ====================

【最有价值的一个测试：test_optimized_matches_naive】
src/tokenizer.py 的 train_bpe 为了性能用了"增量更新"（倒排索引），
逻辑比朴素版复杂得多，很容易写错。这里实现了一个直白但慢的
_naive_train_bpe 作为"标准答案"，断言两者在同一语料上学出的
vocab 和 merges 完全一致——复杂实现只要和简单实现对得上，
基本就可以放心。

【其余测试各卡住一个关键性质】
  - test_train_bpe_expected_layout：词表布局（256 字节 + 特殊 token + 合并项）
  - test_roundtrip_unicode_emoji：  中文/emoji encode->decode 无损往返
  - test_no_oov：                   任意字符编码出的 id 都在词表范围内
  - test_special_token_preserved：  <|endoftext|> 永远是单个 id，不被拆开
  - test_save_load：                json 保存/加载后行为不变

【约定】
  - 语料用 CS336 官方小例子（low/lower/widest/newest），
    合并结果有参考实现可以对照
  - 临时语料文件用 pytest 的 tmp_path fixture，测试结束自动清理
"""
import regex as re
from collections import Counter

from src.tokenizer import BPE, GPT2_SPLIT_PATTERN, train_bpe

# CS336 官方教学语料：几个单词按不同频率重复，合并顺序是可预期的。
# 频率设计：low(5) > newest(6) > widest(3) > lower(2)，
# 高频词的 pair 会先被合并——这正是 BPE "越常见越先合并"的体现。
SAMPLE = (
    "low low low low low lower lower widest widest widest "
    "newest newest newest newest newest newest <|endoftext|>."
)


def _make_corpus(tmp_path, text):
    """把一段文本写成临时语料文件，返回路径。

    为什么用 tmp_path？pytest 会给每个测试分配一个独立的临时目录，
    测试结束自动删除——不用手动清理，多个测试并行跑也不会互相覆盖。
    """
    p = tmp_path / "corpus.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


def _naive_train_bpe(path, vocab_size, special_tokens=None):
    """朴素版 BPE 训练（只用于对照，O(merges × 语料)，慢但一眼能看懂）。

    和 src/tokenizer.py 里优化版的唯一区别：
    每一轮合并都把【全部单词】重新扫一遍、重新统计 pair 频率，
    不做任何增量更新。逻辑上和"BPE 算法定义"逐句对应，
    所以把它当作标准答案，去验证优化版的正确性。
    """
    special_tokens = special_tokens or []
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # 按特殊 token 切 chunk（和优化版完全相同的预处理）
    if special_tokens:
        pat = "(" + "|".join(re.escape(s) for s in special_tokens) + ")"
        chunks = re.split(pat, text)
    else:
        chunks = [text]

    # 预分词 + 词频统计：{单词元组: 出现次数}
    words = Counter()
    for chunk in chunks:
        if not chunk or chunk in special_tokens:
            continue
        for m in re.finditer(GPT2_SPLIT_PATTERN, chunk):
            words[tuple(bytes([b]) for b in m.group().encode("utf-8"))] += 1

    # 词表初始化：256 个单字节 + 特殊 token（与优化版相同）
    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    merges = []
    while len(vocab) < vocab_size and words:
        # 全量重扫：把每个单词的每个相邻 pair 都重新计一次数
        pair_counts = Counter()
        for w, cnt in words.items():
            for i in range(len(w) - 1):
                pair_counts[(w[i], w[i + 1])] += cnt
        if not pair_counts:
            break

        # 选最高频 pair（平局取字节串字典序较大者，与优化版/CS336 对齐）
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        # 把所有单词里的 best 都合并掉（全量重写，不做增量）
        new_words = Counter()
        for w, cnt in words.items():
            nw = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                    nw.append(best[0] + best[1])
                    i += 2        # 跳过已合并的两个，保证"不重叠"合并
                else:
                    nw.append(w[i])
                    i += 1
            new_words[tuple(nw)] += cnt
        words = new_words
    return vocab, merges


def test_train_bpe_expected_layout(tmp_path):
    """验证词表布局：256 单字节 + 1 特殊 token + N 个合并项，id 连续不重叠。"""
    path = _make_corpus(tmp_path, SAMPLE)
    special = ["<|endoftext|>"]
    vocab, merges = train_bpe(path, 263, special)
    # 256 单字节 + 1 特殊 token + 6 次合并 = 263
    assert len(vocab) == 263
    # 特殊 token 必须紧跟在 256 个字节之后，占 id 256
    assert vocab[256] == "<|endoftext|>".encode("utf-8")
    # 257 及以后都是合并产物，长度至少为 2（单字节都在 0~255）
    assert all(len(vocab[i]) >= 2 for i in range(257, 263))


def test_roundtrip_unicode_emoji(tmp_path):
    """encode 再 decode 必须原样还原——包括训练时没见过的中文和 emoji。

    注意 "Hello 世界 🎉" 这些字符在 SAMPLE 语料里根本没出现过，
    字节级 BPE 靠 256 个基础字节兜底，照样能编解码（无 OOV 的直接体现）。
    """
    path = _make_corpus(tmp_path, SAMPLE)
    bpe = BPE.train(path, 263, ["<|endoftext|>"])
    text = "Hello 世界 🎉! <|endoftext|> Nice to meet you."
    assert bpe.decode(bpe.encode(text)) == text


def test_no_oov(tmp_path):
    """任意字符编码出的 id 都不能越界（0 <= id < vocab_size）。"""
    path = _make_corpus(tmp_path, SAMPLE)
    bpe = BPE.train(path, 263, ["<|endoftext|>"])
    for ch in "abcXYZ0123!@# 中文汉字🎉":
        for tid in bpe.encode(ch):
            assert 0 <= tid < 263


def test_special_token_preserved(tmp_path):
    """特殊 token 永远是单独一个 id，绝不和普通文本合并。"""
    path = _make_corpus(tmp_path, SAMPLE)
    bpe = BPE.train(path, 263, ["<|endoftext|>"])
    assert bpe.encode("<|endoftext|>") == [256]
    assert bpe.decode(bpe.encode("hi <|endoftext|> hi")) == "hi <|endoftext|> hi"


def test_save_load(tmp_path):
    """json 保存/加载后，vocab、merges、编解码行为都必须不变。"""
    path = _make_corpus(tmp_path, SAMPLE)
    bpe = BPE.train(path, 263, ["<|endoftext|>"])
    save_path = tmp_path / "bpe.json"
    bpe.save(str(save_path))
    bpe2 = BPE.load(str(save_path))
    assert bpe2.vocab == bpe.vocab
    assert bpe2.merges == bpe.merges
    text = "low low lower newest <|endoftext|>"
    assert bpe2.decode(bpe2.encode(text)) == text


def test_optimized_matches_naive(tmp_path):
    """核心正确性测试：增量优化版和朴素版学出的 vocab/merges 必须完全相同。

    merges 顺序也要一致（用 == 比较列表）：顺序不同说明平局处理
    或更新逻辑有偏差，encode 结果就可能和参考实现不一致。
    """
    path = _make_corpus(tmp_path, SAMPLE)
    v1, m1 = train_bpe(path, 263, ["<|endoftext|>"])
    v2, m2 = _naive_train_bpe(path, 263, ["<|endoftext|>"])
    assert v1 == v2
    assert m1 == m2
