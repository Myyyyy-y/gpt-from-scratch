r"""
字节级 BPE 分词器（Byte-level Byte-Pair Encoding）

==================== 给初学者的整体说明 ====================

【分词器是干什么的？】
模型不认识文字，只认识数字。分词器（tokenizer）负责两件事：
  encode: "Once upon a time"  ->  [13740, 642, 263, 640]   （文字 -> 数字 id 序列）
  decode: [13740, 642, 263, 640]  ->  "Once upon a time"   （数字 id 序列 -> 文字）

【BPE 的核心思想】
1. 先把每个字拆成"字节"（byte）。任何字符（包括中文、emoji）用 UTF-8 编码后
   都是 1~4 个字节，所以 256 种字节就能表示世界上所有文字 —— 这叫"无 OOV"
   （OOV = out of vocabulary，词表外词。按词切的分词器会遇到没见过的词，
   字节级永远不会有这个问题）。
2. 但一个字一个字节太碎了（"hello" 要 5 个 token），效率低。
   BPE 的做法：统计语料里哪些"相邻字节对"出现得最频繁，把它合并成一个新 token。
   比如 (b'l', b'o') 出现最多 -> 合并成 b'lo' 存入词表；下一轮 (b'lo', b'w')
   出现最多 -> 合并成 b'low'……如此反复，直到词表达到目标大小。
   最终常见单词（甚至常见词组）都会变成单个 token。

【本文件的三个部分】
  1. GPT2_SPLIT_PATTERN：预分词正则（先把文本切成"单词块"）
  2. train_bpe()：从语料学习词表 vocab 和合并规则 merges
  3. BPE 类：拿着学好的 vocab/merges 做 encode / decode / 保存 / 加载

依赖：regex 库（Python 内置 re 不支持 \p{L} 这种 Unicode 类别匹配）。
"""

import json
import regex as re            # 第三方正则库，支持 \p{L}（任意语言的字母）
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# GPT-2 / tiktoken 官方预分词正则
#
# 为什么不直接对整段文本做 BPE，而要先"预分词"成单词块？
#   如果跨单词合并，会学出 "e t"（两个单词交界处的空格）这种没意义的 token。
#   所以先用正则把文本切成一块一块（大致=单词），BPE 只在块内部合并。
#
# 正则各分支含义（| 表示"或"，从左到右优先匹配）：
#   '(?:[sdmt]|ll|ve|re)   英文缩写后缀：'s 't 'd 'm 'll 've 're（如 don't -> don + 't）
#    ?\p{L}+               可选空格 + 一串字母（\p{L} = 任何语言的字母，含中文）
#    ?\p{N}+               可选空格 + 一串数字
#    ?[^\s\p{L}\p{N}]+     可选空格 + 一串标点/符号（非空白、非字母、非数字）
#   \s+(?!\S)             末尾空白（(?!\S) 是"后面不是非空白"的否定前瞻：
#                         用来把"单词前的空格"留给下一个词，而句尾空格单独成块）
#   \s+                   兜底：其余空白（换行、制表符等）
# ---------------------------------------------------------------------------
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path, vocab_size, special_tokens=None):
    """从语料文件训练字节级 BPE，返回 (vocab, merges)。

    参数：
      input_path:    纯文本语料路径（utf-8 编码）
      vocab_size:    目标词表大小（含 256 个基础字节和特殊 token）
      special_tokens: 特殊 token 列表，如 ["<|endoftext|>"]。
                      它们直接占一个 id，且作为硬边界——BPE 合并不允许跨越它。

    返回：
      vocab:  {id(int): 字节串(bytes)}，例如 {0: b'\\x00', ..., 256: b'<|endoftext|>', 257: b'lo'}
      merges: [(左字节串, 右字节串), ...]，按学习先后顺序排列。
              顺序很重要！encode 时要按同样顺序应用合并规则。
    """
    special_tokens = special_tokens or []

    # ===== 第 1 步：读入全文，按特殊 token 切成若干 chunk ===================
    # re.split 的模式里有捕获组 (…)，所以切分后特殊 token 本身也会留在结果里。
    # 例如 "abc<|endoftext|>def" -> ["abc", "<|endoftext|>", "def"]
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        # re.escape：把 <|endoftext|> 里的 | 等特殊字符转义成普通字符，
        # 否则 | 会被正则解释成"或"。
        split_pat = "(" + "|".join(re.escape(st) for st in special_tokens) + ")"
        chunks = re.split(split_pat, text)
    else:
        chunks = [text]

    # ===== 第 2 步：预分词 + 词频统计 =======================================
    # 把每个 chunk 用 GPT-2 正则切成"单词块"，每个块转成"单字节组成的元组"。
    # 例："low" -> utf-8 字节 [108, 111, 119] -> (b'l', b'o', b'w')
    #
    # 为什么用元组不用列表？字典的键必须"可哈希"（不可变），列表可变不能当键。
    #
    # words 是一个计数器：{单词元组: 出现次数}。
    # 注意我们统计的是"去重后的单词"——语料里 "the" 出现 100 万次，
    # 也只存一条 (b't',b'h',b'e'): 1000000。这让后面的合并循环快几个数量级。
    words = Counter()
    for chunk in chunks:
        if not chunk or chunk in special_tokens:
            continue                      # 跳过空串和特殊 token 本身
        for m in re.finditer(GPT2_SPLIT_PATTERN, chunk):
            # m.group() 是匹配到的字符串；encode("utf-8") 得到字节序列；
            # bytes([b]) 把单个整数变成长度为 1 的 bytes 对象（如 b'l'）
            w = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            words[w] += 1

    # ===== 第 3 步：初始化词表 ==============================================
    # id 0..255 固定对应 256 个单字节（这是"永远不会遇到生僻字"的根基）
    vocab = {i: bytes([i]) for i in range(256)}
    # 特殊 token 紧接着占 id（256 开始）
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    # ===== 第 4 步：BPE 主循环——反复合并最高频的相邻 pair ==================
    merges = []
    if not words:                         # 防御：空语料直接返回基础词表
        return vocab, merges

    # 【性能关键】如果每轮合并都把全文重新扫一遍，复杂度是 O(轮数 × 全文)，
    # 在真实语料上要跑几天。这里用两个辅助结构做"增量更新"：
    #   pair_counts: {pair: 总出现次数}          —— 谁出现最多？
    #   pair_words:  {pair: {包含它的单词集合}}   —— 倒排索引：这个 pair 在哪些单词里？
    # 每轮合并只需更新"包含被合并 pair 的单词"，其他单词完全不用碰。
    pair_counts = Counter()
    pair_words = defaultdict(set)

    def add_word(w):
        """把单词 w 里所有相邻 pair 登记进 pair_counts / pair_words。"""
        cnt = words[w]                    # 这个单词在语料中出现的次数
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])          # 相邻两个字节组成 pair
            pair_counts[p] += cnt         # 加权计数：出现 N 次的单词，pair 也计 N 次
            pair_words[p].add(w)

    def remove_word(w):
        """把单词 w 的贡献从 pair_counts / pair_words 中撤销。"""
        cnt = words[w]
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_counts[p] -= cnt
            if pair_counts[p] <= 0:
                del pair_counts[p]        # 计数归零就删掉，防止字典越攒越大
            pair_words[p].discard(w)      # discard：不存在也不报错（remove 会报）

    def merge_word(w, pair):
        """把单词 w 中所有（不重叠的）pair 合并成一个新字节串，返回新单词。

        例：w = (b'l', b'o', b'w')，pair = (b'l', b'o')
            -> (b'lo', b'w')
        "不重叠"指 (b'a', b'a', b'a') 合并 (b'a',b'a') 得到 (b'aa', b'a')，
        合并完 i 直接跳 2 格，不会用合并结果继续参与本轮匹配。
        """
        new_w = []
        i = 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                new_w.append(pair[0] + pair[1])   # 字节串相加 = 拼接：b'l'+b'o' = b'lo'
                i += 2
            else:
                new_w.append(w[i])
                i += 1
        return tuple(new_w)

    # 初始登记：把所有单词的 pair 统计一遍（整个训练过程只有这一次全量扫描）
    for w in words:
        add_word(w)

    # 主循环：词表没满且还有 pair 可合并，就一轮一轮合并
    while len(vocab) < vocab_size and pair_counts:
        # --- 选本轮最优 pair ---
        # max 的比较规则是一个元组 (出现次数, pair 本身)：
        # 先比次数；次数相同再比字节串字典序（取较大者）。
        # 这个"平局取字典序"的规则是为了和 CS336 官方参考实现对齐，保证可复现。
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        merges.append(best_pair)                          # 记录合并规则（顺序即优先级）
        vocab[len(vocab)] = best_pair[0] + best_pair[1]   # 新 token 入词表

        # --- 增量更新：只处理包含 best_pair 的单词 ---
        # list(...) 先快照一份再遍历，因为循环体里会修改 pair_words 这个集合，
        # 边遍历边修改会抛 RuntimeError。
        new_word_counts = Counter()
        for w in list(pair_words[best_pair]):
            if w not in words:
                continue                    # 可能已被前面某个单词的合并顺带处理过
            cnt = words[w]
            remove_word(w)                  # 1. 撤销旧单词的全部 pair 贡献
            del words[w]
            new_word_counts[merge_word(w, best_pair)] += cnt   # 2. 合并得到新单词

        # 3. 把合并后的新单词重新登记回去
        #    （不同旧单词可能合并成同一个新单词，所以先按 new_word_counts 聚合）
        for new_w, cnt in new_word_counts.items():
            words[new_w] = words.get(new_w, 0) + cnt
            for i in range(len(new_w) - 1):
                p = (new_w[i], new_w[i + 1])
                pair_counts[p] += cnt
                pair_words[p].add(new_w)

    return vocab, merges


class BPE:
    """字节级 BPE 编码器/解码器：拿着训练好的 vocab/merges 干活。"""

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)          # id(int) -> bytes
        self.merges = list(merges)        # [(left, right), ...] 按学习顺序
        self.special_tokens = list(special_tokens or [])

        # 反查表：bytes -> id。encode 最后一步要把字节串换成数字 id。
        self._id_of = {b: i for i, b in self.vocab.items()}
        # 特殊 token 字符串 -> id，如 {"<|endoftext|>": 256}
        self._special_id = {
            st: self._id_of[st.encode("utf-8")] for st in self.special_tokens
        }

        # 预先编译好特殊 token 的切分正则（否则每次 encode 都重新拼，白花时间）。
        # 按长度降序排：如果特殊 token 互为前缀（如 "<|end|>" 和 "<|endoftext|>"），
        # 正则从左到右匹配，长的在前才能保证匹配到长的那个。
        if self.special_tokens:
            pats = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pat = "(" + "|".join(re.escape(st) for st in pats) + ")"
        else:
            self._special_pat = None

        # 单词 -> id 序列 的缓存。
        # 语料里同一个单词会被 encode 成千上万次（"the" 出现无数次），
        # 缓存后每个不同的单词只真正算一次，编码整个语料快很多。
        self._encode_cache = {}

    @classmethod
    def train(cls, input_path, vocab_size, special_tokens=None):
        """便捷构造：BPE.train("corpus.txt", 8192, ["<|endoftext|>"]) 一步到位。"""
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        """字符串 -> id 列表。"""
        ids = []
        # 先按特殊 token 切开，普通文本和特殊 token 分开处理
        parts = re.split(self._special_pat, text) if self._special_pat else [text]

        for part in parts:
            if not part:
                continue
            if part in self._special_id:
                ids.append(self._special_id[part])   # 特殊 token：直接映射成它的 id
                continue
            # 普通文本：预分词成单词块，逐块编码
            for m in re.finditer(GPT2_SPLIT_PATTERN, part):
                word = tuple(bytes([b]) for b in m.group().encode("utf-8"))
                cached = self._encode_cache.get(word)
                if cached is None:                          # 第一次见这个单词才算
                    cached = self._encode_word(word)
                    self._encode_cache[word] = cached
                ids.extend(cached)
        return ids

    def _encode_word(self, word):
        """对单个预分词后的单词（单字节元组）应用全部 merge 规则。

        做法：按 merges 列表的顺序（= 训练时的学习顺序 = 优先级从高到低），
        每条规则扫一遍当前 token 序列，能合就合。

        为什么"按顺序逐条应用"等价于"每步找当前优先级最高的 pair"？
        因为合并只会产生"更新的 token"，而更新 token 参与的 pair 对应的规则
        一定排在更后面——所以一条规则处理完后，不可能再有它的实例出现。
        这是原始 BPE 论文实现的标准做法。
        """
        tokens = list(word)
        for left, right in self.merges:
            merged = left + right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self._id_of[t] for t in tokens]

    def decode(self, ids):
        """id 列表 -> 字符串。

        两点关键：
        1. 先把所有 token 的字节串拼接起来，再一次性 utf-8 解码。
           因为一个汉字是 3 个字节，可能被拆在相邻两个 token 里，
           逐 token 解码会把汉字切碎报错。
        2. errors="replace"：模型采样生成时可能输出"半个汉字"这种非法字节
           （比如序列被 max_new_tokens 截断），replace 会把坏字节换成 �
           而不是直接抛异常把程序搞崩。
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        """保存到 json。bytes 不能直接 json 序列化，
        先用 latin-1 解码成 str（latin-1 是字节值 0~255 到字符的一一映射，
        保证无损往返），load 时再 encode 回 bytes。"""
        payload = {
            "vocab": {str(k): v.decode("latin-1") for k, v in self.vocab.items()},
            "merges": [[a.decode("latin-1"), b.decode("latin-1")] for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """从 json 加载（save 的逆过程）。"""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab = {int(k): v.encode("latin-1") for k, v in payload["vocab"].items()}
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in payload["merges"]]
        return cls(vocab, merges, payload.get("special_tokens", []))


if __name__ == "__main__":
    # 自测：用 CS336 官方小语料验证合并结果
    text = ("low low low low low lower lower widest widest widest "
            "newest newest newest newest newest newest <|endoftext|>.")
    with open("/tmp/bpe_corpus.txt", "w", encoding="utf-8") as f:
        f.write(text)

    bpe = BPE.train("/tmp/bpe_corpus.txt", vocab_size=263,
                    special_tokens=["<|endoftext|>"])
    print("学到的合并规则：")
    for left, right in bpe.merges:
        print(f"  {left} + {right} -> {left + right}")

    ids = bpe.encode("newest low <|endoftext|>")
    print("ids:", ids)
    print("decode:", repr(bpe.decode(ids)))

    bpe.save("/tmp/bpe.json")
    bpe2 = BPE.load("/tmp/bpe.json")
    assert bpe2.encode("newest low") == bpe.encode("newest low")
    print("save/load 往返一致 ✓")
