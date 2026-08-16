r"""Byte-level BPE tokenizer (GPT-2 style): train/encode/decode/save/load.

Relies on the `regex` package for Unicode property matching (\p{L}).
"""

import json
import regex as re
from collections import Counter, defaultdict

# GPT-2 / tiktoken pretokenization regex: splits text into word-like chunks
# so BPE merges never cross word boundaries.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path, vocab_size, special_tokens=None):
    """Train a byte-level BPE from a corpus file. Returns (vocab, merges).

    vocab:  {id: bytes}
    merges: [(left, right), ...] in learning order; encode() applies them in order.
    """
    special_tokens = special_tokens or []

    # split text on special tokens, keeping the tokens themselves
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        split_pat = "(" + "|".join(re.escape(st) for st in special_tokens) + ")"
        chunks = re.split(split_pat, text)
    else:
        chunks = [text]

    # pretokenize each chunk into word tuples of single bytes, count frequencies
    words = Counter()
    for chunk in chunks:
        if not chunk or chunk in special_tokens:
            continue
        for m in re.finditer(GPT2_SPLIT_PATTERN, chunk):
            w = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            words[w] += 1

    # ids 0..255 are the 256 raw bytes; special tokens follow
    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    # incremental merge loop:
    #   pair_counts: {pair: weighted count}       — most frequent pair wins
    #   pair_words:  {pair: set of words}         — inverted index for updates
    merges = []
    if not words:
        return vocab, merges

    pair_counts = Counter()
    pair_words = defaultdict(set)

    def add_word(w):
        cnt = words[w]
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_counts[p] += cnt
            pair_words[p].add(w)

    def remove_word(w):
        cnt = words[w]
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_counts[p] -= cnt
            if pair_counts[p] <= 0:
                del pair_counts[p]
            pair_words[p].discard(w)

    def merge_word(w, pair):
        new_w = []
        i = 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                new_w.append(pair[0] + pair[1])
                i += 2
            else:
                new_w.append(w[i])
                i += 1
        return tuple(new_w)

    for w in words:
        add_word(w)

    while len(vocab) < vocab_size and pair_counts:
        # tie-break by byte-string order to match the reference implementation
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        # only words containing best_pair need their pair stats refreshed
        new_word_counts = Counter()
        for w in list(pair_words[best_pair]):
            if w not in words:
                continue
            cnt = words[w]
            remove_word(w)
            del words[w]
            new_word_counts[merge_word(w, best_pair)] += cnt

        for new_w, cnt in new_word_counts.items():
            words[new_w] = words.get(new_w, 0) + cnt
            for i in range(len(new_w) - 1):
                p = (new_w[i], new_w[i + 1])
                pair_counts[p] += cnt
                pair_words[p].add(new_w)

    return vocab, merges


class BPE:
    """Byte-level BPE encoder/decoder over a trained vocab + merge list."""

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        self._id_of = {b: i for i, b in self.vocab.items()}
        self._special_id = {
            st: self._id_of[st.encode("utf-8")] for st in self.special_tokens
        }

        # precompiled split regex; longest special tokens first so prefixes match greedily
        if self.special_tokens:
            pats = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pat = "(" + "|".join(re.escape(st) for st in pats) + ")"
        else:
            self._special_pat = None

        # word -> ids cache; the same words repeat across a corpus, so this
        # turns encode() into a cache lookup for common words
        self._encode_cache = {}

    @classmethod
    def train(cls, input_path, vocab_size, special_tokens=None):
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        """str -> list of token ids."""
        ids = []
        parts = re.split(self._special_pat, text) if self._special_pat else [text]

        for part in parts:
            if not part:
                continue
            if part in self._special_id:
                ids.append(self._special_id[part])
                continue
            for m in re.finditer(GPT2_SPLIT_PATTERN, part):
                word = tuple(bytes([b]) for b in m.group().encode("utf-8"))
                cached = self._encode_cache.get(word)
                if cached is None:
                    cached = self._encode_word(word)
                    self._encode_cache[word] = cached
                ids.extend(cached)
        return ids

    def _encode_word(self, word):
        # applying merges in learning order is equivalent to repeatedly picking
        # the highest-priority pair, since merged tokens can only participate
        # in rules that were learned later
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
        # concatenate all bytes first, then utf-8 decode once: a multi-byte
        # character can span two adjacent tokens. errors="replace" guards
        # against truncated sequences during sampling.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        # bytes are not JSON-serializable; latin-1 maps byte values 0..255
        # to characters 1:1, so round-trips losslessly
        payload = {
            "vocab": {str(k): v.decode("latin-1") for k, v in self.vocab.items()},
            "merges": [[a.decode("latin-1"), b.decode("latin-1")] for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab = {int(k): v.encode("latin-1") for k, v in payload["vocab"].items()}
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in payload["merges"]]
        return cls(vocab, merges, payload.get("special_tokens", []))


if __name__ == "__main__":
    # smoke test with the CS336 reference corpus
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
