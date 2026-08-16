"""
训练数据读取：memmap 懒加载 + 随机 batch 采样

==================== 给初学者的整体说明 ====================

【这个文件在数据管线里的位置】
  prepare_data.py 已经把文本变成了磁盘上的 .bin 文件
  （一长串 uint16 整数，每个整数是一个 token id，例如 [262, 261, 32, 260, ...]）
  本文件负责训练时从中"随机切一小段"喂给模型。

【两个核心概念】

1. memmap（内存映射）——为什么不直接 np.load？
   语料编码后可能有几十亿个 token（好几个 GB），全读进内存可能直接爆掉。
   np.memmap 不真的读文件，而是让操作系统"按需取页"：访问哪一段，
   操作系统才把磁盘上对应的那一小块捞进内存。效果：
   代码写得像操作一个大数组，内存占用却几乎为零。
   mode="r" 表示只读（训练时绝不会修改数据，只读还能防止误写）。

2. x / y 错位一位——语言模型的训练样本怎么构造？
   语言模型学的是"给定前面的 token，预测下一个 token"。
   所以从同一位置切两段：
     x = tokens[s     : s + T]      输入：第 s 到 s+T-1 个 token
     y = tokens[s + 1 : s + T + 1]  标签：每个位置的"下一个 token"
   例：原文 [... 我 爱 北 京 ...]，若 x = [我, 爱, 北]，则 y = [爱, 北, 京]
   模型看到 x 的第 0 位"我"，就要让 y 的第 0 位"爱"的概率最大，依此类推。
   一段长度为 T 的切片同时提供了 T 个训练信号，非常高效。

【为什么每次随机抽起点，而不是按顺序读？】
随机抽样保证相邻两个 batch 的内容互不相关，梯度更稳定（SGD 的基本要求），
而且实现简单——不需要维护"读到哪了"的状态。
"""
import numpy as np
import torch


class TokenDataset:
    """对一份 .bin token 文件的只读视图，负责产出训练 batch。"""

    def __init__(self, bin_path, max_tokens=0):
        # dtype 必须和 prepare_data.py 写入时用的 dtype 完全一致（uint16），
        # 否则按错误的字节宽度解读，读出来全是乱码数字。
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        # max_tokens>0 时只暴露前 N 个 token（数据量消融用，其余内存零拷贝）
        self.n_tokens = len(self.tokens) if max_tokens <= 0 else min(max_tokens, len(self.tokens))

    def get_batch(self, batch_size, context_length, device="cpu"):
        """随机采 batch_size 段，每段长 context_length，返回 (x, y)。

        返回形状都是 (batch_size, context_length) 的 int64 张量。
        （PyTorch 的 Embedding/交叉熵要求索引是 int64，所以这里做类型转换。）
        """
        # 起点 s 必须满足 s + context_length + 1 <= n_tokens（y 要比 x 再多读一位）
        max_start = self.n_tokens - context_length - 1
        assert max_start > 0, "语料太短，不足以取 batch"

        starts = np.random.randint(0, max_start, size=batch_size)  # (B,) 个随机起点

        # 【广播索引】一次性取出整个 batch，避免 Python for 循环逐段切片。
        # 原理：starts[:, None] 形状 (B, 1)，offsets[None, :] 形状 (1, T)，
        # 相加时 NumPy 广播成 (B, T) 的索引矩阵——第 i 行就是
        # [starts[i], starts[i]+1, ..., starts[i]+T-1]。
        # 用它索引 self.tokens，直接得到 (B, T) 的结果。
        offsets = np.arange(context_length)                        # (T,)
        idx_x = starts[:, None] + offsets[None, :]                 # (B, T)
        idx_y = idx_x + 1                                          # y 整体右移一位
        x = self.tokens[idx_x].astype(np.int64)
        y = self.tokens[idx_y].astype(np.int64)

        # from_numpy 零拷贝包装成张量，再搬到目标设备（"cpu" 或 "cuda"）
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
