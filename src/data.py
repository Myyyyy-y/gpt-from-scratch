"""Training data: memmap-lazy token file view + random batch sampling."""

import numpy as np
import torch


class TokenDataset:
    """Read-only view over a .bin token file that produces random batches."""

    def __init__(self, bin_path, max_tokens=0):
        # dtype must match what prepare_data.py writes (uint16)
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        # max_tokens>0 exposes only the first N tokens (data-amount ablation)
        self.n_tokens = len(self.tokens) if max_tokens <= 0 else min(max_tokens, len(self.tokens))

    def get_batch(self, batch_size, context_length, device="cpu"):
        """Randomly sample batch_size windows of length context_length -> (x, y).

        y is x shifted right by one: the target is the next token at each position.
        """
        max_start = self.n_tokens - context_length - 1
        assert max_start > 0, "语料太短，不足以取 batch"

        starts = np.random.randint(0, max_start, size=batch_size)

        # broadcast indexing: starts[:, None] (B,1) + offsets (1,T) -> (B,T)
        offsets = np.arange(context_length)
        idx_x = starts[:, None] + offsets[None, :]
        idx_y = idx_x + 1
        x = self.tokens[idx_x].astype(np.int64)
        y = self.tokens[idx_y].astype(np.int64)

        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
