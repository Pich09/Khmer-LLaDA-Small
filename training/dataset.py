"""
Memory-mapped loader for pre-packed uint16 shards (see scripts/pretokenize.py).

Each shard is an .npy of shape (n_sequences, seq_len). No padding (packed).
ResumableLoader restores the exact data position from a checkpoint so a T4 run
that dies mid-epoch continues where it left off.
"""
import glob

import numpy as np
import torch


class PackedShards:
    def __init__(self, shard_glob: str, seq_len: int):
        self.paths = sorted(glob.glob(shard_glob))
        if not self.paths:
            raise FileNotFoundError(f"no shards match {shard_glob}")
        self.arrays = [np.load(p, mmap_mode="r") for p in self.paths]
        for a in self.arrays:
            assert a.ndim == 2 and a.shape[1] == seq_len, f"bad shard shape {a.shape}"
        self.lengths = [a.shape[0] for a in self.arrays]
        self.cum = np.cumsum([0] + self.lengths)
        self.total = int(self.cum[-1])
        self.seq_len = seq_len

    def __len__(self):
        return self.total

    def get(self, idx: int) -> torch.Tensor:
        s = int(np.searchsorted(self.cum, idx, side="right") - 1)
        row = self.arrays[s][idx - self.cum[s]]
        return torch.from_numpy(np.asarray(row, dtype=np.int64))

    def close(self):
        """Release memory-mapped file handles (needed before deleting shards on Windows)."""
        for a in self.arrays:
            mm = getattr(a, "_mmap", None)
            if mm is not None:
                mm.close()
        self.arrays = []


class ResumableLoader:
    """Infinite iterator of (B, seq_len) int64 batches with a per-epoch shuffle."""

    def __init__(self, shards: PackedShards, batch_size: int, seed: int = 42,
                 epoch: int = 0, pos: int = 0):
        self.shards = shards
        self.bs = batch_size
        self.seed = seed
        self.epoch = epoch
        self.pos = pos
        self._build_order()

    def _build_order(self):
        g = np.random.default_rng(self.seed + self.epoch)
        self.order = g.permutation(len(self.shards))

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        if self.pos + self.bs > len(self.shards):
            self.epoch += 1
            self.pos = 0
            self._build_order()
        idx = self.order[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return torch.stack([self.shards.get(int(i)) for i in idx])

    def state_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, s: dict, restore_seed: bool = True):
        """restore_seed=False keeps this loader's own (e.g. rank-distinct) seed and only
        restores epoch/pos — used under DDP so every rank keeps a different data stream
        across a --resume instead of all collapsing onto rank 0's saved seed."""
        if restore_seed:
            self.seed = s["seed"]
        self.epoch = s["epoch"]
        self.pos = s["pos"]
        self._build_order()


def fixed_val_batches(shard_glob: str, seq_len: int, batch_size: int, n_batches: int,
                      seed: int = 0):
    """Deterministic list of val batches — identical across every checkpoint eval."""
    shards = PackedShards(shard_glob, seq_len)
    g = np.random.default_rng(seed)
    idx = g.choice(len(shards), size=min(n_batches * batch_size, len(shards)), replace=False)
    out = []
    for i in range(0, len(idx) - batch_size + 1, batch_size):
        out.append(torch.stack([shards.get(int(j)) for j in idx[i:i + batch_size]]))
    return out
