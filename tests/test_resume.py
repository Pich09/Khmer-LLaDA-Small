"""Checkpoint save/reload must be bit-exact for the model, and the data loader must
resume at the same position."""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada import Config, KhmerLLaDA
from training import checkpoint as ckpt
from training.dataset import PackedShards, ResumableLoader


def _make_shards(d, n_shards=3, seqs=50, seq_len=16):
    for s in range(n_shards):
        arr = (np.arange(seqs * seq_len).reshape(seqs, seq_len) + s * 1000) % 8000
        np.save(os.path.join(d, f"train_{s:04d}.npy"), arr.astype(np.uint16))


def test_model_roundtrip_bit_exact():
    torch.manual_seed(0)
    cfg = Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=128, max_position_embeddings=32, vocab_size=8000)
    m1 = KhmerLLaDA(cfg)
    opt = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    d = tempfile.mkdtemp()
    s1 = s2 = None
    try:
        _make_shards(d)
        s1 = PackedShards(os.path.join(d, "train_*.npy"), 16)
        loader = ResumableLoader(s1, batch_size=4)
        for _ in range(5):
            next(loader)
        p = os.path.join(d, "ck.pt")
        ckpt.save(p, model=m1, optimizer=opt, scaler=scaler, loader=loader,
                  step=5, tokens_seen=999, model_config={}, train_config={})

        m2 = KhmerLLaDA(cfg)
        s2 = PackedShards(os.path.join(d, "train_*.npy"), 16)
        loader2 = ResumableLoader(s2, batch_size=4)
        step, toks = ckpt.load(p, model=m2, optimizer=torch.optim.AdamW(m2.parameters(), lr=1e-3),
                               scaler=torch.amp.GradScaler("cuda", enabled=False), loader=loader2,
                               map_location="cpu")
        assert step == 5 and toks == 999
        for (n, a), (_, b) in zip(m1.named_parameters(), m2.named_parameters()):
            assert torch.equal(a, b), n
        assert torch.equal(next(loader), next(loader2))
    finally:
        if s1:
            s1.close()
        if s2:
            s2.close()
        shutil.rmtree(d, ignore_errors=True)


def test_loader_epoch_rollover():
    d = tempfile.mkdtemp()
    s = None
    try:
        _make_shards(d, n_shards=1, seqs=10, seq_len=16)
        s = PackedShards(os.path.join(d, "train_*.npy"), 16)
        loader = ResumableLoader(s, batch_size=4, seed=1)
        for _ in range(10):
            next(loader)
        assert loader.epoch >= 1  # rolled past the 10-sequence shard
    finally:
        if s:
            s.close()
        shutil.rmtree(d, ignore_errors=True)
