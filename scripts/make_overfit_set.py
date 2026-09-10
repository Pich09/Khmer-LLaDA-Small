"""
Build data/overfit.npy: a few hundred short Khmer sequences at seq_len 128 for the
Milestone-1 gate (training/overfit.py).

  python scripts/make_overfit_set.py --in data/raw/all_text_segmented.txt --n 800
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada.constants import EOS_ID  # noqa: E402
from khmer_llada.tokenizer import KhmerSPTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/all_text_segmented.txt")
    ap.add_argument("--out", default="data/overfit.npy")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--tokenizer-mode", default="sp_direct", choices=["sp_direct", "wrapper"])
    args = ap.parse_args()

    tok = KhmerSPTokenizer(mode=args.tokenizer_mode)
    buf = []
    for line in open(args.inp, encoding="utf-8"):
        ids = tok.encode(line.strip())
        if ids:
            buf.extend(ids + [EOS_ID])
        if len(buf) >= args.n * args.seq_len:
            break
    k = args.n * args.seq_len
    arr = np.asarray(buf[:k], dtype=np.uint16).reshape(args.n, args.seq_len)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, arr)
    print(f"wrote {args.out}  shape {arr.shape}")


if __name__ == "__main__":
    main()
