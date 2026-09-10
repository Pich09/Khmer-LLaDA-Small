"""
Parallel wrapper-mode pretokenization for the new raw (unsegmented) sources
(nphearum/khmer-raw-text-3M-v2, Khmer Wikipedia), appended as additional TRAIN shards
after the existing data/shards/train_*.npy. The existing val_0000.npy (held out from the
original corpus) is left untouched, so val_nll_bound stays comparable across the whole run
history.

khmer-nltk (CRF) segmentation is slow and single-threaded; this uses multiprocessing so it
scales across cores. Each worker loads its own KhmerSPTokenizer(mode="wrapper").

  python scripts/pretokenize_extra_parallel.py --workers 48
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada.constants import EOS_ID, PAD_ID, UNK_ID  # noqa: E402

VAR_LEN_FRAC = 0.01  # paper §2.2: 1% of rows get a random length in [1, seq_len), PAD-filled
_var_len_rng = np.random.default_rng(1)  # different seed than pretokenize.py's stream is fine


def apply_variable_length(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """These new shards are all TRAIN data — see scripts/pretokenize.py::apply_variable_length
    for the full rationale. Truncates ~1% of rows to a random length, PAD-fills the rest."""
    n = arr.shape[0]
    pick = _var_len_rng.random(n) < VAR_LEN_FRAC
    for i in np.nonzero(pick)[0]:
        cut = int(_var_len_rng.integers(1, seq_len))
        arr[i, cut:] = PAD_ID
    return arr


_TOK = None  # per-worker singleton


def _init_worker():
    global _TOK
    from khmer_llada.tokenizer import KhmerSPTokenizer
    _TOK = KhmerSPTokenizer(mode="wrapper")


def _tokenize_chunk(lines):
    global _TOK
    buf = []
    n_tok = n_unk = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ids = _TOK.encode(line)
        except Exception:
            continue
        if not ids:
            continue
        buf.extend(ids)
        buf.append(EOS_ID)
        n_tok += len(ids)
        n_unk += ids.count(UNK_ID)
    return buf, n_tok, n_unk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw_extra/combined_raw.txt")
    ap.add_argument("--out-dir", default="data/shards")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--shard-sequences", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--chunk-lines", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(args.out_dir, "train_*.npy")))
    next_shard_idx = len(existing)
    print(f"existing train shards: {len(existing)} -> new ones start at train_{next_shard_idx:04d}.npy")

    lines = open(args.inp, encoding="utf-8").read().splitlines()
    print(f"{len(lines):,} lines to tokenize")

    chunks = [lines[i:i + args.chunk_lines] for i in range(0, len(lines), args.chunk_lines)]

    n_tok = n_unk = 0
    shard_idx = next_shard_idx
    buf = []
    shard_cap = args.seq_len * args.shard_sequences

    with Pool(args.workers, initializer=_init_worker) as pool:
        for chunk_buf, ctok, cunk in tqdm(pool.imap(_tokenize_chunk, chunks, chunksize=1),
                                          total=len(chunks), desc="tokenize (parallel)"):
            buf.extend(chunk_buf)
            n_tok += ctok
            n_unk += cunk
            while len(buf) >= shard_cap:
                arr = np.asarray(buf[:shard_cap], dtype=np.uint16).reshape(-1, args.seq_len)
                arr = apply_variable_length(arr, args.seq_len)
                np.save(os.path.join(args.out_dir, f"train_{shard_idx:04d}.npy"), arr)
                buf = buf[shard_cap:]
                shard_idx += 1

    # final partial shard
    k = (len(buf) // args.seq_len) * args.seq_len
    if k > 0:
        arr = np.asarray(buf[:k], dtype=np.uint16).reshape(-1, args.seq_len)
        arr = apply_variable_length(arr, args.seq_len)
        np.save(os.path.join(args.out_dir, f"train_{shard_idx:04d}.npy"), arr)
        shard_idx += 1

    print("\n" + "=" * 50)
    print(f"NEW TOKENS   : {n_tok:,}")
    print(f"UNK rate     : {n_unk / max(1, n_tok):.4%}")
    print(f"new shards   : train_{next_shard_idx:04d}.npy .. train_{shard_idx-1:04d}.npy")
    print("=" * 50)


if __name__ == "__main__":
    main()
