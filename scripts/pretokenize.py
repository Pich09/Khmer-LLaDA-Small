"""
Corpus -> packed uint16 shards for training.

Uses the `segmented` corpus file (all_text_segmented.txt) with tokenizer mode "sp_direct":
that file is already khmer-nltk word-segmented + English-masked, which is exactly the form
khmer-sp-8k expects, so a bare SentencePiece pass is correct AND fast (no khmer-nltk in the loop).

For RAW text (your own corpus), use mode "wrapper" instead — slower (khmer-nltk).

Prints TOTAL TOKENS and the <UNK> rate. TOTAL TOKENS picks the model config:
  < 1B  -> configs/small_a.json + 3-4 epochs
  1-3B  -> configs/small_b.json

  python scripts/pretokenize.py --in data/raw/all_text_segmented.txt --seq-len 512
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada.constants import EOS_ID, PAD_ID, UNK_ID  # noqa: E402
from khmer_llada.tokenizer import KhmerSPTokenizer  # noqa: E402

VAR_LEN_FRAC = 0.01  # paper §2.2: 1% of rows get a random length in [1, seq_len], PAD-filled
_var_len_rng = np.random.default_rng(0)


def apply_variable_length(arr: np.ndarray, seq_len: int, split: str) -> np.ndarray:
    """TRAIN split only: truncate a random ~1% of rows to a random length, pad the rest with
    PAD_ID. forward_process(..., pad_id=PAD_ID) then excludes those positions from masking
    and loss — lets the model see shorter-than-context sequences during pretraining, matching
    the trick Nie et al. (and this paper, §2.2) used to make generation length insensitive."""
    if split != "train":
        return arr
    n = arr.shape[0]
    pick = _var_len_rng.random(n) < VAR_LEN_FRAC
    for i in np.nonzero(pick)[0]:
        cut = int(_var_len_rng.integers(1, seq_len))  # length in [1, seq_len)
        arr[i, cut:] = PAD_ID
    return arr


def pack_and_write(buf, seq_len, out_dir, split, shard_idx):
    k = (len(buf) // seq_len) * seq_len
    if k == 0:
        return buf, shard_idx
    arr = np.asarray(buf[:k], dtype=np.uint16).reshape(-1, seq_len)
    arr = apply_variable_length(arr, seq_len, split)
    np.save(os.path.join(out_dir, f"{split}_{shard_idx:04d}.npy"), arr)
    return buf[k:], shard_idx + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw/all_text_segmented.txt")
    ap.add_argument("--out-dir", default="data/shards")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--tokenizer-mode", default="sp_direct", choices=["sp_direct", "wrapper"])
    ap.add_argument("--val-lines", type=int, default=5000, help="held out from the END (shuffled corpus)")
    ap.add_argument("--shard-sequences", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = KhmerSPTokenizer(mode=args.tokenizer_mode)

    # quick round-trip sanity
    for s in ["ខ្ញុំចូលចិត្តភាសាខ្មែរ", "GDP កើនឡើង ៥ ភាគរយ", "សួស្តី!"]:
        rt = tok.decode(tok.encode(s))
        print(f"  round-trip: {s!r} -> {rt!r}")

    lines = open(args.inp, encoding="utf-8").read().splitlines()
    splits = [("val", lines[-args.val_lines:]), ("train", lines[:-args.val_lines])]

    n_tok = n_unk = 0
    shard_cap = args.seq_len * args.shard_sequences
    for split, rows in splits:
        buf, shard_idx = [], 0
        for line in tqdm(rows, desc=f"tokenize {split}"):
            ids = tok.encode(line.strip())
            if not ids:
                continue
            buf.extend(ids)
            buf.append(EOS_ID)
            n_tok += len(ids)
            n_unk += ids.count(UNK_ID)
            if len(buf) >= shard_cap:
                buf, shard_idx = pack_and_write(buf, args.seq_len, args.out_dir, split, shard_idx)
        pack_and_write(buf, args.seq_len, args.out_dir, split, shard_idx)

    print("\n" + "=" * 50)
    print(f"TOTAL TOKENS : {n_tok:,}")
    print(f"UNK rate     : {n_unk / max(1, n_tok):.4%}")
    print(f"sequences @{args.seq_len}: {n_tok // args.seq_len:,}")
    print("=" * 50)
    if n_tok < 1_000_000_000:
        print("-> configs/small_a.json  (+ 3-4 epochs)")
    elif n_tok < 3_000_000_000:
        print("-> configs/small_b.json")
    else:
        print("-> configs/small_b.json or small_c.json")


if __name__ == "__main__":
    main()
