"""
One-off: apply the 1% variable-length trick (paper §2.2) to TRAIN shards that were already
written before this feature existed, so we don't have to re-tokenize 400M+ tokens. Truncates
a random ~1% of rows in each shard to a random length and PAD-fills the remainder; leaves
val_*.npy untouched (val is meant to be held out and unchanged for comparability).

Idempotency: running this twice would double-apply truncation, which is harmless (a row
already PAD-tailed just gets a >= cut, no-op or shorter) but wasteful — this script tags shards
it has touched in a sidecar `.varlen_done` file per shard so re-runs are safe no-ops.

  python scripts/retrofit_variable_length.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from khmer_llada.constants import PAD_ID  # noqa: E402

VAR_LEN_FRAC = 0.01


def main():
    shard_glob = "data/shards/train_*.npy"
    paths = sorted(glob.glob(shard_glob))
    print(f"{len(paths)} train shards found")

    rng = np.random.default_rng(2)
    touched = 0
    for path in paths:
        marker = path + ".varlen_done"
        if os.path.exists(marker):
            continue
        arr = np.load(path)
        n, seq_len = arr.shape
        pick = rng.random(n) < VAR_LEN_FRAC
        n_pick = int(pick.sum())
        for i in np.nonzero(pick)[0]:
            cut = int(rng.integers(1, seq_len))
            arr[i, cut:] = PAD_ID
        np.save(path, arr)
        open(marker, "w").close()
        touched += 1
        print(f"  {os.path.basename(path)}: {n_pick}/{n} rows truncated")

    print(f"\ndone: {touched} shards updated, {len(paths) - touched} already marked done")


if __name__ == "__main__":
    main()
