"""
Extract raw text from the newly-downloaded sources (nphearum/khmer-raw-text-3M-v2,
Khmer Wikipedia) into a single plain-text file, one "line" per row, ready for
pretokenize.py --tokenizer-mode wrapper (these sources are NOT pre-segmented).

  python scripts/extract_extra_corpus.py
"""
import os
import re
import unicodedata

import pandas as pd

OUT = "data/raw_extra/combined_raw.txt"
ZERO_WIDTH = re.compile("[​‌‍﻿]")
WS = re.compile(r"\s+")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH.sub("", text)
    text = WS.sub(" ", text).strip()
    return text


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_lines = 0
    n_chars = 0
    with open(OUT, "w", encoding="utf-8") as out:
        df1 = pd.read_parquet("data/raw_extra/nphearum/train.parquet")
        for t in df1["text"]:
            t = clean(str(t))
            if len(t) < 5:
                continue
            out.write(t + "\n")
            n_lines += 1
            n_chars += len(t)

        df2 = pd.read_parquet("data/raw_extra/wikipedia_km/20231101.km/train-00000-of-00001.parquet")
        for t in df2["text"]:
            t = clean(str(t))
            if len(t) < 5:
                continue
            out.write(t + "\n")
            n_lines += 1
            n_chars += len(t)

    print(f"wrote {OUT}: {n_lines:,} lines, {n_chars:,} chars")


if __name__ == "__main__":
    main()
