"""
Thin wrapper around Panhapich/khmer-sp-8k.

The tokenizer was trained on khmer-nltk word-segmented text, so RAW Khmer must go through
`khmer_segmentation.KhmerTokenizer` first (khmer-nltk + wordninja, guarded by the json files).
A bare SentencePieceProcessor on raw text produces cross-word token fusion.

Two modes:
  - mode="wrapper"    : use khmer_segmentation.KhmerTokenizer  (correct for RAW Khmer)
  - mode="sp_direct"  : bare SentencePiece  (ONLY valid for already-segmented text, e.g.
                        the `segmented` config of Panhapich/khmer-text-corpus)

Vendor these into ./tokenizer/ (see scripts/setup.sh):
    khmer_sp.model  khmer_segmentation.py  gazetteer.json  latin_exceptions.json

NOTE: confirm the real method names in khmer-sp-8k/USAGE.md and adjust `_encode_wrapper` /
`_decode_wrapper` if they differ (e.g. .encode_as_ids / .encode / .tokenize).
"""
import os
import sys
from typing import List

from .constants import PAD_ID, UNK_ID, BOS_ID, EOS_ID, MASK_ID

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer")


class KhmerSPTokenizer:
    def __init__(self, tokenizer_dir: str = _DEFAULT_DIR, mode: str = "wrapper"):
        self.dir = tokenizer_dir
        self.mode = mode
        self.model_path = os.path.join(tokenizer_dir, "khmer_sp.model")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"{self.model_path} not found. Run scripts/setup.sh first."
            )

        self.pad_id, self.unk_id, self.bos_id, self.eos_id, self.mask_id = (
            PAD_ID, UNK_ID, BOS_ID, EOS_ID, MASK_ID,
        )

        if mode == "wrapper":
            if tokenizer_dir not in sys.path:
                sys.path.insert(0, tokenizer_dir)
            try:
                from khmer_segmentation import KhmerTokenizer  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "khmer_segmentation.py not importable from ./tokenizer/. "
                    "Run scripts/setup.sh, or use mode='sp_direct' on pre-segmented text."
                ) from e
            gazetteer_path = os.path.join(tokenizer_dir, "gazetteer.json")
            latin_exceptions_path = os.path.join(tokenizer_dir, "latin_exceptions.json")
            self._impl = KhmerTokenizer(
                self.model_path,
                gazetteer_path,
                latin_exceptions_path if os.path.exists(latin_exceptions_path) else None,
            )
        elif mode == "sp_direct":
            import sentencepiece as spm
            self._impl = spm.SentencePieceProcessor(model_file=self.model_path)
        else:
            raise ValueError(f"unknown mode: {mode}")

    # ------------------------------------------------------------------ encode
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if self.mode == "wrapper":
            ids = self._encode_wrapper(text)
        else:
            ids = self._impl.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def _encode_wrapper(self, text: str) -> List[int]:
        impl = self._impl
        for name in ("encode", "encode_as_ids", "tokenize_to_ids", "to_ids"):
            fn = getattr(impl, name, None)
            if callable(fn):
                out = fn(text)
                # normalize to list[int]
                if out and isinstance(out[0], str):
                    continue
                return list(out)
        raise AttributeError("Could not find an id-encoding method on KhmerTokenizer; "
                             "check khmer-sp-8k/USAGE.md and edit _encode_wrapper.")

    # ------------------------------------------------------------------ decode
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            specials = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
            ids = [i for i in ids if i not in specials]
        if self.mode == "wrapper":
            for name in ("decode", "decode_ids", "detokenize", "ids_to_text"):
                fn = getattr(self._impl, name, None)
                if callable(fn):
                    return fn(ids)
            raise AttributeError("Could not find a decode method on KhmerTokenizer.")
        return self._impl.decode(ids)

    def __len__(self) -> int:
        return 8000
