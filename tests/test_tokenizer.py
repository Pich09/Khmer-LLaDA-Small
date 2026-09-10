"""
Tokenizer checks. SKIPPED automatically until tokenizer/khmer_sp.model exists
(run scripts/setup.sh). See PLAN.md §6c.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer")
_HAVE_MODEL = os.path.exists(os.path.join(_TOK_DIR, "khmer_sp.model"))
pytestmark = pytest.mark.skipif(not _HAVE_MODEL, reason="tokenizer not downloaded (scripts/setup.sh)")

KHMER_SAMPLES = [
    "ខ្ញុំចូលចិត្តរៀនភាសាខ្មែរ",
    "GDP កើនឡើង ៥ ភាគរយ ក្នុងឆ្នាំនេះ",
    "តើអ្នកសុខសប្បាយជាទេ?",
    "លេខ ១២៣ និង 456",
]


def test_specials_are_fixed():
    from khmer_llada.tokenizer import KhmerSPTokenizer
    tok = KhmerSPTokenizer(mode="sp_direct")
    assert (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id, tok.mask_id) == (0, 1, 2, 3, 4)


@pytest.mark.parametrize("mode", ["wrapper", "sp_direct"])
def test_roundtrip(mode):
    from khmer_llada.tokenizer import KhmerSPTokenizer
    try:
        tok = KhmerSPTokenizer(mode=mode)
    except ImportError:
        pytest.skip("khmer_segmentation.py not available")
    for s in KHMER_SAMPLES:
        ids = tok.encode(s)
        assert all(isinstance(i, int) for i in ids)
        back = tok.decode(ids)
        # in-coverage text should round-trip; allow whitespace normalization differences
        assert back.replace(" ", "") == s.replace(" ", ""), f"{mode}: {s!r} -> {back!r}"


def test_unk_rate_small_on_sample():
    from khmer_llada.tokenizer import KhmerSPTokenizer
    tok = KhmerSPTokenizer(mode="sp_direct")
    n = u = 0
    for s in KHMER_SAMPLES:
        ids = tok.encode(s)
        n += len(ids)
        u += ids.count(tok.unk_id)
    assert u / max(1, n) < 0.02
