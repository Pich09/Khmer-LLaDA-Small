# Using the Khmer SentencePiece Model (`khmer-sp-8k`)

A practical guide to loading and using the trained tokenizer.

- **Hub repo:** [`Panhapich/khmer-sp-8k`](https://huggingface.co/Panhapich/khmer-sp-8k) (model)
- **Type:** SentencePiece **unigram**, `vocab_size = 8000`, `character_coverage = 0.9995`
- **Files:** `khmer_sp.model` (load this), `khmer_sp.vocab` (human-readable), `tokenizer_info.json` (manifest)
- **Corpus it was trained on:** [`Panhapich/khmer-text-corpus`](https://huggingface.co/datasets/Panhapich/khmer-text-corpus) — 1,959,780 sentences

---

## 1. Install

```bash
pip install sentencepiece huggingface_hub
```

## 2. Load the model

Pull `khmer_sp.model` straight from the Hub (no auth needed if the repo is public):

```python
from huggingface_hub import hf_hub_download
import sentencepiece as spm

model_path = hf_hub_download("Panhapich/khmer-sp-8k", "khmer_sp.model")
sp = spm.SentencePieceProcessor(model_file=model_path)

print("vocab size:", sp.get_piece_size())   # 8000
```

If you already have `khmer_sp.model` locally, skip the download:

```python
sp = spm.SentencePieceProcessor(model_file="khmer_sp.model")
```

## 3. Encode & decode

```python
text = "ខ្ញុំចូលចិត្តរៀនភាសាខ្មែរ"

ids    = sp.encode(text)                 # -> [5, 970, ...]  list of int ids
pieces = sp.encode(text, out_type=str)   # -> ['▁', 'ខ្ញុំចូលចិត្ត', 'រៀន', 'ភាសា', 'ខ្មែរ']
text2  = sp.decode(ids)                   # -> original string (lossless round-trip)

assert text2 == text
```

- `▁` (U+2581) marks a **word boundary / leading space** — that's normal SentencePiece output, not a bug.
- Encoding is **lossless**: `decode(encode(x)) == x` for in-coverage text.
- Segmentation is subword/whole-word: common Khmer words become a single token
  (e.g. `ប្រទេសកម្ពុជា` → 2 tokens), rare ones split into pieces.

## 4. Special tokens

The model reserves five ids. **The tokenizer does not randomly insert them** — you control them
(the one exception is `<UNK>`).

| Token | id | Appears during normal `encode()`? | What it's for |
|---|---|---|---|
| `<PAD>` | 0 | Never | Pad sequences to equal length in a batch (you add it). |
| `<UNK>` | 1 | **Automatically** | Fallback for any character outside 0.9995 coverage (emoji, rare symbols). |
| `<BOS>` | 2 | Only if requested | Sequence start — via `encode(..., add_bos=True)` or manual insert. |
| `<EOS>` | 3 | Only if requested | Sequence end — via `encode(..., add_eos=True)` or manual insert. |
| `<MASK>` | 4 | Never on its own | Masking slot for the diffusion decoder (you insert it at the id level). |

Look ids up dynamically instead of hardcoding — the manifest is authoritative:

```python
PAD  = sp.piece_to_id("<PAD>")   # 0
UNK  = sp.piece_to_id("<UNK>")   # 1
BOS  = sp.piece_to_id("<BOS>")   # 2
EOS  = sp.piece_to_id("<EOS>")   # 3
MASK = sp.piece_to_id("<MASK>")  # 4
```

### Adding BOS/EOS

```python
ids = sp.encode(text, add_bos=True, add_eos=True)   # [2, ..., 3]
```

### `<UNK>` behaviour

```python
sp.encode("😀 hello")   # -> [..., 1, ...]  the emoji collapses to <UNK> (id 1)
```

### `<MASK>` caveat

`<MASK>` was trained as a **user-defined symbol**, so if the *literal string* `"<MASK>"` appears in
raw input text, SentencePiece converts it to id 4:

```python
sp.encode("ខ្ញុំ <MASK> ភាសា", out_type=str)
# ['▁ខ្ញុំ', '▁', '<MASK>', '▁', 'ភាសា']  -> [..., 4, ...]
```

Don't rely on this for masking. For diffusion, mask at the **id level** (see §6).

## 5. Batching / padding

```python
def encode_batch(texts, max_len):
    out = []
    for t in texts:
        ids = sp.encode(t)[:max_len]
        ids += [PAD] * (max_len - len(ids))   # right-pad with <PAD> (id 0)
        out.append(ids)
    return out
```

For training, build an attention/pad mask so the model ignores `PAD` positions.

## 6. Masking for the diffusion decoder (notebook 02)

The MDLM diffusion stage masks **whole tokens**, never characters, and does it in the tensor —
not through the tokenizer:

```python
import torch, random

ids = torch.tensor(sp.encode(text))          # 1) tokenize CLEAN text
t   = random.random()                        # 2) noise level t ~ U(0,1)
mask = torch.rand(ids.shape) < t             #    choose tokens to hide
noisy = ids.clone()
noisy[mask] = MASK                            # 3) replace chosen ids with 4 (<MASK>)
# model is trained to predict `ids` from `noisy`
```

Because `<MASK>` is a real reserved vocab entry (id 4, distinct from `<UNK>`), the embedding table
has a dedicated row for it — exactly what the diffusion model needs.

## 7. Read the manifest

`tokenizer_info.json` records vocab size, special-token ids, and provenance — use it to stay correct
if the tokenizer is ever retrained:

```python
import json
from huggingface_hub import hf_hub_download

info = json.load(open(hf_hub_download("Panhapich/khmer-sp-8k", "tokenizer_info.json")))
print(info["vocab_size"])                        # 8000
print(info["special_tokens"]["MASK"]["id"])      # 4
```

## 8. Quick sanity checklist

- `sp.get_piece_size() == 8000`
- `sp.piece_to_id("<MASK>") == 4` and `!= sp.unk_id()`
- `sp.decode(sp.encode(x)) == x` for clean Khmer text
- Common words tokenize to 1–2 pieces; `▁` marks word starts

---

_Shared vocabulary for the OCR, ASR, and diffusion-decoder stages of the Khmer multimodal project._
