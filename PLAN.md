# PLAN.md — Khmer-LLaDA-Small (refined)

A resource-efficient **LLaDA-style masked diffusion language model for Khmer**, trained **from scratch**
on a single **T4 16 GB** (Kaggle-compatible), using the `Panhapich/khmer-sp-8k` tokenizer.

This refines the original 59-section plan. Same goal and methodology; the changes are: correct the
diffusion loss, lock the model definition to a known-good source, tie model size to the real token
budget, integrate the tokenizer's mandatory segmentation pipeline properly, and put concrete numbers
on "success" and the schedule.

---

## 0. Scope

- **Text-only language model.** No audio, no Whisper, no ASR. The `Digital-Divide-Data/khmer-speech-dataset`
  and all earlier Kaggle-ASR analysis are out of scope. You need a Khmer **text** corpus.
- If ASR is a later goal, this LM can become the decoder in a Whisper-LLaDA-style system as Phase 2.
- Deliverable is a thesis-grade comparison: **masked diffusion vs autoregressive LM for Khmer at matched
  resources**, plus a tokenizer study and a diffusion-steps study.

---

## 1. The two decisions that de-risk everything

### 1a. Model definition: don't fork the training framework, port the block

- The GitHub `ML-GSAI/LLaDA` repo is **inference/eval only** — `generate.py`, `get_log_likelihood.py`,
  `eval_llada.py`. There is no pretraining framework to adapt.
- Take the **model architecture** from the HF repo `GSAI-ML/LLaDA-8B-Base` (`modeling_llada.py`,
  `configuration_llada.py`, the `trust_remote_code` files). It is a LLaMA-style transformer with
  **no causal mask**, RoPE, RMSNorm, SwiGLU, no attention bias.
- Write a **minimal `modeling.py`** (~300 lines) that reproduces that block at small scale, cross-checked
  against LLaDA's file for norm placement, RoPE base, and init. Full control over gradient checkpointing
  and SDPA is worth more than inheriting an 8B config class.
- **The #1 silent bug:** if you start from an AR codebase and leave the causal mask in, everything trains
  as a broken MLM and still looks like it's converging. Assert bidirectionality in a unit test
  (a token's logits must change when a *later* token is altered).

### 1b. Size the model to the token budget — audit the corpus first

Everything else waits on one number: **how many clean Khmer tokens can you actually assemble?**

| Clean tokens (after dedup+filter) | Target model | Verdict |
|---|---|---|
| < 0.3 B | ~15–30 M ("methods demo") | LM won't be "usable"; still fine for the AR-vs-diffusion comparison |
| 0.3 – 1 B | **Small-A ~70 M** | realistic Khmer low-resource target |
| 1 – 3 B | **Small-B ~110 M** | good target if data allows |
| > 5 B | Small-C ~250 M | only if the corpus genuinely exists |

From-scratch low-resource wants **~50–100 tokens/param**, not Chinchilla's 20. The tokenizer was trained
on ~1.96 M sentences (~40–60 M tokens) — that alone is **~10× too small**. You must combine sources.

**Do this in week 1** (§4). Do not pick a model size before you have the number.

---

## 2. The diffusion objective (get this exactly right)

LLaDA is **not** BERT MLM and **not** next-token prediction. It is masked discrete diffusion with a
`1/t` loss weight that makes the loss an upper bound on NLL.

### 2a. Forward (noising) process — `khmer_llada/diffusion.py`

```python
import torch

MASK_ID = 4  # <MASK> from khmer-sp-8k

def forward_process(x0: torch.Tensor, eps: float = 1e-3):
    """x0: (B, L) clean ids from a PACKED sequence (no padding). Returns noisy ids, mask, p_mask."""
    B, L = x0.shape
    t = torch.rand(B, device=x0.device)                 # per-sequence noise level
    p_mask = (1 - eps) * t + eps                         # in [eps, 1], avoids 1/t blow-up
    p_mask = p_mask[:, None].expand(B, L)                # (B, L)
    masked = torch.rand(B, L, device=x0.device) < p_mask
    xt = torch.where(masked, MASK_ID, x0)
    return xt, masked, p_mask
```

- Per-token independent masking at a **per-sequence** ratio `t ~ U(0,1)` (clamped to `[eps, 1]`).
- **Pack sequences** (§5) so there is no padding; then the normalization below is clean.
- Optional refinement: block-diagonal attention so packed documents don't attend across `<EOS>`
  (bidirectional makes cross-doc leakage slightly more real than in causal LMs). Ablate it.

### 2b. Loss — `training/losses.py`

```python
import torch.nn.functional as F

def diffusion_loss(logits, x0, masked, p_mask):
    """logits: (B, L, V); x0, masked, p_mask: (B, L). Canonical LLaDA pretraining loss."""
    B, L = x0.shape
    tok_ce = F.cross_entropy(
        logits[masked], x0[masked], reduction="none"
    )                                   # (N_masked,)
    tok_ce = tok_ce / p_mask[masked]    # <-- the 1/t weighting; DO NOT omit
    return tok_ce.sum() / (B * L)       # normalize by total tokens, not masked count
```

- Dividing by `p_mask` per token and by `B*L` (not `masked.sum()`) is what yields the NLL bound.
  Verify against `GUIDELINES.md` in the LLaDA repo before the first real run.
- No causal shift anywhere. Targets are the original ids at masked positions only.

### 2c. Training step

```python
xt, masked, p_mask = forward_process(x0)
logits = model(xt).logits                 # bidirectional forward
loss = diffusion_loss(logits, x0, masked, p_mask)
scaler.scale(loss / accum_steps).backward()
# every accum_steps: unscale -> clip_grad_norm_(model.parameters(), 1.0) -> scaler.step -> scaler.update
```

---

## 3. Evaluation metric (must match the objective)

Masked-diffusion LMs have **no standard perplexity**. Report a Monte-Carlo NLL **upper bound**.

```python
@torch.no_grad()
def nll_bound(model, x0, n_samples=128, eps=1e-3):
    """Mean over resampled t of the diffusion_loss. Lower is better. exp(bound) ~ pseudo-perplexity."""
    total = 0.0
    for _ in range(n_samples):
        xt, masked, p_mask = forward_process(x0, eps)
        logits = model(xt).logits
        total += diffusion_loss(logits, x0, masked, p_mask).item()
    return total / n_samples
```

- Port the estimator logic from `get_log_likelihood.py`.
- **Fix the seed and `n_samples`** so the number is comparable across checkpoints and against the AR
  baseline (for AR, report true token-level NLL / perplexity — label both clearly, they are not identical
  quantities but both are "nats/token").
- Track `val_nll_bound` and `exp(val_nll_bound)` every N steps.

---

## 4. Corpus (week 1 — the gate)

### 4a. Candidate Khmer text sources — count tokens for each with the real tokenizer

| Source | Rough scale | Notes |
|---|---|---|
| `Panhapich/khmer-text-corpus` | ~1.96 M sentences (~40–60 M tok) | tokenizer's own corpus; cleanest |
| `nphearum/khmer-raw-text-3M-v2` | 3 M docs (~0.2–0.6 B tok?) | measure it |
| `CulturaX` (km) | ~1–2 B tok | web, noisy but large — likely your bulk |
| `MADLAD-400` (km, clean split) | ~ several 100 M tok | |
| `oscar-2301` / `mc4` (km) | ~ several 100 M tok | heavy dedup needed |
| Khmer Wikipedia | ~15–30 M tok | small, high quality |
| `Sokheng/khmer-synthetic-ocr-v1-100k` | small, OCR | high `<UNK>`; include only after hard filtering |

Target: **≥ 1 B tokens** deduped/filtered; 2–4 B is comfortable for Small-B.

### 4b. Pipeline — `data/preprocess.py`

```
raw docs
  → Unicode NFC only  (NEVER strip combining marks — Khmer coeng/vowels depend on them)
  → whitespace/control cleanup
  → language filter: Khmer-script char ratio > 0.6  (fastText lid optional)
  → quality filter: length, symbol ratio, n-gram repetition, per-doc <UNK> rate < 1%
  → dedup: exact hash + MinHash-LSH at document level
  → HOLD OUT eval set FIRST, then dedup eval against train
  → segment + tokenize (§6) → uint16 id shards + index
```

- Log a corpus report: total tokens, tokens/source, corpus-level `<UNK>` rate, dedup ratio,
  tokens-per-Khmer-character, tokens-per-sentence.

---

## 5. Sequence packing & sharding

- Concatenate each doc's ids + `<EOS>` (id 3). Chunk the stream into fixed `seq_len` windows (start 512).
  Drop the final partial chunk. **No padding** → the §2 loss normalization stays exact.
- Store as memory-mapped `uint16` arrays (`vocab 8000 < 65536`), e.g. `train_000.npy … train_NNN.npy`,
  plus a JSON index of `[shard, n_sequences]`.
- Dataloader returns `(B, seq_len)` int64 tensors. **Resumable:** persist `(shard_idx, seq_offset, rng_state)`
  in the checkpoint so a restart continues the exact data position.
- Never run segmentation in the dataloader (§6 — it's slow).

---

## 6. Tokenizer integration — `Panhapich/khmer-sp-8k`

### What it is
SentencePiece **unigram**, vocab 8000, char-coverage 0.9998, **no byte fallback**.
Specials fixed: `<PAD>=0 <UNK>=1 <BOS>=2 <EOS>=3 <MASK>=4`.
**Requires** `khmer_segmentation.KhmerTokenizer` (khmer-nltk word-seg + wordninja for Latin runs,
guarded by `latin_exceptions.json` / `gazetteer.json`). A bare `SentencePieceProcessor` produces
cross-word token fusion — do not use it directly.

### 6a. Milestone-0 task: `KhmerLLaDATokenizer(PreTrainedTokenizer)`
Vendor into `tokenizer/`: `khmer_sp.model`, `khmer_segmentation.py`, `gazetteer.json`,
`latin_exceptions.json`. Wrap `KhmerTokenizer`; implement `_tokenize`, `_convert_token_to_id`,
`_convert_id_to_token`, `convert_tokens_to_string`, `save_vocabulary`, and set the 5 special tokens
from their fixed ids. Expose `mask_token_id` etc. so §48–49's `from_pretrained`/`save_pretrained`
work and the released model is self-contained (with `trust_remote_code=True`).

### 6b. Pin versions
`khmer-nltk`, `wordninja`, `sentencepiece` — exact pins in `requirements.txt` and the run manifest.
Segmentation drift silently re-tokenizes the corpus and misaligns a resumed run's embeddings.

### 6c. Verify before any training
- `decode(encode(x)) == x` for in-coverage Khmer / Khmer+English / numbers / punctuation / combining chars.
- **Decode returns natural (unspaced) Khmer**, not khmer-nltk-spaced text — otherwise human fluency
  scoring (§13) measures segmented text. If it leaves segmentation spaces, add a de-segmentation step.
- **Deterministic encoding** (Viterbi best path, not unigram sampling). Subword regularization is a
  deliberate ablation, not a default to trip into.
- Corpus-level `<UNK>` rate on a 10k-doc sample. If > ~0.5%, tighten §4 filters.

### 6d. Offline tokenization is a batch job
khmer-nltk is CRF-based and slow. Budget hours-to-days for the full corpus; parallelize across CPU
processes. If it exceeds ~48 h, subsample the noisier web sources.

---

## 7. Model configs

FFN ≈ `round_to_64(8/3 · hidden)` for SwiGLU. **Tie input/output embeddings** at this scale
(saves `vocab·hidden` params; ablate untied later). RoPE (base 10000), RMSNorm, no biases.
GPT-NeoX-style init: `std=0.02`, residual projections scaled by `1/sqrt(2·n_layers)`.

```yaml
# configs/tiny.json  — debugging only, not for quality
{vocab_size: 8000, hidden_size: 256, num_hidden_layers: 4,  num_attention_heads: 4,
 intermediate_size: 683,  max_position_embeddings: 128, tie_word_embeddings: true}

# configs/small_a.json  — ~70M, target if corpus 0.3–1B tokens
{vocab_size: 8000, hidden_size: 640, num_hidden_layers: 12, num_attention_heads: 10,
 intermediate_size: 1707, max_position_embeddings: 512, tie_word_embeddings: true}

# configs/small_b.json  — ~110M, target if corpus 1–3B tokens
{vocab_size: 8000, hidden_size: 768, num_hidden_layers: 12, num_attention_heads: 12,
 intermediate_size: 2048, max_position_embeddings: 512, tie_word_embeddings: true}

# configs/small_c.json  — ~250M, only if corpus >5B tokens
{vocab_size: 8000, hidden_size: 1024, num_hidden_layers: 16, num_attention_heads: 16,
 intermediate_size: 2730, max_position_embeddings: 1024, tie_word_embeddings: true}
```

Compute the exact parameter count after building the model and record embedding vs non-embedding split.

---

## 8. T4 training configuration

```yaml
precision: fp16                 # T4 (Turing) has NO bf16 hardware — fp16 is mandatory
amp: torch.cuda.amp.GradScaler # dynamic loss scaling (fp16 from-scratch needs it)
grad_clip_norm: 1.0            # NOT in the original plan — add it
attention: sdpa               # mem-efficient kernel; FlashAttention-2 unavailable on Turing

optimizer: AdamW  (betas 0.9/0.95, weight_decay 0.1, fused if available)
lr: 3.0e-4                    # ~70–110M from scratch; drop to 2e-4 if loss spikes
schedule: cosine to 10% of peak, warmup 1–2% of total steps
effective_tokens_per_step: ~262k–524k   # = batch * accum * seq_len

# START HERE, then profile (§9) and raise batch:
seq_len: 512
physical_batch: 8            # a ~110M model at seq 512 fits FAR more than batch=1 on 16GB
grad_accum: 32
gradient_checkpointing: false  # enable ONLY at seq 1024 or model >=250M
adam_8bit: false               # 110M optimizer state ~0.9GB — not needed; revisit only if pressured
```

**The original plan is over-conservative** (batch=1 + checkpointing from the start). That leaves ~3–5×
throughput on the table. Measure first.

### Schedule by tokens, not epochs
Track `tokens_seen`, `step`, `train_loss`, `val_nll_bound`. Target ~50–100 tok/param
(Small-B ~110M → ~6–11 B tokens ideal; ~2–3 B is a defensible minimum for the thesis).

### Wall-clock reality
T4 ≈ 20–40k tok/s for ~100M. 2 B tokens ≈ 14–28 h continuous. On Kaggle (12 h sessions, 30 h/week
GPU quota) that is **~1–3 weeks wall-clock** with checkpoint/resume. Plan milestones accordingly.

---

## 9. Bring-up sequence (do not skip, do not reorder)

**M0 — Tokenizer & corpus (week 1).** `KhmerLLaDATokenizer` + tests (§6c). Corpus audit → **total clean
token count**. Pick model size from §1b table. *Gate: you know your token budget.*

**M1 — Pipeline & Tiny model (week 2).** preprocess/dedup/filter/pack/shard. Build `modeling.py` (Tiny).
Nine unit tests, all must pass before scaling:
1. text → ids (round-trip) 5. logits → `diffusion_loss` (finite, ~ln(V) at init)
2. ids → `forward_process` (mask rate ≈ t) 6. loss → `backward()` (grads on all params)
3. masked ids → model (shape (B,L,V)) 7. optimizer step (params change)
4. model forward is **bidirectional** (later-token edit changes earlier logits) 8. checkpoint save/reload (bit-exact resume)
   9. all-MASK → generation → decodes to valid Khmer

**M1 gate — overfit test.** 500–1000 Khmer sentences, Tiny model, no dropout. Masked loss must fall
**< 0.1 nats/token**. If it can't: the bug is in the loss / mask / attention / label path — use the
checklist, do not proceed.

**M2 — Profiling (week 3).** Small-A on ~100 M tokens. Produce the throughput/VRAM table
(seq × batch × peak VRAM × tok/s). LR micro-sweep {2e-4, 3e-4} on 100 M tokens. Lock the config.

**M3 — Pretrain (weeks 4–7).** Small-A or Small-B full run with resume. Val curve every N steps;
fixed-prompt generation dump every M steps. Watch train↓/val↑ divergence.

**M4 — Experiments (weeks 8–10).** AR baseline, tokenizer A/B, diffusion-steps sweep (§12).

**M5 — Evaluation & packaging (weeks 11–12).** Quant + human eval, ablation write-up, final model dir.

---

## 10. Generation — `khmer_llada/generation.py`

Adapt LLaDA `generate.py`: semi-autoregressive **block decoding** + **low-confidence remasking**,
Gumbel-noise argmax, params `{steps, block_length, temperature, remasking, cfg_scale}`.
Unconditional Khmer = start from `<BOS>` + `gen_len` `<MASK>`; iteratively unmask by confidence.
Do **not** invent a new sampler — match the reference so results are interpretable.

Fixed eval prompt set (`prompts/{general,news,education,technology,conversational}.txt`), fixed seed,
fixed steps across checkpoints. Flag every generation for: repetition, invalid Unicode, stray
`<MASK>`/`<PAD>`, English leakage, `<UNK>`.

---

## 11. Autoregressive baseline (the headline comparison)

Same `modeling.py` with `is_causal=True` + next-token shift + standard CE. **Same tokenizer, same
packed data, same token budget, same ~parameter count.** Train alongside Small-B.

Research question: *how does masked diffusion compare with autoregressive LM for Khmer under a fixed
T4 budget* — on val NLL/nats-per-token, generation quality (human), throughput (train tok/s), and
generation latency vs quality. A win on **either** quality-at-equal-tokens **or** speed-at-equal-quality
is a publishable result.

---

## 12. Experiments (matched everything except the variable)

| Exp | Variable | Constant | Primary metrics |
|---|---|---|---|
| A | `khmer-sp-8k` (+ segmentation) vs **byte-level BPE-8k** (no segmentation) | model, data, tokens, procedure | tok/char, val NLL bound, tok/s **incl. preprocessing cost**, human fluency, `<UNK>` rate |
| B | diffusion steps {16, 32, 64, 128, 256} | trained model | gen latency, tok/s, fluency, repetition |
| C | model size Tiny / Small-A / Small-B | tokens seen | val NLL bound vs params, params breakdown |
| D | diffusion (Small-B) vs AR baseline | tokenizer, data, tokens, params | §11 metrics |

Exp A's byte-BPE control is what tells you whether the mandatory khmer-nltk front-end **earns** its
complexity and latency — a genuine finding either way.

---

## 13. Success criteria (concrete — replace the checkbox list)

- **M1:** Tiny overfits 500 sentences to < 0.1 nats/token masked loss; all 9 unit tests pass.
- **M2:** documented T4 throughput ≥ 15k tok/s for the chosen config; stable loss for 100 M tokens with
  no NaN/inf, no loss-scale collapse.
- **M3:** Small-B `val_nll_bound` below the baseline set by its own first 100 M-token checkpoint by a
  clear margin, and still decreasing at the token budget's end (or explicitly compute-bound).
- **Generation:** ≥ 3.5 / 5 mean human fluency on 50 fixed prompts, ≥ 2 raters, inter-rater agreement
  (Cohen's κ or Krippendorff's α) reported; < 5 % of 200 sampled generations contain malformed Unicode
  or stray special tokens.
- **Thesis result (Exp D):** diffusion matches or beats AR on nats/token at equal tokens, **or** reaches
  equal human fluency at lower generation latency. State the operating point.

---

## 14. Risks & kill criteria

| Risk | Trigger | Action |
|---|---|---|
| Corpus too small | < 0.3 B clean tokens after §4 | descope to ~15–30 M "methods-demo" model, or add sources before proceeding |
| Loss / mask / attention bug | Tiny cannot overfit (< 0.1 nats/tok) | stop; run the M1 checklist; do not spend T4 hours |
| Segmentation too slow | full-corpus tokenization > 48 h | parallelize; subsample noisy web sources |
| fp16 instability | recurring loss-scale collapse / NaN | lower LR to 2e-4, tighten grad clip to 0.5, warmup longer; if persistent, borrow an Ampere GPU for bf16 |
| Throughput too low | < 10k tok/s | fix batch/seq/checkpointing before shrinking the model |
| `<UNK>` leakage in generations | > 5 % malformed | harder §4 UNK filter; consider retraining tokenizer with byte fallback (last resort) |

---

## 15. Repository layout

```
khmer-llada-small/
├── configs/            tiny.json, small_a.json, small_b.json, small_c.json, train_t4.yaml
├── tokenizer/          khmer_sp.model, khmer_segmentation.py, gazetteer.json,
│                       latin_exceptions.json, tokenization_khmer_llada.py  (PreTrainedTokenizer)
├── khmer_llada/        __init__.py, configuration.py, modeling.py, diffusion.py, generation.py
├── data/               preprocess.py, dedup.py, filters.py, pack.py, dataset.py
├── training/           train.py (custom loop), losses.py, optim.py, checkpoint.py
├── evaluation/         nll_bound.py, generate_eval.py, tokenizer_eval.py, human_eval_protocol.md
├── scripts/            audit_corpus.py, tokenize_corpus.py, profile_memory.py
├── experiments/        experiment_XXX/{config.yaml, manifest.json, metrics.jsonl, logs/, samples/}
├── tests/              test_tokenizer.py, test_diffusion.py, test_bidirectional.py, test_resume.py
├── requirements.txt    (pinned: torch, transformers, sentencepiece, khmer-nltk, wordninja, ...)
└── PLAN.md
```

Use a **custom training loop**, not HF `Trainer` — the loss is non-standard, and resume-with-exact-data-
position matters more than `Trainer`'s conveniences on a single GPU (~200 lines).

---

## 16. Reproducibility manifest (per run)

`experiments/experiment_XXX/manifest.json`: git commit; Python / torch / transformers / CUDA versions;
`khmer-nltk` / `wordninja` / `sentencepiece` versions; GPU model; tokenizer file hash; corpus shard
hashes + total tokens; random seed; full model + train config; `tokens_seen` and `step` at each saved
checkpoint.

---

## 17. What to reuse vs build

**Reuse / reference (LLaDA):** bidirectional transformer block, RoPE/RMSNorm/SwiGLU details,
`forward_process` + `1/t` loss (from `GUIDELINES.md`), generation algorithm, `get_log_likelihood`
estimator.

**Build (yours):** `KhmerLLaDATokenizer` + segmentation integration, Khmer corpus pipeline
(normalize/filter/dedup/pack), small configs, custom T4 training loop with resumable data position,
Khmer eval (quant + human), the four experiments.

**Do not reuse:** LLaDA-8B weights, tokenizer, or config (different vocab, different scale, trained
from scratch).

---

## 18. Immediate checklist (week 1 only)

```
[ ] Pull modeling_llada.py / configuration_llada.py from HF GSAI-ML/LLaDA-8B-Base for reference
[ ] Read LLaDA GUIDELINES.md — copy the exact pretraining loss snippet
[ ] Vendor khmer-sp-8k files; write KhmerLLaDATokenizer(PreTrainedTokenizer)
[ ] tests/test_tokenizer.py: round-trip, natural-decode, determinism, UNK rate
[ ] scripts/audit_corpus.py: token count per candidate source with the real tokenizer
[ ] >>> DECIDE MODEL SIZE from the total-token number (§1b table) <<<
[ ] Draft khmer_llada/diffusion.py (forward_process) + training/losses.py (diffusion_loss)
[ ] tests/test_bidirectional.py + test_diffusion.py
```

**Rule:** no T4 pretraining run until Tiny overfits a tiny Khmer set and generates valid Khmer.
