#!/usr/bin/env bash
# Download the tokenizer + corpus. Run from the repo root.
set -euo pipefail

mkdir -p tokenizer data/raw

echo "== tokenizer: Panhapich/khmer-sp-8k =="
huggingface-cli download Panhapich/khmer-sp-8k \
  --local-dir tokenizer \
  --include "khmer_sp.model" "khmer_sp.vocab" "khmer_segmentation.py" \
            "gazetteer.json" "latin_exceptions.json" "tokenizer_info.json" "USAGE.md"

echo "== corpus: Panhapich/khmer-text-corpus (segmented + raw) =="
huggingface-cli download Panhapich/khmer-text-corpus --repo-type dataset \
  --local-dir data/raw \
  --include "all_text_segmented.txt" "all_text.txt" "metadata.json"

echo
echo "done. Next:"
echo "  python scripts/pretokenize.py       # prints TOTAL TOKENS -> picks model size"
echo "  python scripts/make_overfit_set.py"
echo "  python -m pytest tests/ -q"
