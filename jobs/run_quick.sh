#!/usr/bin/env bash
# Smoke test (~10-15 min, fits a 4 GB laptop GPU such as the RTX 3050): every script once, small settings.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results}
python scripts/01_micro.py --out $OUT
python scripts/02_latency.py --preset quick --iters 200 --out $OUT
python scripts/03_profile.py --precisions fp16 --impls eager cudagraph --out $OUT
python scripts/06_prelim_conv.py --precisions fp16 --batches 1 --out $OUT
python scripts/02_latency.py --models eou120m --precisions fp16 --impls eager cudagraph --patches none A B AB \
       --iters 200 --tag latency --out $OUT
if [ -d data/LibriSpeech/test-clean ]; then
  python scripts/05_wer.py --max-utts 40 --batch 8 --modes fp32:eager fp16:cudagraph fp16:cudagraph:AB --out $OUT
fi
python scripts/summarize.py --results $OUT
