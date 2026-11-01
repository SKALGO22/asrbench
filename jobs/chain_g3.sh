#!/usr/bin/env bash
# GPU3: accuracy (streaming WER) parity
set -uo pipefail; source ~/asrbench/jobs/chain_common.sh; export CUDA_VISIBLE_DEVICES=3; O=results_g3
python scripts/05_wer.py --max-utts 500 --batch 32 --modes fp32:eager amp16:eager fp16:eager fp16:cudagraph fp16:cudagraph:A fp16:cudagraph:AB --out $O; s wer500
python scripts/05_wer.py --max-utts 0 --batch 32 --modes fp32:eager fp16:cudagraph fp16:cudagraph:AB --out $O; s wer_clean_full
python scripts/05_wer.py --subsets test-other --max-utts 0 --batch 32 --modes fp32:eager fp16:cudagraph:AB --out $O; s wer_other_full
