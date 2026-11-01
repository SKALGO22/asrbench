#!/usr/bin/env bash
# GPU0: headline baselines (eager / CUDA Graph / compile x 4 precisions), batch sweep, then slow max-autotune
set -uo pipefail; source ~/asrbench/jobs/chain_common.sh; export CUDA_VISIBLE_DEVICES=0; O=results_g0
python scripts/02_latency.py --preset full --impls eager cudagraph compile-default compile-reduce-overhead --out $O; s latency
python scripts/02_latency.py --preset batch --out $O; s batch
python scripts/02_latency.py --preset full --out $O; s maxautotune
