#!/usr/bin/env bash
# GPU1: where the time goes + preliminary approaches A/B
set -uo pipefail; source ~/asrbench/jobs/chain_common.sh; export CUDA_VISIBLE_DEVICES=1; O=results_g1
python scripts/03_profile.py --out $O
python scripts/03_profile.py --precisions fp16 --impls eager cudagraph --patches A B AB --out $O; s profile
python scripts/02_latency.py --preset prelim --out $O; s prelim
python scripts/06_prelim_conv.py --precisions fp32 fp16 --batches 1 4 16 --sweep --out $O; s prelim_conv
