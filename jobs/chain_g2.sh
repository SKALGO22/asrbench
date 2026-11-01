#!/usr/bin/env bash
# GPU2: hardware microbenchmarks, ONNX Runtime / TensorRT, 600M model chunk-size sweep
set -uo pipefail; source ~/asrbench/jobs/chain_common.sh; export CUDA_VISIBLE_DEVICES=2; O=results_g2
OUT=$O STAGES="micro pmb" bash jobs/run_full.sh; s micro
python scripts/04_onnx_trt.py --model eou120m --out $O; s onnx
python scripts/02_latency.py --preset chunks --out $O; s chunks600m
