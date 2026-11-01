#!/usr/bin/env bash
# Full baseline + prelim suite for ONE GPU (~3-5 h on an RTX 4090; resumable: rerun to continue).
# Stages can be selected:  STAGES="micro latency profile onnx wer prelim pmb" bash jobs/run_full.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results}
STAGES=${STAGES:-"micro pmb latency profile prelim onnx batch wer"}
GPUMEM=$(python -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/2**30))")
BIG=""; [ "$GPUMEM" -ge 10 ] && BIG="nemotron600m"
CC=$(python -c "import torch;a,b=torch.cuda.get_device_capability(0);print(a*10+b)")
SLUG=$(python -c "import sys;sys.path.insert(0,'.');from asrbench.common import gpu_slug;print(gpu_slug())")
mkdir -p "$OUT/$SLUG"
if [ "$CC" -lt 70 ]; then
  # Pascal (GTX 1080 Ti): no torch.compile/Triton, no bf16, FP16 math runs at 1/64 rate -> FP32 is the real baseline
  echo "compute capability $CC: compile/Triton configs will be skipped; using FP32 for batch/WER defaults"
  BATCH_PREC=${BATCH_PREC:-fp32}
  CHUNK_PREC=${CHUNK_PREC:-fp32}
  WER_MODES=${WER_MODES:-"fp32:eager fp32:cudagraph fp32:cudagraph:A"}
fi
echo "GPU $SLUG (sm_$CC, ${GPUMEM} GB) -> $OUT"
run() { echo "=== $* ==="; "$@" || echo "!!! stage failed: $*"; }
for s in $STAGES; do case $s in
  micro)   run python scripts/01_micro.py --out $OUT ;;
  pmb)     if [ -x micro/pmb ] || command -v nvcc >/dev/null; then
             [ -x micro/pmb ] || run nvcc -O3 -std=c++17 -arch=native micro/persistent_microbench.cu -o micro/pmb
             [ -x micro/pmb ] && \
             ./micro/pmb 2000 > "$OUT/$SLUG/persistent_microbench.json"
           else echo "nvcc not found: skipping persistent microbenchmark"; fi ;;
  latency) run python scripts/02_latency.py --preset full --out $OUT
           [ -n "$BIG" ] && run python scripts/02_latency.py --preset chunks --precisions ${CHUNK_PREC:-fp16} --out $OUT ;;
  profile) run python scripts/03_profile.py --precisions fp32 amp16 fp16 \
               --impls eager cudagraph compile-default compile-reduce-overhead --out $OUT
           run python scripts/03_profile.py --precisions fp16 --impls eager cudagraph --patches A B AB --out $OUT ;;
  prelim)  run python scripts/02_latency.py --preset prelim --out $OUT
           run python scripts/06_prelim_conv.py --precisions fp32 fp16 --batches 1 4 16 --sweep --out $OUT ;;
  onnx)    run python scripts/04_onnx_trt.py --model eou120m --out $OUT ;;
  batch)   run python scripts/02_latency.py --preset batch --precisions ${BATCH_PREC:-fp16} --tag latency --out $OUT ;;
  wer)     if [ -d data/LibriSpeech/test-clean ]; then
             run python scripts/05_wer.py --max-utts ${WER_UTTS:-0} --batch 32 \
                 --modes ${WER_MODES:-fp32:eager amp16:eager fp16:eager fp16:cudagraph fp16:cudagraph:A fp16:cudagraph:B fp16:cudagraph:AB} --out $OUT
           else echo "LibriSpeech missing: run data/get_librispeech.sh"; fi ;;
esac; done
run python scripts/summarize.py --results $OUT
