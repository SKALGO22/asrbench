#!/usr/bin/env bash
# Use all 4 GPUs of one cluster node (e.g. 4x GTX 1080 Ti) in parallel, one stage group per GPU,
# each writing its own results folder; merges at the end. Run inside tmux so it survives disconnects.
#   tmux new -s bench   ->   bash jobs/run_node4.sh   ->   detach: Ctrl-b d   ->   reattach: tmux attach -t bench
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 OUT=results_g0 STAGES="micro pmb latency" bash jobs/run_full.sh > logs/g0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 OUT=results_g1 STAGES="profile onnx batch prelim" bash jobs/run_full.sh > logs/g1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 OUT=results_g2 WER_MODES="fp32:eager fp32:cudagraph fp32:cudagraph:A" STAGES="wer" bash jobs/run_full.sh > logs/g2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 OUT=results_g3 WER_MODES="fp16:eager amp16:eager fp16:cudagraph" STAGES="wer" bash jobs/run_full.sh > logs/g3.log 2>&1 &
wait
python scripts/merge_results.py --into results results_g0 results_g1 results_g2 results_g3
python scripts/summarize.py --results results
echo "all done -> results/SUMMARY.md"
