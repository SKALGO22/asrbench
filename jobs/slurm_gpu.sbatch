#!/usr/bin/env bash
#SBATCH --job-name=asrbench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:50:00
#SBATCH --output=logs/%x-%j.out
# Submit:  mkdir -p logs && sbatch jobs/slurm_gpu.sbatch
# For a specific GPU type add e.g.  #SBATCH --constraint=rtx4090   (or --gres=gpu:rtx2080:1, per your cluster)
# Edit HF_HOME to a shared directory pre-populated by scripts/00_fetch.py on a login node.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=${HF_HOME:-$PWD/.hf}
source venv/bin/activate
nvidia-smi
# Lock clocks if permitted (comment out otherwise); results record whether this happened.
nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits) 2>/dev/null && echo "clocks locked" || echo "clocks not locked"
trap 'nvidia-smi -rgc >/dev/null 2>&1 || true' EXIT
timeout 5.6h bash jobs/run_full.sh
