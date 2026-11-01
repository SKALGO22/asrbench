"""Pre-download models (and print cache location) so compute nodes without internet can run.

Usage (on a login node with internet):  HF_HOME=/shared/hf python scripts/00_fetch.py --models eou120m nemotron600m
Then export the same HF_HOME in your job script.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrbench.model import MODELS, load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["eou120m", "nemotron600m"])
args = ap.parse_args()
for m in args.models:
    load_model(m, "cpu")
    print(f"cached {m} ({MODELS[m][0]})")
print("HF_HOME =", os.environ.get("HF_HOME", "~/.cache/huggingface"))
