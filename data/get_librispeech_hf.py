"""Fallback when openslr.org is slow/blocked: fetch LibriSpeech test sets from the Hugging Face Hub
(openslr/librispeech_asr parquet) and write them in the original LibriSpeech folder layout
(data/LibriSpeech/<subset>/<spk>/<chapter>/<id>.flac + .trans.txt) that 05_wer.py reads.

Usage: python data/get_librispeech_hf.py [--subsets test-clean test-other] [--max-utts 0]
"""
import argparse
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

FILES = {"test-clean": "clean/test/0000.parquet", "test-other": "other/test/0000.parquet"}

ap = argparse.ArgumentParser()
ap.add_argument("--subsets", nargs="+", default=["test-clean", "test-other"])
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "LibriSpeech"))
ap.add_argument("--max-utts", type=int, default=0)
args = ap.parse_args()

import pyarrow.parquet as pq  # noqa: E402

for subset in args.subsets:
    path = hf_hub_download("openslr/librispeech_asr", FILES[subset], repo_type="dataset")
    table = pq.read_table(path)
    rows = table.to_pylist()
    if args.max_utts:
        rows = rows[: args.max_utts]
    trans = defaultdict(list)
    for r in rows:
        spk, chap = str(r["speaker_id"]), str(r["chapter_id"])
        d = Path(args.out) / subset / spk / chap
        d.mkdir(parents=True, exist_ok=True)
        uid = r["id"]
        (d / f"{uid}.flac").write_bytes(r["audio"]["bytes"])
        trans[d / f"{spk}-{chap}.trans.txt"].append(f"{uid} {r['text']}")
    for p, lines in trans.items():
        p.write_text("\n".join(sorted(lines)) + "\n")
    print(f"{subset}: wrote {len(rows)} utterances under {Path(args.out) / subset}")
