"""Merge result folders from several runs/machines into one tree (dedup by key; later files win).

Usage: python scripts/merge_results.py --into results  results_laptop results_gpu0 results_gpu1 ...
Each source must contain <gpu_slug>/*.jsonl (and optionally micro.json, system.json, traces/, plots are ignored).
"""
import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--into", default="results")
ap.add_argument("sources", nargs="+")
args = ap.parse_args()
dst = Path(args.into)
for src in map(Path, args.sources):
    if src.resolve() == dst.resolve():
        continue
    for gdir in [p for p in src.iterdir() if p.is_dir() and p.name != "plots"]:
        out = dst / gdir.name
        out.mkdir(parents=True, exist_ok=True)
        for f in gdir.iterdir():
            if f.suffix == ".jsonl":
                recs = {}
                for p in [out / f.name, f]:
                    if p.exists():
                        for line in p.read_text().splitlines():
                            try:
                                r = json.loads(line)
                                recs[r.get("key", line)] = r
                            except Exception:
                                pass
                (out / f.name).write_text("".join(json.dumps(r) + "\n" for r in recs.values()))
            elif f.is_file():
                shutil.copy2(f, out / f.name)
            elif f.is_dir():
                shutil.copytree(f, out / f.name, dirs_exist_ok=True)
        print(f"merged {gdir} -> {out}")
