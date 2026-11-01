"""Aggregate results/<gpu>/*.jsonl into results/SUMMARY.md and results/plots/*.png (slide-ready).

Usage: python scripts/summarize.py [--results results]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(p: Path):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def md_table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s + "\n"


def f(x, nd=0, pct=False):
    if x is None:
        return "–"
    if pct:
        return f"{x * 100:.{nd}f}%"
    return f"{x:,.{nd}f}"


def gpu_section(gdir: Path, plots: Path):
    md = []
    sysinfo = json.loads((gdir / "system.json").read_text()) if (gdir / "system.json").exists() else {}
    micro = json.loads((gdir / "micro.json").read_text()) if (gdir / "micro.json").exists() else None
    name = sysinfo.get("gpu") or (micro or {}).get("system", {}).get("gpu") or gdir.name
    md.append(f"## {name}\n")
    if sysinfo:
        md.append(f"SMs {sysinfo.get('sm_count')} · L2 {sysinfo.get('l2_mb')} MB · VRAM {sysinfo.get('vram_gb')} GB · "
                  f"torch {sysinfo.get('torch')} · CUDA {sysinfo.get('cuda_runtime')} · NeMo {sysinfo.get('nemo')}\n")
    peak = None
    if micro:
        d, o = micro["dram"], micro["overhead"]
        peak = max(d["copy_gbs"], d["read_gbs"])
        md.append("### Hardware microbenchmarks\n")
        md.append(md_table(["DRAM copy GB/s", "DRAM read GB/s", "eager op: host issue µs", "eager op: wall µs",
                            "CUDA Graph µs/kernel", "single tiny kernel µs"],
                           [[f(d["copy_gbs"]), f(d["read_gbs"]), f(o["eager_host_issue_us_per_op"], 2),
                             f(o["eager_wall_us_per_op"], 2), f(o["graph_us_per_kernel"], 2),
                             f(o["stream_us_per_tiny_kernel"], 2)]]))
        md.append(md_table(["GEMV (K×N, fp16)", "weights MB", "single launch µs", "GB/s", "in graph µs", "GB/s",
                            "% of DRAM peak (graph)"],
                           [[f"{r['k']}×{r['n']}", f(r["weight_mb"], 2), f(r["us_single"], 1), f(r["gbs_single"]),
                             f(r["us_in_graph"], 1), f(r["gbs_in_graph"]), f(r["gbs_in_graph"] / peak, 0, True)]
                            for r in micro["gemv_fp16"]]))
    pmb = gdir / "persistent_microbench.json"
    if pmb.exists():
        try:
            p = json.loads(pmb.read_text())
            md.append("### Persistent-kernel primitives\n")
            md.append(md_table(["launch+sync (1 block) p50 µs", "grid.sync µs", "counter barrier µs",
                                "mailbox round trip 1 block p50 µs", "mailbox full grid p50 µs"],
                               [[f(p.get("launch_sync_1block", {}).get("p50_us"), 1), f(p.get("grid_sync_us"), 2),
                                 f(p.get("counter_barrier_us"), 2), f(p.get("mailbox_roundtrip_1block", {}).get("p50_us"), 1),
                                 f(p.get("mailbox_roundtrip_fullgrid", {}).get("p50_us"), 1)]]))
        except Exception:
            pass

    lat = [r for r in load_jsonl(gdir / "latency.jsonl") if r.get("ok")]
    ort = [r for r in load_jsonl(gdir / "onnx_trt.jsonl") if r.get("ok")]
    fails = [r for r in load_jsonl(gdir / "latency.jsonl") + load_jsonl(gdir / "onnx_trt.jsonl") if not r.get("ok")]
    rows_all = lat + ort
    if rows_all:
        md.append("### Baseline latency per streaming chunk (batch 1, no patches)\n")
        base = [r for r in rows_all if r.get("batch") == 1 and r.get("patch", "none") == "none"]
        by_model = defaultdict(list)
        for r in base:
            by_model[(r["model"], tuple(r["att"]))].append(r)
        for (m, att), rs in sorted(by_model.items()):
            g = rs[0]["geom"]
            md.append(f"**{m}**, att {list(att)}: chunk {g['chunk_ms']} ms, input {g['input_mel']} mel frames\n")
            rs.sort(key=lambda r: r["b2b"]["wall_us"]["p50"])
            md.append(md_table(
                ["impl", "precision", "wall p50 µs", "wall p99 µs", "GPU p50 µs", "host p50 µs", "paced p99 µs",
                 "eff. GB/s", "% DRAM peak", "RTF", "rel.err vs fp32", "build s"],
                [[r["impl"], r["precision"], f(r["b2b"]["wall_us"]["p50"]), f(r["b2b"]["wall_us"]["p99"]),
                  f(r["b2b"]["gpu_us"]["p50"]), f(r["b2b"]["host_us"]["p50"]),
                  f(r.get("paced", {}).get("wall_us", {}).get("p99")), f(r.get("eff_weight_gbs")),
                  f((r.get("eff_weight_gbs") or 0) / peak, 0, True) if peak else "–",
                  f(r["b2b"]["wall_us"]["p50"] / (g["chunk_ms"] * 1e3), 4),
                  f"{r.get('parity_vs_fp32_eager', {}).get('max_rel_err', float('nan')):.1e}", f(r.get("build_s"), 1)]
                 for r in rs]))
            _plot_bars(rs, plots / f"{gdir.name}_{m}_{att[0]}-{att[1]}_latency.png", f"{name} · {m} · {g['chunk_ms']} ms chunk")
        # chunk-size sweep
        sweep = defaultdict(dict)
        for r in base:
            sweep[(r["model"], r["impl"], r["precision"])][r["geom"]["chunk_ms"]] = r["b2b"]["wall_us"]["p50"]
        multi = {k: v for k, v in sweep.items() if len(v) > 1}
        if multi:
            chunks = sorted({c for v in multi.values() for c in v})
            md.append("### Chunk-size sweep (wall p50 µs)\n")
            md.append(md_table(["model", "impl", "precision"] + [f"{c} ms" for c in chunks],
                               [[k[0], k[1], k[2]] + [f(v.get(c)) for c in chunks] for k, v in sorted(multi.items())]))
        # batch sweep / streams per GPU
        bs = defaultdict(dict)
        for r in lat:
            if r.get("patch", "none") == "none":
                bs[(r["model"], tuple(r["att"]), r["impl"], r["precision"])][r["batch"]] = r
        bsw = {k: v for k, v in bs.items() if len(v) > 2}
        if bsw:
            md.append("### Batch sweep and streams per GPU\n")
            rows = []
            for k, v in sorted(bsw.items()):
                chunk_us = next(iter(v.values()))["geom"]["chunk_ms"] * 1e3
                ok10 = [b for b, r in v.items() if r["b2b"]["wall_us"]["p99"] <= 0.1 * chunk_us]
                ok100 = [b for b, r in v.items() if r["b2b"]["wall_us"]["p99"] <= chunk_us]
                rows.append([k[0], k[2], k[3]] + [f(v[b]["b2b"]["wall_us"]["p99"]) if b in v else "–"
                                                  for b in sorted({b for vv in bsw.values() for b in vv})]
                            + [max(ok10) if ok10 else 0, max(ok100) if ok100 else 0])
            bl = sorted({b for vv in bsw.values() for b in vv})
            md.append("p99 wall µs per step for B concurrent streams; last columns: largest tested B with p99 ≤ 10% / ≤ 100% "
                      "of the chunk duration (a lower bound on streams per GPU).\n\n")
            md.append(md_table(["model", "impl", "precision"] + [f"B={b}" for b in bl] + ["streams @10%", "streams @100%"], rows))
            _plot_batch(bsw, plots / f"{gdir.name}_batch_sweep.png", name)
        # prelim patches
        pre = [r for r in lat if r.get("patch", "none") != "none"]
        if pre:
            md.append("### Preliminary approaches inside the full encoder (A = K/V+pos caching, B = fused conv)\n")
            idx = {(r["model"], tuple(r["att"]), r["precision"], r["batch"], r["impl"], r.get("patch", "none")): r for r in lat}
            rows = []
            for r in sorted(pre, key=lambda r: (r["precision"], r["batch"], r["impl"], r["patch"])):
                b = idx.get((r["model"], tuple(r["att"]), r["precision"], r["batch"], r["impl"], "none"))
                sp = b["b2b"]["gpu_us"]["p50"] / r["b2b"]["gpu_us"]["p50"] if b else None
                spw = b["b2b"]["wall_us"]["p50"] / r["b2b"]["wall_us"]["p50"] if b else None
                rows.append([r["precision"], r["batch"], r["impl"], r["patch"], f(r["b2b"]["wall_us"]["p50"]),
                             f(r["b2b"]["gpu_us"]["p50"]), f(sp, 2) + "×" if sp else "–", f(spw, 2) + "×" if spw else "–",
                             f"{r.get('parity_vs_fp32_eager', {}).get('max_rel_err', float('nan')):.1e}"])
            md.append(md_table(["precision", "batch", "impl", "patch", "wall p50 µs", "GPU p50 µs", "GPU speedup",
                                "wall speedup", "rel.err vs fp32"], rows))
    prof = [r for r in load_jsonl(gdir / "profile.jsonl") if r.get("ok")]
    if prof:
        md.append("### Where the time goes (profiler)\n")
        md.append(md_table(["impl", "precision", "patch", "kernels/chunk", "aten ops/chunk", "kernel µs", "wall µs",
                            "GPU busy", "idle µs per kernel", "top category"],
                           [[r["impl"], r["precision"], r.get("patch", "none"), f(r["kernels_per_step"]),
                             f(r["aten_ops_per_step"]), f(r["kernel_us_per_step"]), f(r["wall_us"]["p50"]),
                             f(r["gpu_busy_fraction"], 0, True), f(r["idle_us_per_kernel"], 2),
                             ", ".join(f"{k} {v:.0f}µs" for k, v in list(r["by_category_us_per_step"].items())[:3])]
                            for r in prof]))
        _plot_busy(prof, plots / f"{gdir.name}_busy.png", name)
    pc = [r for r in load_jsonl(gdir / "prelim_conv.jsonl") if r.get("ok") and "variants" in r]
    if pc:
        md.append("### Conv block alone: NeMo vs fused Triton (wall µs with sync / kernels per call)\n")
        names = ["nemo-eager", "nemo-compile", "nemo-cudagraph", "fused-triton", "fused-triton-cudagraph"]
        md.append(md_table(["precision", "batch", "frames"] + names + ["rel.err"],
                           [[r["precision"], r["batch"], r["frames"]] +
                            [(f"{r['variants'][n]['wall_us_sync']:.1f} / {r['variants'][n]['kernels']:.0f}"
                              if n in r["variants"] and "wall_us_sync" in r["variants"][n] else "fail") for n in names]
                            + [f"{r['parity']['max_rel_err_out']:.1e}"] for r in pc]))
    wer = [r for r in load_jsonl(gdir / "wer.jsonl") if r.get("ok")]
    if wer:
        md.append("### Streaming WER parity (LibriSpeech)\n")
        md.append(md_table(["subset", "utts", "mode", "WER", "errors/words", "utts differing from first mode",
                            "encoder ms/chunk (batched)"],
                           [[r["subset"], r["utts"], r["mode"], f(r["wer"], 2, True), f"{r['errors']}/{r['words']}",
                             r.get("utts_differing_from_first_mode"), f(r.get("enc_ms_per_chunk"), 2)] for r in wer]))
    if fails:
        md.append("### Failed configurations (also results)\n")
        md.append(md_table(["key", "error"], [[r["key"], r.get("error", "")[:160].replace("|", "/")] for r in fails]))
    return "\n".join(md)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_bars(rs, path, title):
    try:
        plt = _plt()
        labels = [f"{r['impl']}\n{r['precision']}" for r in rs]
        wall = [r["b2b"]["wall_us"]["p50"] for r in rs]
        gpu = [r["b2b"]["gpu_us"]["p50"] for r in rs]
        p99 = [r["b2b"]["wall_us"]["p99"] for r in rs]
        fig, ax = plt.subplots(figsize=(max(7, len(rs) * 0.8), 4))
        x = range(len(rs))
        ax.bar([i - 0.2 for i in x], wall, 0.4, label="wall p50", color="#2f5d8a")
        ax.bar([i + 0.2 for i in x], gpu, 0.4, label="GPU p50", color="#e07a1f")
        ax.scatter(list(x), p99, marker="_", s=300, color="black", label="wall p99", zorder=5)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("µs per chunk"); ax.set_yscale("log"); ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    except Exception as e:
        print("plot failed", path, e)


def _plot_batch(bsw, path, title):
    try:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        chunk_us = None
        for k, v in sorted(bsw.items()):
            bs = sorted(v)
            ax.plot(bs, [v[b]["b2b"]["wall_us"]["p99"] for b in bs], marker="o", label=f"{k[2]} {k[3]}")
            chunk_us = next(iter(v.values()))["geom"]["chunk_ms"] * 1e3
        if chunk_us:
            ax.axhline(chunk_us, color="grey", ls=":", label="chunk duration")
            ax.axhline(0.1 * chunk_us, color="grey", ls="--", label="10% of chunk")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("concurrent streams (batch)"); ax.set_ylabel("p99 µs per step")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    except Exception as e:
        print("plot failed", path, e)


def _plot_busy(prof, path, title):
    try:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(max(6, len(prof) * 0.7), 3.6))
        labels = [f"{r['impl']}\n{r['precision']}" + ("" if r.get("patch", "none") == "none" else f"\n+{r['patch']}") for r in prof]
        ax.bar(range(len(prof)), [r["gpu_busy_fraction"] * 100 for r in prof], color="#2f5d8a")
        for i, r in enumerate(prof):
            ax.text(i, r["gpu_busy_fraction"] * 100 + 1, f"{r['kernels_per_step']:.0f}k", ha="center", fontsize=7)
        ax.set_xticks(range(len(prof))); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("GPU busy % of chunk time"); ax.set_ylim(0, 110)
        ax.set_title(title + " (label = kernels per chunk)", fontsize=9); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    except Exception as e:
        print("plot failed", path, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    parts = ["# Streaming ASR encoder baselines — summary\n",
             "All latencies are per streaming chunk (one encoder step), steady state, synchronized. "
             "'eff. GB/s' = encoder weight bytes / GPU time; 'RTF' = wall time / chunk duration.\n"]
    for gdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "plots"):
        parts.append(gpu_section(gdir, plots))
    (root / "SUMMARY.md").write_text("\n".join(parts))
    print(f"wrote {root / 'SUMMARY.md'} and plots in {plots}")


if __name__ == "__main__":
    main()
