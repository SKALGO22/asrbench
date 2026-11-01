"""Where does a streaming chunk's time go?  Kernel counts, GPU-busy fraction and a per-category breakdown.

For each (precision, implementation[, patch]) it profiles N steady-state steps with torch.profiler (CUPTI)
and reports per chunk:
  * kernels launched, aten ops dispatched, memcpy/memset count
  * sum of kernel durations ("GPU busy") vs wall time  ->  busy fraction and average gap per kernel
  * time by kernel category (GEMM/GEMV, elementwise, reduction/norm, softmax, copy/cat, triton, other)
  * top-15 kernels; a Chrome trace you can open in https://ui.perfetto.dev
Output: results/<gpu>/profile.jsonl and results/<gpu>/traces/*.json

Usage: python scripts/03_profile.py --models eou120m --precisions fp32 fp16 --impls eager cudagraph compile-default
"""
import argparse
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrbench.common import Results, log, out_dir, summarize, system_info, time_steps, triton_ok  # noqa: E402
from asrbench.model import (StreamState, apply_precision, att_contexts, chunk_geometry, load_model,  # noqa: E402
                            make_step, precision, set_att_context, warm_caches)

CATS = [
    ("gemm", ("gemm", "gemv", "cutlass", "xmma", "cublas", "sgemm", "hgemm", "s884", "s1688", "s16816",
              "ampere_", "sm80_", "sm86_", "sm89_", "sm90_", "turing_", "volta_", "splitk", "dot_kernel",
              "matmul", "convolve", "conv2d", "implicit_convolve", "cudnn")),
    ("triton", ("triton",)),
    ("norm_reduce", ("layer_norm", "layernorm", "reduce", "norm")),
    ("softmax", ("softmax",)),
    ("copy_cat", ("copy", "cat", "concat", "transpose", "index", "gather", "scatter", "fill")),
    ("elementwise", ("elementwise", "vectorized", "unrolled", "where", "masked")),
]


def category(name: str) -> str:
    n = name.lower()
    for cat, keys in CATS:
        if any(k in n for k in keys):
            return cat
    return "other"


def analyse(prof, n_steps: int):
    kernels, memops = [], 0
    kr = prof.profiler.kineto_results
    for e in kr.events():
        if e.device_type() != torch.autograd.DeviceType.CUDA:
            continue
        name = e.name()
        if name.startswith("Memcpy") or name.startswith("Memset") or "memcpy" in name.lower() or "memset" in name.lower():
            memops += 1
            continue
        kernels.append((e.start_ns(), e.duration_ns(), name))
    kernels.sort()
    busy_ns = sum(d for _, d, _ in kernels)
    by_cat, by_name = defaultdict(float), defaultdict(lambda: [0, 0.0])
    for _, d, n in kernels:
        by_cat[category(n)] += d / 1e3
        by_name[n[:110]][0] += 1
        by_name[n[:110]][1] += d / 1e3
    aten = 0
    for ev in prof.key_averages():
        if ev.key.startswith("aten::"):
            aten += ev.count
    top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:15]
    return {
        "kernels_per_step": len(kernels) / n_steps,
        "memops_per_step": memops / n_steps,
        "aten_ops_per_step": aten / n_steps,
        "kernel_us_per_step": busy_ns / 1e3 / n_steps,
        "by_category_us_per_step": {k: v / n_steps for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "top_kernels": [{"name": k, "calls_per_step": c / n_steps, "us_per_step": t / n_steps} for k, (c, t) in top],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["eou120m"])
    ap.add_argument("--precisions", nargs="+", default=["fp32", "amp16", "fp16"])
    ap.add_argument("--impls", nargs="+", default=["eager", "cudagraph", "compile-default", "compile-reduce-overhead"])
    ap.add_argument("--patches", nargs="+", default=["none"])
    ap.add_argument("--att", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-trace", action="store_true")
    args = ap.parse_args()
    sysinfo = system_info()
    od = out_dir(args.out)
    (od / "traces").mkdir(exist_ok=True)
    res = Results(od / "profile.jsonl")
    from asrbench import fused_conv, prelim_attn

    for mname in args.models:
        for pname in args.precisions:
            prec = precision(pname)
            model = load_model(mname, "cuda")
            apply_precision(model, prec)
            att = list(map(int, args.att.split(","))) if args.att else att_contexts(mname, model)[0]
            set_att_context(model, att)
            for patch in args.patches:
                fused_conv.remove(model); prelim_attn.remove(model)
                if "A" in patch:
                    prelim_attn.apply(model)
                if "B" in patch:
                    fused_conv.apply(model)
                for impl in args.impls:
                    key = f"{mname}|{att[0]},{att[1]}|{pname}|{patch}|b{args.batch}|{impl}"
                    if not triton_ok() and (impl.startswith("compile") or "B" in patch):
                        log(f"skip {key} (needs compute capability >= 7.0)")
                        continue
                    if res.has(key):
                        log(f"skip {key}")
                        continue
                    rec = {"key": key, "model": mname, "att": att, "precision": pname, "patch": patch,
                           "batch": args.batch, "impl": impl, "gpu": sysinfo.get("gpu"), "geom": chunk_geometry(model)}
                    try:
                        st = StreamState(model, args.batch, prec)
                        if "A" in patch:
                            prelim_attn.reset(model, st, prec)
                        warm_caches(model, st, prec)
                        step = make_step(impl, model, st, prec)
                        for _ in range(30):
                            step()
                        torch.cuda.synchronize()
                        wall = time_steps(step, 200)
                        rec["wall_us"] = summarize(wall.wall_us)
                        rec["gpu_us"] = summarize(wall.gpu_us)
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                            for _ in range(args.steps):
                                step()
                                torch.cuda.synchronize()
                        a = analyse(prof, args.steps)
                        rec.update(a)
                        w = rec["wall_us"]["p50"]
                        rec["gpu_busy_fraction"] = a["kernel_us_per_step"] / w
                        rec["idle_us_per_kernel"] = (w - a["kernel_us_per_step"]) / max(a["kernels_per_step"], 1)
                        if not args.no_trace:
                            tp = od / "traces" / (key.replace("|", "_").replace(",", "-") + ".json")
                            prof.export_chrome_trace(str(tp))
                            rec["trace"] = str(tp)
                        rec["ok"] = True
                        log(f"{key}: {a['kernels_per_step']:.0f} kernels/chunk, {a['aten_ops_per_step']:.0f} aten ops, "
                            f"kernel time {a['kernel_us_per_step']:.0f} us of wall {w:.0f} us "
                            f"(busy {rec['gpu_busy_fraction']*100:.0f}%), gap/kernel {rec['idle_us_per_kernel']:.2f} us")
                        del step
                    except Exception as e:
                        rec["ok"] = False
                        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
                        rec["traceback"] = traceback.format_exc()[-3000:]
                        log(f"{key}: FAILED {rec['error'][:200]}")
                    torch.cuda.empty_cache()
                    try:
                        torch._dynamo.reset()
                    except Exception:
                        pass
                    res.write(rec)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
