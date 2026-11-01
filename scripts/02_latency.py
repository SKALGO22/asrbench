"""Baseline latency sweep for cache-aware streaming encoders (the core baseline experiment).

For every (model, attention context / chunk size, precision, patch, batch, implementation) it measures one
steady-state streaming step of the encoder:
  * per-chunk latency: wall (incl. sync), GPU-event and host-dispatch time; mean/p50/p90/p99/p99.9
  * back-to-back and real-time-paced runs (GPU idles between chunks like with live audio)
  * effective weight bandwidth = encoder weight bytes / GPU time (vs measured DRAM peak from 01_micro)
  * energy per chunk (NVML), peak memory, build time (capture / compile)
  * numerical parity against fp32 eager over a 30-chunk stream (max abs / rel error)
Results are appended to results/<gpu>/latency.jsonl; reruns skip finished configs (resumable).

Examples
  quick:  python scripts/02_latency.py --models eou120m --precisions fp16 --impls eager cudagraph
  full:   python scripts/02_latency.py --preset full
"""
import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrbench.common import (PowerSampler, Results, log, out_dir, summarize,  # noqa: E402
                             supports_bf16, system_info, time_steps, triton_ok)
from asrbench.model import (IMPLS, StreamState, att_contexts, apply_precision, chunk_geometry,  # noqa: E402
                            compare, encoder_param_bytes, load_model, make_step, parity_with_step,
                            precision, reference_sequence, set_att_context, warm_caches)

PRESETS = {
    "quick": dict(models=["eou120m"], precisions=["fp16"], impls=["eager", "cudagraph", "compile-default"],
                  patches=["none"], batches=[1], iters=200, paced_iters=0),
    "full": dict(models=["eou120m"], precisions=["fp32", "amp16", "fp16", "bf16"],
                 impls=IMPLS, patches=["none"], batches=[1], iters=500, paced_iters=100),
    "chunks": dict(models=["nemotron600m"], precisions=["fp16"],
                   impls=["eager", "cudagraph", "compile-default", "compile-reduce-overhead"],
                   patches=["none", "A"], batches=[1], iters=300, paced_iters=0),
    "batch": dict(models=["eou120m"], precisions=["fp16"], impls=["eager", "cudagraph", "compile-reduce-overhead"],
                  patches=["none"], batches=[1, 2, 4, 8, 16, 32, 64, 128], iters=200, paced_iters=0),
    "prelim": dict(models=["eou120m"], precisions=["fp32", "amp16", "fp16"], impls=["eager", "cudagraph"],
                   patches=["none", "A", "B", "AB"], batches=[1, 4], iters=500, paced_iters=0),
}


def apply_patches(model, patch: str):
    from asrbench import fused_conv, prelim_attn

    fused_conv.remove(model)
    prelim_attn.remove(model)
    if "A" in patch:
        prelim_attn.apply(model)
    if "B" in patch:
        fused_conv.apply(model)


def patch_reset_fn(model, patch, prec):
    if "A" not in patch:
        return None
    from asrbench import prelim_attn

    return lambda st: prelim_attn.reset(model, st, prec)


DEVICE = "cuda"


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        torch._dynamo.reset()
    except Exception:
        pass


def run_one(model, mname, att, prec, patch, batch, impl, args, refs, res: Results, sysinfo):
    key = f"{mname}|{att[0]},{att[1]}|{prec.name}|{patch}|b{batch}|{impl}"
    if DEVICE == "cuda" and not triton_ok() and (impl.startswith("compile") or "B" in patch):
        log(f"skip {key} (needs compute capability >= 7.0)")
        return
    if res.has(key) and not args.force:
        log(f"skip {key}")
        return
    rec = {"key": key, "model": mname, "att": att, "precision": prec.name, "patch": patch, "batch": batch,
           "impl": impl, "gpu": sysinfo.get("gpu"), "geom": chunk_geometry(model),
           "weight_bytes": encoder_param_bytes(model)}
    reset = patch_reset_fn(model, patch, prec)
    step = None
    try:
        cuda = DEVICE == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        state = StreamState(model, batch, prec, device=DEVICE)
        if reset:
            reset(state)
        warm_caches(model, state, prec, steps=40)
        t0 = time.perf_counter()
        step = make_step(impl, model, state, prec)
        for _ in range(args.warmup):  # includes compilation for torch.compile
            step()
        if cuda:
            torch.cuda.synchronize()
        rec["build_s"] = time.perf_counter() - t0
        rec["cache_bytes"] = state.cache_bytes()
        with PowerSampler() as ps:
            t = time_steps(step, args.iters, cuda=cuda)
        rec["power"] = ps.result()
        if rec["power"]["energy_j"] is not None:
            rec["power"]["energy_mj_per_chunk"] = rec["power"]["energy_j"] / args.iters * 1e3
        if not cuda:
            t.gpu_us = t.wall_us
        rec["b2b"] = {"wall_us": summarize(t.wall_us), "gpu_us": summarize(t.gpu_us), "host_us": summarize(t.host_us)}
        rec["b2b_gpu_samples"] = [round(v, 2) for v in t.gpu_us]
        if args.paced_iters and impl in args.paced_impls and batch == 1:
            period = rec["geom"]["chunk_ms"]
            tp = time_steps(step, args.paced_iters, paced_period_ms=period, cuda=cuda)
            if not cuda:
                tp.gpu_us = tp.wall_us
            rec["paced"] = {"period_ms": period, "wall_us": summarize(tp.wall_us), "gpu_us": summarize(tp.gpu_us),
                            "host_us": summarize(tp.host_us)}
        rec["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
        gp50 = rec["b2b"]["gpu_us"]["p50"]
        rec["eff_weight_gbs"] = rec["weight_bytes"] / gp50 / 1e3
        rec["rtf"] = rec["b2b"]["wall_us"]["p50"] / (rec["geom"]["chunk_ms"] * 1e3)
        # parity vs fp32 eager reference (same inputs, fresh stream), reusing the built step
        if args.parity_steps:
            out = parity_with_step(step, args.parity_steps, patch_reset=reset)
            rec["parity_vs_fp32_eager"] = compare(out, refs[(mname, tuple(att), batch)])
        rec["ok"] = True
        log(f"{key}: wall p50 {rec['b2b']['wall_us']['p50']:.0f} us | gpu p50 {gp50:.0f} us | "
            f"host p50 {rec['b2b']['host_us']['p50']:.0f} us | p99 {rec['b2b']['wall_us']['p99']:.0f} us | "
            f"eff {rec['eff_weight_gbs']:.0f} GB/s | build {rec['build_s']:.1f}s"
            + (f" | rel.err {rec['parity_vs_fp32_eager']['max_rel_err']:.1e}" if 'parity_vs_fp32_eager' in rec else ""))
    except Exception as e:  # record failures: they are results too (e.g. graph capture impossible)
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
        rec["traceback"] = traceback.format_exc()[-4000:]
        log(f"{key}: FAILED {rec['error'][:200]}")
    finally:
        del step
        free()
    res.write(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--att", nargs="*", default=None, help="e.g. 70,1 70,13 (default: all supported by model)")
    ap.add_argument("--precisions", nargs="+")
    ap.add_argument("--impls", nargs="+")
    ap.add_argument("--patches", nargs="+", help="none | A (K/V+pos caching) | B (fused conv) | AB")
    ap.add_argument("--batches", nargs="+", type=int)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--paced-iters", type=int)
    ap.add_argument("--paced-impls", nargs="+", default=["eager", "cudagraph", "compile-reduce-overhead"])
    ap.add_argument("--parity-steps", type=int, default=30)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="latency")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matmuls in fp32 mode")
    ap.add_argument("--cpu-smoke", action="store_true", help="debug the harness on CPU (eager only, tiny)")
    args = ap.parse_args()
    global DEVICE
    if args.cpu_smoke:
        DEVICE = "cpu"
    p = PRESETS.get(args.preset or "quick")
    for k, v in p.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    sysinfo = system_info()
    od = out_dir(args.out)
    res = Results(od / f"{args.tag}.jsonl")
    (od / "system.json").write_text(__import__("json").dumps(sysinfo, indent=1))
    log(f"GPU {sysinfo.get('gpu')} | torch {sysinfo['torch']} | nemo {sysinfo.get('nemo')} -> {res.path}")

    for mname in args.models:
        refs = _fp32_refs(mname, args) if args.parity_steps else {}
        for pname in args.precisions:
            if pname == "bf16" and not supports_bf16():
                log("bf16 not supported on this GPU, skipping")
                continue
            prec = precision(pname)
            model = load_model(mname, DEVICE)
            apply_precision(model, prec)
            atts = [list(map(int, a.split(","))) for a in args.att] if args.att else att_contexts(mname, model)
            for patch in args.patches:
                apply_patches(model, patch)
                for att in atts:
                    set_att_context(model, att)
                    for batch in args.batches:
                        for impl in args.impls:
                            run_one(model, mname, att, prec, patch, batch, impl, args, refs, res, sysinfo)
            del model
            free()
    log("done")


def _fp32_refs(mname, args):
    """fp32 eager outputs of a fresh model for every (att, batch): the ground truth for parity checks."""
    m = load_model(mname, DEVICE)
    atts = [list(map(int, a.split(","))) for a in args.att] if args.att else att_contexts(mname, m)
    refs = {}
    for att in atts:
        set_att_context(m, att)
        for b in args.batches:
            refs[(mname, tuple(att), b)] = [o.cpu() for o in reference_sequence(m, precision("fp32"), b, args.parity_steps, device=DEVICE)]
    del m
    free()
    return refs


if __name__ == "__main__":
    main()
