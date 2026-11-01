"""Preliminary approach B at module level: the Conformer convolution block for one streaming chunk.

Compares NeMo's block (LayerNorm + ConformerConvolution + residual, cache-aware) run eager / torch.compile /
CUDA graph against the 2-kernel Triton version (eager-launched and in a CUDA graph). Reports GPU time per call,
kernels per call, effective weight bandwidth and numerical parity, plus a small tile-size sweep.

Usage: python scripts/06_prelim_conv.py --model eou120m --precisions fp32 fp16 --batches 1 4 16
"""
import argparse
import itertools
import os
import sys
import traceback
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrbench.common import Results, cuda_time_us, log, out_dir, system_info, triton_ok  # noqa: E402
from asrbench.fused_conv import FusedConvBlock, reference_block  # noqa: E402
from asrbench.model import apply_precision, att_contexts, chunk_geometry, load_model, precision, set_att_context  # noqa: E402


def graphed(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return lambda: g.replay()


def kernels_per_call(fn, n=10):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    k = 0
    for e in p.profiler.kineto_results.events():
        if e.device_type() == torch.autograd.DeviceType.CUDA and "mem" not in e.name().lower():
            k += 1
    return k / n


def wall_us(fn, n=300):
    import time
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eou120m")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--frames", nargs="*", type=int, default=None, help="new encoder frames per chunk (default: model's)")
    ap.add_argument("--sweep", action="store_true", help="tile-size sweep for the Triton kernels")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if not triton_ok():
        log("Triton kernels need compute capability >= 7.0; skipping prelim conv benchmark on this GPU")
        return
    sysinfo = system_info()
    od = out_dir(args.out)
    res = Results(od / "prelim_conv.jsonl")
    for pname in args.precisions:
        prec = precision(pname)
        model = load_model(args.model, "cuda")
        apply_precision(model, prec)
        set_att_context(model, att_contexts(args.model, model)[0])
        geom = chunk_geometry(model)
        frames = args.frames or [geom["valid_out_len"] or 2]
        L = model.encoder.layers[args.layer]
        norm, conv = L.norm_conv, L.conv
        D = conv.d_model
        fused = FusedConvBlock(norm, conv, add_residual=True)
        wbytes = sum(p.numel() * p.element_size() for p in list(norm.parameters()) + list(conv.parameters()))
        if args.sweep and pname == args.precisions[-1]:
            best = None
            x = torch.randn(1, frames[0], D, device="cuda", dtype=prec.act_dtype)
            cache = torch.randn(1, D, fused.kc, device="cuda", dtype=prec.act_dtype)
            for bc, bk, w in itertools.product([2, 4, 8, 16], [64, 128, 256], [2, 4, 8]):
                os.environ.update(FC_BLOCK_C=str(bc), FC_BLOCK_N=str(bc), FC_BLOCK_K=str(bk), FC_WARPS=str(w))
                try:
                    t = cuda_time_us(lambda: fused(x, None, cache), iters=100)
                except Exception:
                    continue
                if best is None or t < best[0]:
                    best = (t, bc, bk, w)
            log(f"best tiles {pname}: block_c=block_n={best[1]} block_k={best[2]} warps={best[3]} -> {best[0]:.1f} us")
            os.environ.update(FC_BLOCK_C=str(best[1]), FC_BLOCK_N=str(best[1]), FC_BLOCK_K=str(best[2]), FC_WARPS=str(best[3]))
            res.write({"key": f"sweep|{pname}", "best": {"us": best[0], "block": best[1], "block_k": best[2], "warps": best[3]}})
        for B, T in itertools.product(args.batches, frames):
            key = f"{args.model}|L{args.layer}|{pname}|b{B}|T{T}"
            if res.has(key):
                log(f"skip {key}")
                continue
            rec = {"key": key, "model": args.model, "precision": pname, "batch": B, "frames": T, "d_model": D,
                   "weight_bytes": wbytes, "gpu": sysinfo.get("gpu"), "variants": {}}
            try:
                torch.manual_seed(0)
                x = torch.randn(B, T, D, device="cuda", dtype=prec.act_dtype)
                cache = torch.randn(B, D, fused.kc, device="cuda", dtype=prec.act_dtype)
                pm = torch.zeros(B, T, dtype=torch.bool, device="cuda")
                with torch.inference_mode():
                    ref_fn = lambda: reference_block(norm, conv, x, pm, cache)  # noqa: E731
                    with prec.ctx():
                        r_out, r_cache = ref_fn()
                    f_out, f_cache = fused(x, pm, cache)
                    rec["parity"] = {
                        "max_rel_err_out": ((f_out.float() - r_out.float()).norm() / r_out.float().norm()).item(),
                        "max_abs_err_cache": (f_cache.float() - r_cache.float()).abs().max().item()}
                    comp = torch.compile(reference_block, dynamic=False)

                    def c_fn():
                        with prec.ctx():
                            return comp(norm, conv, x, pm, cache)

                    def e_fn():
                        with prec.ctx():
                            return ref_fn()
                    variants = {
                        "nemo-eager": e_fn,
                        "nemo-compile": c_fn,
                        "nemo-cudagraph": graphed(e_fn),
                        "fused-triton": lambda: fused(x, pm, cache),
                        "fused-triton-cudagraph": graphed(lambda: fused(x, pm, cache)),
                    }
                    for name, fn in variants.items():
                        try:
                            gpu = cuda_time_us(fn, iters=200, warmup=20)
                            v = {"gpu_us_per_call": gpu, "wall_us_sync": wall_us(fn), "kernels": kernels_per_call(fn),
                                 "eff_weight_gbs": wbytes / gpu / 1e3}
                        except Exception as e:  # e.g. compile failure
                            v = {"error": f"{type(e).__name__}: {e}"[:500]}
                        rec["variants"][name] = v
                rec["ok"] = True
                ve = rec["variants"]
                log(f"{key}: " + " | ".join(f"{k} {v.get('wall_us_sync', float('nan')):.1f}us/{v.get('kernels', 0):.0f}k"
                                            for k, v in ve.items()) + f" | rel.err {rec['parity']['max_rel_err_out']:.1e}")
            except Exception as e:
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {e}"[:2000]
                rec["traceback"] = traceback.format_exc()[-3000:]
                log(f"{key}: FAILED {rec['error'][:300]}")
            res.write(rec)
            torch._dynamo.reset()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
