"""Megakernel benchmarks (results appended to results/<gpu>/megakernel.jsonl).

  python megakernel/mk_bench.py                       # layers-only + full step vs NeMo CUDA Graph, default grid
  python megakernel/mk_bench.py --sweep               # threads x blocks-per-SM sweep
  python megakernel/mk_bench.py --stages              # per-stage breakdown from in-kernel %globaltimer stamps
  python megakernel/mk_bench.py --paced               # also one chunk every 160 ms (real-time arrival)

Measured
  layers:   the 17 Conformer layers only (one cooperative launch)
  step:     full encoder step = pre-encode (captured in a CUDA Graph) + megakernel launch
  baseline: NeMo's whole encoder step captured in one CUDA Graph (same model, same precision)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mk_runtime import (NSTAGE, STAGE_NAMES, STAGE_WEIGHT_BYTES, MegakernelEncoder, capture_constants,  # noqa: E402
                        model_dtype, nemo_initial, nemo_step, random_chunks)
from mk_test import load  # noqa: E402


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(round((len(s) - 1) * q)))]


def time_fn(fn, n, paced_ms=0.0):
    """Per-call wall (incl. sync) and GPU-event time in µs."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    wall = []
    nxt = time.perf_counter()
    for i in range(n):
        if paced_ms:
            now = time.perf_counter()
            if nxt > now:
                time.sleep(nxt - now)
            nxt += paced_ms / 1e3
        t0 = time.perf_counter()
        starts[i].record()
        fn()
        ends[i].record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1e6)
    gpu = [starts[i].elapsed_time(ends[i]) * 1e3 for i in range(n)]
    st = lambda v: {"p50": pct(v, 0.5), "p99": pct(v, 0.99), "mean": statistics.fmean(v), "min": min(v)}  # noqa: E731
    return {"wall_us": st(wall), "gpu_us": st(gpu)}


class PreEncodeGraph:
    """NeMo pre-encode for one chunk captured in a CUDA Graph, writing straight into the megakernel input buffer."""

    def __init__(self, mk, xlen, drop, feat, T_in):
        self.mel = torch.zeros(1, feat, T_in, device="cuda", dtype=model_dtype(mk.model))
        self.mk, self.xlen, self.drop = mk, xlen, drop
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                mk.x_in.copy_(mk.pre_encode(self.mel, xlen, drop))
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g), torch.inference_mode():
            mk.x_in.copy_(mk.pre_encode(self.mel, xlen, drop))

    @torch.inference_mode()
    def __call__(self, mel=None):
        if mel is not None:
            self.mel.copy_(mel)
        self.g.replay()


class NemoGraph:
    """Baseline: NeMo's full cache-aware step in one CUDA Graph (caches updated in place)."""

    def __init__(self, model, x, xlen, drop):
        self.c, self.t, self.cl = nemo_initial(model, "cuda")
        self.x = x.clone().to(model_dtype(model))
        with torch.inference_mode():
            for _ in range(40):
                _, c, t, cl = nemo_step(model, self.x, xlen, self.c, self.t, self.cl, drop)
                self.c.copy_(c); self.t.copy_(t); self.cl.copy_(cl)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    nemo_step(model, self.x, xlen, self.c, self.t, self.cl, drop)
            torch.cuda.current_stream().wait_stream(s)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                o, c, t, cl = nemo_step(model, self.x, xlen, self.c, self.t, self.cl, drop)
                self.c.copy_(c); self.t.copy_(t); self.cl.copy_(cl)

    @torch.inference_mode()
    def __call__(self):
        self.g.replay()


def make_mk(model, consts, threads, blocks, timers, xs, xlen, drop, flags=2):
    mk = MegakernelEncoder(model, consts, threads=threads, blocks=blocks, timers=timers, flags=flags)
    c, t, cl = nemo_initial(model, "cuda")
    for x in xs[:40]:
        _, c, t, cl = nemo_step(model, x, xlen, c, t, cl, drop)
    mk.state.load_from_nemo(model, c, t, cl)
    mk.state.cache_len = 70      # steady state: keep the steady mask for every timed chunk
    return mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eou120m")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--paced", action="store_true")
    ap.add_argument("--flags", type=int, default=2, help="bit0 L2 prefetch, bit1 fused LN_out")
    ap.add_argument("--ablation", action="store_true", help="layers-only latency for flags 0,1,2,3")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load(args.model, "cuda", args.fp32)
    consts = capture_constants(model)
    xs, xlen, drop = random_chunks(model, 41, "cuda", seed=3)
    from mk_runtime import stream_geometry
    geo = stream_geometry(model)
    gpu = torch.cuda.get_device_name(0)
    slug = "".join(ch for ch in gpu.lower().replace("nvidia", "").replace("geforce", "") if ch.isalnum())
    out = Path(args.out) / slug
    out.mkdir(parents=True, exist_ok=True)
    resf = open(out / "megakernel.jsonl", "a")
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    prec = "fp32" if args.fp32 else "fp16"
    weight_bytes = sum(p.numel() * p.element_size() for p in model.encoder.layers.parameters()) // (2 if args.fp32 else 1)

    def record(rec):
        rec.update(gpu=gpu, precision=prec, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        resf.write(json.dumps(rec) + "\n"); resf.flush()

    # baseline: NeMo CUDA Graph (full step)
    nb = NemoGraph(model, xs[0], xlen, drop)
    for _ in range(30):
        nb()
    rb = time_fn(nb, args.iters)
    print(f"[base] NeMo CUDA Graph full step   wall p50 {rb['wall_us']['p50']:7.0f} us  p99 {rb['wall_us']['p99']:7.0f}")
    record({"kind": "baseline_cudagraph_step", **rb})
    if args.paced:
        rbp = time_fn(nb, 150, paced_ms=160)
        print(f"[base] paced                       wall p50 {rbp['wall_us']['p50']:7.0f} us  p99 {rbp['wall_us']['p99']:7.0f}")
        record({"kind": "baseline_cudagraph_step_paced", **rbp})
    del nb
    torch.cuda.empty_cache()

    configs = [(args.threads, args.blocks or n_sm)]
    if args.sweep:
        configs = []
        for th in (256,):                     # v5 kernel: 256 threads/block (250 registers)
            probe = MegakernelEncoder(model, consts, threads=th, blocks=n_sm)
            for k in (1, 2, 3, 4):
                if k * n_sm <= probe.max_blocks:
                    configs.append((th, k * n_sm))
            del probe
    if args.ablation:
        names = {0: "LN_out separate, no L2 prefetch", 1: "+ L2 prefetch of next layer", 2: "+ fused LN_out",
                 3: "both"}
        th, bl = configs[0]
        for fl in (0, 1, 2, 3):
            mk = make_mk(model, consts, th, bl, False, xs, xlen, drop, flags=fl)
            for _ in range(30):
                mk.layers_step()
            r = time_fn(lambda: mk.layers_step(), args.iters)
            print(f"[abl] flags {fl}  {names[fl]:44s} layers wall p50 {r['wall_us']['p50']:7.0f} us  gpu p50 {r['gpu_us']['p50']:7.0f}")
            record({"kind": "mk_ablation", "flags": fl, "threads": th, "blocks": bl, **r})
            del mk
            torch.cuda.empty_cache()

    for th, bl in configs:
        mk = make_mk(model, consts, th, bl, False, xs, xlen, drop, flags=args.flags)
        for _ in range(30):
            mk.layers_step()
        rl = time_fn(lambda: mk.layers_step(), args.iters)
        bw = weight_bytes / rl["gpu_us"]["p50"] / 1e3
        print(f"[mk ] layers  grid {bl:4d} x {th:3d} f{args.flags}   wall p50 {rl['wall_us']['p50']:7.0f} us  p99 {rl['wall_us']['p99']:7.0f}"
              f"  gpu p50 {rl['gpu_us']['p50']:7.0f}  eff {bw:5.0f} GB/s")
        record({"kind": "mk_layers", "threads": th, "blocks": bl, "flags": args.flags, "eff_weight_gbs": bw, **rl})
        pe = PreEncodeGraph(mk, xlen, drop, geo["feat"], geo["T_in"])

        def full_step():
            pe()
            mk.layers_step()
        for _ in range(30):
            full_step()
        rs = time_fn(full_step, args.iters)
        rp = time_fn(pe, args.iters)
        print(f"[mk ] step    grid {bl:4d} x {th:3d}   wall p50 {rs['wall_us']['p50']:7.0f} us  p99 {rs['wall_us']['p99']:7.0f}"
              f"   (pre-encode graph alone {rp['wall_us']['p50']:.0f} us)")
        record({"kind": "mk_step", "threads": th, "blocks": bl, "flags": args.flags, "pre_encode": rp, **rs})
        if args.paced and (th, bl) == configs[0]:
            rsp = time_fn(full_step, 150, paced_ms=160)
            print(f"[mk ] paced                       wall p50 {rsp['wall_us']['p50']:7.0f} us  p99 {rsp['wall_us']['p99']:7.0f}")
            record({"kind": "mk_step_paced", "threads": th, "blocks": bl, **rsp})
        del pe, mk
        torch.cuda.empty_cache()

    if args.stages:
        th, bl = configs[0]
        mk = make_mk(model, consts, th, bl, True, xs, xlen, drop, flags=args.flags)
        per = [[] for _ in range(NSTAGE)]
        ph = [[[], [], []] for _ in range(NSTAGE - 1)]
        first = []
        for i in range(200):
            mk.layers_step()
            torch.cuda.synchronize()
            d = mk.stage_times_us()
            pp = mk.stage_phases_us()
            if i < 20:
                continue
            first.append(d[0])
            for l in range(mk.nl):
                for s in range(NSTAGE):
                    per[s].append(d[1 + l * NSTAGE + s])
                for s in range(NSTAGE - 1):
                    for k in range(3):
                        ph[s][k].append(pp[l][s][k])
        print(f"[mk ] per-stage (median over layers x chunks), input copy {statistics.median(first):.2f} us")
        print(f"      {'stage':24s} {'total':>8s} {'GB/s':>6s} | block 0: {'load':>6s} {'compute':>8s} {'barrier':>8s}")
        rows = []
        for s in range(NSTAGE):
            m = statistics.median(per[s])
            gbs = STAGE_WEIGHT_BYTES[s] / m / 1e3 if STAGE_WEIGHT_BYTES[s] else 0.0
            r = {"stage": STAGE_NAMES[s], "median_us": m, "weight_bytes": STAGE_WEIGHT_BYTES[s], "gbs": gbs}
            extra = ""
            if s < NSTAGE - 1:
                lo, co, ba = (statistics.median(ph[s][k]) for k in range(3))
                r.update(load_us=lo, compute_us=co, barrier_us=ba)
                extra = f" | {lo:15.2f} {co:8.2f} {ba:8.2f}"
            rows.append(r)
            print(f"      {STAGE_NAMES[s]:24s} {m:8.2f} {gbs:6.0f}{extra}")
        tot = sum(r["median_us"] for r in rows) * mk.nl
        print(f"      sum x {mk.nl} layers = {tot:.0f} us   (barrier column = waiting for the slowest block + grid.sync + next-stage prefetch)")
        record({"kind": "mk_stages", "threads": th, "blocks": bl, "flags": args.flags, "stages": rows, "sum_us": tot})
    resf.close()


if __name__ == "__main__":
    main()
