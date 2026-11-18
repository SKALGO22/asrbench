"""Persistent megakernel vs one launch per chunk (results -> results/<gpu>/megakernel_persistent.jsonl).

Both paths use the same zero-copy host buffers, so the only difference is how a chunk is started:
  launch:      cudaLaunchCooperativeKernel + cudaStreamSynchronize per chunk
  persistent:  kernel launched once and resident; the CPU writes the input + a sequence number into mapped host
               memory and spins until the kernel writes `done` (block 0 polls; other blocks wait in grid.sync)
Measured: correctness (identical outputs from the same state), back-to-back and real-time paced (one chunk every
160 ms) latency, and GPU power / SM clock during the paced runs and while idle.

  python megakernel/mk_persist.py [--flags 2] [--iters 500] [--paced 150]
"""
import argparse
import ctypes
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mk_runtime import (CACHE, D, HostBuffer, MegakernelEncoder, capture_constants, check, lib,  # noqa: E402
                        make_stream, nemo_initial, nemo_step, random_chunks)
from mk_test import load  # noqa: E402


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(round((len(s) - 1) * q)))]


def summ(v):
    return {"p50": pct(v, 0.5), "p99": pct(v, 0.99), "mean": statistics.fmean(v), "min": min(v), "max": max(v)}


class NvmlSampler:
    """Background sampler of GPU power (W) and SM clock (MHz)."""

    def __init__(self, period=0.05):
        import pynvml
        pynvml.nvmlInit()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(int(vis) if vis.isdigit() else 0)
        self.period, self.samples, self.on = period, [], False

    def _run(self):
        while self.on:
            try:
                p = self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0
                c = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
                self.samples.append((p, c))
            except Exception:
                pass
            time.sleep(self.period)

    def __enter__(self):
        self.samples, self.on = [], True
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.on = False
        self.t.join()

    def stats(self):
        if not self.samples:
            return {}
        p = [s[0] for s in self.samples]; c = [s[1] for s in self.samples]
        return {"power_w_mean": statistics.fmean(p), "sm_clock_mhz_mean": statistics.fmean(c),
                "sm_clock_mhz_min": min(c), "n": len(p)}


def run_loop(step, X, n, paced_ms=0.0):
    lat = []
    nxt = time.perf_counter()
    for i in range(n):
        if paced_ms:
            now = time.perf_counter()
            if nxt > now:
                time.sleep(nxt - now)
            nxt += paced_ms / 1e3
        t0 = time.perf_counter()
        step(X[i % len(X)])
        lat.append((time.perf_counter() - t0) * 1e6)
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eou120m")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--flags", type=int, default=2)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--paced", type=int, default=150)
    ap.add_argument("--int8", action="store_true", help="weight-only INT8 megakernel")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    model = load(args.model, "cuda", args.fp32)
    consts = capture_constants(model)
    mels, xlen, drop = random_chunks(model, 70, "cuda", seed=5)
    mk = MegakernelEncoder(model, consts, flags=args.flags, quant="int8" if args.int8 else "fp16")
    c, t, cl = nemo_initial(model, "cuda")
    for x in mels[:40]:
        _, c, t, cl = nemo_step(model, x, xlen, c, t, cl, drop)
    mk.state.load_from_nemo(model, c, t, cl)
    X = [mk.pre_encode(x, xlen, drop).cpu().numpy().astype(np.float32).copy() for x in mels[40:70]]
    snap = (mk.state.kv.clone(), mk.state.conv.clone(), mk.state.head.clone())
    torch.cuda.synchronize()

    L = lib()
    L.mk_post.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.mk_wait.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong]
    n = mk.T * D
    hx, hy, mb = HostBuffer(n), HostBuffer(n), HostBuffer(4, ctypes.c_int)
    stream = make_stream()
    gpu = torch.cuda.get_device_name(0)
    slug = "".join(ch for ch in gpu.lower().replace("nvidia", "").replace("geforce", "") if ch.isalnum())
    outd = Path(args.out) / slug; outd.mkdir(parents=True, exist_ok=True)
    resf = open(outd / "megakernel_persistent.jsonl", "a")

    def record(rec):
        rec.update(gpu=gpu, flags=args.flags, weights="int8" if args.int8 else "fp16", ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        resf.write(json.dumps(rec) + "\n"); resf.flush()

    def restore():
        mk.state.kv.copy_(snap[0]); mk.state.conv.copy_(snap[1]); mk.state.head.copy_(snap[2])
        torch.cuda.synchronize()

    def launch_step(x):
        np.copyto(hx.arr, x.ravel())
        mk.launch_raw(hx.dptr, hy.dptr, stream, flags=args.flags, cache_len=CACHE)
        check(L.mk_stream_sync(stream), "stream sync")
        return hy.arr.copy()

    seq = [0]

    def persist_step(x):
        seq[0] += 1
        L.mk_post(hx.hptr, x.ctypes.data, n, mb.hptr, seq[0], CACHE, 1)
        if L.mk_wait(mb.hptr, seq[0], 20_000_000_000) != 0:
            raise TimeoutError("persistent kernel did not answer")
        return hy.arr.copy()

    def persist_start():
        mb.arr[:] = 0; seq[0] = 0
        mk.launch_raw(hx.dptr, hy.dptr, stream, flags=args.flags | 16, mbox_ptr=mb.dptr)

    def persist_stop():
        seq[0] += 1
        L.mk_post(hx.hptr, None, 0, mb.hptr, seq[0], CACHE, 2)
        check(L.mk_stream_sync(stream), "persistent exit")

    nvs = NvmlSampler()

    # ---- correctness: same inputs, same starting state, both paths
    ref = [launch_step(x) for x in X]
    restore()
    persist_start()
    got = [persist_step(x) for x in X]
    persist_stop()
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
    print(f"[per] correctness: max |launch - persistent| over {len(X)} chunks = {diff:.3e}")
    record({"kind": "persistent_parity", "max_abs_diff": diff, "chunks": len(X)})

    # ---- idle power: nothing resident vs persistent kernel resident and polling
    restore()
    with nvs:
        time.sleep(3.0)
    idle_none = nvs.stats()
    persist_start()
    persist_step(X[0])
    with nvs:
        time.sleep(3.0)
    idle_res = nvs.stats()
    persist_stop()
    print(f"[per] idle: no kernel {idle_none.get('power_w_mean', 0):.0f} W @ {idle_none.get('sm_clock_mhz_mean', 0):.0f} MHz;"
          f" persistent resident {idle_res.get('power_w_mean', 0):.0f} W @ {idle_res.get('sm_clock_mhz_mean', 0):.0f} MHz")
    record({"kind": "idle_power", "no_kernel": idle_none, "persistent_resident": idle_res})

    # ---- latency: launch path
    restore()
    for x in X[:10]:
        launch_step(x)
    lb = run_loop(launch_step, X, args.iters)
    with nvs:
        lp = run_loop(launch_step, X, args.paced, paced_ms=160)
    lp_nv = nvs.stats()
    # ---- latency: persistent path
    restore()
    persist_start()
    for x in X[:10]:
        persist_step(x)
    pb = run_loop(persist_step, X, args.iters)
    with nvs:
        pp = run_loop(persist_step, X, args.paced, paced_ms=160)
    pp_nv = nvs.stats()
    persist_stop()

    rows = [("launch per chunk", lb, lp, lp_nv), ("persistent", pb, pp, pp_nv)]
    print(f"[per] {'mode':18s} {'b2b p50':>9s} {'b2b p99':>9s} | {'paced p50':>9s} {'paced p99':>9s} {'paced max':>9s} | power  SM clock (paced)")
    for name, b, p, nv in rows:
        sb, sp = summ(b), summ(p)
        print(f"[per] {name:18s} {sb['p50']:9.0f} {sb['p99']:9.0f} | {sp['p50']:9.0f} {sp['p99']:9.0f} {sp['max']:9.0f} |"
              f" {nv.get('power_w_mean', 0):4.0f} W {nv.get('sm_clock_mhz_mean', 0):5.0f} MHz (min {nv.get('sm_clock_mhz_min', 0):.0f})")
        record({"kind": "persistent_latency", "mode": name, "b2b_us": sb, "paced_us": sp, "paced_nvml": nv})
    print("[per] latency = host writes input -> host sees output (zero-copy buffers, layers only; pre-encode excluded)")
    resf.close()


if __name__ == "__main__":
    main()
