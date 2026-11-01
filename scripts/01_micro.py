"""Hardware microbenchmarks: the numbers the roofline and overhead arguments rest on.

Measures, on the current GPU:
  1. DRAM bandwidth (copy and read-only), and an effective-bandwidth vs working-set sweep (shows L2).
  2. Per-op cost of eager PyTorch on a tiny tensor (host overhead) vs the same ops replayed in a CUDA Graph
     (per-kernel launch gap on the GPU).
  3. GEMV efficiency at exactly the matrix shapes of a d=512 Conformer layer, single launch vs graph-batched.

Usage: python scripts/01_micro.py [--out results]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrbench.common import cuda_time_us, log, out_dir, system_info  # noqa: E402


def bandwidth(total_mb: int):
    n = total_mb * 2**20 // 4
    a = torch.empty(n, dtype=torch.float32, device="cuda").normal_()
    b = torch.empty_like(a)
    t_copy = cuda_time_us(lambda: b.copy_(a), iters=20)
    t_read = cuda_time_us(lambda: a.sum(), iters=20)
    bytes_ = n * 4
    return {"size_mb": total_mb, "copy_gbs": 2 * bytes_ / t_copy / 1e3, "read_gbs": bytes_ / t_read / 1e3}


def ws_sweep(sizes_mb):
    """Effective read bandwidth when the same buffer is read repeatedly (fits in L2 => higher)."""
    out = []
    for mb in sizes_mb:
        n = int(mb * 2**20) // 2
        a = torch.empty(n, dtype=torch.float16, device="cuda").normal_()
        reps = max(1, int(256 / max(mb, 0.25)))

        def f():
            for _ in range(reps):
                a.sum(dtype=torch.float32)
        t = cuda_time_us(f, iters=5, warmup=2)
        out.append({"size_mb": mb, "read_gbs": n * 2 * reps / t / 1e3})
    return out


def op_overhead(n_ops=2000):
    x = torch.zeros(1, device="cuda")
    for _ in range(100):
        x.add_(1)
    torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    for _ in range(n_ops):
        x.add_(1)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    eager_issue_us = (t1 - t0) / n_ops * 1e6
    eager_total_us = (t2 - t0) / n_ops * 1e6
    # same ops inside one CUDA graph
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x.add_(1)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(n_ops):
            x.add_(1)
    t_graph = cuda_time_us(lambda: g.replay(), iters=10, warmup=3)
    single = cuda_time_us(lambda: x.add_(1), iters=200)
    return {"eager_host_issue_us_per_op": eager_issue_us, "eager_wall_us_per_op": eager_total_us,
            "graph_us_per_kernel": t_graph / n_ops, "stream_us_per_tiny_kernel": single}


def gemv(shapes, dtype=torch.float16, n_in_graph=50):
    rows = []
    for (k, n) in shapes:
        W = torch.randn(n, k, device="cuda", dtype=dtype)
        x = torch.randn(1, k, device="cuda", dtype=dtype)
        f = lambda: torch.nn.functional.linear(x, W)  # noqa: E731
        t1 = cuda_time_us(f, iters=200)
        # many independent GEMVs back to back inside a graph: removes launch cost, keeps kernel ramp/tail
        Ws = [torch.randn(n, k, device="cuda", dtype=dtype) for _ in range(n_in_graph)]
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for w in Ws:
                torch.nn.functional.linear(x, w)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for w in Ws:
                torch.nn.functional.linear(x, w)
        tg = cuda_time_us(lambda: g.replay(), iters=10) / n_in_graph
        bytes_ = n * k * W.element_size()
        rows.append({"k": k, "n": n, "dtype": str(dtype).split(".")[-1], "weight_mb": bytes_ / 2**20,
                     "us_single": t1, "gbs_single": bytes_ / t1 / 1e3,
                     "us_in_graph": tg, "gbs_in_graph": bytes_ / tg / 1e3})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--bw-mb", type=int, default=0, help="buffer size for DRAM test (default: auto)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    od = out_dir(args.out)
    info = system_info()
    log(f"GPU: {info.get('gpu')}  SMs={info.get('sm_count')}  L2={info.get('l2_mb')} MB")
    vram_mb = int(info.get("vram_gb", 4) * 1024)
    bw_mb = args.bw_mb or min(1024, vram_mb // 8)
    res = {"system": info}
    res["dram"] = bandwidth(bw_mb)
    log(f"DRAM copy {res['dram']['copy_gbs']:.0f} GB/s, read {res['dram']['read_gbs']:.0f} GB/s")
    l2 = info.get("l2_mb") or 4
    sizes = sorted({0.5, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, round(l2 / 2, 1), l2})
    res["ws_sweep"] = ws_sweep([s for s in sizes if s * 2 < vram_mb])
    res["overhead"] = op_overhead()
    o = res["overhead"]
    log(f"eager op: issue {o['eager_host_issue_us_per_op']:.2f} us, wall {o['eager_wall_us_per_op']:.2f} us; "
        f"graph per kernel {o['graph_us_per_kernel']:.2f} us")
    shapes = [(512, 2048), (2048, 512), (512, 1024), (512, 512), (1024, 4096), (4096, 1024), (1024, 1024)]
    res["gemv_fp16"] = gemv(shapes, torch.float16)
    res["gemv_fp32"] = gemv(shapes[:4], torch.float32)
    for r in res["gemv_fp16"]:
        log(f"GEMV {r['k']}x{r['n']} fp16 ({r['weight_mb']:.2f} MB): single {r['us_single']:.1f} us "
            f"({r['gbs_single']:.0f} GB/s), in-graph {r['us_in_graph']:.1f} us ({r['gbs_in_graph']:.0f} GB/s)")
    (od / "micro.json").write_text(json.dumps(res, indent=1))
    log(f"wrote {od / 'micro.json'}")


if __name__ == "__main__":
    main()
