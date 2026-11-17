"""Parity: megakernel (or the PyTorch reference of the same algorithm) vs NeMo's own cache-aware streaming step.

Procedure: run NeMo for --warm chunks (fills caches), hand the caches over to the megakernel state, then feed the
next --steps chunks to both and compare encoder outputs chunk by chunk (and caches at the end).

  GPU, kernel:      python megakernel/mk_test.py
  GPU, reference:   python megakernel/mk_test.py --reference
  CPU, reference:   python megakernel/mk_test.py --device cpu --reference --fp32
  first L layers:   python megakernel/mk_test.py --layers 1        (debug: compares after layer L via a hook)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mk_runtime import (CACHE, MegakernelEncoder, capture_constants, nemo_initial, nemo_step,  # noqa: E402
                        random_chunks)


def load(name, device, fp32):
    import logging
    import nemo.collections.asr as nemo_asr
    from nemo.utils import logging as nl
    nl.setLevel(logging.ERROR)
    full = {"eou120m": "nvidia/parakeet_realtime_eou_120m-v1"}.get(name, name)
    m = nemo_asr.models.ASRModel.from_pretrained(full, map_location=device)
    m.eval()
    if not fp32:
        m.encoder.half()
    try:
        m.encoder.set_default_att_context_size([70, 1])
    except Exception:
        pass
    return m


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / (b.norm() + 1e-12)).item(), (a - b).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eou120m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true", help="FP32 NeMo model (kernel still streams FP16 weights)")
    ap.add_argument("--reference", action="store_true", help="run the PyTorch reference instead of the CUDA kernel")
    ap.add_argument("--warm", type=int, default=40)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers (debug); 0 = all")
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--flags", type=int, default=0, help="bit0 L2 prefetch, bit1 fused LN_out, bit3 adjacent rows (default 0)")
    ap.add_argument("--tol", type=float, default=0.0, help="max allowed relative error (default: 2e-2 FP16, 1e-3 FP32)")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = args.device
    model = load(args.model, dev, args.fp32)
    t0 = time.time()
    consts = capture_constants(model)
    print(f"[mk] constants: T={consts['T']} NK={consts['NK']} npos={consts['npos']} "
          f"masks for cache lengths {sorted(consts['masks'])[:3]}...{sorted(consts['masks'])[-2:]} ({time.time()-t0:.1f}s)")
    mk = MegakernelEncoder(model, consts, threads=args.threads, blocks=args.blocks or None,
                           use_kernel=not args.reference, device=dev, flags=args.flags)
    if args.layers:
        mk.nl = args.layers
        mk.layers = mk.layers[:args.layers]
        if mk.use_kernel:
            mk.params = mk.params[:args.layers * len(mk.layers[0])].contiguous()
    print(f"[mk] mode: {'reference (PyTorch)' if args.reference else f'CUDA kernel, grid {mk.blocks} x {mk.threads}'}"
          f", layers {mk.nl}, flags {args.flags}")

    xs, xlen, drop = random_chunks(model, args.warm + args.steps, dev, seed=7)
    c, t, cl = nemo_initial(model, dev)
    for x in xs[:args.warm]:
        _, c, t, cl = nemo_step(model, x, xlen, c, t, cl, drop)
    mk.state.load_from_nemo(model, c, t, cl)

    # debug: capture NeMo's output after layer N
    cap = {}
    hook = None
    if args.layers:
        def fwd_hook(mod, inp, out):
            cap["x"] = (out[0] if isinstance(out, tuple) else out).detach()
        hook = model.encoder.layers[args.layers - 1].register_forward_hook(fwd_hook)

    worst = 0.0
    for i, x in enumerate(xs[args.warm:]):
        ref, c, t, cl = nemo_step(model, x, xlen, c, t, cl, drop)
        if args.layers:
            xin = mk.pre_encode(x, xlen, drop)
            out = mk.layers_step(xin).clone()
            r, mx = rel(out, cap["x"][0])
        else:
            out = mk.step(x, xlen, drop)
            if dev == "cuda":
                torch.cuda.synchronize()
            r, mx = rel(out, ref)
        worst = max(worst, r)
        if i < 3 or i == args.steps - 1:
            print(f"  chunk {i:2d}: rel.err {r:.2e}  max.abs {mx:.2e}")
    if hook:
        hook.remove()
    # caches after the run: compare conv cache directly, attention cache via projected K
    if not args.layers:
      with torch.inference_mode():
        conv_r = rel(mk.state.conv.permute(0, 1, 2), t[:, 0].float())[0]
        head = int(mk.state.head.item())
        order = [(head + j) % CACHE for j in range(CACHE)]
        a0 = model.encoder.layers[0].self_attn
        k_ref = a0.linear_k(c[0, 0]).float()
        k_r = rel(mk.state.kv[0, 0][order], k_ref)[0]
      print(f"  caches after {args.steps} chunks: conv rel.err {conv_r:.2e}, layer-0 K rel.err {k_r:.2e}")
    tol = args.tol or (1e-3 if args.fp32 else 2e-2)
    ok = worst <= tol
    print(f"[mk] worst rel.err {worst:.2e} (tolerance {tol:.0e}) -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
