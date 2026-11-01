"""Vendor-stack baselines: cache-aware encoder step exported to ONNX, run with ONNX Runtime (CUDA EP,
optionally with CUDA graphs) and TensorRT (via ORT's TensorRT EP).

The exported graph is exactly one steady-state streaming step:
  (mel[B,F,T_in], len[B], cache_ch[L,B,C,D], cache_t[L,B,D,K], cache_len[B]) -> (enc, enc_len, caches...)
Inputs/outputs stay on the GPU via IOBinding and share PyTorch's CUDA stream, so timing is comparable
with 02_latency.py. Caches are fed back by a device-to-device copy each step (counted in the latency).

Requirements on the GPU machine:  pip install onnxruntime-gpu   (and for TensorRT: pip install tensorrt-cu12,
then make sure libnvinfer is on LD_LIBRARY_PATH; the script records a failure if the EP is unavailable).

Usage: python scripts/04_onnx_trt.py --model eou120m [--variants ort-cuda ort-cuda-graph ort-cuda-fp16 trt-fp16]
       python scripts/04_onnx_trt.py --export-only --device cpu     (sanity check export without a GPU)
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrbench.common import Results, log, out_dir, summarize, system_info, time_steps  # noqa: E402
from asrbench.model import (StreamState, apply_precision, att_contexts, chunk_geometry, encoder_param_bytes,  # noqa: E402
                            load_model, precision, reference_sequence, set_att_context, warm_caches)

IN_NAMES = ["audio_signal", "length", "cache_last_channel", "cache_last_time", "cache_last_channel_len"]
OUT_NAMES = ["encoded", "encoded_len", "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"]
NP = {torch.float32: np.float32, torch.float16: np.float16, torch.int64: np.int64, torch.int32: np.int32}


class StepModule(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x, xlen, c, t, cl):
        rets = self.enc.forward_internal(x, xlen, cache_last_channel=c, cache_last_time=t, cache_last_channel_len=cl)
        return self.enc.streaming_post_process(rets, keep_all_outputs=False)


def export(model, state: StreamState, path: Path):
    enc = model.encoder
    geom = chunk_geometry(model)
    enc.update_max_seq_length(seq_length=geom["input_mel"] + geom["cache_len"] + 16, device=state.x.device)
    m = StepModule(enc).eval()
    args = (state.x, state.xlen, state.c, state.t, state.cl)
    with torch.inference_mode(False), torch.no_grad():
        kw = dict(input_names=IN_NAMES, output_names=OUT_NAMES, opset_version=17, do_constant_folding=True)
        try:
            torch.onnx.export(m, args, str(path), dynamo=False, **kw)
        except TypeError:  # older torch without the dynamo kwarg
            torch.onnx.export(m, args, str(path), **kw)
    return path


class OrtStep:
    """One streaming step through ONNX Runtime with IOBinding on PyTorch's stream."""

    def __init__(self, onnx_path, state: StreamState, providers, out_shapes, out_dtypes):
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):  # reuse the CUDA/cuDNN libraries that came with PyTorch
            try:
                ort.preload_dlls()
            except Exception:
                pass
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        self.active = self.sess.get_providers()
        self.s = state
        self.outs = [torch.empty(sh, dtype=dt, device="cuda") for sh, dt in zip(out_shapes, out_dtypes)]
        self.io = self.sess.io_binding()
        ins = [state.x, state.xlen, state.c, state.t, state.cl]
        for n, tsr in zip(IN_NAMES, ins):
            self.io.bind_input(n, "cuda", 0, NP[tsr.dtype], list(tsr.shape), tsr.data_ptr())
        for n, tsr in zip(OUT_NAMES, self.outs):
            self.io.bind_output(n, "cuda", 0, NP[tsr.dtype], list(tsr.shape), tsr.data_ptr())

    def __call__(self):
        self.sess.run_with_iobinding(self.io)
        s, o = self.s, self.outs
        s.c.copy_(o[2]); s.t.copy_(o[3]); s.cl.copy_(o[4])
        return o[0]


def providers_for(variant, cache_dir):
    stream = str(torch.cuda.current_stream().cuda_stream)
    cuda_opts = {"device_id": 0, "user_compute_stream": stream, "has_user_compute_stream": "1"}
    if variant.startswith("ort-cuda"):
        o = dict(cuda_opts)
        if variant.endswith("graph"):
            o["enable_cuda_graph"] = "1"
        return [("CUDAExecutionProvider", o)]
    if variant.startswith("trt"):
        t = {"device_id": 0, "user_compute_stream": stream, "has_user_compute_stream": "1",
             "trt_engine_cache_enable": "True", "trt_engine_cache_path": str(cache_dir),
             "trt_fp16_enable": "True" if "fp16" in variant else "False",
             "trt_builder_optimization_level": "5"}
        if variant.endswith("graph"):
            t["trt_cuda_graph_enable"] = "True"
        return [("TensorrtExecutionProvider", t), ("CUDAExecutionProvider", cuda_opts)]
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eou120m")
    ap.add_argument("--att", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--variants", nargs="+",
                    default=["ort-cuda", "ort-cuda-graph", "ort-cuda-fp16", "ort-cuda-fp16-graph", "trt-fp32", "trt-fp16",
                             "trt-fp16-graph"])
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    if args.device == "cpu":  # export sanity check only
        model = load_model(args.model, "cpu")
        att = list(map(int, args.att.split(","))) if args.att else att_contexts(args.model, model)[0]
        set_att_context(model, att)
        prec = precision("fp32")
        st = StreamState(model, args.batch, prec, device="cpu")
        warm_caches(model, st, prec, 3)
        p = export(model, st, Path("/tmp") / f"{args.model}_step.onnx")
        import onnxruntime as ort

        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        feeds = dict(zip(IN_NAMES, [v.numpy() for v in (st.x, st.xlen, st.c, st.t, st.cl)]))
        o = sess.run(None, feeds)
        with torch.inference_mode():
            r = StepModule(model.encoder)(st.x, st.xlen, st.c, st.t, st.cl)
        err = np.abs(o[0] - r[0].numpy()).max()
        log(f"exported {p} ({p.stat().st_size/2**20:.0f} MB); ORT-CPU vs torch max abs err {err:.2e}")
        return

    sysinfo = system_info()
    od = out_dir(args.out)
    cache_dir = od / "trt_cache"
    cache_dir.mkdir(exist_ok=True)
    res = Results(od / "onnx_trt.jsonl")
    onnx_dir = od / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    ref = None
    for variant in args.variants:
        prec = precision("fp16" if "ort-cuda-fp16" in variant else "fp32")  # TRT fp16 builds from the fp32 graph
        model = load_model(args.model, "cuda")
        apply_precision(model, prec)
        att = list(map(int, args.att.split(","))) if args.att else att_contexts(args.model, model)[0]
        set_att_context(model, att)
        key = f"{args.model}|{att[0]},{att[1]}|{variant}|b{args.batch}"
        if res.has(key):
            log(f"skip {key}")
            continue
        rec = {"key": key, "model": args.model, "att": att, "variant": variant, "batch": args.batch,
               "gpu": sysinfo.get("gpu"), "geom": chunk_geometry(model), "impl": variant,
               "precision": "fp16" if "fp16" in variant else "fp32", "patch": "none",
               "weight_bytes": encoder_param_bytes(model) // (2 if "fp16" in variant and prec.name == "fp32" else 1)}
        try:
            import onnxruntime as ort

            rec["ort_version"] = ort.__version__
            st = StreamState(model, args.batch, prec)
            warm_caches(model, st, prec, 40)
            path = onnx_dir / f"{args.model}_{att[0]}-{att[1]}_b{args.batch}_{prec.name}.onnx"
            if not path.exists():
                t0 = time.perf_counter()
                export(model, st, path)
                rec["export_s"] = time.perf_counter() - t0
            with torch.inference_mode():
                r = StepModule(model.encoder)(st.x, st.xlen, st.c, st.t, st.cl)
            shapes, dtypes = [tuple(x.shape) for x in r], [x.dtype for x in r]
            t0 = time.perf_counter()
            step = OrtStep(path, st, providers_for(variant, cache_dir), shapes, dtypes)
            rec["providers_active"] = step.active
            if variant.startswith("trt") and "TensorrtExecutionProvider" not in step.active:
                raise RuntimeError(f"TensorRT EP not available (active: {step.active})")
            for _ in range(30):
                step()
            torch.cuda.synchronize()
            rec["build_s"] = time.perf_counter() - t0
            t = time_steps(step, args.iters)
            rec["b2b"] = {"wall_us": summarize(t.wall_us), "gpu_us": summarize(t.gpu_us), "host_us": summarize(t.host_us)}
            rec["b2b_gpu_samples"] = [round(v, 2) for v in t.gpu_us]
            rec["eff_weight_gbs"] = rec["weight_bytes"] / rec["b2b"]["gpu_us"]["p50"] / 1e3
            # parity: run the ORT step on a fresh stream with the reference inputs
            if ref is None:
                m32 = load_model(args.model, "cuda"); set_att_context(m32, att)
                ref = [o.cpu() for o in reference_sequence(m32, precision("fp32"), args.batch, 30)]
                del m32; torch.cuda.empty_cache()
            gen = torch.Generator(device="cpu").manual_seed(124)
            st.c.zero_(); st.t.zero_(); st.cl.zero_()
            errs_abs, errs_rel = 0.0, 0.0
            for i in range(30):
                st.x.copy_(torch.randn(st.x.shape, generator=gen).to(st.x))
                y = step().float()
                d = (y.cpu() - ref[i])
                errs_abs = max(errs_abs, d.abs().max().item())
                errs_rel = max(errs_rel, (d.norm() / ref[i].norm()).item())
            rec["parity_vs_fp32_eager"] = {"max_abs_err": errs_abs, "max_rel_err": errs_rel}
            rec["ok"] = True
            log(f"{key}: wall p50 {rec['b2b']['wall_us']['p50']:.0f} us, gpu p50 {rec['b2b']['gpu_us']['p50']:.0f} us, "
                f"build {rec['build_s']:.1f}s, rel.err {errs_rel:.1e}  [{','.join(step.active)}]")
            del step
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"[:2000]
            rec["traceback"] = traceback.format_exc()[-3000:]
            log(f"{key}: FAILED {rec['error'][:300]}")
        res.write(rec)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
