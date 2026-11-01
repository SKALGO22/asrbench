"""Streaming word error rate on LibriSpeech, per implementation: proves baselines and our changes compute
the same thing (and quantifies any precision effect).

Runs true cache-aware streaming (NeMo's CacheAwareStreamingAudioBuffer chunking, RNN-T greedy decoding with
partial hypotheses, exactly as conformer_stream_step does) but lets the encoder step be:
  eager (any precision), a CUDA graph for steady-state chunks (eager fallback for the first/last chunk),
  and optionally the prelim patches A (K/V + position caching) and B (fused conv).

Usage:
  bash data/get_librispeech.sh              # downloads test-clean (+ test-other) into data/LibriSpeech
  python scripts/05_wer.py --modes fp32:eager fp16:eager fp16:cudagraph fp16:cudagraph:AB --max-utts 200
"""
import argparse
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrbench.common import Results, log, out_dir, system_info, triton_ok  # noqa: E402
from asrbench.model import (apply_precision, att_contexts, chunk_geometry, encoder_step_fn, load_model,  # noqa: E402
                            precision, set_att_context)


DEVICE = "cuda"


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- data & scoring

def librispeech(root: Path, subset: str, max_utts: int):
    items = []
    for trans in sorted((root / subset).rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            items.append((str(trans.parent / f"{uid}.flac"), text))
    if max_utts:
        items = items[:max_utts]
    return items


def normalize(s: str) -> str:
    s = re.sub(r"<[^>]*>", " ", s.lower())
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def edit_distance(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)]


def wer(refs, hyps):
    errs = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    words = sum(len(r.split()) for r in refs)
    return errs / max(words, 1), errs, words


# ----------------------------------------------------------------------------- encoder runners

class GraphRunner:
    """CUDA graph for steady-state chunks; anything irregular (first chunk, last chunk, short lengths) runs eager."""

    def __init__(self, model, prec, batch):
        self.f = encoder_step_fn(model, prec)
        g = chunk_geometry(model)
        self.T, self.drop = g["input_mel"], g["drop_extra_pre_encoded"]
        enc = model.encoder
        self.x = torch.zeros(batch, enc._feat_in, self.T, device="cuda", dtype=prec.act_dtype)
        self.xlen = torch.full((batch,), self.T, dtype=torch.int64, device="cuda")
        self.c, self.t, self.cl = enc.get_initial_cache_state(batch_size=batch, dtype=prec.act_dtype, device="cuda")
        self.graph = None
        self.batch = batch

    def _capture(self):
        c0, t0, cl0 = self.c.clone(), self.t.clone(), self.cl.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                o = self.f(self.x, self.xlen, self.c, self.t, self.cl, self.drop)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            o = self.f(self.x, self.xlen, self.c, self.t, self.cl, self.drop)
            self.c.copy_(o[2]); self.t.copy_(o[3]); self.cl.copy_(o[4])
            self.out, self.out_len = o[0], o[1]
        self.c.copy_(c0); self.t.copy_(t0); self.cl.copy_(cl0)  # capture must not advance the stream


def plain_greedy(model):
    """Streaming with partial hypotheses needs alignments/timestamps off (the EOU model ships with them on)."""
    import copy

    from omegaconf import open_dict

    cfg = copy.deepcopy(model.cfg.decoding)
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.preserve_alignments = False
        cfg.compute_timestamps = False
    model.change_decoding_strategy(cfg, verbose=False) if "verbose" in model.change_decoding_strategy.__code__.co_varnames \
        else model.change_decoding_strategy(cfg)


def run_mode(model, mname, prec, impl, patch, items, batch, log_every=50):
    from asrbench import fused_conv, prelim_attn
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    fused_conv.remove(model); prelim_attn.remove(model)
    if "A" in patch:
        prelim_attn.apply(model)
    if "B" in patch:
        fused_conv.apply(model)
    plain_greedy(model)
    enc = model.encoder
    cfg = enc.streaming_cfg
    T_full = chunk_geometry(model)["input_mel"]
    hyps_all, refs_all = [], []
    t_enc = 0.0
    n_chunks = 0
    graph_cache = {}
    for i0 in range(0, len(items), batch):
        group = items[i0:i0 + batch]
        buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
        for path, _ in group:
            buf.append_audio_file(path, stream_id=-1)
        B = len(group)
        if impl == "cudagraph":
            if B not in graph_cache:
                graph_cache[B] = GraphRunner(model, prec, B)
            gr = graph_cache[B]
            c, t, cl = gr.c, gr.t, gr.cl
            c.zero_(); t.zero_(); cl.zero_()
        else:
            gr = None
            c, t, cl = enc.get_initial_cache_state(batch_size=B, dtype=prec.act_dtype, device=DEVICE)
        if "A" in patch:
            prelim_attn.reset(model, SimpleNamespace(c=c), prec)
        f = encoder_step_fn(model, prec)
        hyps = None
        for k, (chunk, chunk_len) in enumerate(buf):
            chunk = chunk.to(prec.act_dtype)
            drop = 0 if k == 0 else cfg.drop_extra_pre_encoded
            last = buf.is_buffer_empty()
            _sync(); ts = time.perf_counter()
            with torch.inference_mode():
                steady = (gr is not None and not last and drop == cfg.drop_extra_pre_encoded
                          and chunk.shape == gr.x.shape and bool((chunk_len == T_full).all()))
                if steady:
                    if gr.graph is None:
                        gr._capture()
                        if "A" in patch:
                            prelim_attn.reset(model, SimpleNamespace(c=gr.c), prec)
                    gr.x.copy_(chunk)
                    gr.graph.replay()
                    encoded, enc_len = gr.out.clone(), gr.out_len.clone()
                else:
                    with prec.ctx():
                        o = enc.cache_aware_stream_step(
                            processed_signal=chunk, processed_signal_length=chunk_len,
                            cache_last_channel=c, cache_last_time=t, cache_last_channel_len=cl,
                            keep_all_outputs=last, drop_extra_pre_encoded=drop)
                    encoded, enc_len = o[0], o[1]
                    if gr is not None:
                        c.copy_(o[2]); t.copy_(o[3]); cl.copy_(o[4])
                    else:
                        c, t, cl = o[2], o[3], o[4]
            _sync(); t_enc += time.perf_counter() - ts; n_chunks += 1
            with torch.inference_mode():
                hyps = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded.float(), encoded_lengths=enc_len,
                    return_hypotheses=True, partial_hypotheses=hyps)
        for (path, ref), h in zip(group, hyps):
            hyps_all.append(normalize(h.text if hasattr(h, "text") else str(h)))
            refs_all.append(normalize(ref))
        if (i0 // batch) % max(1, log_every // batch) == 0:
            w, _, _ = wer(refs_all, hyps_all)
            log(f"  {len(hyps_all)}/{len(items)} utts, running WER {w*100:.2f}%")
    fused_conv.remove(model); prelim_attn.remove(model)
    w, e, n = wer(refs_all, hyps_all)
    return {"wer": w, "errors": e, "words": n, "utts": len(hyps_all), "enc_ms_per_chunk": t_enc / n_chunks * 1e3,
            "examples": list(zip(refs_all[:5], hyps_all[:5]))}, hyps_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eou120m")
    ap.add_argument("--att", default=None)
    ap.add_argument("--data", default="data/LibriSpeech")
    ap.add_argument("--subsets", nargs="+", default=["test-clean"])
    ap.add_argument("--max-utts", type=int, default=200, help="0 = full subset")
    ap.add_argument("--batch", type=int, default=16, help="streams processed together (same for every mode)")
    ap.add_argument("--modes", nargs="+", default=["fp32:eager", "amp16:eager", "fp16:eager", "fp16:cudagraph"],
                    help="precision:impl[:patch], impl in {eager,cudagraph}, patch in {A,B,AB}")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cpu-smoke", action="store_true", help="debug on CPU (eager modes only)")
    args = ap.parse_args()
    global DEVICE
    if args.cpu_smoke:
        DEVICE = "cpu"
    sysinfo = system_info()
    od = out_dir(args.out)
    res = Results(od / "wer.jsonl")
    for subset in args.subsets:
        items = librispeech(Path(args.data), subset, args.max_utts)
        if not items:
            raise SystemExit(f"no LibriSpeech files under {args.data}/{subset}; run data/get_librispeech.sh")
        ref_hyps = None
        for mode in args.modes:
            parts = mode.split(":")
            pname, impl = parts[0], parts[1]
            patch = parts[2] if len(parts) > 2 else "none"
            key = f"{args.model}|{subset}|n{len(items)}|b{args.batch}|{mode}"
            if DEVICE == "cuda" and "B" in patch and not triton_ok():
                log(f"skip {key} (Triton needs compute capability >= 7.0)")
                continue
            if res.has(key):
                log(f"skip {key}")
                continue
            prec = precision(pname)
            model = load_model(args.model, DEVICE)
            apply_precision(model, prec)
            att = list(map(int, args.att.split(","))) if args.att else att_contexts(args.model, model)[0]
            set_att_context(model, att)
            rec = {"key": key, "model": args.model, "subset": subset, "mode": mode, "precision": pname, "impl": impl,
                   "patch": patch, "att": att, "batch": args.batch, "gpu": sysinfo.get("gpu")}
            log(f"WER {key}")
            try:
                r, hyps = run_mode(model, args.model, prec, impl, patch, items, args.batch)
                rec.update(r)
                if ref_hyps is None:
                    ref_hyps = hyps
                rec["utts_differing_from_first_mode"] = sum(a != b for a, b in zip(hyps, ref_hyps))
                rec["ok"] = True
                log(f"{key}: WER {r['wer']*100:.2f}% ({r['errors']}/{r['words']}), "
                    f"{rec['utts_differing_from_first_mode']} utts differ from {args.modes[0]}")
            except Exception as e:
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {e}"[:2000]
                rec["traceback"] = traceback.format_exc()[-3000:]
                log(f"{key}: FAILED {rec['error'][:300]}")
            res.write(rec)
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
