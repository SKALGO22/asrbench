"""End-to-end streaming WER: NeMo's own cache-aware streaming vs the same pipeline with the megakernel encoder.

Both modes use NeMo's CacheAwareStreamingAudioBuffer and model.conformer_stream_step (RNN-T greedy decoding with
partial hypotheses). In megakernel mode, model.encoder.cache_aware_stream_step is replaced:
  * first chunk of each utterance: NeMo's encoder (different geometry); its caches are converted into the megakernel
    state (projected K/V ring + conv cache)
  * every later chunk: NeMo pre-encode + the megakernel for all 17 layers (mask computed from the cache length)
  * a short last chunk goes back to NeMo: the kernel also keeps NeMo's attention-input cache, so the state is
    handed back exactly (no approximation anywhere)
Reports WER for both modes, transcripts that differ, and the max encoder-output difference per chunk.

  python megakernel/mk_wer.py --data data/LibriSpeech/test-clean --n 200
  python megakernel/mk_wer.py --synthetic --device cpu --reference --n 2     # plumbing test without data/GPU
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mk_runtime import CACHE, MegakernelEncoder, capture_constants, model_dtype, stream_geometry  # noqa: E402
from mk_test import load  # noqa: E402


# ----------------------------------------------------------------------------- data + scoring
def librispeech_items(root, n):
    trans = sorted(glob.glob(os.path.join(root, "**", "*.trans.txt"), recursive=True))
    items = []
    for tf in trans:
        d = os.path.dirname(tf)
        for line in open(tf):
            uid, text = line.strip().split(" ", 1)
            items.append((os.path.join(d, uid + ".flac"), text))
    items.sort()
    return items[:n] if n else items


def norm(s):
    s = re.sub(r"<[^>]*>", " ", s.lower())          # drop special tokens such as <EOU>
    s = re.sub(r"[^a-z' ]+", " ", s)
    return s.split()


def edit_distance(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)]


# ----------------------------------------------------------------------------- encoder replacement
class EncoderPatch:
    def __init__(self, model, mk, orig, record=None):
        self.model, self.mk, self.enc = model, mk, model.encoder
        self.orig = orig
        g = stream_geometry(model)
        self.drop, self.T_in, self.T = g["drop"], g["T_in"], mk.T
        self.record = record
        self.reset()
        self.stats = {"nemo_chunks": 0, "mk_chunks": 0}

    def reset(self):
        self.ready = False

    @torch.inference_mode()
    def __call__(self, processed_signal, processed_signal_length=None, cache_last_channel=None, cache_last_time=None,
                 cache_last_channel_len=None, keep_all_outputs=True, drop_extra_pre_encoded=None,
                 bypass_pre_encode=False):
        Tm = processed_signal.shape[-1]
        steady = self.ready and drop_extra_pre_encoded == self.drop and Tm == self.T_in
        if not steady:
            # NeMo handles it (first chunk, or a short last chunk). Hand our state back to NeMo exactly (the kernel
            # keeps NeMo's attention-input cache alongside its projected K/V), then take NeMo's caches back.
            if self.ready:
                cache_last_channel, cache_last_time, cache_last_channel_len = self.mk.state.export_to_nemo(model_dtype(self.model))
            out = self.orig(processed_signal=processed_signal, processed_signal_length=processed_signal_length,
                            cache_last_channel=cache_last_channel, cache_last_time=cache_last_time,
                            cache_last_channel_len=cache_last_channel_len, keep_all_outputs=keep_all_outputs,
                            drop_extra_pre_encoded=drop_extra_pre_encoded, bypass_pre_encode=bypass_pre_encode)
            self.mk.state.load_from_nemo(self.model, out[2], out[3], out[4])
            self.ready = True
            self.stats["nemo_chunks"] += 1
            if self.record is not None:
                self.record.append(out[0][..., : int(out[1][0])].float().cpu())
            return out
        x = processed_signal
        dt = model_dtype(self.model)
        xe, le = self.enc.pre_encode(x=x.to(dt).transpose(1, 2), lengths=processed_signal_length)
        xe = xe[:, self.drop:, :]
        le = (le.to(torch.int64) - self.drop).clamp(min=0, max=self.T)
        xe, _ = self.enc.pos_enc(x=xe, cache_len=CACHE)
        y = self.mk.layers_step(xe[0].float().contiguous())
        enc_out = y.clone().unsqueeze(0).transpose(1, 2).to(dt)            # (1, D, T)
        new_cl = (cache_last_channel_len + self.T).clamp(max=CACHE)
        self.stats["mk_chunks"] += 1
        if self.record is not None:
            self.record.append(enc_out[..., : int(le[0])].float().cpu())
        return enc_out, le, cache_last_channel, cache_last_time, new_cl


class Recorder:
    """Wraps NeMo's own encoder step to record its outputs (for the per-chunk comparison)."""

    def __init__(self, model, record):
        self.orig, self.record = model.encoder.cache_aware_stream_step, record

    @torch.inference_mode()
    def __call__(self, **kw):
        out = self.orig(**kw)
        self.record.append(out[0][..., : int(out[1][0])].float().cpu())
        return out


# ----------------------------------------------------------------------------- streaming transcription
@torch.inference_mode()
def transcribe(model, audio, device, on_new_utt=None):
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False)
    buf.append_audio(audio, stream_id=-1)
    c, t, cl = model.encoder.get_initial_cache_state(batch_size=1, dtype=model_dtype(model), device=device)
    drop = model.encoder.streaming_cfg.drop_extra_pre_encoded
    hyps, pred = None, None
    if on_new_utt:
        on_new_utt()
    text = ""
    for step, (chunk, lens) in enumerate(buf):
        out = model.conformer_stream_step(
            processed_signal=chunk.to(device), processed_signal_length=lens.to(device),
            cache_last_channel=c, cache_last_time=t, cache_last_channel_len=cl,
            keep_all_outputs=buf.is_buffer_empty(), previous_hypotheses=hyps, previous_pred_out=pred,
            drop_extra_pre_encoded=0 if step == 0 else drop, return_transcription=True)
        pred, texts, c, t, cl, hyps = out[0], out[1], out[2], out[3], out[4], out[5]
        h = texts[0]
        text = h.text if hasattr(h, "text") else str(h)
    return text


def setup_decoding(model):
    from omegaconf import open_dict
    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.preserve_alignments = False
        cfg.compute_timestamps = False
        if "greedy" in cfg:
            cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eou120m")
    ap.add_argument("--data", default="data/LibriSpeech/test-clean")
    ap.add_argument("--n", type=int, default=200, help="utterances (0 = all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", help="FP16 NeMo model (default FP32 reference)")
    ap.add_argument("--reference", action="store_true", help="PyTorch reference instead of the CUDA kernel")
    ap.add_argument("--synthetic", action="store_true", help="random audio instead of LibriSpeech (plumbing test)")
    ap.add_argument("--flags", type=int, default=2)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    dev = args.device
    model = load(args.model, dev, not args.fp16)
    model.eval()
    setup_decoding(model)
    consts = capture_constants(model)
    mk = MegakernelEncoder(model, consts, use_kernel=not args.reference, device=dev, flags=args.flags)

    if args.synthetic:
        rng = np.random.default_rng(0)
        items = [(rng.standard_normal(16000 * 4).astype(np.float32) * 0.05, "") for _ in range(args.n or 2)]
    else:
        items = librispeech_items(args.data, args.n)
        if not items:
            sys.exit(f"no LibriSpeech transcripts under {args.data} (see README: download test-clean)")
    import soundfile as sf

    def audio_of(it):
        return it[0] if isinstance(it[0], np.ndarray) else sf.read(it[0], dtype="float32")[0]

    rec_nemo, rec_mk = [], []
    t0 = time.time()
    orig = model.encoder.cache_aware_stream_step
    model.encoder.cache_aware_stream_step = Recorder(model, rec_nemo)
    hyp_nemo = [transcribe(model, audio_of(it), dev) for it in items]
    t_nemo = time.time() - t0
    patch = EncoderPatch(model, mk, orig, record=rec_mk)
    model.encoder.cache_aware_stream_step = patch
    t0 = time.time()
    hyp_mk = [transcribe(model, audio_of(it), dev, on_new_utt=patch.reset) for it in items]
    t_mk = time.time() - t0
    model.encoder.cache_aware_stream_step = orig

    diffs = [float((a - b).abs().max()) if a.shape == b.shape else float("nan") for a, b in zip(rec_nemo, rec_mk)]
    finite = [d for d in diffs if d == d]
    rel = [float((a - b).norm() / (a.norm() + 1e-9)) for a, b in zip(rec_nemo, rec_mk) if a.shape == b.shape]
    print(f"[wer] chunks compared {len(finite)}/{len(diffs)}; encoder output max|diff| {max(finite):.3e}, "
          f"median rel.err {np.median(rel):.2e}, worst rel.err {max(rel):.2e}")
    print(f"[wer] megakernel chunks {patch.stats['mk_chunks']}, NeMo chunks (first / short last) {patch.stats['nemo_chunks']}")
    res = {"n_utts": len(items), "chunks": len(diffs), "enc_max_abs_diff": max(finite), "enc_rel_err_median": float(np.median(rel)),
           "enc_rel_err_max": max(rel), **patch.stats, "seconds_nemo": t_nemo, "seconds_mk": t_mk,
           "model_precision": "fp16" if args.fp16 else "fp32", "kernel": not args.reference, "flags": args.flags}
    if not args.synthetic:
        errs_n = errs_m = words = 0
        differ = []
        for (path, ref), hn, hm in zip(items, hyp_nemo, hyp_mk):
            r = norm(ref)
            words += len(r)
            errs_n += edit_distance(r, norm(hn))
            errs_m += edit_distance(r, norm(hm))
            if norm(hn) != norm(hm):
                differ.append((os.path.basename(path), " ".join(norm(hn)), " ".join(norm(hm))))
        print(f"[wer] {len(items)} utterances, {words} words: NeMo streaming WER {100 * errs_n / words:.2f}% | "
              f"megakernel WER {100 * errs_m / words:.2f}% | transcripts differing: {len(differ)}")
        for u, a, b in differ[:5]:
            print(f"      {u}\n        nemo: {a}\n        mk  : {b}")
        res.update(words=words, wer_nemo=errs_n / words, wer_mk=errs_m / words, n_differ=len(differ))
        gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
        slug = "".join(ch for ch in gpu.lower().replace("nvidia", "").replace("geforce", "") if ch.isalnum())
        outd = Path(args.out) / slug; outd.mkdir(parents=True, exist_ok=True)
        with open(outd / "megakernel_wer.jsonl", "a") as f:
            f.write(json.dumps({**res, "gpu": gpu, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    else:
        same = sum(a == b for a, b in zip(hyp_nemo, hyp_mk))
        print(f"[wer] synthetic: identical transcripts {same}/{len(items)}")


if __name__ == "__main__":
    main()
