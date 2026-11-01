"""Load a NeMo cache-aware streaming model and build per-chunk encoder "step" implementations.

Every implementation computes exactly one steady-state streaming step of the encoder:
    (mel chunk incl. pre-encode cache, caches) -> (encoded frames, next caches)
which is what a streaming server runs once per audio chunk.
"""
from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

MODELS = {
    # short name -> (HF/NGC name, list of att_context_size modes to benchmark)
    "eou120m": ("nvidia/parakeet_realtime_eou_120m-v1", [[70, 1]]),
    "nemotron600m": ("nvidia/nemotron-speech-streaming-en-0.6b", [[70, 0], [70, 1], [70, 6], [70, 13]]),
    "hybrid114m": ("nvidia/stt_en_fastconformer_hybrid_large_streaming_multi", [[70, 0], [70, 1], [70, 16], [70, 33]]),
}

PRECISIONS = ["fp32", "amp16", "fp16", "bf16"]


def quiet_nemo():
    for name in ["nemo_logger", "nemo", "NeMo"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_model(name: str, device: str = "cuda"):
    """Load a NeMo ASR model by short name (see MODELS) or full name/.nemo path."""
    quiet_nemo()
    import nemo.collections.asr as nemo_asr
    try:
        from nemo.utils import logging as nemo_logging

        nemo_logging.setLevel(logging.ERROR)
    except Exception:
        pass

    full = MODELS[name][0] if name in MODELS else name
    if full.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(full, map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(full, map_location=device)
    model.eval()
    try:
        model.preprocessor.featurizer.dither = 0.0  # deterministic features
        model.preprocessor.featurizer.pad_to = 0
    except Exception:
        pass
    return model


def att_contexts(name: str, model) -> List[List[int]]:
    if name in MODELS:
        wanted = MODELS[name][1]
    else:
        wanted = [list(model.encoder.att_context_size)]
    supported = [list(a) for a in getattr(model.encoder, "att_context_size_all", [model.encoder.att_context_size])]
    return [a for a in wanted if a in supported] or [list(model.encoder.att_context_size)]


def set_att_context(model, att: List[int]):
    model.encoder.set_default_att_context_size(att)
    cfg = model.encoder.streaming_cfg
    return cfg


def chunk_geometry(model) -> Dict[str, int]:
    """Steady-state step geometry, in mel frames (10 ms each) and encoder frames."""
    cfg = model.encoder.streaming_cfg
    pick = lambda v: v[1] if isinstance(v, (list, tuple)) else v  # noqa: E731
    chunk = pick(cfg.chunk_size)
    shift = pick(cfg.shift_size)
    pre = pick(cfg.pre_encode_cache_size)
    hop_ms = 10
    try:
        hop_ms = int(round(model.cfg.preprocessor.window_stride * 1000))
    except Exception:
        pass
    return {
        "chunk_mel": chunk,
        "shift_mel": shift,
        "pre_encode_mel": pre,
        "input_mel": chunk + pre,
        "chunk_ms": shift * hop_ms,
        "drop_extra_pre_encoded": cfg.drop_extra_pre_encoded,
        "valid_out_len": cfg.valid_out_len,
        "cache_len": cfg.last_channel_cache_size,
    }


# ----------------------------------------------------------------------------- precision


@dataclass
class Precision:
    name: str
    weight_dtype: torch.dtype
    act_dtype: torch.dtype
    autocast_dtype: Optional[torch.dtype]

    def ctx(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        # cache_enabled=False is required for CUDA-graph capture and avoids stale casts.
        return torch.autocast("cuda", dtype=self.autocast_dtype, cache_enabled=False)


def precision(name: str) -> Precision:
    return {
        "fp32": Precision("fp32", torch.float32, torch.float32, None),
        "amp16": Precision("amp16", torch.float32, torch.float32, torch.float16),
        "fp16": Precision("fp16", torch.float16, torch.float16, None),
        "bf16": Precision("bf16", torch.bfloat16, torch.bfloat16, None),
    }[name]


def apply_precision(model, prec: Precision):
    model.encoder.to(prec.weight_dtype)
    return model


def encoder_param_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.encoder.parameters())


# ----------------------------------------------------------------------------- streaming state


class StreamState:
    """Inputs and caches for one steady-state streaming step, batch of B independent streams."""

    def __init__(self, model, batch: int, prec: Precision, device: str = "cuda", seed: int = 0,
                 features: Optional[torch.Tensor] = None):
        enc = model.encoder
        self.geom = chunk_geometry(model)
        self.batch = batch
        g = torch.Generator(device="cpu").manual_seed(seed)
        n_mels = enc._feat_in
        T = self.geom["input_mel"]
        if features is not None:  # real log-mel features (n_mels, frames) to make inputs realistic
            f = features[:, :T]
            x = f.unsqueeze(0).repeat(batch, 1, 1)
        else:
            x = torch.randn(batch, n_mels, T, generator=g)
        self.x = x.to(device=device, dtype=prec.act_dtype).contiguous()
        self.xlen = torch.full((batch,), T, dtype=torch.int64, device=device)
        c, t, cl = enc.get_initial_cache_state(batch_size=batch, dtype=prec.act_dtype, device=device)
        self.c, self.t, self.cl = c.contiguous(), t.contiguous(), cl.contiguous()
        self.drop = self.geom["drop_extra_pre_encoded"]

    def cache_bytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in (self.c, self.t))


def encoder_step_fn(model, prec: Precision):
    """Plain function f(x, xlen, c, t, cl, drop) -> (enc, enc_len, c_next, t_next, cl_next)."""
    enc = model.encoder

    def f(x, xlen, c, t, cl, drop):
        with prec.ctx():
            return enc.cache_aware_stream_step(
                processed_signal=x, processed_signal_length=xlen,
                cache_last_channel=c, cache_last_time=t, cache_last_channel_len=cl,
                keep_all_outputs=False, drop_extra_pre_encoded=drop)

    return f


# ----------------------------------------------------------------------------- step implementations


class Step:
    """A callable that advances the stream by one chunk and keeps caches in `state`."""

    name = "base"

    def __init__(self, model, state: StreamState, prec: Precision):
        self.model, self.s, self.prec = model, state, prec
        self.f = encoder_step_fn(model, prec)
        self.out = None

    def __call__(self):
        raise NotImplementedError


class EagerStep(Step):
    name = "eager"

    @torch.inference_mode()
    def __call__(self):
        s = self.s
        o = self.f(s.x, s.xlen, s.c, s.t, s.cl, s.drop)
        self.out = o[0]
        s.c, s.t, s.cl = o[2], o[3], o[4]
        return self.out


class CudaGraphStep(Step):
    """Whole encoder step captured in one CUDA graph; caches updated in place inside the graph."""

    name = "cudagraph"

    def __init__(self, model, state, prec, warmup: int = 3):
        super().__init__(model, state, prec)
        s = self.s
        with torch.inference_mode():
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(warmup):
                    o = self.f(s.x, s.xlen, s.c, s.t, s.cl, s.drop)
                    s.c.copy_(o[2]); s.t.copy_(o[3]); s.cl.copy_(o[4])
            torch.cuda.current_stream().wait_stream(st)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                o = self.f(s.x, s.xlen, s.c, s.t, s.cl, s.drop)
                s.c.copy_(o[2]); s.t.copy_(o[3]); s.cl.copy_(o[4])
                self.out = o[0]

    def __call__(self):
        self.g.replay()
        return self.out


class CompiledStep(Step):
    """torch.compile of the encoder step. Modes using CUDA graphs reuse output memory, so caches are copied."""

    def __init__(self, model, state, prec, mode: str = "default"):
        super().__init__(model, state, prec)
        self.mode = mode
        self.name = f"compile-{mode}"
        kw = {"dynamic": False}
        if mode != "default":
            kw["mode"] = mode
        self.cf = torch.compile(self.f, **kw)
        self.uses_cg = mode in ("reduce-overhead", "max-autotune")

    @torch.inference_mode()
    def __call__(self):
        s = self.s
        if self.uses_cg:
            torch.compiler.cudagraph_mark_step_begin()
        o = self.cf(s.x, s.xlen, s.c, s.t, s.cl, s.drop)
        if self.uses_cg:
            s.c.copy_(o[2]); s.t.copy_(o[3]); s.cl.copy_(o[4])
            self.out = o[0].clone()
        else:
            s.c, s.t, s.cl = o[2], o[3], o[4]
            self.out = o[0]
        return self.out


IMPLS = ["eager", "cudagraph", "compile-default", "compile-reduce-overhead",
         "compile-max-autotune-no-cudagraphs", "compile-max-autotune"]


def make_step(impl: str, model, state: StreamState, prec: Precision) -> Step:
    if impl == "eager":
        return EagerStep(model, state, prec)
    if impl == "cudagraph":
        return CudaGraphStep(model, state, prec)
    if impl.startswith("compile-"):
        from asrbench.common import triton_ok

        if torch.cuda.is_available() and not triton_ok():
            raise RuntimeError("torch.compile/Inductor needs compute capability >= 7.0")
        return CompiledStep(model, state, prec, mode=impl[len("compile-"):])
    raise ValueError(impl)


def warm_caches(model, state: StreamState, prec: Precision, steps: int = 40):
    """Run eager steps until the attention cache is full (steady state: cache_len == 70)."""
    f = encoder_step_fn(model, prec)
    with torch.inference_mode():
        for _ in range(steps):
            o = f(state.x, state.xlen, state.c, state.t, state.cl, state.drop)
            state.c, state.t, state.cl = o[2].clone(), o[3].clone(), o[4].clone()


@torch.inference_mode()
def reference_sequence(model, prec: Precision, batch: int, n_steps: int, device="cuda", seed=123,
                       impl: str = "eager", patch_reset: Optional[Callable] = None) -> List[torch.Tensor]:
    """Encoder outputs for n_steps consecutive chunks (fresh stream, fixed random inputs).

    Used to check that every implementation computes the same function as fp32 eager.
    """
    s = StreamState(model, batch, prec, device=device, seed=seed)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    xs = [torch.randn(s.x.shape, generator=gen).to(device=device, dtype=prec.act_dtype) for _ in range(n_steps)]
    if patch_reset:
        patch_reset(s)
    if impl == "eager":
        step = EagerStep(model, s, prec)
    else:
        # build the implementation on a scratch state, then reset caches in place to zeros
        step = make_step(impl, model, s, prec)
        s.c.zero_(); s.t.zero_(); s.cl.zero_()
        if patch_reset:
            patch_reset(s)
    outs = []
    for x in xs:
        s.x.copy_(x)
        outs.append(step().detach().float().clone())
    return outs


@torch.inference_mode()
def parity_with_step(step: "Step", n_steps: int, seed: int = 123, patch_reset: Optional[Callable] = None):
    """Re-use an already-built step (no recompilation/recapture): restart its stream from zero caches and
    feed the same inputs reference_sequence() uses. Returns the encoder outputs."""
    s = step.s
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    xs = [torch.randn(s.x.shape, generator=gen).to(device=s.x.device, dtype=s.x.dtype) for _ in range(n_steps)]
    s.c.zero_(); s.t.zero_(); s.cl.zero_()
    if patch_reset:
        patch_reset(s)
    outs = []
    for x in xs:
        s.x.copy_(x)
        outs.append(step().detach().float().clone())
    return outs


def compare(a: List[torch.Tensor], b: List[torch.Tensor]) -> Dict[str, float]:
    mx, rel = 0.0, 0.0
    for x, y in zip(a, b):
        y = y.to(x.device)
        mx = max(mx, (x - y).abs().max().item())
        rel = max(rel, ((x - y).norm() / (y.norm() + 1e-12)).item())
    return {"max_abs_err": mx, "max_rel_err": rel}
