"""Megakernel runtime: builds libmk.so, packs NeMo weights, captures streaming constants, runs the encoder step.

Pieces
  build_lib()                 nvcc -> build/libmk.so (plain shared library, driven through ctypes)
  capture_constants(model)    attention masks per cache length, pos_emb, rel_shift index table (taken from NeMo itself)
  pack_layers(model, ...)     per-layer weights in the exact order of struct LayerParams (FP16 matrices, FP32 vectors)
  MKState                     device streaming state: projected K/V ring, conv cache, ring head
  reference_layers(...)       the SAME stage algorithm in plain PyTorch (validates packing/semantics without a GPU)
  MegakernelEncoder           pre-encode (NeMo) + megakernel for all 17 layers, one launch per chunk
"""
from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

D, DFF, NH, DK, KC, KS, CACHE, TMAX, NSTAGE = 512, 2048, 8, 64, 8, 9, 70, 4, 10
STAGE_NAMES = ["S1 LN+FFN1 up", "S2 FFN1 down", "S3 LN+QKV", "S4 attention", "S5 out-proj+KV commit",
               "S6 LN+PW1+GLU+DWconv", "S7 LN+SiLU+PW2", "S8 LN+FFN2 up", "S9 FFN2 down", "S10 LN out"]
# FP16 weight bytes streamed per stage (one layer, d=512): used for per-stage achieved bandwidth
STAGE_WEIGHT_BYTES = [DFF * D * 2, D * DFF * 2, 3 * D * D * 2, 0, D * D * 2, 2 * D * D * 2, D * D * 2,
                      DFF * D * 2, D * DFF * 2, 0]

LAYER_FIELDS = ["ln_ff1_g", "ln_ff1_b", "w_ff1_1", "b_ff1_1", "w_ff1_2", "b_ff1_2", "ln_att_g", "ln_att_b",
                "w_qkv", "b_qkv", "w_o", "b_o", "pos_u", "pos_v", "p", "ln_conv_g", "ln_conv_b", "w_pw1",
                "b_pw1", "w_dw", "b_dw", "ln_cn_g", "ln_cn_b", "w_pw2", "b_pw2", "ln_ff2_g", "ln_ff2_b",
                "w_ff2_1", "b_ff2_1", "w_ff2_2", "b_ff2_2", "ln_out_g", "ln_out_b"]
MATRIX_FIELDS = {"w_ff1_1", "w_ff1_2", "w_qkv", "w_o", "w_pw1", "w_pw2", "w_ff2_1", "w_ff2_2"}


# ============================================================================== build + ctypes
def build_lib(force: bool = False) -> Path:
    src = HERE / "csrc" / "mk.cu"
    out = HERE / "build" / "libmk.so"
    out.parent.mkdir(exist_ok=True)
    if out.exists() and not force and out.stat().st_mtime > src.stat().st_mtime:
        return out
    arch = os.environ.get("MK_ARCH", "-gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_86,code=sm_86")
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-cudart", "shared", "-lineinfo",
           *arch.split(), str(src), "-o", str(out)]
    print("[mk] building:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


class MKArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in
                ["layers", "x_in", "x_out", "kv", "conv_cache", "ring_head", "mask", "pidx", "resid0", "resid1",
                 "hbuf", "qkv", "att", "ybuf", "timers", "ptimers", "mbox"]] + \
               [("n_layers", ctypes.c_int), ("T", ctypes.c_int), ("npos", ctypes.c_int), ("flags", ctypes.c_int),
                ("cache_len", ctypes.c_int), ("pad3_", ctypes.c_int),
                ("eps", ctypes.c_float), ("fc_factor", ctypes.c_float), ("att_scale", ctypes.c_float),
                ("pad2_", ctypes.c_float)]


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        torch.cuda.init()                       # loads PyTorch's libcudart first, so libmk reuses it
        _LIB = ctypes.CDLL(str(build_lib()))
        _LIB.mk_error_string.restype = ctypes.c_char_p
        _LIB.mk_host_alloc.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        assert _LIB.mk_args_size() == ctypes.sizeof(MKArgs), "MKArgs layout mismatch between C and Python"
        assert _LIB.mk_layer_params_size() == 8 * len(LAYER_FIELDS), "LayerParams layout mismatch"
    return _LIB


def check(err: int, what: str):
    if err != 0:
        raise RuntimeError(f"{what}: CUDA error {err}: {lib().mk_error_string(err).decode()}")


class HostBuffer:
    """Pinned + mapped (zero-copy) host memory: the CPU writes/reads `arr`, the GPU uses `dptr` directly."""

    def __init__(self, n, ctype=ctypes.c_float):
        import numpy as np
        self.hptr, self.dptr = ctypes.c_void_p(), ctypes.c_void_p()
        check(lib().mk_host_alloc(n * ctypes.sizeof(ctype), ctypes.byref(self.hptr), ctypes.byref(self.dptr)),
              "mk_host_alloc")
        self.raw = (ctype * n).from_address(self.hptr.value)
        self.arr = np.ctypeslib.as_array(self.raw)

    def free(self):
        lib().mk_host_free(self.hptr)


def make_stream():
    s = ctypes.c_void_p()
    check(lib().mk_stream_create(ctypes.byref(s)), "mk_stream_create")
    return s


# ============================================================================== NeMo streaming helpers
def stream_geometry(model):
    cfg = model.encoder.streaming_cfg
    pick = lambda v: v[1] if isinstance(v, (list, tuple)) else v  # noqa: E731
    assert getattr(cfg, "cache_drop_size", 0) in (0, None), "kernel assumes cache_drop_size == 0"
    return {"T_in": pick(cfg.chunk_size) + pick(cfg.pre_encode_cache_size), "drop": cfg.drop_extra_pre_encoded,
            "feat": model.encoder._feat_in, "valid_out_len": cfg.valid_out_len}


def model_dtype(model):
    return next(model.encoder.parameters()).dtype


@torch.inference_mode()
def nemo_initial(model, device):
    return model.encoder.get_initial_cache_state(batch_size=1, dtype=model_dtype(model), device=device)


@torch.inference_mode()
def nemo_step(model, mel, mel_len, c, t, cl, drop):
    """One NeMo cache-aware streaming step. Returns (encoded (1,D,T), c_next, t_next, cl_next)."""
    o = model.encoder.cache_aware_stream_step(
        processed_signal=mel.to(model_dtype(model)), processed_signal_length=mel_len,
        cache_last_channel=c, cache_last_time=t, cache_last_channel_len=cl,
        keep_all_outputs=False, drop_extra_pre_encoded=drop)
    return o[0], o[2], o[3], o[4]


def random_chunks(model, n, device, seed=0):
    g = stream_geometry(model)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    xs = [torch.randn(1, g["feat"], g["T_in"], generator=gen).to(device) for _ in range(n)]
    xlen = torch.full((1,), g["T_in"], dtype=torch.int64, device=device)
    return xs, xlen, g["drop"]


@torch.inference_mode()
def capture_constants(model, n_steps: int = 40, device=None) -> Dict:
    """Run NeMo's own streaming step and record, for layer 0, the attention mask (per valid cache length) and pos_emb.

    The rel_shift index table is obtained by running NeMo's rel_shift on an index tensor, so our attention reproduces
    NeMo's relative-position term exactly (including its zero padding).
    """
    enc = model.encoder
    device = device or next(model.parameters()).device
    att0 = enc.layers[0].self_attn
    seen = {}

    def pre_hook(mod, args, kwargs):
        seen["mask"] = kwargs.get("mask")
        seen["pos_emb"] = kwargs.get("pos_emb")

    h = att0.register_forward_pre_hook(pre_hook, with_kwargs=True)
    xs, xlen, drop = random_chunks(model, n_steps, device, seed=1234)
    c, t, cl = nemo_initial(model, device)
    masks, pos_emb, T_seen = {}, None, None
    try:
        for x in xs:
            clv = int(cl[0].item())
            _, c, t, cl = nemo_step(model, x, xlen, c, t, cl, drop)
            m = seen["mask"]
            if m is not None:
                masks[clv] = m[0].to(torch.uint8).cpu().contiguous()       # (T, NK), 1 = masked
                T_seen = m.shape[1]
            pos_emb = seen["pos_emb"]
    finally:
        h.remove()
    T = T_seen
    NK = masks[max(masks)].shape[1]
    npos = pos_emb.shape[1]
    idx = torch.arange(1, npos + 1, dtype=torch.float32, device=pos_emb.device).view(1, 1, 1, npos)
    idx = idx.expand(1, 1, T, npos).contiguous()
    shifted = att0.rel_shift(idx)[..., :NK]
    pidx = (shifted[0, 0].round().long() - 1).to(torch.int32).cpu().contiguous()  # -1 = rel_shift padding (zero)
    return {"masks": masks, "pos_emb": pos_emb.detach().clone(), "pidx": pidx, "T": T, "NK": NK, "npos": npos}


# ============================================================================== weight packing
def _vec(t, n, device):
    if t is None:
        return torch.zeros(n, dtype=torch.float32, device=device)
    return t.detach().reshape(-1).to(device=device, dtype=torch.float32).contiguous()


@torch.inference_mode()
def pack_layers(model, consts, device="cuda", wdtype=torch.float16) -> List[Dict[str, torch.Tensor]]:
    """Per-layer tensors keyed by LAYER_FIELDS. Matrices in `wdtype` (FP16 for the kernel), vectors FP32."""
    layers = []
    pos_emb = consts["pos_emb"]
    for L in model.encoder.layers:
        a, c = L.self_attn, L.conv
        assert c.norm_type == "layer_norm", "kernel implements the LayerNorm conv-module variant"
        assert c.pointwise_activation == "glu_", "kernel implements GLU"
        mat = lambda w: w.detach().reshape(w.shape[0], -1).to(device=device, dtype=wdtype).contiguous()  # noqa: E731
        p = a.linear_pos(pos_emb.to(a.linear_pos.weight.dtype))                 # (1, npos, D)
        p = p.view(-1, NH, DK).transpose(0, 1).to(device=device, dtype=torch.float32).contiguous()  # (NH, npos, DK)
        wq, wk, wv = a.linear_q.weight, a.linear_k.weight, a.linear_v.weight
        bq = _vec(a.linear_q.bias, D, device); bk = _vec(a.linear_k.bias, D, device); bv = _vec(a.linear_v.bias, D, device)
        d = {
            "ln_ff1_g": _vec(L.norm_feed_forward1.weight, D, device), "ln_ff1_b": _vec(L.norm_feed_forward1.bias, D, device),
            "w_ff1_1": mat(L.feed_forward1.linear1.weight), "b_ff1_1": _vec(L.feed_forward1.linear1.bias, DFF, device),
            "w_ff1_2": mat(L.feed_forward1.linear2.weight), "b_ff1_2": _vec(L.feed_forward1.linear2.bias, D, device),
            "ln_att_g": _vec(L.norm_self_att.weight, D, device), "ln_att_b": _vec(L.norm_self_att.bias, D, device),
            "w_qkv": torch.cat([mat(wq), mat(wk), mat(wv)], 0).contiguous(), "b_qkv": torch.cat([bq, bk, bv]).contiguous(),
            "w_o": mat(a.linear_out.weight), "b_o": _vec(a.linear_out.bias, D, device),
            "pos_u": _vec(a.pos_bias_u, NH * DK, device), "pos_v": _vec(a.pos_bias_v, NH * DK, device),
            "p": p,
            "ln_conv_g": _vec(L.norm_conv.weight, D, device), "ln_conv_b": _vec(L.norm_conv.bias, D, device),
            "w_pw1": mat(c.pointwise_conv1.weight), "b_pw1": _vec(c.pointwise_conv1.bias, 2 * D, device),
            "w_dw": c.depthwise_conv.weight.detach().reshape(D, KS).to(device=device, dtype=torch.float32).contiguous(),
            "b_dw": _vec(c.depthwise_conv.bias, D, device),
            "ln_cn_g": _vec(c.batch_norm.weight, D, device), "ln_cn_b": _vec(c.batch_norm.bias, D, device),
            "w_pw2": mat(c.pointwise_conv2.weight), "b_pw2": _vec(c.pointwise_conv2.bias, D, device),
            "ln_ff2_g": _vec(L.norm_feed_forward2.weight, D, device), "ln_ff2_b": _vec(L.norm_feed_forward2.bias, D, device),
            "w_ff2_1": mat(L.feed_forward2.linear1.weight), "b_ff2_1": _vec(L.feed_forward2.linear1.bias, DFF, device),
            "w_ff2_2": mat(L.feed_forward2.linear2.weight), "b_ff2_2": _vec(L.feed_forward2.linear2.bias, D, device),
            "ln_out_g": _vec(L.norm_out.weight, D, device), "ln_out_b": _vec(L.norm_out.bias, D, device),
        }
        assert set(d) == set(LAYER_FIELDS)
        layers.append(d)
    return layers


def model_scalars(model):
    L0 = model.encoder.layers[0]
    eps = {L0.norm_feed_forward1.eps, L0.norm_self_att.eps, L0.norm_conv.eps, L0.conv.batch_norm.eps, L0.norm_out.eps}
    assert len(eps) == 1, f"kernel assumes one LayerNorm eps, got {eps}"
    return {"eps": float(eps.pop()), "fc_factor": float(L0.fc_factor), "att_scale": 1.0 / math.sqrt(DK)}


# ============================================================================== streaming state
class MKState:
    """Device-side streaming state for the megakernel (B = 1)."""

    def __init__(self, n_layers, device="cuda"):
        self.kv = torch.zeros(n_layers, 2, CACHE, D, dtype=torch.float32, device=device)
        self.conv = torch.zeros(n_layers, D, KC, dtype=torch.float32, device=device)
        self.head = torch.zeros(1, dtype=torch.int32, device=device)
        self.cache_len = 0

    @torch.inference_mode()
    def load_from_nemo(self, model, c, t, cl):
        """Convert NeMo caches (attention INPUTS, conv inputs) into projected K/V ring + conv cache."""
        for l, L in enumerate(model.encoder.layers):
            a = L.self_attn
            x = c[l, 0].to(a.linear_k.weight.dtype)                  # (CACHE, D): stored normalized inputs
            self.kv[l, 0] = a.linear_k(x).float()
            self.kv[l, 1] = a.linear_v(x).float()
            self.conv[l] = t[l, 0].float()
        self.head.zero_()
        self.cache_len = int(cl[0].item())


# ============================================================================== reference (same algorithm, PyTorch)
def _ln(x, g, b, eps):
    return F.layer_norm(x, (x.shape[-1],), g, b, eps)


@torch.inference_mode()
def reference_layers(layers, x, kv, conv, head, mask, pidx, sc) -> torch.Tensor:
    """Exactly the stage algorithm of mk.cu, in PyTorch. Mutates kv/conv in place, returns the layer-stack output."""
    T = x.shape[0]
    NK = CACHE + T
    order = [(head + j) % CACHE for j in range(CACHE)]
    m = mask.bool().to(x.device)
    pi = pidx.long().to(x.device)
    for l, P in enumerate(layers):
        W = {k: (v.float() if k in MATRIX_FIELDS else v) for k, v in P.items()}
        h = F.silu(F.linear(_ln(x, W["ln_ff1_g"], W["ln_ff1_b"], sc["eps"]), W["w_ff1_1"], W["b_ff1_1"]))
        x = x + sc["fc_factor"] * F.linear(h, W["w_ff1_2"], W["b_ff1_2"])
        qkv = F.linear(_ln(x, W["ln_att_g"], W["ln_att_b"], sc["eps"]), W["w_qkv"], W["b_qkv"])
        q, kn, vn = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        K = torch.cat([kv[l, 0][order], kn], 0)
        V = torch.cat([kv[l, 1][order], vn], 0)
        att = torch.zeros(T, D, device=x.device)
        for hh in range(NH):
            s_ = slice(hh * DK, (hh + 1) * DK)
            qh = q[:, s_]
            ac = (qh + W["pos_u"][s_]) @ K[:, s_].T                                    # (T, NK)
            raw = (qh + W["pos_v"][s_]) @ W["p"][hh].T                                 # (T, npos)
            bd = torch.where(pi >= 0, torch.gather(raw, 1, pi.clamp(min=0)), torch.zeros_like(ac))
            s = (ac + bd) * sc["att_scale"]
            s = s.masked_fill(m, float("-inf"))
            allm = m.all(dim=1, keepdim=True)
            w = torch.softmax(s.masked_fill(allm, 0.0), dim=1).masked_fill(m | allm, 0.0)
            att[:, s_] = w @ V[:, s_]
        x = x + F.linear(att, W["w_o"], W["b_o"])
        for t in range(T):                                                             # commit into oldest slots
            slot = (head + t) % CACHE
            kv[l, 0, slot] = kn[t]
            kv[l, 1, slot] = vn[t]
        z = F.linear(_ln(x, W["ln_conv_g"], W["ln_conv_b"], sc["eps"]), W["w_pw1"], W["b_pw1"])
        g = z[:, :D] * torch.sigmoid(z[:, D:])                                         # (T, D)
        hist = torch.cat([conv[l], g.T], 1)                                            # (D, KC + T)
        y = torch.stack([W["b_dw"] + (W["w_dw"] * hist[:, t:t + KS]).sum(1) for t in range(T)], 0)  # (T, D)
        conv[l] = hist[:, T:T + KC]
        x = x + F.linear(F.silu(_ln(y, W["ln_cn_g"], W["ln_cn_b"], sc["eps"])), W["w_pw2"], W["b_pw2"])
        h = F.silu(F.linear(_ln(x, W["ln_ff2_g"], W["ln_ff2_b"], sc["eps"]), W["w_ff2_1"], W["b_ff2_1"]))
        x = x + sc["fc_factor"] * F.linear(h, W["w_ff2_2"], W["b_ff2_2"])
        x = _ln(x, W["ln_out_g"], W["ln_out_b"], sc["eps"])
    return x


# ============================================================================== pre-encode (NeMo) + megakernel
class MegakernelEncoder:
    """One streaming encoder step = NeMo pre-encode (subsampling + pos-enc scaling) + the megakernel for all layers."""

    def __init__(self, model, consts, threads: int = 256, blocks: Optional[int] = None, timers: bool = False,
                 use_kernel: bool = True, device="cuda", wdtype=None, flags: int = 2):
        self.model, self.enc, self.consts = model, model.encoder, consts
        self.device = device
        self.T = consts["T"]
        assert self.T <= TMAX
        self.NK = consts["NK"]
        self.npos = consts["npos"]
        self.sc = model_scalars(model)
        if wdtype is None:   # the kernel needs FP16 matrices; the reference can use the model's own precision
            wdtype = torch.float16 if use_kernel else model_dtype(model)
        self.layers = pack_layers(model, consts, device=device, wdtype=wdtype)
        self.nl = len(self.layers)
        self.state = MKState(self.nl, device)
        self.pidx = consts["pidx"].to(device)
        self.mask_by_len = {k: v.to(device) for k, v in consts["masks"].items()}
        self.use_kernel = use_kernel
        self.flags = flags          # bit0: L2 prefetch next layer, bit1: fused LN_out, bit3: adjacent rows (default 0 since v6.1)
        T = self.T
        z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=device)  # noqa: E731
        self.x_in, self.x_out = z(T, D), z(T, D)
        self.bufs = {"resid0": z(T, D), "resid1": z(T, D), "hbuf": z(T, DFF), "qkv": z(T, 3 * D), "att": z(T, D),
                     "ybuf": z(T, D)}
        self.timers = torch.zeros(self.nl * NSTAGE + 3, dtype=torch.int64, device=device) if timers else None
        self.ptimers = torch.zeros(self.nl * NSTAGE * 2, dtype=torch.int64, device=device) if timers else None
        if use_kernel:
            assert threads == 256, "v5 kernel is compiled for 256 threads per block (__launch_bounds__(256), 250 regs)"
            L = lib()
            check(L.mk_prepare(threads), "mk_prepare")
            per_sm = ctypes.c_int(0)
            check(L.mk_max_blocks_per_sm(threads, self.T, ctypes.byref(per_sm)), "occupancy")
            n_sm = torch.cuda.get_device_properties(0).multi_processor_count
            self.max_blocks = per_sm.value * n_sm
            self.blocks = blocks or n_sm
            assert self.blocks <= self.max_blocks, f"grid {self.blocks} > co-resident limit {self.max_blocks}"
            self.threads = threads
            ptrs = []
            for P in self.layers:
                ptrs += [P[f].data_ptr() for f in LAYER_FIELDS]
            self.params = torch.tensor(ptrs, dtype=torch.int64, device=device)

    def _mask(self):
        cl = min(self.state.cache_len, CACHE)
        if cl not in self.mask_by_len:
            raise KeyError(f"no captured attention mask for cache length {cl}; captured: {sorted(self.mask_by_len)}")
        return self.mask_by_len[cl]

    def layers_step(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (T, D) float32 on device (already pre-encoded), or None if self.x_in was already filled (e.g. by a
        captured pre-encode graph). Returns (T, D) float32 view of the output buffer."""
        mask = self._mask()
        if x is None:
            x = self.x_in
        if not self.use_kernel:
            head = int(self.state.head.item())
            out = reference_layers(self.layers, x, self.state.kv, self.state.conv, head, mask, self.pidx, self.sc)
            self.state.head.fill_((head + self.T) % CACHE)
            self.state.cache_len += self.T
            self.x_out.copy_(out)
            return self.x_out
        if x.data_ptr() != self.x_in.data_ptr():
            self.x_in.copy_(x)
        a = MKArgs()
        a.layers = self.params.data_ptr(); a.x_in = self.x_in.data_ptr(); a.x_out = self.x_out.data_ptr()
        a.kv = self.state.kv.data_ptr(); a.conv_cache = self.state.conv.data_ptr(); a.ring_head = self.state.head.data_ptr()
        a.mask = mask.data_ptr(); a.pidx = self.pidx.data_ptr()
        for k, v in self.bufs.items():
            setattr(a, k, v.data_ptr())
        a.timers = self.timers.data_ptr() if self.timers is not None else None
        a.ptimers = self.ptimers.data_ptr() if self.ptimers is not None else None
        a.n_layers, a.T, a.npos, a.flags = self.nl, self.T, self.npos, self.flags
        a.cache_len = min(self.state.cache_len, CACHE)
        a.mbox = None
        a.eps, a.fc_factor, a.att_scale = self.sc["eps"], self.sc["fc_factor"], self.sc["att_scale"]
        stream = torch.cuda.current_stream().cuda_stream
        check(lib().mk_launch(ctypes.byref(a), self.blocks, self.threads, ctypes.c_void_p(stream)), "mk_launch")
        self.state.cache_len += self.T
        return self.x_out

    def launch_raw(self, x_in_ptr, x_out_ptr, stream, flags=None, mbox_ptr=None, cache_len=CACHE):
        """Launch with caller-supplied input/output pointers (e.g. zero-copy host buffers) on `stream`.
        flags | 16 starts the resident persistent kernel (returns immediately; drive it through the mailbox).
        The attention mask is synthesised in-kernel from cache_len (mask = nullptr)."""
        a = MKArgs()
        a.layers = self.params.data_ptr(); a.x_in = x_in_ptr; a.x_out = x_out_ptr
        a.kv = self.state.kv.data_ptr(); a.conv_cache = self.state.conv.data_ptr(); a.ring_head = self.state.head.data_ptr()
        a.mask = None; a.pidx = self.pidx.data_ptr()
        for k, v in self.bufs.items():
            setattr(a, k, v.data_ptr())
        a.timers = None; a.ptimers = None; a.mbox = mbox_ptr
        a.n_layers, a.T, a.npos = self.nl, self.T, self.npos
        a.flags = self.flags if flags is None else flags
        a.cache_len = cache_len
        a.eps, a.fc_factor, a.att_scale = self.sc["eps"], self.sc["fc_factor"], self.sc["att_scale"]
        check(lib().mk_launch(ctypes.byref(a), self.blocks, self.threads, stream), "mk_launch")

    @torch.inference_mode()
    def pre_encode(self, mel: torch.Tensor, mel_len: torch.Tensor, drop: int) -> torch.Tensor:
        """NeMo's own subsampling + positional-encoding input scaling for one chunk. Returns (T, D) float32."""
        enc = self.enc
        x = mel.to(model_dtype(self.model)).transpose(1, 2)                  # (B, T_mel, F)
        x, _ = enc.pre_encode(x=x, lengths=mel_len)
        if drop:
            x = x[:, drop:, :]
        x, _ = enc.pos_enc(x=x, cache_len=CACHE)
        return x[0].float()

    @torch.inference_mode()
    def step(self, mel: torch.Tensor, mel_len: torch.Tensor, drop: int) -> torch.Tensor:
        """Full encoder step. Returns encoded (1, D, T) like NeMo's cache_aware_stream_step."""
        x = self.pre_encode(mel, mel_len, drop)
        y = self.layers_step(x)
        out = y.unsqueeze(0)
        if getattr(self.enc, "out_proj", None) is not None:
            out = self.enc.out_proj(out.to(self.enc.out_proj.weight.dtype)).float()
        return out.transpose(1, 2)

    def stage_times_us(self) -> Optional[List[float]]:
        if self.timers is None:
            return None
        t = self.timers.cpu().double()
        return ((t[1:] - t[:-1]) / 1e3).tolist()

    def stage_phases_us(self):
        """Per (layer, stage): (input load, compute, barrier+prefetch) in µs as seen by block 0 (S10 excluded)."""
        if self.ptimers is None:
            return None
        t = self.timers.cpu().double().tolist()
        p = self.ptimers.cpu().double().view(self.nl, NSTAGE, 2).tolist()
        out = []
        for l in range(self.nl):
            row = []
            for s in range(NSTAGE - 1):
                start = t[1 + l * NSTAGE + s]           # end of the previous stage (or of the input copy)
                end = t[2 + l * NSTAGE + s]
                a0, a1 = p[l][s][0], p[l][s][1]
                row.append(((a0 - start) / 1e3, (a1 - a0) / 1e3, (end - a1) / 1e3))
            out.append(row)
        return out
