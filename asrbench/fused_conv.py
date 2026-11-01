"""Preliminary approach B: the Conformer convolution block for a streaming chunk in 2 Triton kernels.

Replaces (eval, cache-aware path)
    residual -> LayerNorm(norm_conv) -> pointwise_conv1 (D->2D) -> GLU -> mask pads -> causal depthwise
    conv (k=9, uses cache_last_time) -> LayerNorm(conv.batch_norm) -> SiLU -> pointwise_conv2 (D->D) -> + residual
which NeMo runs as ~15 kernels, by

  kernel 1  grid (B, D/BLOCK_C): each program owns BLOCK_C channels for the whole call. It recomputes the
            LayerNorm statistics of the full input vector itself (no grid sync), computes rows c and c+D of
            pointwise_conv1, applies GLU, and runs the causal depthwise conv for its channels using their
            cached history. Also writes the next conv cache for its channels.
  kernel 2  grid (B*T, D/BLOCK_N): LayerNorm over channels (needs all channels -> the one unavoidable
            dependency), SiLU, pointwise_conv2 rows, optional residual add.

This is the "one dependency per conv module" mapping from the proposal, written in Triton so it can be
validated quickly; the CUDA megakernel version comes later.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_k1(X, LNW, LNB, W1, PAD, CACHE, WDW, H, Y, CACHE_OUT,
             T, D: tl.constexpr, eps,
             KC: tl.constexpr, KSIZE: tl.constexpr, T_PAD: tl.constexpr,
             BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, HAS_PAD: tl.constexpr):
    b = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < D
    offs_k = tl.arange(0, BLOCK_K)
    L = KC + T  # length of [cache | new frames] per channel in scratch H
    h_base = H + b * D * L
    for t in range(T):
        x_row = X + (b * T + t) * D
        # LayerNorm statistics over all D channels (two-pass, fp32), recomputed by every program
        s = tl.zeros([BLOCK_K], dtype=tl.float32)
        for k0 in range(0, D, BLOCK_K):
            km = (k0 + offs_k) < D
            s += tl.load(x_row + k0 + offs_k, mask=km, other=0.0).to(tl.float32)
        mean = tl.sum(s, axis=0) / D
        s2 = tl.zeros([BLOCK_K], dtype=tl.float32)
        for k0 in range(0, D, BLOCK_K):
            km = (k0 + offs_k) < D
            xv = tl.load(x_row + k0 + offs_k, mask=km, other=0.0).to(tl.float32)
            dv = tl.where(km, xv - mean, 0.0)
            s2 += dv * dv
        rstd = 1.0 / tl.sqrt(tl.sum(s2, axis=0) / D + eps)
        # rows c (value half) and c + D (gate half) of pointwise_conv1
        acc_a = tl.zeros([BLOCK_C], dtype=tl.float32)
        acc_g = tl.zeros([BLOCK_C], dtype=tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k = k0 + offs_k
            km = k < D
            xv = tl.load(x_row + k, mask=km, other=0.0).to(tl.float32)
            lw = tl.load(LNW + k, mask=km, other=0.0).to(tl.float32)
            lb = tl.load(LNB + k, mask=km, other=0.0).to(tl.float32)
            xn = tl.where(km, (xv - mean) * rstd * lw + lb, 0.0)
            wm = cm[:, None] & km[None, :]
            wa = tl.load(W1 + c[:, None] * D + k[None, :], mask=wm, other=0.0).to(tl.float32)
            wg = tl.load(W1 + (c[:, None] + D) * D + k[None, :], mask=wm, other=0.0).to(tl.float32)
            acc_a += tl.sum(wa * xn[None, :], axis=1)
            acc_g += tl.sum(wg * xn[None, :], axis=1)
        h = acc_a * tl.sigmoid(acc_g)                                   # GLU
        if HAS_PAD:
            pm = tl.load(PAD + b * T + t)
            h = tl.where(pm != 0, 0.0, h)
        tl.store(h_base + c * L + KC + t, h, mask=cm)
    # history: H[:, 0:KC] = cache
    for j in tl.static_range(KC):
        cv = tl.load(CACHE + (b * D + c) * KC + j, mask=cm, other=0.0).to(tl.float32)
        tl.store(h_base + c * L + j, cv, mask=cm)
    tl.debug_barrier()
    # causal depthwise conv: y[t, c] = sum_j w[c, j] * H[c, t + j]
    tt = tl.arange(0, T_PAD)
    tm = tt < T
    acc = tl.zeros([BLOCK_C, T_PAD], dtype=tl.float32)
    for j in tl.static_range(KSIZE):
        v = tl.load(h_base + c[:, None] * L + tt[None, :] + j, mask=cm[:, None] & tm[None, :], other=0.0)
        w = tl.load(WDW + c * KSIZE + j, mask=cm, other=0.0).to(tl.float32)
        acc += w[:, None] * v
    tl.store(Y + (b * T + tt[None, :]) * D + c[:, None], acc, mask=cm[:, None] & tm[None, :])
    # next cache = last KC entries of [cache | new]
    for j in tl.static_range(KC):
        nv = tl.load(h_base + c * L + T + j, mask=cm, other=0.0)
        tl.store(CACHE_OUT + (b * D + c) * KC + j, nv.to(CACHE_OUT.dtype.element_ty), mask=cm)


@triton.jit
def _conv_k2(Y, LNW, LNB, W2, R, OUT, D: tl.constexpr, eps,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ADD_RES: tl.constexpr):
    row = tl.program_id(0)            # b * T + t
    nb = tl.program_id(1)
    n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = n < D
    offs_k = tl.arange(0, BLOCK_K)
    y_row = Y + row * D
    s = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        km = (k0 + offs_k) < D
        s += tl.load(y_row + k0 + offs_k, mask=km, other=0.0)
    mean = tl.sum(s, axis=0) / D
    s2 = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        km = (k0 + offs_k) < D
        dv = tl.where(km, tl.load(y_row + k0 + offs_k, mask=km, other=0.0) - mean, 0.0)
        s2 += dv * dv
    rstd = 1.0 / tl.sqrt(tl.sum(s2, axis=0) / D + eps)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k = k0 + offs_k
        km = k < D
        yv = tl.load(y_row + k, mask=km, other=0.0)
        lw = tl.load(LNW + k, mask=km, other=0.0).to(tl.float32)
        lb = tl.load(LNB + k, mask=km, other=0.0).to(tl.float32)
        z = (yv - mean) * rstd * lw + lb
        z = tl.where(km, z * tl.sigmoid(z), 0.0)                       # SiLU
        w = tl.load(W2 + n[:, None] * D + k[None, :], mask=nm[:, None] & km[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * z[None, :], axis=1)
    if ADD_RES:
        acc += tl.load(R + row * D + n, mask=nm, other=0.0).to(tl.float32)
    tl.store(OUT + row * D + n, acc.to(OUT.dtype.element_ty), mask=nm)


def _cfg():
    return dict(
        block_c=int(os.environ.get("FC_BLOCK_C", 4)),
        block_n=int(os.environ.get("FC_BLOCK_N", 4)),
        block_k=int(os.environ.get("FC_BLOCK_K", 128)),
        warps=int(os.environ.get("FC_WARPS", 4)),
    )


class FusedConvBlock(nn.Module):
    """LayerNorm(norm_conv) + ConformerConvolution (+ optional residual) for the cache-aware path."""

    def __init__(self, norm_conv: nn.LayerNorm, conv: nn.Module, add_residual: bool = True):
        super().__init__()
        assert conv.pointwise_activation == "glu_", "kernel implements the GLU variant"
        self.norm_conv, self.conv = norm_conv, conv
        self.add_residual = add_residual
        dw = conv.depthwise_conv
        self.ksize = dw.kernel_size[0]
        self.kc = dw._left_padding
        assert dw._right_padding == 0 and getattr(dw, "cache_drop_size", 0) in (None, 0), "only causal conv supported"
        assert conv.norm_type == "layer_norm", "kernel implements the LayerNorm variant (fold BatchNorm otherwise)"
        assert conv.pointwise_conv1.bias is None and conv.pointwise_conv2.bias is None and dw.bias is None
        self.d = conv.d_model

    def weights(self):
        c = self.conv
        return (self.norm_conv.weight, self.norm_conv.bias,
                c.pointwise_conv1.weight.view(2 * self.d, self.d),
                c.depthwise_conv.weight.view(self.d, self.ksize),
                c.batch_norm.weight, c.batch_norm.bias,
                c.pointwise_conv2.weight.view(self.d, self.d))

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                cache: Optional[torch.Tensor] = None, residual_already_normed: bool = False):
        """x: (B, T, D) residual stream. cache: (B, D, KC). Returns (out, next_cache)."""
        if cache is None:  # non-streaming path: defer to NeMo
            y = self.conv(self.norm_conv(x), pad_mask=pad_mask)
            return (x + y) if self.add_residual else y
        B, T, D = x.shape
        cfg = _cfg()
        x = x.contiguous()
        lnw, lnb, w1, wdw, l2w, l2b, w2 = self.weights()
        H = torch.empty(B, D, self.kc + T, device=x.device, dtype=torch.float32)
        Y = torch.empty(B, T, D, device=x.device, dtype=torch.float32)
        cache_out = torch.empty_like(cache)
        has_pad = pad_mask is not None
        pad = pad_mask.to(torch.uint8).contiguous() if has_pad else Y  # dummy pointer when unused
        _conv_k1[(B, triton.cdiv(D, cfg["block_c"]))](
            x, lnw, lnb, w1, pad, cache.contiguous(), wdw, H, Y, cache_out,
            T, D, self.norm_conv.eps,
            KC=self.kc, KSIZE=self.ksize, T_PAD=triton.next_power_of_2(T),
            BLOCK_C=cfg["block_c"], BLOCK_K=cfg["block_k"], HAS_PAD=has_pad, num_warps=cfg["warps"])
        out = torch.empty_like(x)
        _conv_k2[(B * T, triton.cdiv(D, cfg["block_n"]))](
            Y, l2w, l2b, w2, x, out, D, self.conv.batch_norm.eps,
            BLOCK_N=cfg["block_n"], BLOCK_K=cfg["block_k"], ADD_RES=self.add_residual, num_warps=cfg["warps"])
        return out, cache_out


class _ConvPatch(nn.Module):
    """Drop-in replacement for layer.conv when layer.norm_conv is replaced by Identity."""

    def __init__(self, fused: FusedConvBlock):
        super().__init__()
        self.fused = fused

    def forward(self, x, pad_mask=None, cache=None):
        return self.fused(x, pad_mask=pad_mask, cache=cache)


def apply(model) -> int:
    """Patch every ConformerLayer so its conv block runs the fused kernels (residual added by NeMo)."""
    n = 0
    for layer in model.encoder.layers:
        if isinstance(layer.conv, _ConvPatch):
            continue
        fused = FusedConvBlock(layer.norm_conv, layer.conv, add_residual=False)
        layer._orig_norm_conv, layer._orig_conv = layer.norm_conv, layer.conv
        layer.norm_conv = nn.Identity()
        layer.conv = _ConvPatch(fused)
        n += 1
    return n


def remove(model):
    for layer in model.encoder.layers:
        if hasattr(layer, "_orig_conv"):
            layer.norm_conv, layer.conv = layer._orig_norm_conv, layer._orig_conv
            del layer._orig_norm_conv, layer._orig_conv


def reference_block(norm_conv, conv, x, pad_mask, cache):
    """NeMo's own computation of the same block (for parity tests and module-level baselines)."""
    y, nc = conv(norm_conv(x), pad_mask=pad_mask, cache=cache)
    return x + y, nc
