"""Preliminary approach A: remove redundant work in NeMo's cache-aware attention (exact).

NeMo's RelPositionMultiHeadAttention keeps the *inputs* of the last C frames (C = 70) as its cache and, on
every chunk, recomputes  K = W_k [cache; x],  V = W_v [cache; x]  for all C + T frames, plus
P = W_pos pos_emb  for 2(C+T)-1 relative positions. With T = 1-2 new frames that is ~35-140x more
projection work than necessary.

Because W_k, W_v, W_pos are fixed linear maps:
  * cache the *projected* keys/values and only project the T new frames (as LLM KV caches do);
  * pos_emb depends only on shapes in steady state, so W_pos pos_emb is computed once and reused.
Both are exact up to floating-point summation order. The patch keeps NeMo's API: it still returns the
input cache NeMo expects, so it works with cache_aware_stream_step, CUDA graphs and torch.compile.

Projected caches are updated IN PLACE, so a captured CUDA graph stays valid. Call `reset(model, state)`
whenever a new stream starts (or after caches are modified outside the patched attention).
"""
from __future__ import annotations

import math
import types
from typing import Optional

import torch

from nemo.utils.cast_utils import avoid_float16_autocast_context

try:  # NeMo constant used to mask scores
    from nemo.collections.asr.parts.submodules.multi_head_attention import INF_VAL
except Exception:  # pragma: no cover
    INF_VAL = 10000.0


def _patched_forward(self, query, key, value, mask, pos_emb, cache=None):
    if cache is None:
        return self._orig_forward(query, key, value, mask, pos_emb, cache)

    B, T, _ = query.shape
    q_keep = T - self.cache_drop_size
    new_cache = torch.cat([cache[:, q_keep:, :], query[:, :q_keep, :]], dim=1)  # what NeMo returns

    if torch.is_autocast_enabled():
        query = query.to(torch.float32)

    with avoid_float16_autocast_context():
        st = self._kv
        if st["k"] is None or st["k"].shape[0] != B:
            # lazy init (eager only): project the given input cache once
            kc = self.linear_k(cache.to(query.dtype))
            vc = self.linear_v(cache.to(query.dtype))
            st["k"], st["v"] = kc.contiguous(), vc.contiguous()

        q = self.linear_q(query).view(B, -1, self.h, self.d_k).transpose(1, 2)        # (B,h,T,dk)
        k_new = self.linear_k(query)                                                  # (B,T,D)
        v_new = self.linear_v(query)
        k_all = torch.cat([st["k"].to(k_new.dtype), k_new], dim=1)                     # (B,C+T,D)
        v_all = torch.cat([st["v"].to(v_new.dtype), v_new], dim=1)
        # slide projected caches exactly like the input cache, in place
        st["k"].copy_(torch.cat([st["k"][:, q_keep:, :], k_new[:, :q_keep, :].to(st["k"].dtype)], dim=1))
        st["v"].copy_(torch.cat([st["v"][:, q_keep:, :], v_new[:, :q_keep, :].to(st["v"].dtype)], dim=1))
        k = k_all.view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = v_all.view(B, -1, self.h, self.d_k).transpose(1, 2)

        # constant positional projection
        key_p = (tuple(pos_emb.shape), pos_emb.dtype, pos_emb.device, torch.is_autocast_enabled())
        p = st["p"].get(key_p)
        if p is None:
            n_batch_pos = pos_emb.size(0)
            p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2).contiguous()
            st["p"][key_p] = p
        q_with_bias_u = (q.transpose(1, 2) + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q.transpose(1, 2) + self.pos_bias_v).transpose(1, 2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        if self.use_pytorch_sdpa:
            scale_factor = 1 / math.sqrt(q_with_bias_u.size(-1))
            matrix_bd = matrix_bd[:, :, :, : k.size(-2)] * scale_factor
            if mask is not None:
                mask = mask.unsqueeze(1)
                matrix_bd.masked_fill_(mask, -INF_VAL)
            if getattr(self, "use_pytorch_sdpa_backends", None):
                with torch.nn.attention.sdpa_kernel(self.use_pytorch_sdpa_backends):
                    out = torch.nn.functional.scaled_dot_product_attention(q_with_bias_u, k, v, attn_mask=matrix_bd)
            else:
                out = torch.nn.functional.scaled_dot_product_attention(q_with_bias_u, k, v, attn_mask=matrix_bd)
            if mask is not None:
                all_masked_rows = torch.all(mask, dim=-1).unsqueeze(-1)
                all_masked_rows = all_masked_rows.expand(-1, out.size(1), -1, out.size(-1))
                out = out.masked_fill(all_masked_rows, 0.0)
            out = out.transpose(1, 2).reshape(B, -1, self.h * self.d_k)
            out = self.linear_out(out)
        else:
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
            matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
            scores = (matrix_ac + matrix_bd) / self.s_d_k
            out = self.forward_attention(v, scores, mask)

    return out, new_cache


def apply(model):
    """Patch every RelPositionMultiHeadAttention in the encoder. Returns number of patched modules."""
    from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention

    n = 0
    for layer in model.encoder.layers:
        att = layer.self_attn
        if type(att) is not RelPositionMultiHeadAttention:
            continue
        if not hasattr(att, "_orig_forward"):
            att._orig_forward = att.forward
            att.forward = types.MethodType(_patched_forward, att)
        att._kv = {"k": None, "v": None, "p": {}}
        n += 1
    return n


@torch.inference_mode()
def reset(model, state=None, prec=None):
    """Re-synchronise projected caches with the input caches in `state` (in place if they exist).

    `prec` (asrbench.model.Precision) reproduces the autocast context used by the step, so the
    projected cache is computed exactly as the patched forward would compute it.
    """
    import contextlib

    for lth, layer in enumerate(model.encoder.layers):
        att = layer.self_attn
        if not hasattr(att, "_kv"):
            continue
        st = att._kv
        if state is None:
            st["k"] = st["v"] = None
            continue
        cache = state.c[lth]
        ctx = prec.ctx() if prec is not None else contextlib.nullcontext()
        with ctx:
            x = cache.to(torch.float32) if torch.is_autocast_enabled() else cache
            with avoid_float16_autocast_context():
                kc, vc = att.linear_k(x), att.linear_v(x)
        if st["k"] is None or st["k"].shape != kc.shape or st["k"].dtype != kc.dtype:
            st["k"], st["v"] = kc.contiguous(), vc.contiguous()
        else:
            st["k"].copy_(kc); st["v"].copy_(vc)


def remove(model):
    for layer in model.encoder.layers:
        att = layer.self_attn
        if hasattr(att, "_orig_forward"):
            att.forward = att._orig_forward
            del att._orig_forward
            del att._kv
