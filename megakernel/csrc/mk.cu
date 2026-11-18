// Megakernel v7.2 (v7.1 + NeMo-format attention-input ring so the state can be handed back to NeMo exactly): the 17 Conformer layers of one cache-aware streaming step in ONE cooperative launch.
//
// Stages per layer (each separated by a grid-wide barrier, 10 per layer):
//   S1  LN_ff1(resid) -> FFN1 up (512->2048) + SiLU                -> h
//   S2  FFN1 down (2048->512); resid += 0.5 * out
//   S3  LN_att(resid) -> fused Q|K|V projection (512->1536)        -> qkv
//   S4  attention core per (head, query): projected-K/V ring cache + new frames, rel-pos term
//   S5  output projection; resid += out; commit new K/V into the ring
//   S6  LN_conv(resid) -> PW1 rows c, c+512 -> GLU -> causal depthwise conv (+ conv cache)  -> y
//   S7  LN_cn(y) + SiLU -> PW2 (512->512); resid += out
//   S8  LN_ff2(resid) -> FFN2 up + SiLU                             -> h
//   S9  FFN2 down; resid += 0.5 * out
//   S10 LN_out(resid) -> next layer's residual (ping-pong buffer)
// Every block recomputes LayerNorm statistics itself (no extra barrier); matrix rows are spread over all warps.
// Weights FP16, activations/accumulation FP32. Streaming state (K/V ring, conv cache, ring head) lives on device.
//
// Built as a plain shared library (no PyTorch headers) and driven from Python through ctypes.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <string.h>

namespace cg = cooperative_groups;

#define D 512
#define DFF 2048
#define NH 8
#define DK 64
#define KC 8
#define KS 9
#define CACHE 70
#define TMAX 4
#define NKEY_PAD 80          // >= CACHE + TMAX, per-warp attention scratch
#define NSTAGE 10
#define NEG_INF (__int_as_float(0xff800000))
#define MASKED (-1000000)

struct LayerParams {          // ALL fields are pointers: the host packs an int64 array in exactly this order
  const float *ln_ff1_g, *ln_ff1_b;
  const half *w_ff1_1;  const float *b_ff1_1;   // [DFF, D]
  const half *w_ff1_2;  const float *b_ff1_2;   // [D, DFF]
  const float *ln_att_g, *ln_att_b;
  const half *w_qkv;    const float *b_qkv;     // [3D, D]  rows: q | k | v
  const half *w_o;      const float *b_o;       // [D, D]
  const float *pos_u, *pos_v;                   // [NH, DK]
  const float *p;                               // [NH, npos, DK]  linear_pos(pos_emb), precomputed
  const float *ln_conv_g, *ln_conv_b;
  const half *w_pw1;    const float *b_pw1;     // [2D, D]
  const float *w_dw, *b_dw;                     // [D, KS], [D]
  const float *ln_cn_g, *ln_cn_b;               // conv-module LayerNorm
  const half *w_pw2;    const float *b_pw2;     // [D, D]
  const float *ln_ff2_g, *ln_ff2_b;
  const half *w_ff2_1;  const float *b_ff2_1;
  const half *w_ff2_2;  const float *b_ff2_2;
  const float *ln_out_g, *ln_out_b;
};

struct MKArgs {
  const LayerParams* layers;
  const float* x_in;          // [T, D]
  float* x_out;               // [T, D]
  float* kv;                  // [L, 2, CACHE, D] projected K/V ring
  float* conv_cache;          // [L, D, KC]
  int* ring_head;             // [4]: ring head, then the persistent command relay {seq, cmd, cache_len}
  const uint8_t* mask;        // [T, CACHE+T]  1 = masked (NeMo semantics)
  const int* pidx;            // [T, CACHE+T]  index into p, -1 = zero (rel_shift padding)
  float* resid0;              // [T, D]
  float* resid1;              // [T, D]
  float* hbuf;                // [T, DFF]
  float* qkv;                 // [T, 3D]
  float* att;                 // [T, D]
  float* ybuf;                // [T, D]
  unsigned long long* timers; // [n_layers*NSTAGE + 3] or nullptr (block 0 records %globaltimer)
  unsigned long long* ptimers;// [n_layers*NSTAGE*2] or nullptr: block 0 stamps after its input load and after its compute
  int* mbox;                  // persistent mode: mapped host memory {seq, cmd, done, cache_len}
  float* xring;               // [L, CACHE, D] or nullptr: NeMo-format attention-input cache (LN_att rows), so the
                              // state can be handed back to NeMo exactly (e.g. for a short last chunk)
  int n_layers;
  int T;
  int npos;
  int flags;                  // bit0 L2-prefetch next layer, bit1 fused LN_out, bit3 adjacent rows, bit4 persistent
  int cache_len;              // valid cached frames before this chunk (used when mask == nullptr)
  int pad3_;
  float eps;
  float fc_factor;
  float att_scale;
  float pad2_;
};

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// Reduce n (<= 2*TMAX) per-thread values across the block in ONE round of barriers. red: >= 2*TMAX*32 floats.
__device__ void block_reduce_arr(float* v, int n, float* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int i = 0; i < n; ++i) v[i] = warp_sum(v[i]);
  __syncthreads();
  if (lane == 0)
    for (int i = 0; i < n; ++i) red[i * 32 + warp] = v[i];
  __syncthreads();
  for (int i = 0; i < n; ++i) {
    float s = 0.f;
    for (int w = 0; w < nw; ++w) s += red[i * 32 + w];
    v[i] = s;
  }
}

// Copy src[T][K] into shared memory with 16-byte loads, all issued before any is used (one L2 round trip instead of
// one per element); optionally LayerNorm each row (g != nullptr) and apply SiLU (act == 1).
template <int K, int TT>
__device__ void block_load_k(const float* __restrict__ src, const float* g, const float* b, float eps, int act,
                             float* xs, float* red) {
  constexpr int T = TT;
  constexpr int K4 = K / 4;
  constexpr int PER = (K4 + 255) / 256;          // float4 per thread per row (requires blockDim >= 256)
  constexpr int PK = (K == D) ? K / 256 : 1;     // LayerNorm only ever runs on D-wide rows
  float v[2 * TT];
  float4 buf[TT][PER];
  float gv[PK], bv[PK];
#pragma unroll
  for (int i = 0; i < 2 * TT; ++i) v[i] = 0.f;
  if (g != nullptr) {                             // gamma/beta loads go out together with the activation loads
#pragma unroll
    for (int j = 0; j < PK; ++j) { gv[j] = g[threadIdx.x + j * 256]; bv[j] = b[threadIdx.x + j * 256]; }
  }
  const float4* s4 = reinterpret_cast<const float4*>(src);
  float4* x4 = reinterpret_cast<float4*>(xs);
#pragma unroll
  for (int t = 0; t < TT; ++t)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int i = threadIdx.x + j * blockDim.x;
      if (i < K4) buf[t][j] = s4[t * K4 + i];
    }
#pragma unroll
  for (int t = 0; t < TT; ++t)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int i = threadIdx.x + j * blockDim.x;
      if (i < K4) {
        const float4 q = buf[t][j];
        x4[t * K4 + i] = q;
        v[2 * t] += q.x + q.y + q.z + q.w;
        v[2 * t + 1] += q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
      }
    }
  if (g == nullptr) { __syncthreads(); return; }
  block_reduce_arr(v, 2 * T, red);
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float mean = v[2 * t] / K;
    const float rstd = rsqrtf(fmaxf(v[2 * t + 1] / K - mean * mean, 0.f) + eps);
    float* row = xs + t * K;
#pragma unroll
    for (int j = 0; j < PK; ++j) {
      const int k = threadIdx.x + j * 256;
      float y = (row[k] - mean) * rstd * gv[j] + bv[j];
      if (act == 1) y = silu(y);
      row[k] = y;
    }
  }
  __syncthreads();
}

// LayerNorm rows of xs (already in shared memory) in place.
__device__ void block_ln_inplace(float* xs, int T, int K, const float* g, const float* b, float eps, int act, float* red) {
  float v[2 * TMAX];
#pragma unroll
  for (int i = 0; i < 2 * TMAX; ++i) v[i] = 0.f;
  for (int t = 0; t < T; ++t)
    for (int k = threadIdx.x; k < K; k += blockDim.x) { const float x = xs[t * K + k]; v[2 * t] += x; v[2 * t + 1] += x * x; }
  block_reduce_arr(v, 2 * T, red);
  for (int t = 0; t < T; ++t) {
    const float mean = v[2 * t] / K;
    const float rstd = rsqrtf(fmaxf(v[2 * t + 1] / K - mean * mean, 0.f) + eps);
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      float y = (xs[t * K + k] - mean) * rstd * g[k] + b[k];
      if (act == 1) y = silu(y);
      xs[t * K + k] = y;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// Fire-and-forget L2 prefetch of one layer's eight weight matrices (~12.6 MB), spread over every thread of the grid.
__device__ void prefetch_layer(const LayerParams& P, int gtid, int nthreads) {
  const half* mats[8] = {P.w_ff1_1, P.w_ff1_2, P.w_qkv, P.w_o, P.w_pw1, P.w_pw2, P.w_ff2_1, P.w_ff2_2};
  const int nel[8] = {DFF * D, D * DFF, 3 * D * D, D * D, 2 * D * D, D * D, DFF * D, D * DFF};
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    const char* base = reinterpret_cast<const char*>(mats[m]);
    const int lines = nel[m] * 2 / 128;
    for (int i = gtid; i < lines; i += nthreads) prefetch_l2(base + (size_t)i * 128);
  }
}

// acc[t] = dot(W_row, xs[t]) for t < T; result valid in ALL lanes. K must be a multiple of 256.
__device__ __forceinline__ void row_dot(const half* __restrict__ wr, int K, const float* xs, int T, int lane,
                                        float acc[TMAX]) {
#pragma unroll
  for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
  for (int k = lane * 8; k < K; k += 256) {
    const uint4 wv = __ldg(reinterpret_cast<const uint4*>(wr + k));
    const half2* w2 = reinterpret_cast<const half2*>(&wv);
    float wf[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float2 f = __half22float2(w2[i]); wf[2 * i] = f.x; wf[2 * i + 1] = f.y; }
#pragma unroll
    for (int t = 0; t < TMAX; ++t) {
      if (t < T) {
        const float4* xr = reinterpret_cast<const float4*>(xs + t * K + k);
        const float4 a0 = xr[0], a1 = xr[1];
        acc[t] += wf[0] * a0.x + wf[1] * a0.y + wf[2] * a0.z + wf[3] * a0.w
                + wf[4] * a1.x + wf[5] * a1.y + wf[6] * a1.z + wf[7] * a1.w;
      }
    }
  }
#pragma unroll
  for (int t = 0; t < TMAX; ++t) acc[t] = warp_sum(acc[t]);
}

__device__ __forceinline__ int ring_slot(int head_ring, int j) { int s = head_ring + j; return s >= CACHE ? s - CACHE : s; }

// One (head, query) attention item, computed by one whole block:
//   scores: warp w takes keys w, w+wpb, ...; lanes split the 64 head dims (coalesced loads), warp-reduce per key
//   softmax: warp 0 over <= 74 scores in shared memory
//   output:  thread (d, g) sums keys g, g+G, ... for dim d; groups combined through shared memory
// scratch: >= 128 + NKEY_PAD + (blockDim/64)*64 floats
__device__ void attention_block(const MKArgs& a, const LayerParams& P, const float* kvK, const float* kvV,
                                int head_ring, int h, int t, float* scratch) {
  const int T = a.T, NK = CACHE + T;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wpb = blockDim.x >> 5;
  float* qu = scratch;
  float* qv = scratch + DK;
  float* sc = scratch + 2 * DK;
  float* part = sc + NKEY_PAD;
  if (threadIdx.x < DK) {
    const float qd = a.qkv[t * 3 * D + h * DK + threadIdx.x];
    qu[threadIdx.x] = qd + P.pos_u[h * DK + threadIdx.x];
    qv[threadIdx.x] = qd + P.pos_v[h * DK + threadIdx.x];
  }
  __syncthreads();
  {
    const int kq = lane >> 3, sub = lane & 7;          // 4 keys per warp per round, 8 lanes per key, 8 dims per lane
    const float4 qu0 = *reinterpret_cast<const float4*>(qu + sub * 8), qu1 = *reinterpret_cast<const float4*>(qu + sub * 8 + 4);
    const float4 qv0 = *reinterpret_cast<const float4*>(qv + sub * 8), qv1 = *reinterpret_cast<const float4*>(qv + sub * 8 + 4);
    for (int base = warp * 4; base < NK; base += wpb * 4) {
      const int j = base + kq;
      const bool valid = (j < NK) && (a.mask[t * NK + j] == 0);
      float v = 0.f;
      if (valid) {
        const float* krow = ((j < CACHE) ? kvK + ring_slot(head_ring, j) * D : a.qkv + (j - CACHE) * 3 * D + D) + h * DK + sub * 8;
        const float4 k0 = *reinterpret_cast<const float4*>(krow), k1 = *reinterpret_cast<const float4*>(krow + 4);
        v = qu0.x * k0.x + qu0.y * k0.y + qu0.z * k0.z + qu0.w * k0.w + qu1.x * k1.x + qu1.y * k1.y + qu1.z * k1.z + qu1.w * k1.w;
        const int pi = a.pidx[t * NK + j];
        if (pi >= 0) {
          const float* prow = P.p + ((size_t)h * a.npos + pi) * DK + sub * 8;
          const float4 p0 = *reinterpret_cast<const float4*>(prow), p1 = *reinterpret_cast<const float4*>(prow + 4);
          v += qv0.x * p0.x + qv0.y * p0.y + qv0.z * p0.z + qv0.w * p0.w + qv1.x * p1.x + qv1.y * p1.y + qv1.z * p1.z + qv1.w * p1.w;
        }
      }
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      if (sub == 0 && j < NK) sc[j] = valid ? v * a.att_scale : NEG_INF;
    }
  }
  __syncthreads();
  if (warp == 0) {
    float m = NEG_INF;
    for (int j = lane; j < NK; j += 32) m = fmaxf(m, sc[j]);
    m = warp_max(m);
    float sum = 0.f;
    for (int j = lane; j < NK; j += 32) {
      const float e = (m == NEG_INF || sc[j] == NEG_INF) ? 0.f : __expf(sc[j] - m);
      sc[j] = e;
      sum += e;
    }
    sum = warp_sum(sum);
    const float inv = sum > 0.f ? 1.f / sum : 0.f;          // all keys masked -> zeros (NeMo semantics)
    for (int j = lane; j < NK; j += 32) sc[j] *= inv;
  }
  __syncthreads();
  const int d = threadIdx.x % DK, g = threadIdx.x / DK, G = blockDim.x / DK;
  float o = 0.f;
#pragma unroll 4
  for (int j = g; j < NK; j += G) {                    // branch-free: masked keys have weight 0
    const float* vrow = ((j < CACHE) ? kvV + ring_slot(head_ring, j) * D : a.qkv + (j - CACHE) * 3 * D + 2 * D) + h * DK;
    o += sc[j] * vrow[d];
  }
  part[g * DK + d] = o;
  __syncthreads();
  if (threadIdx.x < DK) {
    float acc = 0.f;
    for (int gg = 0; gg < G; ++gg) acc += part[gg * DK + threadIdx.x];
    a.att[t * D + h * DK + threadIdx.x] = acc;
  }
  __syncthreads();
}

// Split-K attention (flash-decoding style). Item (h, t, s) handles keys [s*per, (s+1)*per) of query t, head h, and
// writes an UNNORMALISED partial: max m, sum l = sum exp(score - m), o[d] = sum exp(score - m) * v[d].
// The consumer stage (S5) merges the S partials while loading its input, so no extra barrier is needed.
__device__ void attention_split(const MKArgs& a, const LayerParams& P, const float* kvK, const float* kvV,
                                int head_ring, int h, int t, int s, int S, float* scratch, const int* mp) {
  const int T = a.T, NK = CACHE + T;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wpb = blockDim.x >> 5;
  const int per = (NK + S - 1) / S;
  const int j0 = s * per;
  const int nkeys = max(0, min(NK, j0 + per) - j0);
  const int item = (h * T + t) * S + s;
  float* qu = scratch;
  float* qv = scratch + DK;
  float* sc = scratch + 2 * DK;
  float* part = sc + NKEY_PAD;
  float* bc = part + 8 * DK;                          // broadcast m, l
  if (threadIdx.x < DK) {
    const float qd = a.qkv[t * 3 * D + h * DK + threadIdx.x];
    qu[threadIdx.x] = qd + P.pos_u[h * DK + threadIdx.x];
    qv[threadIdx.x] = qd + P.pos_v[h * DK + threadIdx.x];
  }
  __syncthreads();
  {
    const int kq = lane >> 3, sub = lane & 7;
    const float4 qu0 = *reinterpret_cast<const float4*>(qu + sub * 8), qu1 = *reinterpret_cast<const float4*>(qu + sub * 8 + 4);
    const float4 qv0 = *reinterpret_cast<const float4*>(qv + sub * 8), qv1 = *reinterpret_cast<const float4*>(qv + sub * 8 + 4);
    for (int base = warp * 4; base < nkeys; base += wpb * 4) {
      const int jj = base + kq, j = j0 + jj;
      const int code = (jj < nkeys) ? mp[t * NK + j] : MASKED;     // shared memory: no global round trip
      const bool valid = code != MASKED;
      float v = 0.f;
      if (valid) {
        const float* krow = ((j < CACHE) ? kvK + ring_slot(head_ring, j) * D : a.qkv + (j - CACHE) * 3 * D + D) + h * DK + sub * 8;
        const float4 k0 = *reinterpret_cast<const float4*>(krow), k1 = *reinterpret_cast<const float4*>(krow + 4);
        v = qu0.x * k0.x + qu0.y * k0.y + qu0.z * k0.z + qu0.w * k0.w + qu1.x * k1.x + qu1.y * k1.y + qu1.z * k1.z + qu1.w * k1.w;
        const int pi = code;
        if (pi >= 0) {
          const float* prow = P.p + ((size_t)h * a.npos + pi) * DK + sub * 8;
          const float4 p0 = *reinterpret_cast<const float4*>(prow), p1 = *reinterpret_cast<const float4*>(prow + 4);
          v += qv0.x * p0.x + qv0.y * p0.y + qv0.z * p0.z + qv0.w * p0.w + qv1.x * p1.x + qv1.y * p1.y + qv1.z * p1.z + qv1.w * p1.w;
        }
      }
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      v += __shfl_xor_sync(0xffffffffu, v, 2);
      v += __shfl_xor_sync(0xffffffffu, v, 1);
      if (sub == 0 && jj < nkeys) sc[jj] = valid ? v * a.att_scale : NEG_INF;
    }
  }
  __syncthreads();
  if (warp == 0) {
    float m = NEG_INF;
    for (int jj = lane; jj < nkeys; jj += 32) m = fmaxf(m, sc[jj]);
    m = warp_max(m);
    float l = 0.f;
    for (int jj = lane; jj < nkeys; jj += 32) {
      const float e = (m == NEG_INF || sc[jj] == NEG_INF) ? 0.f : __expf(sc[jj] - m);
      sc[jj] = e;
      l += e;
    }
    l = warp_sum(l);
    if (lane == 0) { bc[0] = m; bc[1] = l; }
  }
  __syncthreads();
  const int d = threadIdx.x % DK, g = threadIdx.x / DK, G = blockDim.x / DK;
  float o = 0.f;
#pragma unroll 4
  for (int jj = g; jj < nkeys; jj += G) {
    const int j = j0 + jj;
    const float* vrow = ((j < CACHE) ? kvV + ring_slot(head_ring, j) * D : a.qkv + (j - CACHE) * 3 * D + 2 * D) + h * DK;
    o += sc[jj] * vrow[d];
  }
  part[g * DK + d] = o;
  __syncthreads();
  if (threadIdx.x < DK) {
    float acc = 0.f;
    for (int gg = 0; gg < G; ++gg) acc += part[gg * DK + threadIdx.x];
    a.hbuf[item * DK + threadIdx.x] = acc;          // partial o   (hbuf is free between S3 and S6)
  }
  if (threadIdx.x == 0) { a.att[2 * item] = bc[0]; a.att[2 * item + 1] = bc[1]; }   // partial m, l
  __syncthreads();
}

// Merge the S attention partials into xs[T][D] (normalised), every block redundantly; replaces S5's input load.
// All partial values a thread needs are loaded before any arithmetic, so the merge costs ~one L2 round trip.
template <int TT>
__device__ void attn_combine(const MKArgs& a, int S, float* xs) {
  constexpr int PER = (TT * D + 255) / 256;          // outputs per thread (blockDim = 256)
  float mv[PER][4], lv[PER][4], ov[PER][4];
#pragma unroll
  for (int p = 0; p < PER; ++p) {
    const int idx = threadIdx.x + p * 256;
    const int t = idx / D, hd = idx % D, h = hd / DK, d = hd % DK;
    const int item0 = (h * TT + t) * S;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      if (idx < TT * D && s < S) {
        mv[p][s] = a.att[2 * (item0 + s)];
        lv[p][s] = a.att[2 * (item0 + s) + 1];
        ov[p][s] = a.hbuf[(item0 + s) * DK + d];
      } else {
        mv[p][s] = NEG_INF; lv[p][s] = 0.f; ov[p][s] = 0.f;
      }
    }
  }
#pragma unroll
  for (int p = 0; p < PER; ++p) {
    const int idx = threadIdx.x + p * 256;
    if (idx < TT * D) {
      float M = NEG_INF;
#pragma unroll
      for (int s = 0; s < 4; ++s) M = fmaxf(M, mv[p][s]);
      float L = 0.f, o = 0.f;
      if (M != NEG_INF) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const float w = (mv[p][s] == NEG_INF) ? 0.f : __expf(mv[p][s] - M);
          L += w * lv[p][s];
          o += w * ov[p][s];
        }
      }
      xs[idx] = (L > 0.f) ? o / L : 0.f;            // all keys masked -> zeros (NeMo semantics)
    }
  }
  __syncthreads();
}

#define STAMP(idx) do { if (a.timers && blockIdx.x == 0 && threadIdx.x == 0) a.timers[(idx)] = gtimer(); } while (0)
#define PSTAMP(stage, k) do { if (a.ptimers && blockIdx.x == 0 && threadIdx.x == 0) a.ptimers[((l) * NSTAGE + (stage)) * 2 + (k)] = gtimer(); } while (0)

// ------------------------------------------------------------------------------------------- multi-row GEMV
// A warp computes R rows at once (R*K/256 16-byte loads per lane, all issued before the first FMA), so a stage pays
// ~one DRAM round trip instead of one per loop iteration per row. R = 4 for K = 512, 1 for K = 2048 (32 registers).
// PAIR: each work item is a channel c needing rows c and c+D (conv PW1 + GLU); R counts channels then.
template <int N_> struct Ops { float v[N_]; };      // per-row epilogue operands (bias, residual, conv params)

// contig: a warp takes R ADJACENT rows (one contiguous R*K*2-byte stretch) instead of rows nwarps apart.
// pre(r) loads the row's epilogue operands right after its weight loads are issued, so they arrive in the same
// round trip instead of after the math; epi(r, acc, acc2, ops) consumes them.
template <int K, int R, int PAIR, int TT, typename Pre, typename Epi>
__device__ __forceinline__ void gemv_rows(const half* __restrict__ W, int N, const float* xs, int gwarp,
                                          int nwarps, int lane, bool contig, Pre pre, Epi epi) {
  constexpr int IT = K / 256;
  constexpr int NQ = PAIR ? 2 : 1;
  const int di = contig ? 1 : nwarps;
  for (int base = contig ? gwarp * R : gwarp; base < N; base += R * nwarps) {
    uint4 w[R][NQ][IT];
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const int r = base + i * di;
      if (r < N) {
#pragma unroll
        for (int q = 0; q < NQ; ++q)
#pragma unroll
          for (int it = 0; it < IT; ++it)
            w[i][q][it] = __ldg(reinterpret_cast<const uint4*>(W + (size_t)(r + q * D) * K + lane * 8 + it * 256));
      }
    }
    decltype(pre(0)) ops[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const int r = base + i * di;
      if (r < N) ops[i] = pre(r);
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
      const int r = base + i * di;
      if (r < N) {                                   // warp-uniform condition
        float acc[NQ][TT];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
#pragma unroll
          for (int t = 0; t < TT; ++t) acc[q][t] = 0.f;
#pragma unroll
          for (int it = 0; it < IT; ++it) {
            const half2* w2 = reinterpret_cast<const half2*>(&w[i][q][it]);
            float wf[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) { const float2 f = __half22float2(w2[e]); wf[2 * e] = f.x; wf[2 * e + 1] = f.y; }
            const int k = lane * 8 + it * 256;
#pragma unroll
            for (int t = 0; t < TT; ++t) {
              const float4* xr = reinterpret_cast<const float4*>(xs + t * K + k);
              const float4 a0 = xr[0], a1 = xr[1];
              acc[q][t] += wf[0] * a0.x + wf[1] * a0.y + wf[2] * a0.z + wf[3] * a0.w
                         + wf[4] * a1.x + wf[5] * a1.y + wf[6] * a1.z + wf[7] * a1.w;
            }
          }
#pragma unroll
          for (int t = 0; t < TT; ++t) acc[q][t] = warp_sum(acc[q][t]);
        }
        epi(r, acc[0], acc[NQ - 1], ops[i]);
      }
    }
  }
}

template <int TT>
__device__ __forceinline__ void mk_body(const MKArgs& a, const int cache_len) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ float smem[];
  float* xs = smem;                           // TMAX*DFF floats
  float* red = xs + TMAX * DFF;               // 2*TMAX*32 = 256 floats
  float* attw = red + 256;                    // 1024 floats attention scratch
  int* mp = reinterpret_cast<int*>(attw + 1024);   // TMAX*NKEY_PAD: pidx, or MASKED for masked keys
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wpb = blockDim.x >> 5;
  const int gwarp = blockIdx.x * wpb + warp, nwarps = gridDim.x * wpb;
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x, nthreads = gridDim.x * blockDim.x;
  constexpr int T = TT;                       // frames per chunk is a compile-time constant (one kernel per T)
  const int S_att = max(1, min(4, (int)gridDim.x / (NH * TT)));   // attention key splits (blocks per head x query)
  const int head_ring = *a.ring_head;
  const bool do_l2pf = (a.flags & 1) != 0, fuse_ln = (a.flags & 2) != 0;   // bit2 (register prefetch) retired in v5
  const bool contig = (a.flags & 8) != 0;                                     // bit3: adjacent rows per warp

  STAMP(0);
  {
    const int NK = CACHE + T, invalid = CACHE - min(cache_len, CACHE);     // oldest slots not yet filled
    for (int i = threadIdx.x; i < T * NK; i += blockDim.x) {
      const bool masked = a.mask ? (a.mask[i] != 0) : ((i % NK) < invalid);
      mp[i] = masked ? MASKED : a.pidx[i];
    }
  }
  if (do_l2pf) prefetch_layer(a.layers[0], gtid, nthreads);
  for (int i = gtid; i < T * D; i += nthreads) a.resid0[i] = a.x_in[i];
  grid.sync(); STAMP(1);
  float* resid = a.resid0;
  float* resid_next = a.resid1;

  for (int l = 0; l < a.n_layers; ++l) {
    const LayerParams& P = a.layers[l];
    float* kvK = a.kv + (size_t)l * 2 * CACHE * D;
    float* kvV = kvK + (size_t)CACHE * D;
    float* cc = a.conv_cache + (size_t)l * D * KC;
    const int ts = 2 + l * NSTAGE;
    if (do_l2pf && l + 1 < a.n_layers) prefetch_layer(a.layers[l + 1], gtid, nthreads);

    // S1: [LN_out of the previous layer, if fused] + LN_ff1 + FFN1 up + SiLU
    if (fuse_ln && l > 0) {
      block_load_k<D, TT>(resid, a.layers[l - 1].ln_out_g, a.layers[l - 1].ln_out_b, a.eps, 0, xs, red);
      for (int i = gtid; i < T * D; i += nthreads) resid_next[i] = xs[i];   // this layer's residual input
      block_ln_inplace(xs, T, D, P.ln_ff1_g, P.ln_ff1_b, a.eps, 0, red);
      float* tmp = resid; resid = resid_next; resid_next = tmp;
    } else {
      block_load_k<D, TT>(resid, P.ln_ff1_g, P.ln_ff1_b, a.eps, 0, xs, red);
    }
    PSTAMP(0, 0);
    gemv_rows<D, 4, 0, TT>(P.w_ff1_1, DFF, xs, gwarp, nwarps, lane, contig,
      [&](int r) { Ops<1> o; o.v[0] = P.b_ff1_1[r]; return o; },
      [&](int r, float* acc, float*, const Ops<1>& o) {
        if (lane == 0) for (int t = 0; t < T; ++t) a.hbuf[t * DFF + r] = silu(acc[t] + o.v[0]);
      });
    PSTAMP(0, 1);
    grid.sync(); STAMP(ts + 0);

    // S2: FFN1 down; resid += fc * out
    block_load_k<DFF, TT>(a.hbuf, nullptr, nullptr, a.eps, 0, xs, red);
    PSTAMP(1, 0);
    gemv_rows<DFF, 1, 0, TT>(P.w_ff1_2, D, xs, gwarp, nwarps, lane, contig,
      [&](int r) { Ops<1 + TT> o; o.v[0] = P.b_ff1_2[r]; for (int t = 0; t < TT; ++t) o.v[1 + t] = resid[t * D + r]; return o; },
      [&](int r, float* acc, float*, const Ops<1 + TT>& o) {
        if (lane == 0) for (int t = 0; t < T; ++t) resid[t * D + r] = o.v[1 + t] + a.fc_factor * (acc[t] + o.v[0]);
      });
    PSTAMP(1, 1);
    grid.sync(); STAMP(ts + 1);

    // S3: LN_att + Q|K|V
    block_load_k<D, TT>(resid, P.ln_att_g, P.ln_att_b, a.eps, 0, xs, red);
    if (a.xring) {                                  // keep NeMo's cache format alongside (slots of the oldest frames)
      float* xr = a.xring + (size_t)l * CACHE * D;
      for (int i = gtid; i < T * D; i += nthreads) xr[((head_ring + i / D) % CACHE) * D + (i % D)] = xs[i];
    }
    PSTAMP(2, 0);
    gemv_rows<D, 4, 0, TT>(P.w_qkv, 3 * D, xs, gwarp, nwarps, lane, contig,
      [&](int r) { Ops<1> o; o.v[0] = P.b_qkv[r]; return o; },
      [&](int r, float* acc, float*, const Ops<1>& o) {
        if (lane == 0) for (int t = 0; t < T; ++t) a.qkv[t * 3 * D + r] = acc[t] + o.v[0];
      });
    PSTAMP(2, 1);
    grid.sync(); STAMP(ts + 2);

    // S4: attention core
    PSTAMP(3, 0);
    for (int it = blockIdx.x; it < NH * T * S_att; it += gridDim.x) {
      const int s_ = it % S_att, ht = it / S_att;
      attention_split(a, P, kvK, kvV, head_ring, ht / T, ht % T, s_, S_att, attw, mp);
    }
    PSTAMP(3, 1);
    grid.sync(); STAMP(ts + 3);

    // S5: output projection; resid += out; commit new K/V into the oldest ring slots
    attn_combine<TT>(a, S_att, xs);
    PSTAMP(4, 0);
    gemv_rows<D, 1, 0, TT>(P.w_o, D, xs, gwarp, nwarps, lane, contig,
      [&](int r) { Ops<1 + TT> o; o.v[0] = P.b_o[r]; for (int t = 0; t < TT; ++t) o.v[1 + t] = resid[t * D + r]; return o; },
      [&](int r, float* acc, float*, const Ops<1 + TT>& o) {
        if (lane == 0) for (int t = 0; t < T; ++t) resid[t * D + r] = o.v[1 + t] + acc[t] + o.v[0];
      });
    for (int i = gtid; i < T * D; i += nthreads) {
      const int t = i / D, d = i % D, slot = (head_ring + t) % CACHE;
      kvK[slot * D + d] = a.qkv[t * 3 * D + D + d];
      kvV[slot * D + d] = a.qkv[t * 3 * D + 2 * D + d];
    }
    PSTAMP(4, 1);
    grid.sync(); STAMP(ts + 4);

    // S6: conv module part 1 (per-channel, owned by one warp): PW1 rows c and c+D, GLU, causal DW conv
    block_load_k<D, TT>(resid, P.ln_conv_g, P.ln_conv_b, a.eps, 0, xs, red);
    PSTAMP(5, 0);
    gemv_rows<D, 1, 1, TT>(P.w_pw1, D, xs, gwarp, nwarps, lane, contig,
      [&](int c) {                                    // b_pw1 (a, gate), b_dw, w_dw[KS], conv cache[KC]
        Ops<3 + KS + KC> o;
        o.v[0] = P.b_pw1[c]; o.v[1] = P.b_pw1[c + D]; o.v[2] = P.b_dw[c];
#pragma unroll
        for (int j = 0; j < KS; ++j) o.v[3 + j] = P.w_dw[c * KS + j];
#pragma unroll
        for (int j = 0; j < KC; ++j) o.v[3 + KS + j] = cc[c * KC + j];
        return o;
      },
      [&](int c, float* ga, float* gb, const Ops<3 + KS + KC>& o) {
        if (lane == 0) {
          float hist[KC + TT];
#pragma unroll
          for (int j = 0; j < KC; ++j) hist[j] = o.v[3 + KS + j];
#pragma unroll
          for (int t = 0; t < TT; ++t) hist[KC + t] = (ga[t] + o.v[0]) / (1.f + __expf(-(gb[t] + o.v[1])));   // GLU
#pragma unroll
          for (int t = 0; t < TT; ++t) {
            float acc = o.v[2];
#pragma unroll
            for (int j = 0; j < KS; ++j) acc += o.v[3 + j] * hist[t + j];
            a.ybuf[t * D + c] = acc;
          }
#pragma unroll
          for (int j = 0; j < KC; ++j) cc[c * KC + j] = hist[TT + j];   // last KC of [cache | new]
        }
      });
    PSTAMP(5, 1);
    grid.sync(); STAMP(ts + 5);

    // S7: conv module part 2: LN_cn(y) + SiLU -> PW2; resid += out
    block_load_k<D, TT>(a.ybuf, P.ln_cn_g, P.ln_cn_b, a.eps, 1, xs, red);
    PSTAMP(6, 0);
    gemv_rows<D, 1, 0, TT>(P.w_pw2, D, xs, gwarp, nwarps, lane, contig,
      [&](int r) { Ops<1 + TT> o; o.v[0] = P.b_pw2[r]; for (int t = 0; t < TT; ++t) o.v[1 + t] = resid[t * D + r]; return o; },
      [&](int r, float* acc, float*, const Ops<1 + TT>& o) {
        if (lane == 0) for (int t = 0; t < T; ++t) resid[t * D + r] = o.v[1 + t] + acc[t] + o.v[0];
      });
    PSTAMP(6, 1);
    grid.sync(); STAMP(ts + 6);

    // S8: LN_ff2 + FFN2 up + SiLU
    block_load_k<D, TT>(resid, P.ln_ff2_g, P.ln_ff2_b, a.eps, 0, xs, red);
    PSTAMP(7, 0);
    gemv_rows<D, 4, 0, TT>(P.w_ff2_1, DFF, xs, gwarp, nwarps, lane, contig,
      [&](int r) { Ops<1> o; o.v[0] = P.b_ff2_1[r]; return o; },
      [&](int r, float* acc, float*, const Ops<1>& o) {
        if (lane == 0) for (int t = 0; t < T; ++t) a.hbuf[t * DFF + r] = silu(acc[t] + o.v[0]);
      });
    PSTAMP(7, 1);
    grid.sync(); STAMP(ts + 7);

    // S9: FFN2 down; resid += fc * out
    block_load_k<DFF, TT>(a.hbuf, nullptr, nullptr, a.eps, 0, xs, red);
    PSTAMP(8, 0);
    gemv_rows<DFF, 1, 0, TT>(P.w_ff2_2, D, xs, gwarp, nwarps, lane, contig,
      [&](int r) { Ops<1 + TT> o; o.v[0] = P.b_ff2_2[r]; for (int t = 0; t < TT; ++t) o.v[1 + t] = resid[t * D + r]; return o; },
      [&](int r, float* acc, float*, const Ops<1 + TT>& o) {
        if (lane == 0) for (int t = 0; t < T; ++t) resid[t * D + r] = o.v[1 + t] + a.fc_factor * (acc[t] + o.v[0]);
      });
    PSTAMP(8, 1);
    grid.sync(); STAMP(ts + 8);

    // S10: LN_out -> next residual (ping-pong: other blocks may still be reading resid for their statistics).
    // When fused, the next layer's S1 applies it instead; only the last layer still runs it (into x_out).
    if (!fuse_ln || l + 1 == a.n_layers) {
      block_load_k<D, TT>(resid, P.ln_out_g, P.ln_out_b, a.eps, 0, xs, red);
      float* dst = (fuse_ln) ? a.x_out : resid_next;
      for (int i = gtid; i < T * D; i += nthreads) dst[i] = xs[i];
      grid.sync();
      if (!fuse_ln) { float* tmp = resid; resid = resid_next; resid_next = tmp; }
    }
    STAMP(ts + 9);
  }

  if (!fuse_ln)
    for (int i = gtid; i < T * D; i += nthreads) a.x_out[i] = resid[i];
  if (gtid == 0) *a.ring_head = (head_ring + T) % CACHE;
  STAMP(2 + a.n_layers * NSTAGE);
}

// Persistent mode (flag bit4): launched once, stays resident. Thread 0 of block 0 polls a mailbox in mapped
// (zero-copy) host memory and copies the command into device memory (ring_head[1..3]); every other block waits in
// grid.sync(), so all SMs stay busy and clocks stay up between chunks. Only that one thread ever reads host memory:
// v7 let all 21.5k threads read the mailbox over PCIe after each barrier, which cost ~1 ms per chunk.
// Host protocol: write x_in, cache_len, cmd, then seq += 1; the kernel runs the step and sets done = seq.
template <int TT>
__device__ void mk_persistent(const MKArgs& a) {
  cg::grid_group grid = cg::this_grid();
  volatile int* mb = a.mbox;                        // host memory: {seq, cmd, done, cache_len}
  volatile int* cmdv = a.ring_head + 1;             // device memory: {seq, cmd, cache_len}
  __shared__ int s_cmd[3];
  int last = 0;
  while (true) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      int sq;
      while ((sq = mb[0]) == last) { }
      __threadfence_system();
      cmdv[0] = sq; cmdv[1] = mb[1]; cmdv[2] = mb[3];
      __threadfence();
    }
    grid.sync();
    if (threadIdx.x == 0) { s_cmd[0] = cmdv[0]; s_cmd[1] = cmdv[1]; s_cmd[2] = cmdv[2]; }
    __syncthreads();
    const int seq = s_cmd[0], cmd = s_cmd[1], cl = s_cmd[2];
    if (cmd == 2) break;
    last = seq;
    mk_body<TT>(a, cl);
    grid.sync();                                    // all output and state writes done
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      __threadfence_system();
      mb[2] = seq;
    }
  }
}

template <int TT>
__device__ __forceinline__ void mk_entry(const MKArgs& a) {
  if (a.flags & 16) mk_persistent<TT>(a);
  else mk_body<TT>(a, a.cache_len);
}

extern "C" __global__ void __launch_bounds__(256) mk_kernel_t1(MKArgs a) { mk_entry<1>(a); }
extern "C" __global__ void __launch_bounds__(256) mk_kernel_t2(MKArgs a) { mk_entry<2>(a); }
extern "C" __global__ void __launch_bounds__(256) mk_kernel_t3(MKArgs a) { mk_entry<3>(a); }
extern "C" __global__ void __launch_bounds__(256) mk_kernel_t4(MKArgs a) { mk_entry<4>(a); }

static const void* kernel_for(int T) {
  switch (T) {
    case 1: return (const void*)mk_kernel_t1;
    case 2: return (const void*)mk_kernel_t2;
    case 3: return (const void*)mk_kernel_t3;
    default: return (const void*)mk_kernel_t4;
  }
}

// ------------------------------------------------------------------------------------------- host API (ctypes)
extern "C" int mk_smem_bytes(int threads) { (void)threads; return (TMAX * DFF + 256 + 1024 + TMAX * NKEY_PAD) * (int)sizeof(float); }

extern "C" int mk_prepare(int threads) {
  const int smem = mk_smem_bytes(threads);
  for (int t = 1; t <= TMAX; ++t) {
    const int e = (int)cudaFuncSetAttribute(kernel_for(t), cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e) return e;
  }
  return 0;
}

extern "C" int mk_max_blocks_per_sm(int threads, int T, int* out) {
  return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, kernel_for(T), threads, mk_smem_bytes(threads));
}

extern "C" int mk_launch(const MKArgs* args, int blocks, int threads, void* stream) {
  MKArgs local = *args;
  void* kargs[] = {(void*)&local};
  return (int)cudaLaunchCooperativeKernel(kernel_for(local.T), dim3(blocks), dim3(threads), kargs,
                                          (size_t)mk_smem_bytes(threads), (cudaStream_t)stream);
}

extern "C" const char* mk_error_string(int e) { return cudaGetErrorString((cudaError_t)e); }

// Zero-copy host buffers (pinned + mapped) for inputs, outputs and the persistent-mode mailbox.
extern "C" int mk_host_alloc(size_t bytes, void** hptr, void** dptr) {
  cudaError_t e = cudaHostAlloc(hptr, bytes, cudaHostAllocMapped);
  if (e != cudaSuccess) return (int)e;
  memset(*hptr, 0, bytes);
  return (int)cudaHostGetDevicePointer(dptr, *hptr, 0);
}
extern "C" int mk_host_free(void* hptr) { return (int)cudaFreeHost(hptr); }
extern "C" int mk_stream_create(void** s) { return (int)cudaStreamCreateWithFlags((cudaStream_t*)s, cudaStreamNonBlocking); }
extern "C" int mk_stream_sync(void* s) { return (int)cudaStreamSynchronize((cudaStream_t)s); }
extern "C" int mk_stream_query(void* s) { return (int)cudaStreamQuery((cudaStream_t)s); }

// Persistent-mode mailbox helpers (host side, C so the stores are ordered and the spin is cheap).
extern "C" void mk_post(float* dst, const float* src, int n, int* mb, int seq, int cache_len, int cmd) {
  if (n > 0) memcpy(dst, src, (size_t)n * sizeof(float));
  mb[3] = cache_len;
  mb[1] = cmd;
  __sync_synchronize();                             // input + fields visible before seq
  ((volatile int*)mb)[0] = seq;
}
extern "C" int mk_wait(int* mb, int seq, long long spin_limit) {
  volatile int* v = mb;
  long long i = 0;
  while (v[2] != seq) { if (++i > spin_limit) return -1; }
  __sync_synchronize();
  return 0;
}

extern "C" int mk_args_size() { return (int)sizeof(MKArgs); }
extern "C" int mk_layer_params_size() { return (int)sizeof(LayerParams); }
