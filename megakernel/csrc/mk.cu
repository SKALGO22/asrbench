// Megakernel v3 (v2 + L2 prefetch of the next layer's weights, LayerNorm-out fused into the next layer, faster attention): the 17 Conformer layers of one cache-aware streaming step in ONE cooperative launch.
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
  int* ring_head;             // [1]
  const uint8_t* mask;        // [T, CACHE+T]  1 = masked (NeMo semantics)
  const int* pidx;            // [T, CACHE+T]  index into p, -1 = zero (rel_shift padding)
  float* resid0;              // [T, D]
  float* resid1;              // [T, D]
  float* hbuf;                // [T, DFF]
  float* qkv;                 // [T, 3D]
  float* att;                 // [T, D]
  float* ybuf;                // [T, D]
  unsigned long long* timers; // [n_layers*NSTAGE + 3] or nullptr (block 0 records %globaltimer)
  int n_layers;
  int T;
  int npos;
  int flags;                  // bit0: L2-prefetch next layer's weights, bit1: fuse LN_out into the next layer's S1
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

// Copy src[T][K] into shared memory; optionally LayerNorm each row (g != nullptr) and apply SiLU (act == 1).
// Single-pass statistics (sum and sum of squares for all T rows reduced together): 2 block barriers per LayerNorm.
__device__ void block_load(const float* __restrict__ src, int T, int K, const float* g, const float* b,
                           float eps, int act, float* xs, float* red) {
  float v[2 * TMAX];
#pragma unroll
  for (int i = 0; i < 2 * TMAX; ++i) v[i] = 0.f;
  for (int t = 0; t < T; ++t)
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      const float x = src[t * K + k];
      xs[t * K + k] = x;
      v[2 * t] += x;
      v[2 * t + 1] += x * x;
    }
  if (g == nullptr) { __syncthreads(); return; }
  block_reduce_arr(v, 2 * T, red);
  for (int t = 0; t < T; ++t) {
    const float mean = v[2 * t] / K;
    const float rstd = rsqrtf(fmaxf(v[2 * t + 1] / K - mean * mean, 0.f) + eps);
    float* row = xs + t * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {      // each thread normalises the elements it loaded
      float y = (row[k] - mean) * rstd * g[k] + b[k];
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

#define STAMP(idx) do { if (a.timers && blockIdx.x == 0 && threadIdx.x == 0) a.timers[(idx)] = gtimer(); } while (0)

extern "C" __global__ void __launch_bounds__(512) mk_kernel(MKArgs a) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ float smem[];
  float* xs = smem;                           // TMAX*DFF floats
  float* red = xs + TMAX * DFF;               // 2*TMAX*32 = 256 floats
  float* attw = red + 256;                    // 1024 floats attention scratch
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wpb = blockDim.x >> 5;
  const int gwarp = blockIdx.x * wpb + warp, nwarps = gridDim.x * wpb;
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x, nthreads = gridDim.x * blockDim.x;
  const int T = a.T;
  const int head_ring = *a.ring_head;
  float acc[TMAX];

  STAMP(0);
  const bool do_prefetch = (a.flags & 1) != 0, fuse_ln = (a.flags & 2) != 0;
  if (do_prefetch) prefetch_layer(a.layers[0], gtid, nthreads);
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

    if (do_prefetch && l + 1 < a.n_layers) prefetch_layer(a.layers[l + 1], gtid, nthreads);

    // S1: [LN_out of the previous layer, if fused] + LN_ff1 + FFN1 up + SiLU
    if (fuse_ln && l > 0) {
      block_load(resid, T, D, a.layers[l - 1].ln_out_g, a.layers[l - 1].ln_out_b, a.eps, 0, xs, red);
      for (int i = gtid; i < T * D; i += nthreads) resid_next[i] = xs[i];   // this layer's residual input
      block_ln_inplace(xs, T, D, P.ln_ff1_g, P.ln_ff1_b, a.eps, 0, red);
      float* tmp = resid; resid = resid_next; resid_next = tmp;              // S2.. read/update the new residual
    } else {
      block_load(resid, T, D, P.ln_ff1_g, P.ln_ff1_b, a.eps, 0, xs, red);
    }
    for (int r = gwarp; r < DFF; r += nwarps) {
      row_dot(P.w_ff1_1 + (size_t)r * D, D, xs, T, lane, acc);
      if (lane == 0) { const float bb = P.b_ff1_1[r]; for (int t = 0; t < T; ++t) a.hbuf[t * DFF + r] = silu(acc[t] + bb); }
    }
    grid.sync(); STAMP(ts + 0);

    // S2: FFN1 down; resid += fc * out
    block_load(a.hbuf, T, DFF, nullptr, nullptr, a.eps, 0, xs, red);
    for (int r = gwarp; r < D; r += nwarps) {
      row_dot(P.w_ff1_2 + (size_t)r * DFF, DFF, xs, T, lane, acc);
      if (lane == 0) { const float bb = P.b_ff1_2[r]; for (int t = 0; t < T; ++t) resid[t * D + r] += a.fc_factor * (acc[t] + bb); }
    }
    grid.sync(); STAMP(ts + 1);

    // S3: LN_att + Q|K|V
    block_load(resid, T, D, P.ln_att_g, P.ln_att_b, a.eps, 0, xs, red);
    for (int r = gwarp; r < 3 * D; r += nwarps) {
      row_dot(P.w_qkv + (size_t)r * D, D, xs, T, lane, acc);
      if (lane == 0) { const float bb = P.b_qkv[r]; for (int t = 0; t < T; ++t) a.qkv[t * 3 * D + r] = acc[t] + bb; }
    }
    grid.sync(); STAMP(ts + 2);

    // S4: attention core
    for (int it = blockIdx.x; it < NH * T; it += gridDim.x)
      attention_block(a, P, kvK, kvV, head_ring, it / T, it % T, attw);
    grid.sync(); STAMP(ts + 3);

    // S5: output projection; resid += out; commit new K/V into the oldest ring slots
    block_load(a.att, T, D, nullptr, nullptr, a.eps, 0, xs, red);
    for (int r = gwarp; r < D; r += nwarps) {
      row_dot(P.w_o + (size_t)r * D, D, xs, T, lane, acc);
      if (lane == 0) { const float bb = P.b_o[r]; for (int t = 0; t < T; ++t) resid[t * D + r] += acc[t] + bb; }
    }
    for (int i = gtid; i < T * D; i += nthreads) {
      const int t = i / D, d = i % D, slot = (head_ring + t) % CACHE;
      kvK[slot * D + d] = a.qkv[t * 3 * D + D + d];
      kvV[slot * D + d] = a.qkv[t * 3 * D + 2 * D + d];
    }
    grid.sync(); STAMP(ts + 4);

    // S6: conv module part 1 (per-channel, owned by one warp): PW1 rows c and c+D, GLU, causal DW conv
    block_load(resid, T, D, P.ln_conv_g, P.ln_conv_b, a.eps, 0, xs, red);
    for (int c = gwarp; c < D; c += nwarps) {
      float ga[TMAX], gb[TMAX];
      row_dot(P.w_pw1 + (size_t)c * D, D, xs, T, lane, ga);
      row_dot(P.w_pw1 + (size_t)(c + D) * D, D, xs, T, lane, gb);
      if (lane == 0) {
        float hist[KC + TMAX];
        for (int j = 0; j < KC; ++j) hist[j] = cc[c * KC + j];
        for (int t = 0; t < T; ++t) {
          const float av = ga[t] + P.b_pw1[c], gv = gb[t] + P.b_pw1[c + D];
          hist[KC + t] = av / (1.f + __expf(-gv));                     // GLU: a * sigmoid(b)
        }
        for (int t = 0; t < T; ++t) {
          float s = P.b_dw[c];
          for (int j = 0; j < KS; ++j) s += P.w_dw[c * KS + j] * hist[t + j];
          a.ybuf[t * D + c] = s;
        }
        for (int j = 0; j < KC; ++j) cc[c * KC + j] = hist[T + j];      // last KC of [cache | new]
      }
    }
    grid.sync(); STAMP(ts + 5);

    // S7: conv module part 2: LN_cn(y) + SiLU -> PW2; resid += out
    block_load(a.ybuf, T, D, P.ln_cn_g, P.ln_cn_b, a.eps, 1, xs, red);
    for (int r = gwarp; r < D; r += nwarps) {
      row_dot(P.w_pw2 + (size_t)r * D, D, xs, T, lane, acc);
      if (lane == 0) { const float bb = P.b_pw2[r]; for (int t = 0; t < T; ++t) resid[t * D + r] += acc[t] + bb; }
    }
    grid.sync(); STAMP(ts + 6);

    // S8: LN_ff2 + FFN2 up + SiLU
    block_load(resid, T, D, P.ln_ff2_g, P.ln_ff2_b, a.eps, 0, xs, red);
    for (int r = gwarp; r < DFF; r += nwarps) {
      row_dot(P.w_ff2_1 + (size_t)r * D, D, xs, T, lane, acc);
      if (lane == 0) { const float bb = P.b_ff2_1[r]; for (int t = 0; t < T; ++t) a.hbuf[t * DFF + r] = silu(acc[t] + bb); }
    }
    grid.sync(); STAMP(ts + 7);

    // S9: FFN2 down; resid += fc * out
    block_load(a.hbuf, T, DFF, nullptr, nullptr, a.eps, 0, xs, red);
    for (int r = gwarp; r < D; r += nwarps) {
      row_dot(P.w_ff2_2 + (size_t)r * DFF, DFF, xs, T, lane, acc);
      if (lane == 0) { const float bb = P.b_ff2_2[r]; for (int t = 0; t < T; ++t) resid[t * D + r] += a.fc_factor * (acc[t] + bb); }
    }
    grid.sync(); STAMP(ts + 8);

    // S10: LN_out -> next residual (ping-pong: other blocks may still be reading resid for their statistics).
    // When fused, the next layer's S1 applies it instead; only the last layer still runs it (into x_out).
    if (!fuse_ln || l + 1 == a.n_layers) {
      block_load(resid, T, D, P.ln_out_g, P.ln_out_b, a.eps, 0, xs, red);
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

// ------------------------------------------------------------------------------------------- host API (ctypes)
extern "C" int mk_smem_bytes(int threads) { (void)threads; return (TMAX * DFF + 256 + 1024) * (int)sizeof(float); }

extern "C" int mk_prepare(int threads) {
  const int smem = mk_smem_bytes(threads);
  return (int)cudaFuncSetAttribute(mk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
}

extern "C" int mk_max_blocks_per_sm(int threads, int* out) {
  return (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(out, mk_kernel, threads, mk_smem_bytes(threads));
}

extern "C" int mk_launch(const MKArgs* args, int blocks, int threads, void* stream) {
  MKArgs local = *args;
  void* kargs[] = {(void*)&local};
  return (int)cudaLaunchCooperativeKernel((void*)mk_kernel, dim3(blocks), dim3(threads), kargs,
                                          (size_t)mk_smem_bytes(threads), (cudaStream_t)stream);
}

extern "C" const char* mk_error_string(int e) { return cudaGetErrorString((cudaError_t)e); }

extern "C" int mk_args_size() { return (int)sizeof(MKArgs); }
extern "C" int mk_layer_params_size() { return (int)sizeof(LayerParams); }
