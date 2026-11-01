// Persistent-kernel feasibility microbenchmarks (the "why should it work" numbers for the megakernel).
//
//  1. launch      : host launch + sync round trip of an empty kernel (1 block and full grid)
//  2. grid_sync   : cost of cooperative-groups grid.sync() with one block per SM
//  3. counter     : cost of a counter-based all-blocks barrier (atomic arrive + acquire spin, monotone target)
//  4. mailbox     : host -> resident kernel -> host round trip through a flag in mapped pinned memory,
//                   with 1 block, and with the full grid (poller fans out + grid sync before "done")
//
// Build:  nvcc -O3 -std=c++17 -arch=native persistent_microbench.cu -o pmb      (CUDA >= 11.8)
//         (older nvcc without -arch=native: -gencode arch=compute_86,code=sm_86 etc.)
// Run:    ./pmb > results/<gpu>/persistent_microbench.json
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CK(x)                                                                                     \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) {                                                                      \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);    \
      exit(1);                                                                                    \
    }                                                                                             \
  } while (0)

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void k_empty() {}

__global__ void k_gridsync(int iters, unsigned long long* out) {
  cg::grid_group g = cg::this_grid();
  unsigned long long t0 = 0;
  g.sync();
  if (g.thread_rank() == 0) t0 = gtimer();
  for (int i = 0; i < iters; ++i) g.sync();
  if (g.thread_rank() == 0) out[0] = gtimer() - t0;
}

__global__ void k_counter(int iters, unsigned* counter, unsigned long long* out) {
  const unsigned nb = gridDim.x;
  unsigned long long t0 = 0;
  if (blockIdx.x == 0 && threadIdx.x == 0) t0 = gtimer();
  for (int i = 1; i <= iters; ++i) {
    __syncthreads();
    if (threadIdx.x == 0) {
      cuda::atomic_ref<unsigned, cuda::thread_scope_device> c(*counter);
      c.fetch_add(1u, cuda::memory_order_release);
      while (c.load(cuda::memory_order_acquire) < (unsigned)i * nb) {
      }
    }
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = gtimer() - t0;
}

struct Mailbox {
  unsigned seq;
  unsigned pad0[31];
  unsigned done;
  unsigned pad1[31];
};

// Resident kernel: block 0 / thread 0 polls the host flag, fans out through a device flag, all blocks
// grid-sync (stand-in for "run the encoder"), then block 0 publishes `done` to the host.
__global__ void k_mailbox(Mailbox* mb, unsigned* go, int iters, int use_grid_sync) {
  unsigned last = 0;
  for (int i = 0; i < iters; ++i) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      cuda::atomic_ref<unsigned, cuda::thread_scope_system> s(mb->seq);
      unsigned v;
      while ((v = s.load(cuda::memory_order_acquire)) == last) {
      }
      cuda::atomic_ref<unsigned, cuda::thread_scope_device> g(*go);
      g.store(v, cuda::memory_order_release);
    }
    if (threadIdx.x == 0) {
      cuda::atomic_ref<unsigned, cuda::thread_scope_device> g(*go);
      while (g.load(cuda::memory_order_acquire) == last) {
      }
    }
    __syncthreads();
    const unsigned v = last + 1;
    if (use_grid_sync) cg::this_grid().sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      __threadfence_system();
      cuda::atomic_ref<unsigned, cuda::thread_scope_system> d(mb->done);
      d.store(v, cuda::memory_order_release);
    }
    last = v;
  }
}

static double pct(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[(size_t)((v.size() - 1) * q)];
}

static void stats_json(const char* name, const std::vector<double>& v, bool comma = true) {
  printf("  \"%s\": {\"p50_us\": %.3f, \"p99_us\": %.3f, \"min_us\": %.3f, \"n\": %zu}%s\n", name, pct(v, 0.5),
         pct(v, 0.99), pct(v, 0.0), v.size(), comma ? "," : "");
}

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 2000;
  cudaSetDeviceFlags(cudaDeviceMapHost);  // no-op/ignored if a context exists; UVA maps pinned memory anyway
  cudaGetLastError();
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  int coop = 0;
  CK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
  const int nsm = p.multiProcessorCount;
  const int threads = 256;
  printf("{\n  \"gpu\": \"%s\", \"sm_count\": %d, \"cooperative_launch\": %d,\n", p.name, nsm, coop);

  // 1. launch + sync round trips
  for (int nb : {1, nsm}) {
    std::vector<double> t;
    for (int i = 0; i < iters + 50; ++i) {
      auto a = clk::now();
      k_empty<<<nb, threads>>>();
      CK(cudaDeviceSynchronize());
      auto b = clk::now();
      if (i >= 50) t.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    }
    stats_json(nb == 1 ? "launch_sync_1block" : "launch_sync_fullgrid", t);
  }
  if (!coop) {
    printf("  \"error\": \"cooperative launch not supported\"\n}\n");
    return 0;
  }

  unsigned long long* d_out;
  unsigned* d_counter;
  CK(cudaMalloc(&d_out, sizeof(unsigned long long)));
  CK(cudaMalloc(&d_counter, sizeof(unsigned)));

  // 2. grid.sync cost
  {
    int n = iters;
    void* args[] = {&n, &d_out};
    CK(cudaLaunchCooperativeKernel((void*)k_gridsync, nsm, threads, args, 0, 0));
    CK(cudaDeviceSynchronize());
    unsigned long long ns;
    CK(cudaMemcpy(&ns, d_out, sizeof(ns), cudaMemcpyDeviceToHost));
    printf("  \"grid_sync_us\": %.3f,\n", ns / 1e3 / iters);
  }
  // 3. counter barrier cost
  {
    CK(cudaMemset(d_counter, 0, sizeof(unsigned)));
    int n = iters;
    void* args[] = {&n, &d_counter, &d_out};
    CK(cudaLaunchCooperativeKernel((void*)k_counter, nsm, threads, args, 0, 0));
    CK(cudaDeviceSynchronize());
    unsigned long long ns;
    CK(cudaMemcpy(&ns, d_out, sizeof(ns), cudaMemcpyDeviceToHost));
    printf("  \"counter_barrier_us\": %.3f,\n", ns / 1e3 / iters);
  }
  // 4. mailbox round trips
  Mailbox* h_mb;
  CK(cudaHostAlloc((void**)&h_mb, sizeof(Mailbox), cudaHostAllocMapped));
  Mailbox* d_mb;
  CK(cudaHostGetDevicePointer((void**)&d_mb, h_mb, 0));
  unsigned* d_go;
  CK(cudaMalloc(&d_go, sizeof(unsigned)));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  struct Cfg { const char* name; int blocks; int gsync; };
  Cfg cfgs[] = {{"mailbox_roundtrip_1block", 1, 0}, {"mailbox_roundtrip_fullgrid", nsm, 1}};
  for (int ci = 0; ci < 2; ++ci) {
    const Cfg& c = cfgs[ci];
    __atomic_store_n(&h_mb->seq, 0u, __ATOMIC_SEQ_CST);
    __atomic_store_n(&h_mb->done, 0u, __ATOMIC_SEQ_CST);
    CK(cudaMemset(d_go, 0, sizeof(unsigned)));
    CK(cudaDeviceSynchronize());
    int n = iters + 50, gs = c.gsync;
    void* args[] = {&d_mb, &d_go, &n, &gs};
    CK(cudaLaunchCooperativeKernel((void*)k_mailbox, c.blocks, threads, args, 0, st));
    std::vector<double> t;
    for (unsigned i = 1; i <= (unsigned)n; ++i) {
      auto a = clk::now();
      __atomic_store_n(&h_mb->seq, i, __ATOMIC_RELEASE);
      while (__atomic_load_n(&h_mb->done, __ATOMIC_ACQUIRE) != i) {
      }
      auto b = clk::now();
      if (i > 50) t.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    }
    CK(cudaStreamSynchronize(st));
    stats_json(c.name, t, ci == 0);
  }
  printf("}\n");
  return 0;
}
