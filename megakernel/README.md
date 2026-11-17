# megakernel/ — the 17 Conformer layers of a streaming step in one GPU launch

## Status log

| Version | Change | RTX 6000 Ada result |
| --- | --- | --- |
| v1 | one cooperative launch, 10 grid barriers per layer | correct (FP32 rel.err 6.6e-4, FP16 3.5e-3); layers 2.49 ms, step 2.61 ms vs NeMo CUDA Graph 1.82 ms. Per-stage: attention 95.7 µs/layer = 65% of the time (16 warps, uncoalesced per-lane dot products); other stages 3.3–6.9 µs, matrix stages 260–330 GB/s |
| v2 | attention: one block per (head, query), lanes along head dims (coalesced), warps over keys; single-pass LayerNorm statistics | correct (same errors); layers 1.13 ms, step 1.24 ms vs NeMo CUDA Graph 1.82 ms (1.47×). Attention 95.7 → 14.3 µs/layer. A weight-free stage (S10) still costs 3.6 µs: fixed per-stage cost ≈ half of all time; matrix stages 250–315 GB/s |
| v3 | flags: bit0 = L2 prefetch of the next layer's 12.6 MB at each layer start; bit1 = LN_out fused into the next layer's S1 (9 barriers/layer); attention 4 keys per warp per round, float4 | correct. Ablation (layers): none 1.124 ms, +L2 prefetch 1.202 (worse: issuing ~100k prefetches costs ~9 µs per layer in S1 and the matrix stages did not speed up, so they are not DRAM-bound), +fused LN_out 1.097 (best), both 1.176. Attention 14.3 → 10.2 µs. Approach B on the same GPU: 1.565 ms; NeMo CUDA Graph 1.828 ms |
| v4 | bit2 = register prefetch: before every barrier each warp loads the weight rows it needs in the NEXT stage into registers | RTX A6000 (84 SMs): correct; layers flags0 1.507 ms, +L2pf 1.559, +fused 1.488 (best), +regpf 1.730 (worse), fused+regpf 1.717. Best step ≈1.64 ms vs NeMo CUDA Graph 2.30 ms (1.40x) and B 1.99 ms (1.21x). Default flags back to 2 |
| v4.1 | per-stage phase timers (block 0): input load / compute / barrier | A6000, flags 2, per layer ≈ load 30 µs, compute 50 µs, barrier 9 µs (~1 µs each; grid.sync alone 0.64 µs). Loading 4–16 KB of activations took 3–7 µs and a 4 KB-row GEMV 5 µs: both are serialized round trips (one load in flight at a time), not bandwidth. Register prefetch (flag 4) shortened compute but grew the barrier phase, so the fence in grid.sync waits for the prefetch loads |
| v5 | activation loads: 16-byte, all issued before use (one L2 round trip); GEMV: each warp computes up to 4 rows (K=512) or 1 row (K=2048) with every weight load issued before the first FMA. Register prefetch retired. 256 threads/block, 250 registers, no spills | expected: load ≈ 1 µs and compute ≈ 2–3 µs per stage, layers ≈ 0.8–1.0 ms on the A6000 |

v1 (this folder) runs all 17 layers of `parakeet_realtime_eou_120m` for one 160 ms chunk in **one cooperative
launch**: 142 thread blocks (one per SM on an L40S) stay alive for the whole chunk and hand work to each other through
grid-wide barriers instead of kernel boundaries. Pre-encode (subsampling) stays in NeMo, captured in a CUDA Graph.

## What is verified

* `reference_layers()` (the exact stage algorithm of `csrc/mk.cu`, written in PyTorch) matches NeMo's own streaming
  step to **rel. err 3e-7** in FP32 over 30+ chunks, also from an almost empty cache (masks per cache length), and the
  caches match too. This validates weight packing, the projected-K/V ring, the relative-position index table taken
  from NeMo's `rel_shift`, the masks, the conv cache, and the pre-encode path.
* With FP16 weights (what the kernel streams) the same algorithm is within **rel. err ~1.5e-4** of FP32 NeMo.
* `csrc/mk.cu` device code compiles for sm_86 and sm_89 (NVRTC). **The kernel has not run on a GPU yet**: run
  `mk_test.py` first.

## Files

| File | Purpose |
| --- | --- |
| `csrc/mk.cu` | the megakernel + a tiny C API (`mk_launch`, occupancy query) used through ctypes |
| `mk_runtime.py` | build (nvcc), weight packing, constants captured from NeMo, device state, PyTorch reference, `MegakernelEncoder` |
| `mk_test.py` | parity vs NeMo (whole encoder, or the first N layers for debugging) |
| `mk_bench.py` | latency (layers only, full step, NeMo CUDA Graph baseline), per-stage breakdown, grid sweep, paced mode |

## Commands (on a GPU node, from the repo root)

```bash
source venv/bin/activate
export PATH=~/cuda126/bin:$PATH            # the nvcc you installed with micromamba
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HF_HUB_OFFLINE=1
python megakernel/mk_test.py --reference     # sanity: PyTorch reference on GPU, no kernel
python megakernel/mk_test.py --fp32          # kernel vs FP32 NeMo, tolerance 1e-3 (expect ~1e-4)
python megakernel/mk_test.py                 # kernel vs FP16 NeMo, tolerance 2e-2 (expect ~7e-3, NeMo's own FP16 error)
python megakernel/mk_test.py --fp32 --layers 1   # if the full run fails: find the first diverging layer
python megakernel/mk_bench.py --stages --paced   # latency, per-stage breakdown, real-time mode
python megakernel/mk_bench.py --sweep            # threads x blocks-per-SM
python megakernel/mk_bench.py --ablation         # flag combinations: L2 prefetch, fused LN_out, register prefetch
```

The library is rebuilt automatically when `csrc/mk.cu` changes (`build/libmk.so`). Override the target with
`MK_ARCH="-gencode=arch=compute_89,code=sm_89"` and the compiler with `NVCC=...`.

## Design (v1)

Per layer, 10 stages separated by `grid.sync()` (170 per chunk):

| Stage | Work | Weights |
| --- | --- | --- |
| S1 | LayerNorm + FFN1 up (512→2048) + SiLU | 2 MB |
| S2 | FFN1 down (2048→512), residual += ½·out | 2 MB |
| S3 | LayerNorm + fused Q·K·V (512→1536) | 1.5 MB |
| S4 | attention: one warp per (head, query) over 70 cached + 2 new keys, rel-pos term from a precomputed table | – |
| S5 | output projection, residual += out, commit new K/V into the ring | 0.5 MB |
| S6 | LayerNorm + PW1 rows c, c+512 + GLU + causal depthwise conv (warp owns channel c and its cache) | 1 MB |
| S7 | LayerNorm + SiLU + PW2, residual += out | 0.5 MB |
| S8–S9 | FFN2 like S1–S2 | 4 MB |
| S10 | final LayerNorm → next layer's residual (ping-pong buffer) | – |

* Every block recomputes LayerNorm statistics itself (cheap) so no extra barrier is needed.
* Matrix rows are spread over all warps of the grid; each warp streams its rows with 16-byte loads and reuses each
  weight for all T = 2 frames; FP32 accumulation.
* Attention caches **projected** K/V (experiment A, exact) in a ring of 70 slots; `linear_pos(pos_emb)` is precomputed
  per layer; NeMo's `rel_shift` is reproduced with an index table built by running NeMo's own function.
* State (K/V ring, conv cache, ring head) lives on the GPU; the host only launches.

## Roadmap and the experiments that go with each step

| Step | Change | Experiment / question |
| --- | --- | --- |
| v1 | one launch, `grid.sync` per stage | parity; per-stage time and GB/s (done: attention dominated) |
| v2 | block-parallel attention, single-pass LayerNorm | done: fixed per-stage cost and sub-peak bandwidth dominate |
| v3 | L2 prefetch of next layer + fused LN_out + faster attention | done: stages are latency-chain bound, not bandwidth bound |
| v4 | register prefetch of the next stage's rows | done: no (fence waits for it) |
| v5 | one-round-trip loads inside each stage (this folder) | phase table: are load/compute now ~1 round trip each? |
| v2 | fewer barriers: fuse S10 into the next S1, S5's commit into S6, LN stats reuse | latency vs number of barriers (170 → ~120) |
| v3 | counters instead of `grid.sync`: a warp waits only for the rows it needs (e.g. S2 rows wait for S1) | barrier cost vs dependency-wait cost |
| v4 | prefetch next stage's weights into shared memory with `cp.async` during the barrier | achieved bandwidth per stage |
| v5 | persistent: stay resident between chunks, mailbox in pinned memory | paced p99, host CPU time, idle power |
| v6 | INT8 / INT4 weights pinned in L2 (`cudaAccessPolicyWindow`) | latency vs bytes; WER |
| — | grid size sweep (blocks per SM, threads per block) | occupancy vs per-warp memory parallelism |

## Known limits of v1

* d_model 512, d_ff 2048, 8 heads, conv kernel 9, attention cache 70, T ≤ 4 frames per chunk (the 120M model).
* Steady-shape chunks only (all chunks after the first). Masks are captured for every cache length that steady
  chunks produce; the first chunk of an utterance and a short last chunk should run through NeMo eager (planned for
  the WER integration, like `05_wer.py`'s CUDA-Graph runner).
* Batch 1.
