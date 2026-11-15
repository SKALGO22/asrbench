# megakernel/ — the 17 Conformer layers of a streaming step in one GPU launch

## Status log

| Version | Change | RTX 6000 Ada result |
| --- | --- | --- |
| v1 | one cooperative launch, 10 grid barriers per layer | correct (FP32 rel.err 6.6e-4, FP16 3.5e-3); layers 2.49 ms, step 2.61 ms vs NeMo CUDA Graph 1.82 ms. Per-stage: attention 95.7 µs/layer = 65% of the time (16 warps, uncoalesced per-lane dot products); other stages 3.3–6.9 µs, matrix stages 260–330 GB/s |
| v2 | attention: one block per (head, query), lanes along head dims (coalesced), warps over keys; single-pass LayerNorm statistics (2 barriers per LayerNorm instead of 4T) | expected: attention ~5 µs/layer, layers ≈ 0.9–1.0 ms |

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
| v2 | block-parallel attention, single-pass LayerNorm (this folder) | per-stage table again: what dominates once attention is fixed? |
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
