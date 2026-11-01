# asrbench — baselines and preliminary results for persistent megakernels in streaming ASR

Benchmarks one **steady-state streaming step** of NVIDIA's cache-aware FastConformer encoders
(`parakeet_realtime_eou_120m-v1`, plus `nemotron-speech-streaming-en-0.6b` for chunk-size scaling) under every
reasonable existing execution strategy, measures where the time goes, checks accuracy, and evaluates two
preliminary optimizations. Output is one JSONL file per experiment per GPU, aggregated into `results/SUMMARY.md`
and slide-ready plots.

## What is verified vs untested

Validated on CPU (no GPU was available while writing this):
* model geometry and streaming API (NeMo 3.0): 17 layers, d = 512, 109.5 M encoder params, steady-state step =
  160 ms chunk = 25 mel frames in (16 new + 9 pre-encode cache) → 2 encoder frames out, attention cache 70 frames;
* **prelim A** (cache projected K/V + precomputed positional projection) matches NeMo to rel. err 1.4e-6 over
  40 chunks at batch 2 (31 % less CPU time per step — not a GPU number);
* **prelim B** (Triton fused conv block, 2 kernels) matches NeMo to rel. err ≈ 5e-7 for B ∈ {1,2,3}, T ∈ {1,2,5},
  with/without padding (Triton interpreter);
* prelim A + B together inside the NeMo encoder (3-layer truncation for speed, 6 chunks, batch 2): rel. err 2.6e-7;
* the latency harness end-to-end (`02_latency.py --cpu-smoke`, eager with and without patch A, parity path);
* true streaming WER (`05_wer.py --cpu-smoke`): 0.79 % on 4 test-clean utterances, identical transcripts with
  patch A. (Needed fix: the EOU model ships with alignments/timestamps on, which breaks partial hypotheses in
  NeMo 3.0 streaming; the script switches decoding to plain greedy_batch);
* ONNX export of the cache-aware step: ORT-CPU matches PyTorch to 2.2e-6; the summarizer on synthetic results;
* the persistent microbenchmark's device code compiles (NVRTC, sm_75/86/89).

**Not yet run on a GPU:** CUDA-graph capture, `torch.compile`, ORT-CUDA/TensorRT IOBinding, CUDA profiler parsing,
NVML. Expect small fixes on the first run; every script records failures as results instead of crashing.

## Setup and running

**See `RUNBOOK.md` for every command** (laptop RTX 3050, IIITH 4x GTX 1080 Ti node, JarvisLabs L4, merging).
Short version:

```bash
bash setup_env.sh && source venv/bin/activate      # torch 2.14.0+cu126 everywhere (last release supporting Pascal)
python scripts/00_fetch.py && bash data/get_librispeech.sh
bash jobs/run_quick.sh                             # 15-min smoke test
bash jobs/run_full.sh                              # full suite on one GPU (resumable)
bash jobs/run_node4.sh                             # 4 GPUs of one node in parallel
python scripts/merge_results.py --into results results_*   &&   python scripts/summarize.py
```

On compute capability < 7.0 (GTX 1080 Ti) `torch.compile`, Triton (prelim B) and BF16 are skipped automatically,
and FP32 is used as the main precision (Pascal GeForce runs FP16 arithmetic at 1/64 rate).

## Experiments

| Script | Question it answers | Output |
| --- | --- | --- |
| `01_micro.py` | What are this GPU's real DRAM/L2 bandwidth, eager per-op cost, CUDA-graph per-kernel cost, and how efficient is a single tiny GEMV at our layer shapes? | `micro.json` |
| `micro/persistent_microbench.cu` | How cheap are the megakernel's primitives: grid.sync, counter barrier, host→resident-kernel mailbox round trip vs launch+sync? | `persistent_microbench.json` |
| `02_latency.py` | **Main baselines.** Per-chunk latency (wall / GPU / host, p50–p99.9, back-to-back and paced at real time), effective weight GB/s, energy, parity vs fp32. Implementations: eager, CUDA Graph, `torch.compile` default / reduce-overhead / max-autotune(±cudagraphs) × fp32 / amp16 / fp16 / bf16 × chunk sizes × batch × patches | `latency.jsonl` |
| `03_profile.py` | Where does a chunk's time go? Kernels and aten ops per chunk, GPU-busy fraction, idle gap per kernel, time by kernel category, Perfetto traces | `profile.jsonl`, `traces/` |
| `04_onnx_trt.py` | Vendor stack: ONNX Runtime CUDA EP (±CUDA graphs, fp32/fp16), TensorRT EP (fp32/fp16, ±CUDA graphs) | `onnx_trt.jsonl` |
| `05_wer.py` | Does every implementation give the same transcripts? True streaming WER on LibriSpeech per mode | `wer.jsonl` |
| `06_prelim_conv.py` | Prelim B in isolation: NeMo conv block (eager/compile/graph) vs fused Triton (eager/graph), kernels per call, tile sweep | `prelim_conv.jsonl` |
| `summarize.py` | Tables + plots for the presentation | `SUMMARY.md`, `plots/` |

Presets of `02_latency.py`: `quick` (smoke), `full` (all impls × precisions, 120M), `chunks` (600M, 4 chunk sizes),
`batch` (B = 1…128 for streams-per-GPU), `prelim` (patches none/A/B/AB × eager/graph).

## Measurement methodology (say this in the talk)

* One step = what a streaming server runs per chunk: encoder forward with caches, synchronized at the end.
* Steady state: caches are filled (40 warm steps) before timing; compile/capture time is reported separately.
* Three clocks: **wall** (call → GPU done), **GPU** (CUDA events), **host** (time until the Python call returns:
  dispatch cost). Wall ≫ GPU means the CPU is the bottleneck.
* **Paced** runs start chunks on a real-time schedule (160 ms) so the GPU idles between chunks like with live audio.
* **Effective GB/s** = encoder weight bytes ÷ GPU time, compared with the DRAM peak measured by `01_micro.py`.
* **Parity**: every implementation is run on the same 30-chunk input stream from zero caches and compared with fp32
  eager (max relative error), and WER is measured on LibriSpeech.
* Precision notes: `amp16` autocasts but NeMo forces attention to bf16 (Ampere+) or fp32 (Turing);
  `fp16`/`bf16` cast the encoder weights. bf16 is skipped on GPUs without support (RTX 2080).

## Preliminary approaches

* **A — exact redundancy removal in cache-aware attention** (`asrbench/prelim_attn.py`). NeMo caches attention
  *inputs* and re-projects K and V for all 72 frames plus the positional projection for ~143 positions every chunk.
  The patch caches projected K/V (updated in place, CUDA-graph safe) and computes `W_pos · pos_emb` once.
* **B — fused conv block** (`asrbench/fused_conv.py`). LayerNorm → PW1 → GLU → causal depthwise conv (with cache)
  → LayerNorm → SiLU → PW2 (+ residual) in 2 Triton kernels instead of ~15; each program recomputes the LayerNorm
  statistics itself, so the only dependency is before PW2 — the same mapping the megakernel will use.
* Both can be applied inside the full NeMo encoder (`--patches A B AB`) and inside CUDA graphs.

## Layout

```
asrbench/   common.py (timing, stats, NVML, results) · model.py (loading, streaming state, step impls)
            prelim_attn.py (A) · fused_conv.py (B)
scripts/    00_fetch · 01_micro · 02_latency · 03_profile · 04_onnx_trt · 05_wer · 06_prelim_conv · summarize
micro/      persistent_microbench.cu
jobs/       run_quick.sh · run_full.sh · slurm_gpu.sbatch
data/       get_librispeech.sh
```
