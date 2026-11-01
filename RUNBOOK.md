# RUNBOOK — every command, end to end

Three machines, one codebase (via a private GitHub repo), one software stack (PyTorch 2.14.0 + CUDA 12.6,
the last release that still supports the GTX 1080 Ti), results merged on the laptop.

| Machine | GPU (arch) | What it runs | What it cannot run |
| --- | --- | --- | --- |
| Laptop | RTX 3050 (Ampere sm_86, 2 MB L2, 4–6 GB) | Everything for the 120M model; development | 600M model (needs ≥ 10 GB) |
| IIITH node | 4× GTX 1080 Ti (Pascal sm_61, 11 GB) | FP32 eager / CUDA Graph / prelim A, profiling, microbenchmarks, long WER runs on 4 GPUs in parallel | `torch.compile`, Triton (prelim B), BF16, TensorRT 10; FP16 math runs at 1/64 speed |
| JarvisLabs | L4 (Ada sm_89, 48 MB L2, 24 GB) | Everything incl. 600M model and TensorRT; the "inference GPU, large L2" data point | — |

Placeholders: `<you>` GitHub user, `<login>` cluster login host (probably `ada.iiit.ac.in`),
`<jl-host>`/`<jl-port>` from the JarvisLabs instance page.

---

## 0. Laptop: put the code in a private GitHub repo (once)

Linux or WSL2 Ubuntu (recommended on Windows: Triton is not officially supported on native Windows).

```bash
unzip asrbench.zip && cd asrbench
git init -b main && git add . && git commit -m "asrbench: baselines + prelim"
# create an EMPTY private repo named asrbench on github.com, then:
git remote add origin git@github.com:<you>/asrbench.git      # or https://github.com/<you>/asrbench.git
git push -u origin main
```

Cloning on other machines over HTTPS needs a GitHub personal access token as the password
(github.com → Settings → Developer settings → Fine-grained tokens, read-only on this repo). No GitHub? Use the
`scp` alternatives shown in each section.

---

## 1. Laptop (RTX 3050)

### 1.1 Driver and CUDA compiler

```bash
nvidia-smi                       # need driver >= 560. On WSL2 install the driver on WINDOWS only, never inside WSL.
# nvcc 12.6 for the persistent microbenchmark (WSL2 Ubuntu; for native Ubuntu replace wsl-ubuntu with ubuntu2204 or ubuntu2404):
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-6
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
nvcc --version                   # 12.6
sudo apt-get -y install tmux git curl unzip
```

### 1.2 Environment, models, data

```bash
cd ~/asrbench
WITH_TRT=1 bash setup_env.sh               # ~10 min; ends by printing torch/GPU/NeMo/ORT versions
source venv/bin/activate
python scripts/00_fetch.py --models eou120m # ~450 MB into ~/.cache/huggingface
bash data/get_librispeech.sh               # test-clean + test-other; falls back to Hugging Face if openslr is slow
```

### 1.3 Smoke test first (~15 min) — send me any failures before using cluster/cloud time

```bash
OUT=results_laptop bash jobs/run_quick.sh 2>&1 | tee logs_quick.txt
grep -h '"ok": false' results_laptop/*/*.jsonl | head        # failed configs, if any
```

### 1.4 Full run (~3–5 h; plug in the charger, set Windows/OS power mode to "Best performance")

```bash
tmux new -s bench
OUT=results_laptop bash jobs/run_full.sh 2>&1 | tee logs_laptop_full.txt
# detach: Ctrl-b d      reattach later: tmux attach -t bench
nvidia-smi -q -d PERFORMANCE     # in another terminal: check for thermal/power throttling during the run
```

Resumable: if anything stops, run the same command again; finished configurations are skipped.

---

## 2. IIITH cluster node (4× GTX 1080 Ti)

### 2.1 Check storage, internet, Python on the node

```bash
ssh <you>@<login>               # then get a GPU node the way you already do (you are on gnode012)
df -h ~ /scratch /share* 2>/dev/null ; quota -s 2>/dev/null   # need ~12 GB: venv 6 GB, models 3 GB, data 1 GB
curl -sI https://huggingface.co | head -1                      # "HTTP/2 200" = this node has internet
```

If home is small, work in a bigger directory and point caches there:

```bash
export WORK=$HOME            # or e.g. /scratch/$USER (node-local: copy results out before the session ends)
export HF_HOME=$WORK/.hf UV_CACHE_DIR=$WORK/.uv-cache
echo "export HF_HOME=$HF_HOME UV_CACHE_DIR=$UV_CACHE_DIR" >> ~/.bashrc
```

### 2.2 Get the code

```bash
cd $WORK && git clone https://github.com/<you>/asrbench.git && cd asrbench
# without GitHub, from the laptop:  scp asrbench.zip <you>@<login>:~/  then on the cluster: unzip asrbench.zip
```

### 2.3 Environment, models, data
If the GPU node has no internet, run this block on the login node (home is shared), then come back to the node.

```bash
cd $WORK/asrbench
bash setup_env.sh                          # pins torch 2.14.0+cu126 (works on sm_61); no TensorRT here
source venv/bin/activate
python scripts/00_fetch.py --models eou120m nemotron600m
bash data/get_librispeech.sh
```

### 2.4 nvcc 12.x for the persistent microbenchmark (CUDA 13 cannot target Pascal)

```bash
module avail cuda 2>&1 | tr ' ' '\n' | grep -i cuda    # if a 12.x module exists:
module load cuda/12.6                                  # (use the exact name shown)
nvcc --version
# no module? user-space nvcc via micromamba (no root):
cd $WORK && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
./bin/micromamba create -y -p $WORK/cuda126 -c nvidia/label/cuda-12.6.3 cuda-nvcc cuda-cudart-dev
export PATH=$WORK/cuda126/bin:$PATH && nvcc --version && cd $WORK/asrbench
nvcc -O3 -std=c++17 -arch=sm_61 micro/persistent_microbench.cu -o micro/pmb && ./micro/pmb 200   # quick check
```

### 2.5 Run on all 4 GPUs in parallel (~3–4 h)

```bash
cd $WORK/asrbench && source venv/bin/activate
nvidia-smi                                   # all 4 GPUs idle?
tmux new -s bench
bash jobs/run_node4.sh                       # GPU0: micro+pmb+latency, GPU1: profile+onnx+batch+prelim, GPU2/3: WER
# Ctrl-b d to detach; watch progress:
tail -f logs/g0.log logs/g1.log logs/g2.log logs/g3.log
```

Pascal is handled automatically: `torch.compile`, Triton and BF16 configurations are skipped, and batch/WER/600M
runs default to FP32 (GeForce Pascal executes FP16 arithmetic at 1/64 rate, so FP16 numbers there are not a fair
baseline — keep them only as a curiosity). If the session has a time limit, rerun `bash jobs/run_node4.sh`
in the next session: everything resumes.

Single GPU instead of four: `CUDA_VISIBLE_DEVICES=0 OUT=results bash jobs/run_full.sh`.

### 2.6 Copy results back to the laptop

```bash
# on the cluster: commit results (small JSON files; traces/onnx/engines are git-ignored)
cd $WORK/asrbench && git add results && git commit -m "results: gtx1080ti" && git push
# or, from the LAPTOP without git:
scp -r <you>@<login>:'~/asrbench/results' ~/asrbench/results_cluster
```

---

## 3. JarvisLabs (L4)

Prices to verify on jarvislabs.ai/in/gpu-cloud-pricing: L4 about ₹36/h, A5000 about ₹40/h, billed per minute;
a paused instance pays only storage. One full run on an L4 ≈ 4–5 h ≈ ₹150–200, so ₹2,800 also covers a second
GPU type and the megakernel experiments later.

### 3.1 Create the instance (web UI)

1. jarvislabs.ai → Create Instance → framework **PyTorch** → GPU **L4 ×1** → storage **50 GB** → Launch.
2. On the instance page, copy the SSH command (needs your public key added under Settings → SSH keys), e.g.
   `ssh -p <jl-port> root@<jl-host>`. The web terminal / JupyterLab terminal also works.

### 3.2 Set up and run

```bash
ssh -p <jl-port> root@<jl-host>
cd /home                                   # persistent across pause/resume
git clone https://github.com/<you>/asrbench.git && cd asrbench
# without GitHub, from the laptop:  scp -P <jl-port> asrbench.zip root@<jl-host>:/home/
apt-get update -qq && apt-get install -y -qq tmux unzip
WITH_TRT=1 bash setup_env.sh && source venv/bin/activate
python scripts/00_fetch.py --models eou120m nemotron600m
bash data/get_librispeech.sh
nvcc --version || true                     # if the image has CUDA 12.x nvcc, the pmb stage builds automatically
OUT=results_l4 bash jobs/run_quick.sh      # 15 min sanity check
tmux new -s bench
OUT=results_l4 bash jobs/run_full.sh 2>&1 | tee logs_l4.txt          # ~4-5 h, includes the 600M model
```

### 3.3 Bring results home and stop paying

```bash
# on the instance
git add results_l4 && git commit -m "results: l4" && git push
# or from the LAPTOP:
scp -P <jl-port> -r root@<jl-host>:/home/asrbench/results_l4 ~/asrbench/
```

Then **Pause** (keeps /home, storage-only billing) or **Destroy** the instance from the web UI.

Optional second cloud GPU for the L2-size contrast (A6000 6 MB L2 vs L4 48 MB): repeat 3.1–3.3 with
`OUT=results_a6000`.

---

## 4. Laptop: merge, summarize, make plots

```bash
cd ~/asrbench && source venv/bin/activate
git pull                                         # if results were pushed from the other machines
python scripts/merge_results.py --into results results_laptop results_cluster results_l4
python scripts/summarize.py --results results    # -> results/SUMMARY.md and results/plots/*.png
git add results && git commit -m "merged results" && git push
```

Open traces from `results/<gpu>/traces/*.json` at https://ui.perfetto.dev to screenshot the kernel timelines
(eager vs CUDA Graph) for the presentation.

---

## 5. Running pieces by hand

```bash
python scripts/01_micro.py                                            # bandwidth, launch costs, GEMV efficiency
python scripts/02_latency.py --preset quick                           # presets: quick | full | chunks | batch | prelim
python scripts/02_latency.py --models eou120m --precisions fp16 --impls eager cudagraph --patches none A AB
python scripts/03_profile.py --precisions fp16 --impls eager cudagraph
python scripts/04_onnx_trt.py --model eou120m --variants ort-cuda ort-cuda-graph trt-fp16
python scripts/05_wer.py --max-utts 200 --modes fp32:eager fp16:cudagraph fp16:cudagraph:AB
python scripts/06_prelim_conv.py --precisions fp16 --batches 1 4 --sweep
./micro/pmb 2000                                                      # after building it (sections 1.1 / 2.4)
python scripts/summarize.py
```

Useful flags: `--force` (re-run finished configs), `--att 70,13` (chunk size of the 600M model),
`--iters N`, `--out DIR`.

---

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `no kernel image is available` or "sm_61 is not compatible" on the 1080 Ti | torch is not the cu126 build: `pip show torch` must say 2.14.0; rerun `bash setup_env.sh` in a fresh venv |
| `CUDA out of memory` on the laptop | use `--batches 1 2 4 8 16` for the batch preset; the 600M model is skipped automatically below 10 GB |
| Hugging Face download fails on the node | run `00_fetch.py` on the login node with the same `HF_HOME`, or copy `~/.cache/huggingface` with `scp -r` |
| openslr.org hangs | `python data/get_librispeech_hf.py` |
| `TensorrtExecutionProvider not available` | expected without `WITH_TRT=1`, and always on Pascal; the other ORT variants still run |
| `nvcc fatal: Unsupported gpu architecture 'compute_61'` | you are using CUDA 13 nvcc; use a 12.x nvcc (section 2.4) |
| Very noisy p99 on the laptop | close other apps, charger in, run again with `--force`; report laptop numbers as indicative |
| A run was interrupted | rerun the same command: finished configurations are skipped |
