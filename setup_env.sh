#!/usr/bin/env bash
# One-time environment setup, identical on every machine (laptop / IIITH cluster / JarvisLabs).
#   bash setup_env.sh [VENV_DIR]          (default: ./venv)   then:  source VENV_DIR/bin/activate
#
# Pins PyTorch 2.14.0 + CUDA 12.6 everywhere: it is the LAST PyTorch release with wheels for Pascal
# (GTX 1080 Ti, sm_61) and it also runs on Turing/Ampere/Ada (RTX 3050, L4, A6000, 4090), so all GPUs
# share one software stack. Needs an NVIDIA driver >= 560 (any CUDA 12.6+/13.x driver).
# Uses `uv` (single static binary, no root) to get Python 3.11 and to install packages fast.
set -euo pipefail
VENV=${1:-venv}
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
source "$VENV/bin/activate"
cat > /tmp/asrbench_constraints.txt <<'C'
torch==2.14.0
C
uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install -c /tmp/asrbench_constraints.txt "nemo_toolkit[asr]" pynvml soundfile matplotlib pyarrow onnx
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo 0)
uv pip install -c /tmp/asrbench_constraints.txt onnxruntime-gpu || echo "onnxruntime-gpu install failed (04_onnx_trt will record it)"
if [ "${WITH_TRT:-0}" = "1" ]; then   # TensorRT 10 needs compute capability >= 7.5 (not the 1080 Ti)
  uv pip install -c /tmp/asrbench_constraints.txt tensorrt-cu12 || echo "tensorrt install failed"
fi
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| CUDA runtime", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0) if ok else None,
      "| capability", torch.cuda.get_device_capability(0) if ok else None)
if ok:
    x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
    print("triton/torch.compile usable:", torch.cuda.get_device_capability(0) >= (7, 0))
import nemo; print("nemo", nemo.__version__)
try:
    import onnxruntime as ort; print("onnxruntime", ort.__version__, ort.get_available_providers())
except Exception as e:
    print("onnxruntime unavailable:", e)
PY
echo "compute capability: $CC   (activate with: source $VENV/bin/activate)"
