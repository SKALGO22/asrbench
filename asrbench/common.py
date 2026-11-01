"""Shared utilities: system info, timing statistics, power sampling, result files."""
from __future__ import annotations

import json
import os
import platform
import socket
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

# ----------------------------------------------------------------------------- system info


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def gpu_slug() -> str:
    """Short, filesystem-safe GPU identifier, e.g. 'rtx4090', 'gtx1080ti', 'l4'."""
    n = gpu_name().lower()
    for k in ["4090", "3090", "2080 ti", "2080", "1080 ti", "1080", "3050", "a6000", "a5000", "a100", "h100", "l40s",
              "l4", "t4", "a10"]:
        if k in n:
            pre = "rtx" if "rtx" in n else ("gtx" if "gtx" in n else "")
            return pre + k.replace(" ", "")
    return "".join(c for c in n if c.isalnum())[:16] or "cpu"


def compute_capability() -> tuple:
    return torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)


def triton_ok() -> bool:
    """torch.compile (Inductor) and our Triton kernels need compute capability >= 7.0 (not Pascal)."""
    return compute_capability() >= (7, 0)


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cpu": platform.processor() or platform.machine(),
    }
    try:
        import nemo  # noqa

        info["nemo"] = nemo.__version__
    except Exception:
        info["nemo"] = None
    try:
        import triton  # noqa

        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(
            gpu=p.name,
            sm=f"{p.major}.{p.minor}",
            sm_count=p.multi_processor_count,
            vram_gb=round(p.total_memory / 2**30, 2),
            l2_mb=round(getattr(p, "L2_cache_size", 0) / 2**20, 2),
        )
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,clocks.max.mem,power.limit",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
            info["nvidia_smi"] = out
        except Exception:
            pass
    return info


def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


# ----------------------------------------------------------------------------- statistics


def summarize(samples: List[float]) -> Dict[str, float]:
    """Latency statistics in the unit of the input (we use microseconds)."""
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)

    def pct(q: float) -> float:
        if n == 1:
            return s[0]
        k = (n - 1) * q
        f, c = int(k), min(int(k) + 1, n - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    return {
        "n": n,
        "mean": statistics.fmean(s),
        "std": statistics.pstdev(s) if n > 1 else 0.0,
        "min": s[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "p999": pct(0.999),
        "max": s[-1],
    }


# ----------------------------------------------------------------------------- timing


@dataclass
class StepTimes:
    wall_us: List[float] = field(default_factory=list)   # call start -> GPU finished (synchronize)
    gpu_us: List[float] = field(default_factory=list)    # CUDA event start -> event end
    host_us: List[float] = field(default_factory=list)   # call start -> Python call returned (dispatch cost)


def time_steps(step: Callable[[], Any], n: int, paced_period_ms: Optional[float] = None,
               cuda: bool = True) -> StepTimes:
    """Run `step` n times; record wall, GPU-event and host-dispatch time per step.

    Each step is synchronized, which is what a streaming server does (it needs the result before
    the next chunk). With `paced_period_ms`, steps start on a fixed real-time schedule so the GPU idles
    between chunks as it would with live audio (exposes clock ramp-down and cold-cache effects).
    """
    st = StepTimes()
    if cuda:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    next_t = time.perf_counter()
    for i in range(n):
        if paced_period_ms:
            now = time.perf_counter()
            if next_t > now:
                time.sleep(next_t - now)
            next_t += paced_period_ms / 1e3
        t0 = time.perf_counter()
        if cuda:
            starts[i].record()
        step()
        if cuda:
            ends[i].record()
        t1 = time.perf_counter()
        if cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        st.host_us.append((t1 - t0) * 1e6)
        st.wall_us.append((t2 - t0) * 1e6)
    if cuda:
        st.gpu_us = [starts[i].elapsed_time(ends[i]) * 1e3 for i in range(n)]
    return st


def cuda_time_us(fn: Callable[[], Any], iters: int = 20, warmup: int = 5) -> float:
    """Mean GPU time of fn() in microseconds, measured with CUDA events over `iters` back-to-back calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


# ----------------------------------------------------------------------------- power


class PowerSampler:
    """Samples board power with NVML in a background thread. Energy = mean power x time."""

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ok = False
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
                                                        if os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].isdigit() else 0)
            self.ok = True
        except Exception:
            self.ok = False

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        if self.ok:
            self.samples = []
            self._t0 = time.perf_counter()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.ok and self._thread is not None:
            self._stop.set()
            self._thread.join()
            self.duration_s = time.perf_counter() - self._t0
        return False

    def result(self) -> Dict[str, Optional[float]]:
        if not self.ok or not self.samples:
            return {"power_mean_w": None, "energy_j": None}
        p = statistics.fmean(self.samples)
        return {"power_mean_w": p, "energy_j": p * self.duration_s}


# ----------------------------------------------------------------------------- results


class Results:
    """Append-only JSONL results file; lets interrupted runs resume by skipping done keys."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    self.done.add(json.loads(line).get("key"))
                except Exception:
                    pass

    def has(self, key: str) -> bool:
        return key in self.done

    def write(self, rec: Dict[str, Any]):
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.done.add(rec.get("key"))


def out_dir(base: str = "results") -> Path:
    d = Path(base) / gpu_slug()
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
