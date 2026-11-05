"""Serve a checkpoint behind an OpenAI-compatible endpoint so the inference-study measurement
tasks (Inspect AI) run unchanged against a trained model.

    with ServedModel(checkpoint, config['serve'], tag='M001') as served:
        inspect_eval(task, model=served.model, ...)

`served.model` is "openai-api/vllm/<tag>"; Inspect's openai-api provider reads VLLM_BASE_URL
and VLLM_API_KEY, which the context manager sets. backend "scripted" serves nothing and returns
"scripted/<tag>" for offline tests (the fake model is registered by the bench's tests).
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def release_gpu(wait_seconds: float = 60.0, target_fraction: float = 0.9) -> float | None:
    """Return the parent process's CUDA allocations before a server subprocess asks for most of
    the device, and report the free fraction (None if it cannot be measured). A training stage
    that just finished in this process can leave tens of GB held; vLLM then refuses to start
    ("Free memory ... is less than desired GPU memory utilization"). Only touches CUDA if torch is
    already imported and initialised; waits up to wait_seconds for target_fraction to be free."""
    import gc
    import sys
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return _free_fraction_nvidia_smi()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    deadline = time.time() + wait_seconds
    while True:
        free, total = torch.cuda.mem_get_info()
        if free / total >= target_fraction or time.time() >= deadline:
            break
        gc.collect(); torch.cuda.empty_cache(); time.sleep(2)
    if free / total < target_fraction:
        print(f"[serving] warning: only {free / 2**30:.1f} of {total / 2**30:.1f} GiB free before starting vLLM", flush=True)
    return free / total


def _free_fraction_nvidia_smi() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()[0]
        used, total = (float(x) for x in out.split(","))
        return 1.0 - used / total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class ServedModel:
    def __init__(self, checkpoint: str, config: dict, tag: str, log_dir: str | Path | None = None):
        self.checkpoint, self.config, self.tag = str(checkpoint), config, tag
        self.log_dir = Path(log_dir) if log_dir else None
        self.process = None
        self.model = None

    def __enter__(self):
        backend = self.config.get("backend", "vllm")
        if backend == "scripted":
            self.model = f"scripted/{self.tag}"
            return self
        if backend != "vllm":
            raise ValueError(f"unknown serving backend {backend!r}")
        free_fraction = release_gpu()
        port = int(self.config.get("port", 8010))
        engine = self.config.get("vllm_engine", {})
        utilization = float(engine.get("gpu_memory_utilization", 0.7))
        if free_fraction is not None and free_fraction - 0.05 < utilization:
            # vLLM refuses to start if the requested fraction exceeds what is free; serve with what there is
            fitted = round(max(0.3, free_fraction - 0.05), 2)
            print(f"[serving] {self.tag}: only {free_fraction:.0%} of the GPU is free; using "
                  f"--gpu-memory-utilization {fitted} instead of {utilization}", flush=True)
            utilization = fitted
        cmd = [self.config["vllm_python"], "-m", "vllm.entrypoints.openai.api_server", "--model", self.checkpoint,
               "--served-model-name", self.tag, "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
               "--max-model-len", str(engine.get("max_model_len", 32768)),
               "--gpu-memory-utilization", str(utilization),
               "--max-num-seqs", str(engine.get("max_num_seqs", 32))]
        if not engine.get("enable_prefix_caching", True):
            cmd.append("--no-enable-prefix-caching")
        cmd += list(self.config.get("extra_args", []))
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"serve-{self.tag}.log", "a") if self.log_dir else subprocess.DEVNULL
        # The inference venv's bin holds ninja (and the CUDA bin under CUDA_HOME holds nvcc), which
        # FlashInfer's JIT runs as subprocesses; put them on PATH as vllm_session.py does.
        env = os.environ.copy()
        extra = [str(Path(self.config["vllm_python"]).resolve().parent)]
        if env.get("CUDA_HOME"):
            extra.append(str(Path(env["CUDA_HOME"]) / "bin"))
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        self.process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + self.config.get("startup_timeout_seconds", 900)
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM server exited with code {self.process.returncode}; see serve-{self.tag}.log")
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(3)
        else:
            self.close()
            raise RuntimeError("vLLM server did not become healthy in time")
        os.environ["VLLM_BASE_URL"] = base + "/v1"
        os.environ["VLLM_API_KEY"] = self.config.get("api_key", "local")
        self.model = f"openai-api/vllm/{self.tag}"
        return self

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def __exit__(self, *exc):
        self.close()
