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
        port = int(self.config.get("port", 8010))
        engine = self.config.get("vllm_engine", {})
        cmd = [self.config["vllm_python"], "-m", "vllm.entrypoints.openai.api_server", "--model", self.checkpoint,
               "--served-model-name", self.tag, "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
               "--max-model-len", str(engine.get("max_model_len", 32768)),
               "--gpu-memory-utilization", str(engine.get("gpu_memory_utilization", 0.7)),
               "--max-num-seqs", str(engine.get("max_num_seqs", 32))]
        if not engine.get("enable_prefix_caching", True):
            cmd.append("--no-enable-prefix-caching")
        cmd += list(self.config.get("extra_args", []))
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"serve-{self.tag}.log", "a") if self.log_dir else subprocess.DEVNULL
        self.process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
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
