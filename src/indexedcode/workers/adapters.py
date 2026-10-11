"""Backend resolution and worker command building (plan §3, §4).

Managed MLX workers run only on Apple Silicon/macOS; CUDA workers only on
a Linux host with an NVIDIA GPU; Vulkan workers (llama.cpp) only on a
Linux host with an AMD GPU; ROCm workers only with a HIP runtime. The CPU
backend runs anywhere. A GPU backend is refused when requested but
unavailable — never a silent CPU fallback. The worker
authentication secret travels via environment, never via argv.
"""

from __future__ import annotations

import platform
import shutil
import sys

from indexedcode.workers.errors import BackendUnavailableError
from indexedcode.workers.gpu import AMD, NVIDIA, detect_gpu

WORKER_MODULE = "indexedcode.workers.serve"


def _default_has_hip() -> bool:
    return shutil.which("hipcc") is not None or shutil.which("rocm-smi") is not None


def backend_available(
    backend: str,
    system: str | None = None,
    machine: str | None = None,
    has_hip=None,
    gpu=None,
) -> bool:
    """Raise BackendUnavailableError unless the backend may run here."""
    system = system if system is not None else platform.system().lower()
    machine = machine if machine is not None else platform.machine().lower()
    if backend == "mlx":
        if system == "darwin" and machine in ("arm64", "aarch64"):
            return True
        raise BackendUnavailableError(
            f"MLX workers require Apple Silicon macOS"
            f" (got {system}/{machine})"
        )
    if backend == "rocm":
        if system != "linux":
            raise BackendUnavailableError(
                f"ROCm workers require Linux (got {system})"
            )
        check = has_hip if has_hip is not None else _default_has_hip
        if not check():
            raise BackendUnavailableError(
                "ROCm requested but no HIP runtime was found;"
                " refusing to fall back to CPU"
            )
        return True
    if backend in ("cuda", "vulkan"):
        wanted = NVIDIA if backend == "cuda" else AMD
        if system != "linux":
            raise BackendUnavailableError(
                f"{backend} workers require Linux (got {system})"
            )
        found = gpu if gpu is not None else detect_gpu()
        if found != wanted:
            raise BackendUnavailableError(
                f"{backend} requested but no {wanted.upper()} GPU was"
                f" detected (found {found}); refusing to fall back to CPU"
            )
        return True
    if backend == "cpu":
        return True
    raise BackendUnavailableError(f"unknown backend {backend!r}")


def default_backend(system: str | None = None,
                    machine: str | None = None,
                    gpu: str | None = None) -> str:
    """MLX on Apple Silicon, CUDA with an NVIDIA GPU, Vulkan with an AMD
    GPU, the portable CPU backend otherwise."""
    system = system if system is not None else platform.system().lower()
    machine = machine if machine is not None else platform.machine().lower()
    if system == "darwin" and machine in ("arm64", "aarch64"):
        return "mlx"
    if system == "linux":
        found = gpu if gpu is not None else detect_gpu()
        if found == NVIDIA:
            return "cuda"
        if found == AMD:
            return "vulkan"
    return "cpu"


# llama.cpp loads GGUF files, so the Vulkan backend needs its own default
# models ("repo_id:filename-glob").
_VULKAN_MODELS = {
    "summary": "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:*q8_0.gguf",
    "embedding": "second-state/All-MiniLM-L6-v2-Embedding-GGUF:*Q8_0.gguf",
}
_DEFAULT_MODELS = {
    "summary": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
}


def default_model(role: str, backend: str) -> str:
    """Default model for ``role`` (``summary``/``embedding``) on a backend."""
    table = _VULKAN_MODELS if backend == "vulkan" else _DEFAULT_MODELS
    return table[role]


def worker_command(role: str, backend: str, port: int | None = None,
                   model_path: str = "", *extra: str) -> list[str]:
    """Build the managed-worker argv (secret is passed via env, not argv)."""
    argv = [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--role",
        role,
        "--backend",
        backend,
    ]
    if port is not None:
        argv += ["--port", str(port)]
    if model_path:
        argv += ["--model", model_path]
    argv.extend(extra)
    return argv
