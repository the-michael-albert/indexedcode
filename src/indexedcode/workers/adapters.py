"""Backend resolution and worker command building (plan §3, §4).

Managed MLX workers run only on Apple Silicon/macOS; managed ROCm workers
only on a compatible AMD/Linux host with HIP available. The CPU backend
runs anywhere. ROCm is refused
when requested but unavailable — never a silent CPU fallback. The worker
authentication secret travels via environment, never via argv.
"""

from __future__ import annotations

import platform
import shutil
import sys

from indexedcode.workers.errors import BackendUnavailableError

WORKER_MODULE = "indexedcode.workers.serve"


def _default_has_hip() -> bool:
    return shutil.which("hipcc") is not None or shutil.which("rocm-smi") is not None


def backend_available(
    backend: str,
    system: str | None = None,
    machine: str | None = None,
    has_hip=None,
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
    if backend == "cpu":
        return True
    raise BackendUnavailableError(f"unknown backend {backend!r}")


def default_backend(system: str | None = None,
                    machine: str | None = None) -> str:
    """MLX on Apple Silicon macOS, the portable CPU backend elsewhere."""
    system = system if system is not None else platform.system().lower()
    machine = machine if machine is not None else platform.machine().lower()
    if system == "darwin" and machine in ("arm64", "aarch64"):
        return "mlx"
    return "cpu"


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
