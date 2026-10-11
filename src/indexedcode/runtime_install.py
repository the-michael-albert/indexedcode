"""Install the inference runtime a backend needs into *this* environment.

Wheels cannot run post-install hooks or pick dependencies by GPU, so the
runtime is installed on first use (``check_runtime``) or explicitly with
``indexedcode install-runtime``. Set ``INDEXEDCODE_AUTO_INSTALL=0`` to
disable the automatic install.

Backends: MLX is a normal dependency on Apple Silicon; CUDA uses PyTorch's
default Linux wheels; CPU uses PyTorch's CPU-only wheels on Linux; Vulkan
builds llama.cpp (llama-cpp-python) with its Vulkan backend, which needs
cmake, a C/C++ compiler, the Vulkan headers and glslc.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

log = logging.getLogger("indexedcode.runtime")

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
VULKAN_BUILD_ENV = {"CMAKE_ARGS": "-DGGML_VULKAN=on", "FORCE_CMAKE": "1"}
VULKAN_PREREQS = ("cmake, a C/C++ compiler, the Vulkan development headers"
                  " (libvulkan-dev) and glslc (glslang-tools/shaderc)")


class RuntimeInstallError(RuntimeError):
    """The runtime could not be installed automatically."""


@dataclass(frozen=True)
class Step:
    packages: tuple[str, ...]
    index_url: str | None = None
    env: dict = field(default_factory=dict)


def install_plan(backend: str, system: str | None = None) -> list[Step] | None:
    """Install steps for ``backend``, or ``None`` if it is not auto-installable."""
    system = system if system is not None else platform.system().lower()
    if backend == "mlx":
        return [Step(("mlx-lm", "mlx-embeddings"))]
    if backend == "cuda":
        return [Step(("torch", "transformers"))]
    if backend == "cpu":
        steps = []
        if system == "linux":  # the default Linux wheel drags in CUDA
            steps.append(Step(("torch",), index_url=TORCH_CPU_INDEX))
        else:
            steps.append(Step(("torch",)))
        steps.append(Step(("transformers",)))
        return steps
    if backend == "vulkan":
        return [Step(("llama-cpp-python", "huggingface-hub"),
                     env=dict(VULKAN_BUILD_ENV))]
    return None  # rocm: the right torch wheel depends on the ROCm version


def _installer(python: str, which=shutil.which):
    """argv prefix that installs into ``python``'s environment."""
    uv = which("uv")
    if uv:
        return [uv, "pip", "install", "--python", python]
    return [python, "-m", "pip", "install"]


def auto_install_enabled(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    return environ.get("INDEXEDCODE_AUTO_INSTALL", "1").strip().lower() not in (
        "0", "false", "no", "off")


def install_runtime(backend: str, run=subprocess.run, python: str | None = None,
                    which=shutil.which, system: str | None = None) -> None:
    steps = install_plan(backend, system)
    if steps is None:
        raise RuntimeInstallError(
            f"the {backend} runtime cannot be installed automatically;"
            " install torch and transformers for your ROCm build")
    prefix = _installer(python or sys.executable, which)
    for step in steps:
        argv = list(prefix)
        if step.index_url:
            argv += ["--index-url", step.index_url]
        argv += list(step.packages)
        log.warning("installing %s runtime: %s", backend, " ".join(argv))
        try:
            result = run(argv, env={**os.environ, **step.env},
                         capture_output=True, text=True)
        except OSError as exc:
            raise RuntimeInstallError(
                f"could not run the installer: {exc}") from exc
        if result.returncode != 0:
            hint = (f" Building the Vulkan backend needs {VULKAN_PREREQS}."
                    if backend == "vulkan" else "")
            tail = (result.stderr or result.stdout or "").strip()[-800:]
            raise RuntimeInstallError(
                f"installing {' '.join(step.packages)} failed"
                f" (exit {result.returncode}).{hint}\n{tail}")
    importlib.invalidate_caches()
