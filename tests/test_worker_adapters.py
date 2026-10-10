"""Backend adapter resolution (plan §3, §4).

Managed MLX workers run only on Apple Silicon/macOS; managed ROCm workers
only on compatible AMD/Linux hosts. ROCm must be refused when requested but
unavailable — never silently use the CPU.
"""

import pytest

from indexedcode.workers.adapters import (
    BackendUnavailableError,
    backend_available,
    worker_command,
)


def test_mlx_available_on_apple_silicon():
    assert backend_available("mlx", system="darwin", machine="arm64")


def test_mlx_unavailable_on_intel_mac():
    with pytest.raises(BackendUnavailableError):
        backend_available("mlx", system="darwin", machine="x86_64")


def test_mlx_unavailable_on_linux():
    with pytest.raises(BackendUnavailableError):
        backend_available("mlx", system="linux", machine="arm64")


def test_rocm_requires_linux_and_hip():
    assert backend_available(
        "rocm", system="linux", machine="x86_64", has_hip=lambda: True
    )


def test_rocm_refused_when_hip_missing():
    with pytest.raises(BackendUnavailableError, match="[Rr][Oo][Cc][Mm]"):
        backend_available(
            "rocm", system="linux", machine="x86_64", has_hip=lambda: False
        )


def test_rocm_refused_on_macos():
    with pytest.raises(BackendUnavailableError):
        backend_available("rocm", system="darwin", machine="arm64")


def test_unknown_backend_rejected():
    with pytest.raises(BackendUnavailableError):
        backend_available("cuda", system="linux", machine="x86_64")


def test_command_uses_python_module_entrypoint():
    argv = worker_command(
        role="summary", backend="mlx", port=8765, model_path="/tmp/qwen"
    )
    assert "-m" in argv
    assert "indexedcode.workers.serve" in argv
    assert "--role" in argv and "summary" in argv
    assert "--port" in argv and "8765" in argv


def test_command_never_puts_the_secret_in_argv():
    argv = worker_command(
        role="embedding", backend="rocm", port=8765, model_path="/tmp/m"
    )
    for arg in argv:
        assert "secret" not in arg.lower()
        assert "token" not in arg.lower()


def test_command_includes_backend_and_model_path():
    argv = worker_command(
        role="embedding", backend="mlx", port=1, model_path="/models/mini"
    )
    assert "mlx" in argv
    assert "/models/mini" in argv
