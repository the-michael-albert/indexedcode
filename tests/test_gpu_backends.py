"""GPU detection, backend defaults and runtime install planning."""

import pytest

from indexedcode import runtime_install, semantic
from indexedcode.config import load_config
from indexedcode.workers.adapters import (
    BackendUnavailableError, backend_available, default_backend, default_model)
from indexedcode.workers.gpu import detect_gpu


def _drm(tmp_path, *vendors):
    for i, vendor in enumerate(vendors):
        d = tmp_path / f"card{i}" / "device"
        d.mkdir(parents=True)
        (d / "vendor").write_text(vendor + "\n")
    return str(tmp_path / "card*" / "device" / "vendor")


@pytest.mark.parametrize("vendors,smi,expected", [
    (["0x10de"], None, "nvidia"),
    (["0x1002"], None, "amd"),
    (["0x1002", "0x10de"], None, "nvidia"),
    (["0x8086"], None, "none"),
    ([], None, "none"),
    ([], "/usr/bin/nvidia-smi", "nvidia"),
])
def test_detect_gpu(tmp_path, vendors, smi, expected):
    assert detect_gpu(_drm(tmp_path, *vendors), which=lambda n: smi,
                      environ={}) == expected


def test_detect_gpu_env_override(tmp_path):
    assert detect_gpu(_drm(tmp_path, "0x10de"),
                      environ={"INDEXEDCODE_GPU": "none"}) == "none"


@pytest.mark.parametrize("gpu,expected", [
    ("nvidia", "cuda"), ("amd", "vulkan"), ("none", "cpu")])
def test_linux_default_backend_follows_gpu(gpu, expected):
    assert default_backend("linux", "x86_64", gpu=gpu) == expected


def test_gpu_ignored_off_linux():
    assert default_backend("darwin", "x86_64", gpu="nvidia") == "cpu"


def test_cuda_and_vulkan_refuse_wrong_gpu():
    assert backend_available("cuda", system="linux", gpu="nvidia")
    assert backend_available("vulkan", system="linux", gpu="amd")
    with pytest.raises(BackendUnavailableError):
        backend_available("cuda", system="linux", gpu="amd")
    with pytest.raises(BackendUnavailableError):
        backend_available("vulkan", system="linux", gpu="none")
    with pytest.raises(BackendUnavailableError):
        backend_available("cuda", system="darwin", gpu="nvidia")


def test_vulkan_defaults_to_gguf_models(tmp_path):
    assert "GGUF" in default_model("summary", "vulkan")
    assert "GGUF" in default_model("embedding", "vulkan")
    path = tmp_path / "c.toml"
    path.write_text('[embedding]\nbackend = "vulkan"\n')
    assert "GGUF" in load_config(path)["embedding"]["model"]
    path.write_text('[embedding]\nbackend = "cuda"\n')
    assert load_config(path)["embedding"]["model"].startswith("sentence-")


def test_install_plans():
    assert install_pkgs("cuda") == [("torch", "transformers")]
    assert install_pkgs("cpu", "linux")[0] == ("torch",)
    cpu = runtime_install.install_plan("cpu", "linux")
    assert cpu[0].index_url == runtime_install.TORCH_CPU_INDEX
    assert runtime_install.install_plan("cpu", "darwin")[0].index_url is None
    vulkan = runtime_install.install_plan("vulkan")[0]
    assert vulkan.env["CMAKE_ARGS"] == "-DGGML_VULKAN=on"
    assert runtime_install.install_plan("rocm") is None


def install_pkgs(backend, system="linux"):
    return [s.packages for s in runtime_install.install_plan(backend, system)]


class _Result:
    def __init__(self, code, err=""):
        self.returncode, self.stderr, self.stdout = code, err, ""


def test_install_runtime_uses_uv_and_index():
    calls = []
    runtime_install.install_runtime(
        "cpu", run=lambda argv, **kw: calls.append((argv, kw)) or _Result(0),
        python="/py", which=lambda n: "/bin/uv", system="linux")
    assert calls[0][0] == ["/bin/uv", "pip", "install", "--python", "/py",
                           "--index-url", runtime_install.TORCH_CPU_INDEX,
                           "torch"]
    assert calls[1][0][-1] == "transformers"


def test_install_runtime_falls_back_to_pip_and_reports_failure():
    with pytest.raises(runtime_install.RuntimeInstallError,
                       match="libvulkan-dev") as exc:
        runtime_install.install_runtime(
            "vulkan", run=lambda argv, **kw: _Result(1, "boom"),
            python="/py", which=lambda n: None)
    assert "boom" in str(exc.value)


def test_check_runtime_auto_installs_once(monkeypatch):
    monkeypatch.setenv("INDEXEDCODE_AUTO_INSTALL", "1")
    monkeypatch.setattr(semantic, "_INSTALL_ATTEMPTED", set())
    installed = []
    monkeypatch.setattr(runtime_install, "install_runtime",
                        lambda backend: installed.append(backend))
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    cfg = {"mode": "managed", "backend": "cuda"}
    for role in ("embedding", "summary"):
        with pytest.raises(RuntimeError, match="install-runtime"):
            semantic.check_runtime(role, cfg)
    assert installed == ["cuda"]
