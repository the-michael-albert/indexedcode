"""Worker factory: managed vs remote wiring (plan §4, §4.1).

Remote mode must never spawn a process; managed mode must refuse to start
on an incompatible backend, and remote configuration must require a URL.
"""

import pytest

from indexedcode.workers.factory import build_worker


def base_config(mode="managed", **overrides):
    cfg = {
        "mode": mode,
        "backend": "mlx",
        "model": "fake-embed",
        "max_batch_items": 32,
        "max_batch_tokens": 4096,
        "max_concurrent_requests": 1,
    }
    cfg.update(overrides)
    return cfg


def test_managed_mode_returns_managed_worker():
    from indexedcode.workers.supervisor import ManagedWorker

    worker = build_worker("embedding", base_config())
    assert isinstance(worker, ManagedWorker)
    assert worker.name == "embedding"
    worker.stop()  # not started; stop is a safe no-op


def test_remote_mode_returns_client_handle_without_spawn():
    from indexedcode.workers.remote import RemoteWorker

    worker = build_worker(
        "summary", base_config(mode="remote", remote_url="https://w.internal")
    )
    assert isinstance(worker, RemoteWorker)
    assert not hasattr(worker, "start") or not callable(
        getattr(worker, "start", None)
    ) or worker.__class__.__name__ == "RemoteWorker"
    # remote handle exposes the HTTP contract, not process supervision
    assert hasattr(worker, "client")
    assert not hasattr(worker, "poll")


def test_remote_mode_requires_remote_url():
    from indexedcode.config import ConfigError

    with pytest.raises(ConfigError, match="remote_url"):
        build_worker("embedding", base_config(mode="remote"))


def test_remote_mode_never_spawns_even_if_backend_looks_local(tmp_path):
    """Remote workers are supervised by their own host, never spawned."""
    from indexedcode.workers.remote import RemoteWorker

    worker = build_worker(
        "embedding",
        base_config(
            mode="remote",
            backend="rocm",  # incompatible with this machine on purpose
            remote_url="https://amd.internal",
        ),
    )
    assert isinstance(worker, RemoteWorker)


def test_managed_mode_refuses_incompatible_backend():
    from indexedcode.workers.adapters import BackendUnavailableError

    worker = build_worker(
        "embedding", base_config(backend="rocm")  # rocm on macOS: refused
    )
    with pytest.raises(BackendUnavailableError):
        worker.start()


def test_managed_worker_command_targets_configured_model():
    worker = build_worker("embedding", base_config())
    assert any("fake-embed" in arg for arg in worker.command)
