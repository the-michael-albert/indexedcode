"""Worker wiring from validated configuration (plan §4.1)."""

from __future__ import annotations

from indexedcode.workers.adapters import worker_command
from indexedcode.workers.client import WorkerClient
from indexedcode.workers.fingerprint import model_fingerprint
from indexedcode.workers.remote import RemoteWorker
from indexedcode.workers.supervisor import ManagedWorker


def _expected_fingerprint(section_cfg: dict) -> str:
    if section_cfg.get("expected_fingerprint"):
        return section_cfg["expected_fingerprint"]
    details = {
        "model": section_cfg["model"],
        "revision": section_cfg.get("revision", "unpinned"),
        "backend": section_cfg["backend"],
    }
    if section_cfg.get("dimension"):
        details["dimension"] = section_cfg["dimension"]
    if section_cfg.get("quantization"):
        details["quantization"] = section_cfg["quantization"]
    return model_fingerprint(**details)


def build_worker(name: str, section_cfg: dict):
    """Return a ManagedWorker (managed) or RemoteWorker (remote)."""
    from indexedcode.config import ConfigError

    mode = section_cfg["mode"]
    fingerprint = _expected_fingerprint(section_cfg)

    if mode == "remote":
        remote_url = section_cfg.get("remote_url")
        if not remote_url:
            raise ConfigError(
                f'{name}.mode = "remote" requires a remote_url'
            )
        client = WorkerClient(
            base_url=remote_url,
            secret=section_cfg.get("remote_token") or "",
            expected_fingerprint=fingerprint,
        )
        return RemoteWorker(name, client)

    return ManagedWorker(
        name=name,
        command=worker_command(
            role=name,
            backend=section_cfg["backend"],
            model_path=section_cfg["model"],
        ),
        expected_fingerprint=fingerprint,
        backend=section_cfg["backend"],
    )
