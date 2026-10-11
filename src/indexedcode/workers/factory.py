"""Worker wiring from validated configuration (plan §4.1)."""

from __future__ import annotations

import os

from indexedcode.workers.adapters import worker_command
from indexedcode.workers.client import WorkerClient
from indexedcode.workers.fingerprint import model_fingerprint
from indexedcode.workers.remote import RemoteWorker
from indexedcode.workers.supervisor import ManagedWorker


def _expected_fingerprint(section_cfg: dict, name: str = "embedding") -> str:
    if section_cfg.get("expected_fingerprint"):
        return section_cfg["expected_fingerprint"]
    if section_cfg["mode"] == "managed":
        from indexedcode.workers.serve import build_model_info

        role = "summary" if name.startswith("summar") else "embedding"
        return model_fingerprint(**build_model_info(
            role, section_cfg["backend"], section_cfg["model"]))
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


def _build_api_worker(name: str, section_cfg: dict):
    from indexedcode.config import ConfigError
    from indexedcode.workers.api import (
        DEFAULT_KEY_ENV,
        ApiEmbeddingClient,
        ApiSummaryClient,
    )

    provider = section_cfg["provider"]
    key_env = section_cfg.get("api_key_env") or DEFAULT_KEY_ENV[provider]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ConfigError(
            f"{name}: set the {key_env} environment variable to your"
            f" {provider} API key")
    common = dict(provider=provider, base_url=section_cfg.get("base_url"),
                  api_key=api_key, model=section_cfg["model"])
    if name.startswith("summar"):
        client = ApiSummaryClient(**common)
    else:
        client = ApiEmbeddingClient(
            dimensions=section_cfg.get("dimensions"), **common)
    return RemoteWorker(name, client)


def build_worker(name: str, section_cfg: dict):
    """Return a ManagedWorker (managed), or a RemoteWorker handle for a
    remote worker endpoint or a hosted API."""
    from indexedcode.config import ConfigError

    mode = section_cfg["mode"]
    if mode == "api":
        return _build_api_worker(name, section_cfg)
    fingerprint = _expected_fingerprint(section_cfg, name)

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
            role="summary" if name.startswith("summar") else "embedding",
            backend=section_cfg["backend"],
            model_path=section_cfg["model"],
        ),
        expected_fingerprint=fingerprint,
        backend=section_cfg["backend"],
    )
