"""Validated configuration loading (plan §4.1) using stdlib tomllib."""

from __future__ import annotations

import tomllib
from urllib.parse import urlsplit

from indexedcode.workers.adapters import default_backend, default_model
from indexedcode.workers.api import PROVIDERS, default_api_model

MODES = ("managed", "remote", "api")
API_CONCURRENCY = 8
BACKENDS = ("mlx", "cuda", "vulkan", "rocm", "cpu")

DEFAULTS = {
    "embedding": {
        "mode": "managed",
        "backend": default_backend(),
        "model": default_model("embedding", default_backend()),
        "max_batch_items": 32,
        "max_batch_tokens": 4096,
        "max_concurrent_requests": 1,
        "remote_url": None,
        "remote_token": None,
        "provider": None,
        "base_url": None,
        "api_key_env": None,
        "api_key": None,
        "dimensions": None,
    },
    "summarization": {
        "mode": "managed",
        "backend": default_backend(),
        "model": default_model("summary", default_backend()),
        "quantization": "recommended-backend-quantized",
        "max_input_tokens": 4096,
        "max_output_tokens": 256,
        "max_concurrent_requests": 1,
        "max_batch_items": 1,
        "remote_url": None,
        "remote_token": None,
        "provider": None,
        "base_url": None,
        "api_key_env": None,
        "api_key": None,
    },
    "memory": {
        "managed_soft_limit_mb": 2300,
        "managed_hard_limit_mb": 2560,
        "sqlite_cache_kib": 32768,
        "max_file_bytes": 1048576,
    },
}

# The documented managed-local budget profile (plan §4.5): exceeding any of
# these requires an explicit allow_over_budget opt-in.
BUDGET_PROFILE = {
    ("summarization", "max_input_tokens"): 4096,
    ("summarization", "max_output_tokens"): 256,
    ("summarization", "max_concurrent_requests"): 1,
    ("summarization", "max_batch_items"): 1,
    ("embedding", "max_batch_items"): 32,
    ("embedding", "max_batch_tokens"): 4096,
    ("embedding", "max_concurrent_requests"): 1,
}

_POSITIVE_INTS = {
    ("embedding", "max_batch_items"),
    ("embedding", "max_batch_tokens"),
    ("embedding", "max_concurrent_requests"),
    ("summarization", "max_input_tokens"),
    ("summarization", "max_output_tokens"),
    ("summarization", "max_concurrent_requests"),
    ("summarization", "max_batch_items"),
    ("memory", "managed_soft_limit_mb"),
    ("memory", "managed_hard_limit_mb"),
    ("memory", "sqlite_cache_kib"),
    ("memory", "max_file_bytes"),
}


class ConfigError(ValueError):
    """The configuration file is invalid."""


def _secure_url(url) -> bool:
    """https anywhere, or plain http to this machine (no network hop)."""
    parts = urlsplit(str(url))
    return parts.scheme == "https" or (
        parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1"))


def _resolve_api_section(section, cfg, given):
    """Validate a hosted-API section and fill provider-specific defaults."""
    provider = cfg.get("provider")
    if provider not in PROVIDERS:
        raise ConfigError(
            f'{section}.mode = "api" requires provider to be one of'
            f" {PROVIDERS}, got {provider!r}"
        )
    role = "summary" if section == "summarization" else "embedding"
    if "model" not in given:
        model = default_api_model(provider, role)
        if model is None:
            raise ConfigError(
                f"{section}.provider = {provider!r} does not offer"
                f" {role} models"
            )
        cfg["model"] = model
    if "max_concurrent_requests" not in given:
        cfg["max_concurrent_requests"] = API_CONCURRENCY
    base_url = cfg.get("base_url")
    if base_url and not _secure_url(base_url):
        raise ConfigError(
            f"{section}.base_url must use https:// (API keys are sent as"
            " bearer credentials); plain http is allowed for localhost only"
        )


def load_config(path):
    """Load and validate an indexedcode.toml file; returns plain dicts."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    allow_over_budget = bool(raw.get("allow_over_budget", False))
    cfg = {}
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **raw.get(section, {})}
        # an explicit backend without an explicit model gets that
        # backend's default model (Vulkan loads GGUF files)
        given = raw.get(section, {})
        if "backend" in given and "model" not in given and "model" in defaults:
            role = "summary" if section == "summarization" else "embedding"
            cfg[section]["model"] = default_model(role, given["backend"])

    for section in ("embedding", "summarization"):
        if cfg[section]["mode"] not in MODES:
            raise ConfigError(
                f"{section}.mode must be one of {MODES},"
                f" got {cfg[section]['mode']!r}"
            )
        if cfg[section]["mode"] == "api":
            _resolve_api_section(section, cfg[section], raw.get(section, {}))
            continue
        if cfg[section]["backend"] not in BACKENDS:
            raise ConfigError(
                f"{section}.backend must be one of {BACKENDS},"
                f" got {cfg[section]['backend']!r}"
            )
        if (
            cfg[section]["mode"] == "remote"
            and not cfg[section].get("remote_url")
        ):
            raise ConfigError(
                f'{section}.mode = "remote" requires a remote_url'
            )
        remote_url = cfg[section].get("remote_url")
        if remote_url and not _secure_url(remote_url):
            raise ConfigError(
                f"{section}.remote_url must use https://"
                " (bearer authentication requires encrypted transport);"
                " plain http is allowed for localhost only"
            )

    for section, key in _POSITIVE_INTS:
        value = cfg[section][key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(
                f"{section}.{key} must be a positive integer, got {value!r}"
            )

    memory = cfg["memory"]
    if memory["managed_soft_limit_mb"] >= memory["managed_hard_limit_mb"]:
        raise ConfigError(
            "memory.managed_soft_limit_mb must be below"
            " managed_hard_limit_mb"
        )

    if not allow_over_budget:
        for (section, key), limit in BUDGET_PROFILE.items():
            # remote/api inference runs on another host: the managed-local
            # budget profile does not apply and needs no opt-in
            if cfg[section]["mode"] in ("remote", "api"):
                continue
            if cfg[section][key] > limit:
                raise ConfigError(
                    f"{section}.{key}={cfg[section][key]} exceeds the"
                    f" managed-local budget profile ({limit}); set"
                    " allow_over_budget = true to opt in explicitly"
                )

    cfg["allow_over_budget"] = allow_over_budget
    return cfg
