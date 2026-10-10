"""Validated configuration loading (plan §4.1) using stdlib tomllib."""

from __future__ import annotations

import tomllib

MODES = ("managed", "remote")
BACKENDS = ("mlx", "rocm")

DEFAULTS = {
    "embedding": {
        "mode": "managed",
        "backend": "mlx",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "max_batch_items": 32,
        "max_batch_tokens": 4096,
        "max_concurrent_requests": 1,
        "remote_url": None,
        "remote_token": None,
    },
    "summarization": {
        "mode": "managed",
        "backend": "mlx",
        "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "quantization": "recommended-backend-quantized",
        "max_input_tokens": 4096,
        "max_output_tokens": 256,
        "max_concurrent_requests": 1,
        "max_batch_items": 1,
        "remote_url": None,
        "remote_token": None,
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


def load_config(path):
    """Load and validate an indexedcode.toml file; returns plain dicts."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    allow_over_budget = bool(raw.get("allow_over_budget", False))
    cfg = {}
    for section, defaults in DEFAULTS.items():
        cfg[section] = {**defaults, **raw.get(section, {})}

    for section in ("embedding", "summarization"):
        if cfg[section]["mode"] not in MODES:
            raise ConfigError(
                f"{section}.mode must be one of {MODES},"
                f" got {cfg[section]['mode']!r}"
            )
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
        if remote_url and not str(remote_url).startswith("https://"):
            raise ConfigError(
                f"{section}.remote_url must use https://"
                " (bearer authentication requires encrypted transport)"
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
            # remote inference runs on another host: the managed-local
            # budget profile does not apply and needs no opt-in
            if cfg[section]["mode"] == "remote":
                continue
            if cfg[section][key] > limit:
                raise ConfigError(
                    f"{section}.{key}={cfg[section][key]} exceeds the"
                    f" managed-local budget profile ({limit}); set"
                    " allow_over_budget = true to opt in explicitly"
                )

    cfg["allow_over_budget"] = allow_over_budget
    return cfg
