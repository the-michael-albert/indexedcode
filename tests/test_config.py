"""Validated configuration tests (plan §4.1 config shape)."""

from __future__ import annotations

import pytest

from indexedcode.config import ConfigError, load_config

VALID_TOML = """
[embedding]
mode = "managed"
backend = "mlx"
model = "sentence-transformers/all-MiniLM-L6-v2"
max_batch_items = 32
max_batch_tokens = 4096
max_concurrent_requests = 1

[summarization]
mode = "managed"
backend = "mlx"
model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
quantization = "recommended-backend-quantized"
max_input_tokens = 4096
max_output_tokens = 256
max_concurrent_requests = 1
max_batch_items = 1

[memory]
managed_soft_limit_mb = 2300
managed_hard_limit_mb = 2560
sqlite_cache_kib = 32768
max_file_bytes = 1048576
"""


def write_config(tmp_path, content):
    path = tmp_path / "indexedcode.toml"
    path.write_text(content)
    return path


def test_valid_config_loads(tmp_path):
    cfg = load_config(write_config(tmp_path, VALID_TOML))
    assert cfg["embedding"]["mode"] == "managed"
    assert cfg["summarization"]["max_output_tokens"] == 256
    assert cfg["memory"]["managed_soft_limit_mb"] == 2300


def test_defaults_applied_for_omitted_sections(tmp_path):
    path = write_config(tmp_path, "")
    cfg = load_config(path)
    assert cfg["memory"]["managed_soft_limit_mb"] == 2300
    assert cfg["memory"]["managed_hard_limit_mb"] == 2560


def test_unknown_mode_rejected(tmp_path):
    content = VALID_TOML.replace('mode = "managed"', 'mode = "embedded"')
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, content))


def test_unknown_backend_rejected(tmp_path):
    content = VALID_TOML.replace('backend = "mlx"', 'backend = "cuda"')
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, content))


def test_soft_limit_must_be_below_hard_limit(tmp_path):
    content = VALID_TOML.replace(
        "managed_soft_limit_mb = 2300", "managed_soft_limit_mb = 3000"
    )
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, content))


def test_over_budget_profile_requires_explicit_opt_in(tmp_path):
    content = VALID_TOML.replace("max_output_tokens = 256", "max_output_tokens = 2048")
    with pytest.raises(ConfigError, match="allow_over_budget"):
        load_config(write_config(tmp_path, content))
    content = "allow_over_budget = true\n\n" + VALID_TOML.replace(
        "max_output_tokens = 256", "max_output_tokens = 2048"
    )
    cfg = load_config(write_config(tmp_path, content))
    assert cfg["summarization"]["max_output_tokens"] == 2048


def test_concurrency_over_one_requires_opt_in(tmp_path):
    content = VALID_TOML.replace(
        """[summarization]
mode = "managed"
backend = "mlx"
model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
quantization = "recommended-backend-quantized"
max_input_tokens = 4096
max_output_tokens = 256
max_concurrent_requests = 1""",
        """[summarization]
mode = "managed"
backend = "mlx"
model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
quantization = "recommended-backend-quantized"
max_input_tokens = 4096
max_output_tokens = 256
max_concurrent_requests = 4""",
    )
    with pytest.raises(ConfigError, match="allow_over_budget"):
        load_config(write_config(tmp_path, content))


def test_nonpositive_limits_rejected(tmp_path):
    content = VALID_TOML.replace("max_input_tokens = 4096", "max_input_tokens = 0")
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, content))
