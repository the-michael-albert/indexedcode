"""Phase 3 configuration additions: remote URLs and remote budget scope."""

import pytest


def write(tmp_path, text):
    path = tmp_path / "indexedcode.toml"
    path.write_text(text)
    return path


def test_remote_mode_requires_remote_url(tmp_path):
    from indexedcode.config import ConfigError, load_config

    path = write(tmp_path, """
[embedding]
mode = "remote"
backend = "rocm"
# no remote_url
""")
    with pytest.raises(ConfigError, match="remote_url"):
        load_config(path)


def test_remote_mode_with_url_loads(tmp_path):
    from indexedcode.config import load_config

    path = write(tmp_path, """
[embedding]
mode = "remote"
backend = "rocm"
remote_url = "https://amd.internal"
[summarization]
mode = "managed"
backend = "mlx"
""")
    cfg = load_config(path)
    assert cfg["embedding"]["remote_url"] == "https://amd.internal"
    assert cfg["summarization"]["remote_url"] is None


def test_remote_section_exempts_from_local_budget_gate(tmp_path):
    """Over-budget settings are a managed-local concern: remote inference
    runs on another host, so they need no allow_over_budget opt-in."""
    from indexedcode.config import load_config

    path = write(tmp_path, """
[summarization]
mode = "remote"
backend = "rocm"
remote_url = "https://amd.internal"
max_input_tokens = 8192
max_output_tokens = 512
""")
    cfg = load_config(path)  # must not raise
    assert cfg["summarization"]["max_input_tokens"] == 8192


def test_managed_section_over_budget_still_requires_opt_in(tmp_path):
    from indexedcode.config import ConfigError, load_config

    path = write(tmp_path, """
[summarization]
mode = "managed"
backend = "mlx"
max_output_tokens = 512
""")
    with pytest.raises(ConfigError, match="allow_over_budget"):
        load_config(path)


def test_remote_worker_token_is_accepted_and_defaults_to_none(tmp_path):
    from indexedcode.config import load_config

    path = write(tmp_path, """
[embedding]
mode = "remote"
backend = "rocm"
remote_url = "https://w.internal"
remote_token = "tok"
""")
    cfg = load_config(path)
    assert cfg["embedding"]["remote_token"] == "tok"
    assert cfg["summarization"]["remote_token"] is None
