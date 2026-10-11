from pathlib import Path

import pytest

from indexedcode.setup_ui import (
    SemanticSettings,
    StageSettings,
    default_db_path,
    load_semantic_settings,
    render_config,
    validate_semantic,
    validate_setup,
    write_config,
)


def test_default_db_path_is_stable_and_outside_repository(tmp_path, monkeypatch):
    root = tmp_path / "my repo"
    root.mkdir()
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    first = default_db_path(root)
    second = default_db_path(root)

    assert first == second
    assert first.name == "index.sqlite"
    assert first.is_relative_to(data_home / "indexedcode")
    assert not first.is_relative_to(root)


def test_default_db_paths_distinguish_same_named_repositories(tmp_path, monkeypatch):
    first = tmp_path / "one" / "repo"
    second = tmp_path / "two" / "repo"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    assert default_db_path(first) != default_db_path(second)


def test_validate_setup_normalizes_values(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    db = tmp_path / "state" / "index.sqlite"

    assert validate_setup(str(root), str(db), "0") == (
        root.resolve(), db.resolve(), 0,
    )


@pytest.mark.parametrize("port", ["-1", "65536", "nope"])
def test_validate_setup_rejects_invalid_port(tmp_path, port):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError, match="port"):
        validate_setup(str(root), str(tmp_path / "index.sqlite"), port)


def test_validate_setup_rejects_database_inside_repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        validate_setup(str(root), str(root / "index.sqlite"), "8765")


def test_setup_command_delegates_selection_to_serve(monkeypatch):
    from indexedcode import cli

    monkeypatch.setattr(
        "indexedcode.setup_ui.launch_setup",
        lambda **kwargs: ("/repo", "/state/index.sqlite", 9123,
                          SemanticSettings(enabled=True, config_path=None)),
    )
    seen = {}

    def serve(args):
        seen.update(root=args.root, db=args.db, port=args.http_port)
        return 17

    monkeypatch.setattr(cli, "_cmd_serve", serve)
    assert cli.main(["setup"]) == 17
    assert seen == {"root": "/repo", "db": "/state/index.sqlite", "port": 9123}


def test_setup_command_cancel_is_success(monkeypatch):
    from indexedcode import cli

    monkeypatch.setattr("indexedcode.setup_ui.launch_setup", lambda **kwargs: None)
    assert cli.main(["setup"]) == 0


def test_setup_app_accepts_valid_values(tmp_path):
    import asyncio

    from indexedcode.setup_ui import SetupApp
    from textual.widgets import Input

    root = tmp_path / "repo"
    root.mkdir()
    db = tmp_path / "state" / "index.sqlite"
    app = SetupApp(root=str(root), db=str(db), port=8123)

    async def exercise():
        async with app.run_test(size=(120, 50)) as pilot:
            app.query_one("#root", Input).value = str(root)
            app.query_one("#db", Input).value = str(db)
            app.query_one("#port", Input).value = "8123"
            await pilot.click("#start")

    asyncio.run(exercise())
    selected = app.return_value
    assert selected[:3] == (str(root.resolve()), str(db.resolve()), 8123)
    assert selected[3].enabled and selected[3].config_path is None
    assert db.parent.is_dir()


def _api_settings():
    settings = SemanticSettings()
    settings.stages["summarization"] = StageSettings(
        mode="api", provider="anthropic", api_key="sk-secret")
    settings.stages["embedding"] = StageSettings(
        mode="remote", url="http://127.0.0.1:8802", api_key="tok",
        parallel="4")
    return settings


def test_render_config_round_trips_through_the_loader(tmp_path):
    from indexedcode.config import load_config

    settings = _api_settings()
    path = tmp_path / "indexedcode.toml"
    assert write_config(path, settings) == str(path)
    assert path.stat().st_mode & 0o777 == 0o600
    cfg = load_config(path)
    assert cfg["summarization"]["mode"] == "api"
    assert cfg["summarization"]["api_key"] == "sk-secret"
    assert cfg["embedding"]["remote_url"] == "http://127.0.0.1:8802"
    assert cfg["embedding"]["remote_token"] == "tok"
    assert cfg["embedding"]["max_concurrent_requests"] == 4
    reloaded = load_semantic_settings(path)
    assert reloaded.stages["summarization"].provider == "anthropic"
    assert reloaded.stages["embedding"].url == "http://127.0.0.1:8802"


def test_all_local_writes_nothing_and_removes_stale_file(tmp_path):
    path = tmp_path / "indexedcode.toml"
    path.write_text("[embedding]\n")
    assert render_config(SemanticSettings()) is None
    assert write_config(path, SemanticSettings()) is None
    assert not path.exists()


def test_validate_semantic_reports_loader_errors(tmp_path):
    settings = SemanticSettings()
    settings.stages["embedding"] = StageSettings(
        mode="api", provider="anthropic")  # anthropic has no embeddings
    with pytest.raises(ValueError, match="embedding models"):
        validate_semantic(settings, tmp_path)
    settings.stages["embedding"] = StageSettings(mode="remote")
    with pytest.raises(ValueError, match="URL"):
        validate_semantic(settings, tmp_path)
    settings.stages["embedding"] = StageSettings(
        mode="remote", url="http://example.com")
    with pytest.raises(ValueError, match="https"):
        validate_semantic(settings, tmp_path)


def test_setup_command_applies_semantic_settings(monkeypatch):
    from indexedcode import cli

    monkeypatch.setattr(
        "indexedcode.setup_ui.launch_setup",
        lambda **kwargs: ("/r", "/s/index.sqlite", 1,
                          SemanticSettings(enabled=True,
                                           config_path="/s/indexedcode.toml")),
    )
    seen = {}
    monkeypatch.setattr(cli, "_cmd_serve", lambda a: seen.update(
        config=a.config, embeddings=a.embeddings) or 0)
    assert cli.main(["setup"]) == 0
    assert seen == {"config": "/s/indexedcode.toml", "embeddings": True}

    monkeypatch.setattr(
        "indexedcode.setup_ui.launch_setup",
        lambda **kwargs: ("/r", "/s/index.sqlite", 1,
                          SemanticSettings(enabled=False)),
    )
    assert cli.main(["setup"]) == 0
    assert seen["embeddings"] is False


def test_setup_app_saves_api_settings(tmp_path):
    import asyncio

    from indexedcode.setup_ui import SetupApp
    from textual.widgets import Input, Select

    root = tmp_path / "repo"
    root.mkdir()
    db = tmp_path / "state" / "index.sqlite"
    app = SetupApp(root=str(root), db=str(db), port=8123)

    async def exercise():
        async with app.run_test(size=(120, 60)) as pilot:
            app.query_one("#summarization-mode", Select).value = "api"
            app.query_one("#summarization-provider", Select).value = (
                "anthropic")
            await pilot.pause()
            app.query_one("#summarization-key", Input).value = "sk-1"
            await pilot.click("#start")

    asyncio.run(exercise())
    semantic = app.return_value[3]
    assert semantic.config_path == str(db.parent / "indexedcode.toml")
    assert "sk-1" in (db.parent / "indexedcode.toml").read_text()
    # reopening prefills from the saved file
    again = SetupApp(root=str(root), db=str(db), port=8123)
    assert again.initial_semantic.stages["summarization"].api_key == "sk-1"
