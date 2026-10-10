from pathlib import Path

import pytest

from indexedcode.setup_ui import default_db_path, validate_setup


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
        lambda **kwargs: ("/repo", "/state/index.sqlite", 9123),
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
    assert app.return_value == (str(root.resolve()), str(db.resolve()), 8123)
    assert db.parent.is_dir()
