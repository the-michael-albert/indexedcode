"""CLI adapters (plan §6): index once, search/context/references/status,
and an orchestrator-serve boot/shutdown smoke test."""

import json
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

from indexedcode.cli import main

from tests.conftest import free_port, write_repo_sources, wait_until


def test_index_once_then_queries(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    write_repo_sources(root)
    db = tmp_path / "index.sqlite"

    rc = main([
        "index", "once", "--root", str(root), "--db", str(db),
    ])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["files_ready"] == 3

    rc = main(["search", "parse_config", "--db", str(db), "--root", str(root)])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["results"][0]["qualified_name"] == "parse_config"

    rc = main([
        "context", "src/config_loader.py", "--db", str(db),
        "--root", str(root),
    ])
    assert rc == 0
    context = json.loads(capsys.readouterr().out)
    assert "parse_config" in context["source"]

    rc = main([
        "references", "parse_config", "--db", str(db), "--root", str(root),
    ])
    assert rc == 0
    references = json.loads(capsys.readouterr().out)
    assert references["references"]

    rc = main(["status", "--db", str(db), "--root", str(root)])
    assert rc == 0
    status = json.loads(capsys.readouterr().out)
    assert status["coverage"]["ready_files"] == 3


def test_index_once_is_idempotent(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    write_repo_sources(root)
    db = tmp_path / "index.sqlite"
    main(["index", "once", "--root", str(root), "--db", str(db)])
    capsys.readouterr()
    rc = main(["index", "once", "--root", str(root), "--db", str(db)])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["files_refreshed"] == 0  # nothing changed


def test_version_uses_package_metadata(monkeypatch, capsys):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.2.3")
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out == "indexedcode 1.2.3\n"


def test_module_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "indexedcode", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "orchestrator" in result.stdout


def test_unknown_command_exits_nonzero(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate", "--db", str(tmp_path / "x.sqlite")])
    assert excinfo.value.code != 0


def test_serve_boots_reconciles_and_shuts_down(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    write_repo_sources(root)
    db = tmp_path / "index.sqlite"
    port = free_port()

    proc = subprocess.Popen(
        [sys.executable, "-m", "indexedcode.cli", "orchestrator", "serve",
         "--root", str(root), "--db", str(db), "--http-port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    try:
        url = f"http://127.0.0.1:{port}/v1/status"
        assert wait_until(lambda: _try_get(url), timeout=30), (
            "orchestrator serve never answered /v1/status"
        )
        body = _try_get(url)
        assert body["coverage"]["ready_files"] == 3
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise AssertionError("serve did not exit on SIGTERM")


def _try_get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read())
    except Exception:
        return None
