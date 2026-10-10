"""CLI logging: records go to stderr, stdout stays machine-readable."""

import json
import logging

from indexedcode.cli import main
from indexedcode.logs import configure_logging, level_for

from tests.conftest import write_repo_sources


def test_level_for_flags():
    assert level_for() == logging.INFO
    assert level_for(verbose=1) == logging.DEBUG
    assert level_for(quiet=True) == logging.WARNING


def test_configure_logging_is_idempotent():
    for _ in range(3):
        configure_logging()
    handlers = [h for h in logging.getLogger("indexedcode").handlers
                if getattr(h, "_indexedcode_cli", False)]
    assert len(handlers) == 1


def test_index_once_logs_to_stderr_and_keeps_stdout_json(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    write_repo_sources(root)
    assert main(["index", "once", "--root", str(root),
                 "--db", str(tmp_path / "i.sqlite")]) == 0
    captured = capsys.readouterr()
    json.loads(captured.out)                         # stdout is pure JSON
    assert "indexed src/config_loader.py" in captured.err
    assert "done: 3 indexed" in captured.err


def test_quiet_suppresses_info_but_keeps_failures(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ok.py").write_text("def a():\n    return 1\n")
    (root / "bad.py").write_text("def b(:\n")
    main(["-q", "index", "once", "--root", str(root),
          "--db", str(tmp_path / "i.sqlite")])
    err = capsys.readouterr().err
    assert "indexed ok.py" not in err and "indexing " not in err
    assert "ERROR" in err and "bad.py" in err and "gave up after 5" in err


def test_verbose_shows_each_retry(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "bad.py").write_text("def b(:\n")
    main(["-v", "index", "once", "--root", str(root),
          "--db", str(tmp_path / "i.sqlite")])
    err = capsys.readouterr().err
    assert "retry bad.py" in err and "attempt 1/5" in err
    main(["index", "once", "--root", str(root),
          "--db", str(tmp_path / "j.sqlite")])
    assert "retry bad.py" not in capsys.readouterr().err   # hidden by default


def test_verbose_and_quiet_are_mutually_exclusive(capsys):
    import pytest

    with pytest.raises(SystemExit) as excinfo:
        main(["-v", "-q", "status"])
    assert excinfo.value.code == 2
