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
    assert "QUEUED   #" in captured.err
    assert "refresh_file src/config_loader.py" in captured.err
    assert "RUNNING  #" in captured.err
    assert "DONE     #" in captured.err
    assert "done: 3 indexed" in captured.err


def test_removed_file_logs_full_job_lifecycle(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "gone.py"
    source.write_text("def gone():\n    pass\n")
    db_path = tmp_path / "i.sqlite"
    main(["index", "once", "--root", str(root), "--db", str(db_path)])
    capsys.readouterr()

    source.unlink()
    main(["index", "once", "--root", str(root), "--db", str(db_path)])
    err = capsys.readouterr().err
    assert "QUEUED   #" in err and "remove_file gone.py" in err
    assert "RUNNING  #" in err and "DONE     #" in err


def test_quiet_suppresses_info_but_keeps_failures(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ok.py").write_text("def a():\n    return 1\n")
    (root / "bad.py").write_text("def b(:\n")
    main(["-q", "index", "once", "--root", str(root),
          "--db", str(tmp_path / "i.sqlite")])
    err = capsys.readouterr().err
    assert "QUEUED" not in err and "RUNNING" not in err and "indexing " not in err
    assert "ERROR" in err and "FAILED" in err and "bad.py" in err
    assert "gave up after 5" in err


def test_retries_are_visible_by_default(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "bad.py").write_text("def b(:\n")
    main(["index", "once", "--root", str(root),
          "--db", str(tmp_path / "i.sqlite")])
    err = capsys.readouterr().err
    assert err.count("RETRY    #") == 4
    assert "refresh_file bad.py" in err
    assert "next attempt in 1 ms" in err
    assert "FAILED   #" in err and "gave up after 5 attempts" in err


def test_verbose_and_quiet_are_mutually_exclusive(capsys):
    import pytest

    with pytest.raises(SystemExit) as excinfo:
        main(["-v", "-q", "status"])
    assert excinfo.value.code == 2
