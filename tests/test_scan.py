"""Repository scan contract tests (plan §9 Phase 2 gate): nested
.gitignore behavior, default excludes, symlinks, binaries, oversized files,
unsupported languages, and test-file classification."""

from __future__ import annotations

import os

import pytest

from indexedcode.scan import scan_repository


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    pass\n")
    return tmp_path


def included(result):
    return {f.relative_path: f for f in result.files}


def skipped(result):
    return {s.relative_path: s.reason for s in result.skipped}


def test_includes_supported_source_file(repo):
    r = scan_repository(repo)
    f = included(r)["src/main.py"]
    assert f.language == "python"
    assert f.is_test is False


def test_nested_gitignore_honored(repo):
    (repo / "gen").mkdir()
    (repo / "gen" / ".gitignore").write_text("out.py\n")
    (repo / "gen" / "out.py").write_text("x = 1\n")
    (repo / "gen" / "kept.py").write_text("y = 2\n")
    r = scan_repository(repo)
    assert "gen/out.py" in skipped(r)
    assert skipped(r)["gen/out.py"] == "ignored"
    assert "gen/kept.py" in included(r)


def test_sibling_gitignore_does_not_leak(repo):
    (repo / "a_cache").mkdir()
    (repo / "a_cache" / ".gitignore").write_text("*\n")
    (repo / "a_cache" / "c.py").write_text("x = 1\n")
    r = scan_repository(repo)
    assert "src/main.py" in included(r)
    assert skipped(r)["a_cache/c.py"] == "ignored"


def test_deep_gitignore_does_not_leak_after_return(repo):
    (repo / "a" / "b").mkdir(parents=True)
    (repo / "a" / "b" / ".gitignore").write_text("*\n")
    (repo / "a" / "b" / "x.py").write_text("x = 1\n")
    (repo / "a" / "top.py").write_text("x = 1\n")
    (repo / "z.py").write_text("x = 1\n")
    r = scan_repository(repo)
    assert "a/top.py" in included(r)
    assert "z.py" in included(r)
    assert "a/b/x.py" in skipped(r)


def test_gitignore_in_excluded_dir_has_no_effect(repo):
    (repo / "build").mkdir()
    (repo / "build" / ".gitignore").write_text("*\n")
    (repo / "build" / "o.py").write_text("x = 1\n")
    r = scan_repository(repo)
    assert skipped(r)["build/o.py"] == "ignored"
    assert "src/main.py" in included(r)


def test_deeper_negation_overrides_ancestor(repo):
    (repo / ".gitignore").write_text("*.gen.py\n")
    (repo / "src" / ".gitignore").write_text("!keep.gen.py\n")
    (repo / "src" / "keep.gen.py").write_text("x = 1\n")
    (repo / "src" / "drop.gen.py").write_text("x = 1\n")
    r = scan_repository(repo)
    assert "src/keep.gen.py" in included(r)
    assert skipped(r)["src/drop.gen.py"] == "ignored"


def test_root_gitignore_honored(repo):
    (repo / ".gitignore").write_text("*.log\nscratch/\n")
    (repo / "debug.log").write_text("x")
    (repo / "scratch").mkdir()
    (repo / "scratch" / "s.py").write_text("x = 1\n")
    r = scan_repository(repo)
    assert skipped(r)["debug.log"] == "ignored"
    assert skipped(r)["scratch/s.py"] == "ignored"


def test_default_excludes(repo):
    for d in (".git", "node_modules", "__pycache__", "vendor", "build",
              "dist", "target"):
        (repo / d).mkdir(exist_ok=True)
        (repo / d / "x.py").write_text("x = 1\n")
    r = scan_repository(repo)
    for d in (".git", "node_modules", "__pycache__", "vendor", "build",
              "dist", "target"):
        assert skipped(r)[f"{d}/x.py"] == "ignored"


def test_symlink_rejected(repo):
    target = repo / "src" / "main.py"
    (repo / "link.py").symlink_to(target)
    (repo / "src" / "loop").symlink_to(repo / "src", target_is_directory=True)
    r = scan_repository(repo)
    assert skipped(r)["link.py"] == "symlink"
    assert "src/loop/main.py" not in included(r)


def test_binary_file_skipped(repo):
    (repo / "blob.py").write_bytes(b"\x00\x01\x02python\x00")
    r = scan_repository(repo)
    assert skipped(r)["blob.py"] == "binary"


def test_oversized_file_skipped(repo):
    (repo / "big.py").write_bytes(b"# padding\n" * 10)
    r = scan_repository(repo, max_file_bytes=20)
    assert skipped(r)["big.py"] == "oversized"


def test_unsupported_language_skipped_with_reason(repo):
    (repo / "notes.txt").write_text("hello")
    (repo / "README.md").write_text("# hi")
    r = scan_repository(repo)
    assert skipped(r)["notes.txt"] == "unsupported_language"
    assert skipped(r)["README.md"] == "unsupported_language"


@pytest.mark.parametrize(
    ("path", "is_test"),
    [
        ("tests/test_foo.py", True),
        ("pkg/test_foo.py", True),
        ("pkg/foo_test.py", True),
        ("pkg/foo_test.go", True),
        ("pkg/foo.test.ts", True),
        ("pkg/foo.test.tsx", True),
        ("pkg/FooTest.java", True),
        ("pkg/main.py", False),
        ("pkg/main.go", False),
    ],
)
def test_test_file_classification(repo, path, is_test):
    full = repo / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("x = 1\n")
    r = scan_repository(repo)
    if is_test:
        assert included(r)[path].is_test is True
    else:
        assert included(r)[path].is_test is False


def test_paths_are_normalized_relative_posix(repo):
    (repo / "src" / "deep").mkdir()
    (repo / "src" / "deep" / "a.py").write_text("x = 1\n")
    r = scan_repository(repo)
    assert "src/deep/a.py" in included(r)


def test_scan_does_not_modify_repository(repo):
    before = sorted((str(p.relative_to(repo)), p.stat().st_mtime_ns)
                    for p in repo.rglob("*"))
    scan_repository(repo)
    after = sorted((str(p.relative_to(repo)), p.stat().st_mtime_ns)
                   for p in repo.rglob("*"))
    assert before == after
