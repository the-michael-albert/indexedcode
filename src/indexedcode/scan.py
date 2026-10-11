"""Repository scan: safe enumeration with explicit skip reasons (plan §7.1).

Honors nested .gitignore files and default directory excludes, rejects
symlinks, detects binaries, enforces the size cap, and classifies test
files. The scan never writes to the repository.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from pathspec import GitIgnoreSpec

from indexedcode.models import RowModel

DEFAULT_EXCLUDES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "vendor",
        "build",
        "dist",
        "target",
        ".mypy_cache",
        ".pytest_cache",
    }
)

TEST_DIR_NAMES = frozenset({"test", "tests", "__tests__", "spec"})
_TEST_SUFFIXES = (
    "_test.py", "_test.go", "_test.rs", "_test.c", "_test.cpp", "_test.cc",
    "_test.java", "_test.js", "_test.jsx", "_test.ts", "_test.tsx",
)
_TEST_NAME_SUFFIXES = ("Test.java", "Tests.java", "IT.java")
BINARY_SNIFF_BYTES = 8192
DEFAULT_MAX_FILE_BYTES = 1_048_576


class ScannedFile(RowModel):
    relative_path: str
    language: str
    is_test: bool


class SkippedFile(RowModel):
    relative_path: str
    reason: str


class ScanResult(RowModel):
    files: list[ScannedFile] = []
    skipped: list[SkippedFile] = []


def _is_test_file(rel_path: str) -> bool:
    parts = rel_path.split("/")
    name = parts[-1]
    if any(p in TEST_DIR_NAMES for p in parts[:-1]):
        return True
    if name.startswith("test_"):
        return True
    if name.endswith(_TEST_SUFFIXES) or name.endswith(_TEST_NAME_SUFFIXES):
        return True
    if ".test." in name or ".spec." in name:
        return True
    return False


def _gitignored(rel_parts, ignore_stack) -> bool:
    """Apply each .gitignore only beneath its owning directory.

    ``ignore_stack`` holds ``(owner_parts, spec)`` pairs ordered outermost
    first. Within one path the last matching rule wins (so a deeper negation
    overrides an ancestor pattern), and a file under an ignored directory
    stays ignored, as in git.
    """
    for end in range(1, len(rel_parts) + 1):
        is_dir = end < len(rel_parts)
        ignored = False
        for owner, spec in ignore_stack:
            n = len(owner)
            if end <= n or tuple(rel_parts[:n]) != owner:
                continue
            rel = "/".join(rel_parts[n:end]) + ("/" if is_dir else "")
            result = spec.check_file(rel)
            if result.include is not None:
                ignored = result.include
        if ignored:
            return True
    return False


def scan_repository(root, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
                    extra_excludes: tuple[str, ...] = ()) -> ScanResult:
    """Enumerate indexable source files under root (read-only)."""
    from indexedcode.extract import detect_language

    root = Path(root)
    excludes = DEFAULT_EXCLUDES | set(extra_excludes)
    files: list[ScannedFile] = []
    skipped: list[SkippedFile] = []
    ignore_stack: list[tuple[tuple[str, ...], GitIgnoreSpec]] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        abs_dir = Path(dirpath)
        rel_parts = list(abs_dir.relative_to(root).parts)
        dirnames.sort()
        filenames.sort()

        # drop rules owned by directories we have left
        while ignore_stack and tuple(rel_parts[: len(ignore_stack[-1][0])]) != ignore_stack[-1][0]:
            ignore_stack.pop()

        # nested gitignore rules apply only beneath their owning directory;
        # ignore files inside excluded trees never contribute rules
        if ".gitignore" in filenames and not any(p in excludes for p in rel_parts):
            lines = (abs_dir / ".gitignore").read_text().splitlines()
            spec = GitIgnoreSpec.from_lines(
                line for line in lines if line.strip() and not line.startswith("#")
            )
            ignore_stack.append((tuple(rel_parts), spec))

        for filename in filenames:
            file_parts = rel_parts + [filename]
            rel_path = "/".join(file_parts)

            abs_file = abs_dir / filename
            if os.path.islink(abs_file):
                skipped.append(SkippedFile(relative_path=rel_path, reason="symlink"))
                continue
            if any(p in excludes for p in rel_parts):
                skipped.append(SkippedFile(relative_path=rel_path, reason="ignored"))
                continue
            if _gitignored(file_parts, ignore_stack):
                skipped.append(SkippedFile(relative_path=rel_path, reason="ignored"))
                continue
            try:
                size = abs_file.stat().st_size
            except OSError:
                skipped.append(SkippedFile(relative_path=rel_path, reason="unreachable"))
                continue
            if size > max_file_bytes:
                skipped.append(SkippedFile(relative_path=rel_path, reason="oversized"))
                continue
            try:
                with open(abs_file, "rb") as fh:
                    head = fh.read(BINARY_SNIFF_BYTES)
            except OSError:
                # unreadable (e.g. permissions): skip explicitly — a
                # permission error never means the file was deleted
                skipped.append(SkippedFile(relative_path=rel_path, reason="unreachable"))
                continue
            if b"\x00" in head:
                skipped.append(SkippedFile(relative_path=rel_path, reason="binary"))
                continue
            language = detect_language(PurePosixPath(rel_path).name)
            if language is None:
                skipped.append(
                    SkippedFile(relative_path=rel_path, reason="unsupported_language")
                )
                continue
            files.append(
                ScannedFile(
                    relative_path=rel_path,
                    language=language,
                    is_test=_is_test_file(rel_path),
                )
            )

    return ScanResult(files=files, skipped=skipped)
