"""The committed indexing-fixture corpus as a reusable module.

Wraps the deterministic generator in ``eval/make_fixture.py`` so tests
and benchmarks can build the same corpus (abstract classes, inheritance
chains, interfaces/traits, arrow functions) at any scale.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_generator():
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "eval" / "make_fixture.py"
    spec = importlib.util.spec_from_file_location(
        "_indexedcode_fixture_generator", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def build_corpus(dest, **scale) -> dict:
    """Write the corpus under ``dest``; returns a per-language file count."""
    return _load_generator().build_corpus(Path(dest), **scale)


def write_zip(zip_path) -> Path:
    """Build the corpus and zip it deterministically."""
    return _load_generator().write_zip(Path(zip_path))
