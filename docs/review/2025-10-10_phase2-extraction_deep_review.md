# Deep review — Phase 2 extraction + scan (commit 93fb69e)

**Branch:** main · **Base:** b50f195 (RED suites) · **Date:** 2025-10-10
**Scope:** `src/indexedcode/{extract,scan}.py`; 147/147 tests green.

## TL;DR

| # | Severity | Issue | Status |
|---|---|---|---|
| 1 | P2 | `_line_snippet` was O(lines) per relationship (O(n²) on large files) | Fixed (bisect) |
| 2 | P3 | `_emit_class` lacks the `has_error` guard `_emit_function` has | Deferred |
| 3 | P3 | Scan reads 8 KiB head of every non-excluded file before language check | Deferred |

## Notes

1. Fixed in-review: `extract.py` now uses `bisect_right` over the line-start index.
2. A class whose body contains a syntax error is still emitted (with `parse_ok=False` at file level). The Phase 4 publisher must treat `parse_ok=False` as "do not replace a known-good snapshot" per plan §7.2 — tracked for Phase 4 tests.
3. Unsupported-language files (`.md`, `.txt`) are read before being skipped. Harmless at Phase 2 scale; reorder the language check before the binary sniff if scan latency ever matters.

## Correctness spot-checks performed

- Gitignore stack never pops, but rules are scoped by `len(rel_parts) <= base_len`, and top-down walk guarantees deeper visits are descendants — verified against nested + root + directory-pattern tests.
- Symlinked directories are not descended (`followlinks=False`); symlinked files skipped explicitly.
- Byte-coordinate exactness verified for Unicode sources (test suite) and snippet extraction uses byte offsets against the byte index, not string indices.
- Relationship dedup key `(relation, source, target, line)` prevents parser-recovery duplicates.
- Scan is read-only (mtime snapshot test).

## Merge gating

Nothing blocking. All three items are tracked above; #2 belongs in the Phase 4 gate.
