# Deep review — Phase 6: operations and evaluation

- **Scope:** commits `a2cce71` (RED suite + fixture zip) → `3f3457b`
  (implementation), plus the review refactor applied during this review.
- **Companion docs:** [Phase 1](2025-10-10_phase1-store_deep_review.md),
  [2](2025-10-10_phase2-extraction_deep_review.md),
  [3](2025-10-10_phase3-workers_deep_review.md),
  [4](2025-10-10_phase4-publication_deep_review.md),
  [5](2025-10-10_phase5-retrieval_deep_review.md).
- **Verification:** `.venv/bin/pytest` → 318 passed, 5 skipped;
  `compileall` clean; `ruff --select F` clean (remaining ruff
  categories are deliberate patterns: blind-excepts on best-effort
  metric appends per plan §5.4, pydantic mutable defaults).

## Measured results (this machine, Apple Silicon, fake deterministic
embeddings — no MLX hardware claim)

Benchmark: `run_benchmarks` on a generated 2,151-file /
11,402-function corpus:

| Measurement | Result | Plan target |
|---|---|---|
| Initial throughput | 638 files/s (2,151 files in 3.4 s) | reported |
| Warm symbol lookup p50 | 2.4 ms | < 100 ms (excl. embedding gen) |
| Warm vector lookup p50 | 156 ms | blob-scan; sqlite-vec is the sanctioned upgrade |
| Update inference, unchanged content | **0 calls** | no needless inference |
| Update inference, one changed function | exactly 1 call | changed-only |
| Idle watcher poll | 0.012 ms | reported |

The managed-local 1–2.5 GB memory profile is **unverified on this
host**: MLX/ROCm smoke gates are opt-in
(`INDEXEDCODE_SMOKE_MLX`/`_ROCM`) and skipped, per the gate's
"missing hardware is reported unverified, never passed".

## Issues

### Fixed during this review

- **P2 — fixture module was not installable** (`src/indexedcode/fixture.py`
  originally loaded `eval/make_fixture.py` by a repo-relative path via
  importlib, which breaks under `pip install` — site-packages has no
  `eval/`). The generator (templates + build/zip) now lives entirely in
  the package; `eval/make_fixture.py` is a thin wrapper, and the
  committed zip is regenerated through it.
- **P3 — benchmark mutated a C++ file with Python content**: the update
  measurement selected the first ready file by id (C++), whose parse
  failure measured retries instead of re-inference. Now selects a
  Python file; `inference_calls_on_changed` went 0 → 1.

### P3 — advisory (documented, not fixed)

1. **Vector lookup is O(entities) per query** (155 ms p50 at 11.4k
   functions): the plan's exact blob-backed similarity until
   measurement justifies sqlite-vec; this measurement is now on
   record as the justification trigger.
2. **`ruff` is not pinned as a dev dependency** — the gate sweep was
   run ad hoc. Adding `[tool.ruff]` configuration is a packaging
   decision left to the maintainer.
3. **The evaluation of agent usefulness** (plan §9: fixed
   locate/behavior/debug/test tasks, retrieval-call counts, citation
   currency) requires live agents and real embeddings; only the
   opt-in semantic smoke test exists. Unverified, not claimed.

## What's correct and worth keeping

- **The fixture corpus is deterministic and extractor-verified**: same
  bytes every build (tree-hash pinned by test), 7 languages, 16+
  abstract/base classes, 39 inheritance edges, cross-file test callers —
  and it round-trips through the committed zip.
- **The benchmark harness runs the real pipeline** (reconcile →
  run_until_idle → retrieval) — no mocking of the system under
  measurement — and its own behavior is pinned by tests, including
  "unchanged content infers nothing" and "idle system does no work".
- **Ops docs are pinned by tests**: MLX local profile with the 1–2.5 GB
  measured-target language, ROCm remote with mandatory https + refusal
  semantics, launchd/systemd units honoring graceful SIGTERM, and
  whole-set-only backup/rebuild/rollback with fingerprint-era guidance.

## Merge gating

| Severity | Issue | Action |
|---|---|---|
| Must fix | none remaining | — |
| Should fix | none remaining | — |
| Defer | P3 #1–#3 | tracked here |
