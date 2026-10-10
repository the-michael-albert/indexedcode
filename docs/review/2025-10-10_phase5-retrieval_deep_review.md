# Deep review — Phase 5: retrieval, MCP, CLI, and skill

- **Scope:** commits `ba2a2aa` (RED suite) → `b383c2c` (implementation), plus review fixes applied during this review.
- **Companion docs:** [Phase 1](2025-10-10_phase1-store_deep_review.md), [Phase 2](2025-10-10_phase2-extraction_deep_review.md), [Phase 3](2025-10-10_phase3-workers_deep_review.md), [Phase 4](2025-10-10_phase4-publication_deep_review.md).
- **Verification:** `.venv/bin/pytest` → 303 passed, 5 skipped (4 hardware smoke, 1 real-embedding semantic quality — opt-in, unverified on this host by design).

## Architecture orientation

Phase 5 adds the read plane over the Phase 1–4 write plane: one retrieval
core (`retrieval.py`) with three adapters (HTTP loopback server, MCP stdio
server, argparse CLI) and a skill document. All four tools are read-only
and bounded. Freshness is enforced exactly as the plan requires: indexed
source is only "current" when the on-disk hash matches
`files.indexed_hash`.

## File map

| File | Role |
|---|---|
| `retrieval.py` | search/context/references/status; cursors; freshness; bounds |
| `http_api.py` | loopback HTTP adapter; bearer auth; payload caps |
| `mcp.py` | FastMCP stdio server (`python -m indexedcode.mcp`) |
| `cli.py` | `orchestrator serve`, `index once`, search/context/references/status |
| `db.py` | `connect(threaded=True)` for server worker threads |
| `pipeline.py` | exported `run_until_idle` (shared by CLI + serve) |
| `skills/indexedcode/SKILL.md` | agent skill; untrusted-content rule |

## Issues

### Fixed during this review

- **P2 — negative `start_line` wraparound** ([retrieval.py:308]()):
  `offset_line = start_line or 1` let a request like `start_line=-5`
  index Python slices from the end, silently returning the wrong window.
  Now clamped to ≥ 1.
- **Test gap — non-loopback bind rule untested**: `create_server`
  refuses a token-less non-loopback bind ([http_api.py:23]()); now
  pinned by `test_non_loopback_bind_requires_a_token`.

### P3 — advisory (documented, not fixed)

1. **Semantic tier is a full table scan per query**: every
   `entity_index` blob is unpacked and scored in Python. This is the
   plan's deliberate "exact blob-backed similarity at initial scale"
   (§5.3); sqlite-vec is the sanctioned upgrade after measurement.
   Also true of `get_status`, which hashes every ready file per call.
2. **`orchestrator serve` watches its own database**: watchfiles events
   on `index.sqlite`/WAL files trigger reconcile no-ops. Wasteful, not
   incorrect (scan skips them as unsupported languages).
3. **CLI queries without an embedding model**: `index once` builds no
   `entity_index` rows, so CLI search is symbol-tier-only until real
   workers are wired into `serve` (planned; see Phase 6 examples).
4. **Bearer comparison uses `==`** (non-constant-time) — acceptable on
   loopback; relevant only for remote deployments behind TLS.

## What's correct and worth keeping

- **A real MCP client** (mcp SDK 1.x, pinned `<2` in pyproject) lists
  and calls all four tools over stdio, including the oversized-query
  tool-error path — not a mock.
- **Freshness is honest**: entity context serves the last indexed
  snapshot marked `stale` after a disk edit; explicit path reads return
  current disk source with `origin: "disk"`; status lists stale paths
  by the same hash comparison.
- **Cursors invalidate, never corrupt**: bound to query + a
  sum-of-timestamps index version; any publish or in-place update
  raises `StaleCursorError`, and a dedicated test mutates one row to
  prove `MAX`-based versions would have missed it.
- **Auth and limits are enforced at the edges**: 401 on missing/wrong
  token, 413 before reading oversized bodies, responses bounded with an
  explicit `truncated` flag, path traversal and symlink escapes
  rejected (including intermediate-directory symlinks via resolve
  containment).
- **Degradation is explicit**: semantic mode without an embedder fails
  with a clear message; hybrid flags `semantic_unavailable` and still
  answers lexically — exactly the plan's contract.
- **`serve` is a real integration**: subprocess boots, reconciles and
  indexes a repo, answers `/v1/status` with coverage, and exits on
  SIGTERM within grace — under WriterLock, with the Phase 4 watcher and
  pipeline doing the work.

## Merge gating

| Severity | Issue | Action |
|---|---|---|
| Must fix | none remaining | — |
| Should fix | none remaining | — |
| Defer | P3 #1–#4 | tracked here for Phase 6 measurement/examples |
