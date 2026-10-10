# Deep review — Phase 1 store (commit 21d7cd5)

**Branch:** main · **Base:** 72f86d0 (tests only) · **Date:** 2025-10-10
**Scope:** `src/indexedcode/{db,dal,config,embeddings}.py` + two corrected tests (70/70 green, `.venv/bin/pytest`)

## TL;DR

| # | Severity | Issue |
|---|---|---|
| 1 | **P1** | Success job-attempt metrics can never persist: `jobs` deletion and `metrics` FK/CHECK are mutually exclusive in SQLite. Needs a semantics decision before Phase 4. |
| 2 | P2 | `claim_next_job` SELECT-then-UPDATE is not atomic across connections; safe only under the single-writer invariant the plan mandates. |
| 3 | P2 | `enqueue_job`/`claim_next_job` run outside explicit transactions; same single-writer assumption. |
| 4 | P3 | Over-budget gate applies budget-profile limits even in `remote` mode where local memory is unaffected. |
| 5 | P3 | `dal.py` imports `sqlite3` without using it (docstring-only reference). |

## Finding 1 — success job metrics are structurally dropped

`dal.publish_file_refresh` calls `complete_job` (DELETEs the `jobs` row) inside the publication transaction, then appends the success metric after COMMIT (`src/indexedcode/dal.py:504`, metric block below it). With `PRAGMA foreign_keys=ON`, inserting `metric_kind='job'` with `job_id` pointing at the deleted row violates the FK; inserting with `job_id=NULL` violates the kind CHECK. The exception is swallowed, so success rows are silently never written — contradicting plan §5.4 ("one row whenever a claimed job attempt reaches a terminal disposition: success, ...").

Verified empirically: the plan's own DDL combination is internally inconsistent — `ON DELETE SET NULL` updates the metrics row, SQLite re-validates the CHECK (`kind='job' AND job_id IS NOT NULL`), and the *job deletion itself* fails once any job metric exists. So neither ordering works:

```
delete ok -> metric insert FK failure (current code, metric lost)
metric insert ok -> job delete CHECK failure (reordered code, publication breaks)
```

**Fix shape (needs a decision):** either (a) retain terminal jobs — add a `done` status to the `jobs` CHECK, filter claims to `pending/retry`, and keep rows for metrics/history; or (b) accept that success metrics omit `job_id` by relaxing the kind CHECK. Option (a) matches the plan's FK design intent.

## Finding 2/3 — claim and enqueue atomicity

`dal.claim_next_job` (`dal.py:113`) SELECTs the next due id then UPDATEs by id with no `BEGIN IMMEDIATE`; two connections could claim the same job (double lease, double attempt_count). Same pattern in `enqueue_job` coalescing. The plan's single-writer + OS `WriterLock` invariant makes this unreachable in production; the WAL busy-timeout does not serialize the read-modify-write. Acceptable for Phase 1; wrap both in `BEGIN IMMEDIATE` when the Phase 4 worker arrives.

## Finding 4 — remote mode over-budget gate

`config.BUDGET_PROFILE` (`config.py:50`) rejects e.g. `max_concurrent_requests > 1` without `allow_over_budget` even when `mode = "remote"`, where the memory stays on the remote host. Plan §4.5 scopes the budget to the managed-local profile. Advisory; tighten in Phase 3 when worker config lands.

## What is correct and worth keeping

- Publication transaction (`publish_file_refresh`) correctly deletes prior snapshot before inserting, rolls back fully on failure, and readers never see partial snapshots (tests prove it).
- `complete_job` enforces lease token AND desired revision before disposition — the plan's stale-publication guard.
- Metrics `detail_json` whitelist is a single frozen constant enforced at the single write boundary.
- `WriterLock` is OS-enforced (flock), verified across processes by test.
- Embedding validation (dimension/finite/nonzero/normalized) matches plan §5.2 exactly.

## Merge gating

| Gate | Issue | Action |
|---|---|---|
| Should fix before Phase 4 | #1 | Decide job retention vs metric FK semantics; implement + test |
| Defer | #2, #3 | Wrap in `BEGIN IMMEDIATE` when multi-connection workers exist |
| Defer | #4, #5 | Tighten remote-mode budget gate; drop unused import |
