# Deep review — Phase 4: watcher and durable publication

- **Scope:** commits `37ee75b` (RED suite) → `992d233` (implementation), plus the review fixes applied during this review. 9 files, +834/−23 core diff.
- **Companion docs:** [Phase 1](2025-10-10_phase1-store_deep_review.md), [Phase 2](2025-10-10_phase2-extraction_deep_review.md), [Phase 3](2025-10-10_phase3-workers_deep_review.md).
- **Verification:** `.venv/bin/pytest` → 251 passed, 4 skipped (opt-in hardware smoke). Adversarial probes: import/call publication, class linkage, nested-function parents, entity-index reuse across edits.

## Architecture orientation

Phase 4 wires Phases 1–3 into a processing loop: `reconcile` turns the
Phase 2 scan into coalesced job commands, `pipeline` claims them under
leases and publishes through the Phase 1 atomic publication transaction,
with changed-entity-only inference and stable snapshots in between.
`watcher` contributes the pure debouncing logic the Phase 5 orchestrator
CLI will drive from real file events. The only pre-existing surfaces
touched are `dal.publish_file_refresh` (qname linkage, in-transaction
entity_index, durations, id map return), the models it validates, and a
narrow `scan.py` error-handling fix.

## File map

| File | Role |
|---|---|
| `pipeline.py` | claim → stable read → extract → changed-only inference → revalidate → atomic publish |
| `reconcile.py` | initial/reconciliation scans; cheap filters; authoritative removals |
| `watcher.py` | EventWatcher burst coalescing; ParentDebouncer |
| `stable.py` | stat/read/stat snapshots |
| `dal.py` | publish extended (qname linkage, entity_index, durations, id map); ensure_module/file_row |
| `models.py` | FunctionDef/ClassDef qname+summary fields; RelationshipRecord optional source_qname |
| `scan.py` | unreadable files now skipped with reason instead of raising |

## Issues

### P1 — Any file containing imports failed publication entirely (fixed)

The Phase 2 extractor emits import edges with `source_qname="file"`; the
Phase 4 publication path resolved source ids only against function/class
qname maps, producing `NULL` for `relationships.source_id` (NOT NULL,
[db.py:149](../../src/indexedcode/db.py#L149)). The entire publication
transaction rolled back, so a plain `import os` made a file permanently
unindexable. Verified empirically before the fix. Fixed in
`dal.publish_file_refresh`: unresolved sources fall back to the origin
file row. Regression test:
`test_pipeline.py::test_imports_and_calls_publish_with_file_level_sources`.

### P2 — Changed-entity snapshot read scanned the whole entity_index table (fixed)

`pipeline._existing_snapshot` fetched every function/class embedding row
in the database per job to match by qname. Correct but O(total entities)
per file refresh — at the plan's 10k-function benchmark that is 10k rows
per edit. Now one indexed point-lookup per entity of the file being
refreshed.

### P3 — advisory (documented, not fixed)

1. **Filesystem/SQLite race window**: the source can change between the
   pre-publish revalidation read and `BEGIN IMMEDIATE`. The plan accepts
   this (§7.3) — Phase 5 retrieval must compare the on-disk hash before
   serving indexed source as current.
2. **Delete vs pending refresh**: a file deleted while a refresh job is
   pending will burn its 5 attempts (`file_missing`) before the remove
   job wins. Correct end state, wasteful retries; could cancel pending
   refreshes when enqueueing removal.
3. **Revision bookkeeping with coalescing**: a watcher-enqueued newer
   revision processed under an older job's `desired_revision` publishes
   current content with an older revision number. CHECK constraints hold;
   revision is a routing hint, not a content hash.
4. **`search_text` unbounded**: full function source is stored as
   `search_text` (≤ max_file_bytes per file); the `embedding_truncated`
   flag is always False. Phase 5 retrieval should bound it and set the
   flag.
5. **`reconcile` removal is conservative**: skipped paths (ignored,
   binary, unsupported, oversized, unreachable) are never treated as
   removed even when the plan permits exclusion-based removal. Safe
   default; revisit if tombstones accumulate for vendored trees.

## What's correct and worth keeping

- **Changed-entity-only inference is proven, not assumed**: the counting
  fake embedder shows zero calls for location-only edits and exactly one
  text (the changed sibling) for content edits, with the unchanged
  sibling's embedding blob and input_hash byte-identical after
  republishing.
- **Last-good snapshots are never replaced**: parse errors and transient
  inference failures leave the prior indexed_hash, entities, and
  embeddings intact, with the job on capped exponential backoff and the
  file marked error — and a first-index parse failure publishes nothing.
- **Stale in-flight work discards itself**: a file mutated *during*
  embedding fails the hash revalidation and retries instead of
  publishing.
- **Removal is authoritative-only**: unreadable files are skipped with an
  explicit reason (the `scan.py` fix), and only a complete scan missing a
  path enqueues `remove_file`; tombstones revive on reappearance.
- **Metrics discipline**: success/retry/failed job rows carry
  queue/process durations and categorical error codes; a dedicated test
  greps every metric field for path and source leakage.

## Merge gating

| Severity | Issue | Action |
|---|---|---|
| Must fix | none remaining | — |
| Should fix | none remaining | — |
| Defer | P3 #1–#5 | tracked here for Phases 5–6 |
