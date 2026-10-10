# Deep review — Phase 3: managed workers, remote clients, adapters, memory admission

- **Scope:** commit `9e68653` (base `b100131`) — 16 files, +947/−13, plus review fixes applied during this review.
- **Companion docs:** [Phase 1 store review](2025-10-10_phase1-store_deep_review.md), [Phase 2 extraction review](2025-10-10_phase2-extraction_deep_review.md).
- **Verification:** full suite `.venv/bin/pytest` → 213 passed, 4 skipped (opt-in MLX/ROCm smoke, no hardware on this host — unverified by design).

## Architecture orientation

Phase 3 adds a control plane under the orchestrator: a `workers` package that
owns (a) the versioned HTTP contract shared by managed loopback children and
remote endpoints, (b) process supervision for managed children, and (c)
admission control over inference based on sampled memory. The DAL/store and
extraction layers from Phases 1–2 are untouched except for `config.py`, which
gains remote endpoint fields and a remote exemption from the managed-local
budget gate. No schema change.

## File map

| File | Role |
|---|---|
| `workers/errors.py` | exception hierarchy (transient vs fatal vs auth vs fingerprint) |
| `workers/fingerprint.py` | canonical-JSON sha256 model fingerprints |
| `workers/protocol.py` | pydantic shapes for `/v1/model`, embed, summarize |
| `workers/client.py` | stdlib bearer-auth HTTP client + retry classification |
| `workers/supervisor.py` | `ManagedWorker`: spawn/readiness/restart/drain/stop |
| `workers/remote.py` | `RemoteWorker` handle (client only, never spawns) |
| `workers/adapters.py` | platform gating (MLX/ROCm) + worker argv builder |
| `workers/memory.py` | soft/hard admission with cooldown resume |
| `workers/factory.py` | config → ManagedWorker / RemoteWorker wiring |
| `workers/hardwares.py` | hardware-gated MLX/ROCm inference wrappers |
| `workers/serve.py` | `python -m indexedcode.workers.serve` worker entrypoint |
| `config.py` | +remote_url/remote_token, https enforcement, remote budget exemption |
| `tests/fake_worker.py` | deterministic stdlib HTTP fake worker (all failure modes) |

## Issues

### P1 — Remote endpoints accepted cleartext URLs (fixed in this review)

`config.py` validated only the presence of `remote_url`. A configured
`http://` endpoint would send the bearer `remote_token` in cleartext,
violating plan §4.4 ("Remote workers must require bearer authentication and
encrypted transport"). Fixed at the earliest gate: `load_config` now rejects
any non-`https://` remote URL; regression test
`tests/test_config_phase3.py::test_remote_url_must_use_https`.

### P2 — Restart path could leak non-Worker exceptions from `poll()` (fixed)

`supervisor.py` `_handle_unexpected_exit` wrapped the respawn in
`except WorkerError`, so an `OSError` from `Popen` (EMFILE, ENOMEM) escaped
`poll()` and left the object in state `starting` with a dead child —
permanently unrestartable and unreported. Now `except Exception` →
`state="failed"`, `_terminal_failure=True`.

### P3 — advisory (documented, not fixed)

1. **Port TOCTOU** (`supervisor.py:_free_loopback_port`): the ephemeral port
   is released before the child binds it; a racing binder makes readiness
   fail cleanly (`WorkerStartupError`) but retry-looping on `start()` is the
   caller's job today. Consider Unix sockets (plan permits) in Phase 6.
2. **Worker stdout/stderr → DEVNULL** (`supervisor.py:_spawn_and_wait_ready`):
   crash diagnostics from managed children are lost. Capture the tail for
   `status()`/metrics in Phase 4 wiring.
3. **`_wait_ready` deadline overshoot**: the deadline is checked before the
   sleep, so total wait can exceed `startup_timeout_s` by one poll interval.
   Cosmetic.
4. **`workers/hardwares.py` API usage is unverified**: `mlx_embeddings` /
   `transformers` call shapes were written against docs, not hardware; only
   the opt-in smoke tests can validate them. Reported unverified, never
   passed (per gate).
5. **`remote_token` default is `""`**: a remote section without a token gets
   401s at first request. Config could require a token for remote mode when
   Phase 5 wires real callers.
6. **`serve.py` requires the package importable** in the child (installed or
   PYTHONPATH). Fine under pytest (`pythonpath=src`); revisit at packaging
   (Phase 6).

## What's correct and worth keeping

- **Secret handling**: per-run `secrets.token_urlsafe(32)` via env only;
  asserted absent from argv (`ps` check) and from `status()`.
- **Fingerprint gate before use**: `WorkerClient` validates the fingerprint
  on model info *and* every embed/summarize response; mismatch is fatal at
  supervisor startup and the child is killed and reaped.
- **Retry classification** is centralized (`classify_status`/`is_retryable`)
  with connection errors transient and 401 fatal — matches the plan's
  retry-classification requirement and is unit-pinned.
- **MemoryAdmission semantics** exactly match plan §4.5: pressure never
  kills, resume requires the full cooldown below soft, excursions restart
  the cooldown, and status is dimension-bounded (no paths/environment).
- **Remote never spawns**: `RemoteWorker` has no process surface at all;
  factory test pins it across a deliberately incompatible backend.
- **Deterministic testing**: real child processes and real sockets
  throughout; the only hardware boundary (MLX/ROCm) is smoke-gated and
  honestly skipped.

## Merge gating

| Severity | Issue | Action |
|---|---|---|
| Must fix | none remaining | — |
| Should fix | none remaining | — |
| Defer | P3 #1–#6 | tracked in this doc for Phases 4–6 |
