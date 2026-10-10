# Backup, rebuild, and rollback

The database (plus its `-wal`/`-shm` sidecars), the configuration, and
the model cache are copies of repository source — protect them with the
same care as the source itself.

## Backup (online)

The store runs in WAL mode; a safe online backup uses SQLite's backup
API rather than copying live files:

```bash
sqlite3 index.sqlite ".backup 'backups/index-$(date +%F).sqlite'"
```

Or stop the writer (`launchctl unload` / `systemctl stop` / SIGTERM to
`orchestrator serve`), copy `index.sqlite` plus any `-wal`/`-shm`
sidecars, and restart. Retention: keep the most recent N backups plus
one known-good from each fingerprint era.

## Rebuild (fingerprint change or corruption)

A changed embedding model fingerprint/dimension, summary prompt change,
or extractor upgrade requires a **fresh database** — never an in-place
migration:

1. Stop the writer (SIGTERM; confirm `status` reports the port closed).
2. Move the old set aside: `index.sqlite{,.wal,.shm}` →
   `index.sqlite.old/`.
3. Start `orchestrator serve` (or `index once`); the initial scan
   rebuilds from the repository tree.
4. Validate: `indexedcode status` shows full coverage, zero stale paths,
   zero failed jobs, and the new model fingerprint.

## Rollback

1. Stop the writer.
2. Swap the complete database set back:
   `mv index.sqlite{,.wal,.shm} index.failed/ && mv index.sqlite.old/* .`
3. Restart and re-run `status`. If the rolled-back index predates the
   current fingerprint, retrieval marks affected rows stale by hash
   comparison — a full rebuild is then required.

Swaps are whole-file-set only. There is no partial or per-table
rollback: readers must see either the previous complete database or the
next one.
