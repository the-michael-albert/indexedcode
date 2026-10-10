# Backup, rebuild, and rollback

The index database (and its `-wal` / `-shm` sidecar files) contains copies of
your repository source. Protect backups with the same care as the source
itself.

The examples use absolute paths. Set them once:

```bash
ROOT="/absolute/path/to/repository"
DB="/absolute/path/to/index.sqlite"     # keep this outside $ROOT
```

SQLite names the sidecars by appending to the full database filename:
`$DB-wal` and `$DB-shm` (for example `index.sqlite-wal`), not `index.wal`.
The writer also keeps a lock file, `$DB.writerlock`; it is not part of the
data and is not backed up.

## Backup (online)

The store runs in WAL mode. Use SQLite's backup command instead of copying
live files, which can produce an inconsistent copy:

```bash
mkdir -p "$(dirname "$DB")/backups"
sqlite3 "$DB" ".backup '$(dirname "$DB")/backups/index-$(date +%F).sqlite'"
```

This is safe while `orchestrator serve` is running. To copy files directly
instead, stop the writer first (Ctrl-C or SIGTERM to the process, or
`launchctl unload` / `systemctl --user stop` for a service), then copy
`$DB` together with any existing `$DB-wal` and `$DB-shm`. After a clean
shutdown the sidecars may not exist.

Keep the most recent few backups, plus one known-good backup from before each
upgrade.

## Rebuild (corruption, extractor change, or a new release)

Releases do not promise cross-version database migration. If a release note
says the index must be rebuilt, or the database is damaged, rebuild into a
**fresh database** rather than modifying the old one in place. The index is
derived data: the repository is the source of truth.

1. Stop the writer and confirm the process has exited (for example
   `pgrep -fl "indexedcode.*serve"` prints nothing).
2. Move the whole old database set aside:

   ```bash
   mkdir -p "$DB.old"
   for f in "$DB" "$DB-wal" "$DB-shm"; do
     [ -e "$f" ] && mv "$f" "$DB.old/"
   done
   ```

3. Rebuild from the repository tree:

   ```bash
   indexedcode index once --root "$ROOT" --db "$DB"
   ```

   (or start `orchestrator serve`, which performs the same initial scan).
4. Validate with `indexedcode status --root "$ROOT" --db "$DB"`:
   - `pending_jobs` is `0`;
   - `stale_paths` is empty;
   - `failures.failed_jobs` and `failures.error_files` are `0`;
   - `coverage.ready_files` matches what you expect.

   `model_fingerprint` is `null` for the core CLI, which does not run an
   embedding model. It only carries a value when an index was built with one.

Files that cannot be indexed (binary, oversized, unsupported language) are
skipped by design and are not failures.

## Rollback

1. Stop the writer and confirm the process has exited.
2. Move the failed set aside, then restore the previous complete set:

   ```bash
   mkdir -p "$DB.failed"
   for f in "$DB" "$DB-wal" "$DB-shm"; do
     [ -e "$f" ] && mv "$f" "$DB.failed/"
   done
   cp "$DB.old/$(basename "$DB")"* "$(dirname "$DB")/"
   ```

   To roll back from a `.backup` file instead, copy it to `$DB`; a backup is a
   single self-contained file and needs no sidecars.
3. Reinstall the previous release if you are rolling back an upgrade (see the
   README), restart, and run `status`.
4. If the restored index predates files changed since, retrieval compares the
   on-disk hash and marks those entries stale. The next `index once` or the
   running server's reconcile refreshes them; rebuild fully if the status
   check still shows problems.

Swaps are whole-set only. There is no partial or per-table rollback: readers
must see either the previous complete database or the next one.
