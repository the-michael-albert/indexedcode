# IndexedCode

IndexedCode builds a local, queryable SQLite index of a source repository. It extracts modules, classes, functions, and relationships with Tree-sitter, keeps the index current while files change, and exposes results through a CLI, a loopback HTTP API, and MCP.

## Support and scope

The current release supports **macOS and Linux with Python 3.11 or newer**. It provides structural indexing plus symbol and lexical search without a GPU or model download. Native Windows is not yet supported because the writer lock uses POSIX file locking.

The repository contains experimental MLX and ROCm worker components, but the core CLI does **not** currently load `indexedcode.toml`, start those workers, or connect them to indexing and search. Adding TOML configuration does not enable inference. Consequently, `hybrid` search reports `semantic_unavailable` and falls back to symbol/lexical search. Use `--mode lexical` or `--mode symbol` for an explicit core-only mode; `--mode semantic` fails without an embedding model. The core install includes no GPU or inference dependencies.

Supported source languages include Python, JavaScript, TypeScript, Go, Rust, Java, C, and C++. References are syntactic candidates, not a compiler-backed call graph.

## Install a release

Wheel and source-distribution assets are published on [GitHub Releases](https://github.com/the-michael-albert/indexedcode/releases), **not PyPI**. Versioned URLs become available only after the corresponding release is published; a 404 before publication is expected. The examples pin version `1.0.0a1`; replace both version components when upgrading. CI tests Linux and macOS on Python 3.11–3.14.

### uv (recommended)

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv tool install --python 3.12 \
  https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a1/indexedcode-1.0.0a1-py3-none-any.whl
```

### pipx

With Python 3.11+ and [pipx](https://pipx.pypa.io/stable/installation/) installed:

```bash
pipx install \
  https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a1/indexedcode-1.0.0a1-py3-none-any.whl
```

Both methods download Python dependencies during installation. Verify the commands are available:

```bash
indexedcode --version
indexedcode --help
indexedcode-mcp --help
```

The package also supports `python -m indexedcode` with the Python interpreter in the environment where it is installed. Your system Python does not automatically see uv/pipx tool environments. If pipx selects an older interpreter, add `--python /absolute/path/to/python3.12` to the install command.

## Start watching a repository

The guided setup is the easiest first run:

```bash
indexedcode setup
```

The Textual interface lets you select a repository, choose an index database, and set the HTTP port. Its default database is under `~/.local/share/indexedcode/` (or `$XDG_DATA_HOME/indexedcode/`), outside the watched repository. Select **Start watching** to index the repository, watch for changes, and start the API. Press Ctrl+C to stop it.

Keeping the database outside the repository prevents SQLite database and WAL updates from retriggering the watcher. You can also start the same service non-interactively:

```bash
ROOT="$(cd /path/to/repository && pwd)"
DB="$HOME/.local/share/indexedcode/my-project/index.sqlite"
mkdir -p "$(dirname "$DB")"
indexedcode orchestrator serve --root "$ROOT" --db "$DB" --http-port 8765
```

The server prints its loopback URL on stderr. Port `0` asks the operating system to select a free port. It performs an initial scan, watches changes, and periodically reconciles the tree. Ctrl-C or SIGTERM stops the service. The default HTTP port is `8765`, debounce is `0.25` seconds, and reconciliation interval is `5` seconds; use `--http-port`, `--debounce-s`, and `--poll-interval-s` to override them.

The core command defaults remain `--root .` and `--db index.sqlite`, relative to the working directory. Explicit absolute paths and an external-to-repository database are deployment recommendations, not changed CLI defaults. Use a separate database per repository.

HTTP has no authentication by default; keep it local. `--http-token` enables bearer authentication, but command-line secrets may be visible to local process inspection. Do not expose the service publicly. See [service-manager examples](docs/ops/service-units.md) for launchd and systemd.

## One-shot indexing and queries

Use the same absolute root and external database paths for every command. For one-shot use, stop the watcher first: only one writer may own a database. Run `index once` or `orchestrator serve` before querying a new database. Replace the example symbols and repository-relative context path with real ones:

```bash
indexedcode index once --root "$ROOT" --db "$DB"
indexedcode search parse_config --mode lexical --root "$ROOT" --db "$DB"
indexedcode context src/config.py --root "$ROOT" --db "$DB"
indexedcode references parse_config --root "$ROOT" --db "$DB"
indexedcode status --root "$ROOT" --db "$DB"
```

Commands print bounded JSON, making them suitable for scripts. The HTTP server exposes the equivalent read-only endpoints under `http://127.0.0.1:8765/v1/`.

## MCP client configuration

Index the repository first, then configure an MCP client to run the installed stdio server:

```json
{
  "mcpServers": {
    "indexedcode": {
      "command": "/absolute/path/to/indexedcode-mcp",
      "args": [
        "--root", "/absolute/path/to/repository",
        "--db", "/absolute/path/to/index.sqlite"
      ]
    }
  }
}
```

Find the executable with `command -v indexedcode-mcp`. GUI applications often do not inherit the shell's `PATH`, so an absolute command path is more reliable. The MCP server offers `code_search`, `code_context`, `code_references`, and `code_status`; it writes protocol data only to stdout. It reads an existing index and does not start a watcher or inference workers. Run `orchestrator serve` separately for continuous updates, and keep the MCP database outside the repository.

## Upgrade, back up, and uninstall

Before an upgrade, stop the watcher and back up the complete database using SQLite's backup command rather than copying only the main file while it is live:

```bash
sqlite3 "$DB" ".backup '$DB.backup'"
```

The backup requires the `sqlite3` command and is safe even while a writer is active. Choose a unique destination to retain earlier backups. Protect databases and backups like repository source: they contain indexed source text.

Upgrade using the exact wheel URL of a **published** release and the installer that owns your installation. These commands reinstall the documented version; replace both version components to select a newer release:

```bash
uv tool install --force --python 3.12 \
  https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a1/indexedcode-1.0.0a1-py3-none-any.whl
# Or:
pipx install --force \
  https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a1/indexedcode-1.0.0a1-py3-none-any.whl
```

A direct URL pins a release; do not assume `uv tool upgrade` or `pipx upgrade` discovers newer GitHub assets. Check `indexedcode --version`, restart the writer, and run `status` with your root/database paths. Read release notes for rebuild requirements; cross-version database migration is not promised. To rebuild safely, retain the old database and use `index once` with a new external database path. To roll back, stop the writer, reinstall the previous wheel, and use its known-good backup at a separate path.

Uninstall the application (indexes are intentionally retained):

```bash
uv tool uninstall indexedcode
# or: pipx uninstall indexedcode
```

Uninstalling retains your repository, database, backups, and custom service definitions. Stop and disable any service first. Delete index data only after verifying its absolute location and that no process uses it. SQLite sidecars use the suffixes `-wal` and `-shm`.

## Troubleshooting

- **`indexedcode: command not found`:** open a new shell after installing, run `uv tool update-shell`, or run `pipx ensurepath`. Use `uv tool dir --bin` or `pipx environment` to inspect tool paths.
- **Python version error:** install through `uv tool install --python 3.12 ...`, or ensure pipx uses Python 3.11+.
- **Database cannot be opened:** create its parent directory and ensure it is writable. Guided setup does this automatically.
- **Address already in use:** choose another `--http-port`, or use `0` for a free port.
- **Writer lock error:** only one indexing/watching writer may use a database at a time. Read-only CLI and MCP queries can run alongside it.
- **Semantic search unavailable:** this is expected in the core release; use lexical or symbol mode.

## Development

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

Hardware smoke tests are opt-in and are not part of the core release claim. See [release instructions](docs/releasing.md) for artifact validation and publication.

Licensed under the [Apache License 2.0](LICENSE).
