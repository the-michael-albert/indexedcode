# IndexedCode

IndexedCode builds a local, queryable SQLite index of a source repository. It extracts modules, classes, functions, and relationships with Tree-sitter, keeps the index current while files change, and exposes results through a CLI, a loopback HTTP API, and MCP.

## Support and scope

The current release supports **macOS and Linux with Python 3.11 or newer**. It provides structural indexing plus symbol and lexical search without a GPU or model download. Native Windows is not yet supported because the writer lock uses POSIX file locking.

Semantic search is optional. When enabled, indexing is **scan → summarize → embed**: `setup`, `orchestrator serve`, `index once`, and `search` start a summary worker and an embedding worker (MLX on Apple Silicon, CPU elsewhere); every function and class is summarized, and the generated summary (never the raw source) is what gets embedded. Raw source stays available for structural and lexical retrieval. Queries for `hybrid` and `semantic` search are embedded with the same embedding model. Files that are already indexed are backfilled: missing or outdated summaries are regenerated even when vectors exist, and vectors are recomputed whenever the summary or embedding model changes. Both stages need their runtime in the *same Python environment that runs `indexedcode`*, and the right one is picked and installed for you. Apple Silicon gets MLX (`mlx-lm`, `mlx-embeddings`) with the package itself. On Linux the GPU is detected at first run: **NVIDIA → CUDA** (PyTorch), **AMD → Vulkan** (llama.cpp with GGUF models; the install compiles `llama-cpp-python` and needs cmake, a C/C++ compiler, `libvulkan-dev` and `glslc`), **no GPU → CPU** (CPU-only PyTorch). The missing runtime is installed automatically the first time a summary or embedding worker starts; run `indexedcode install-runtime` to do it up front (`--backend` overrides detection), set `INDEXEDCODE_GPU=nvidia|amd|none` to override GPU detection, or `INDEXEDCODE_AUTO_INSTALL=0` to turn the automatic install off. A failed install is reported as a warning and indexing continues structurally. ROCm (`backend = \"rocm\"`) is still available but is not installed automatically. Pass `--config indexedcode.toml` to choose other backends or models. If either stage cannot start, a warning names the stage and the reason, indexing continues structurally (symbol/lexical search only; `hybrid` reports `semantic_unavailable`), and `GET /v1/status` plus the dashboard report `inference` per stage (state, reason, and `done`/`total` entities) so an empty queue never implies summaries and embeddings exist. Restart after installing the runtime and existing files are backfilled. Pass `--no-embeddings` to opt out explicitly. The first run downloads the models. The core wheel carries no CUDA/CPU/Vulkan inference dependencies; those are installed on demand as described above.

### Hosted embedding and summarization APIs

Local models are slow on modest hardware. To offload either stage to a hosted API, set `mode = "api"` in the file you pass with `--config`; no local runtime is installed or started for that stage. Requests run in parallel (`max_concurrent_requests`, default 8), and 429/5xx responses are retried with backoff.

```toml
[summarization]
mode = "api"
provider = "anthropic"            # or "openai" (any OpenAI-compatible endpoint)
# model = "claude-haiku-5-5"      # default for anthropic; openai: gpt-4o-mini
# api_key_env = "ANTHROPIC_API_KEY"

[embedding]
mode = "api"
provider = "openai"               # embeddings: OpenAI-compatible only
# model = "text-embedding-3-small"
# base_url = "https://api.openai.com/v1"   # Voyage, Together, Ollama, vLLM, ...
# dimensions = 512                # optional, for models that support it
```

API keys are read from environment variables (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` by default, or the variable named by `api_key_env`); `indexedcode setup` can instead save a key as `api_key` in an owner-only `indexedcode.toml` next to the index database. `base_url` must be `https://` (plain `http://` only for localhost). You can mix modes, for example a hosted summarizer with a local embedder. **Entity source code is sent to the provider for summarization, and generated summaries are sent for embedding**; use local mode for code that must not leave the machine. Changing provider or model changes the fingerprint, so affected summaries and vectors are regenerated.

### Worker pools in Docker (Vulkan)

For an AMD/Intel GPU box, `deploy/docker/docker-compose.yml` runs the summarization and embedding services as scalable pools of workers behind a least-connections gateway; see [docs/ops/docker-vulkan.md](docs/ops/docker-vulkan.md). Quick start: `cd deploy/docker && cp .env.example .env` (set the token), then `docker compose up -d --build`. `indexedcode setup` has a switch for semantic search and, per stage, a choice of local models, the remote worker pool (URL and token) or a hosted API (provider, URL, key, model, parallel requests).

Supported source languages include Python, JavaScript, TypeScript, Go, Rust, Java, C, and C++. References are syntactic candidates, not a compiler-backed call graph.

## Install a release

Wheel and source-distribution assets are published on [GitHub Releases](https://github.com/the-michael-albert/indexedcode/releases), **not PyPI**. Versioned URLs become available only after the corresponding release is published; a 404 before publication is expected. The examples pin version `1.0.0a9`; replace both version components when upgrading. CI tests Linux and macOS on Python 3.11–3.14.

### uv (recommended)

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv tool install --python 3.12 \
  https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a9/indexedcode-1.0.0a9-py3-none-any.whl
```

### pipx

With Python 3.11+ and [pipx](https://pipx.pypa.io/stable/installation/) installed:

```bash
pipx install \
  https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a9/indexedcode-1.0.0a9-py3-none-any.whl
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

## Logs

`indexedcode setup`, `orchestrator serve`, and `index once` print progress to **stderr** as it happens:

```text
16:49:40 INFO    watching /path/to/repo
16:49:40 INFO    QUEUED   #1 refresh_file one.py
16:49:40 INFO    RUNNING  #1 refresh_file one.py (attempt 1/5)
16:49:40 INFO    DONE     #1 refresh_file one.py (1 functions, 0 classes, 1 ms)
16:49:44 INFO    QUEUED   #2 refresh_file added.py
16:49:44 INFO    RUNNING  #2 refresh_file added.py (attempt 1/5)
16:49:44 INFO    DONE     #2 refresh_file added.py (1 functions, 0 classes, 2 ms)
```

The default log shows every job being queued, started, completed, retried, or failed. Use `-v` for extra debug detail (HTTP requests and scans that found nothing) and `-q` for warnings and errors only; put the flag before the command, e.g. `indexedcode -v orchestrator serve ...`. stdout is reserved for JSON results, so `indexedcode search ... | jq` is unaffected, and `2>server.log` captures the log. A file with a syntax error is reported once and tried again only after it changes.

## Web dashboard and `llms.txt`

While `indexedcode setup` or `orchestrator serve` is running, open **http://127.0.0.1:8765/** (or the port you chose) for a live dashboard: index coverage, the queued jobs with the functions waiting to be re-indexed, when each job was queued and last updated, recently indexed functions, and a search box. It refreshes every two seconds.

`http://127.0.0.1:8765/llms.txt` describes the API for AI agents, with this server's real URL filled in. The page and `llms.txt` contain no index data, so they load without a token; when the server was started with `--http-token`, the dashboard asks for it and keeps it only in that browser tab. Without a token, the server only answers requests whose `Host` is a loopback name, which blocks DNS-rebinding pages from reading your source.

`GET /v1/queue` returns the same queue data as JSON.

### Graph view and GraphQL

The dashboard also has a **Graph** section: pick a module scope, optionally type a semantic query to highlight matching functions, and press Draw to see the call graph (files, classes, functions; test files dimmed). It is fed by `POST /v1/graphql`, a read-only GraphQL endpoint (no mutations; token rules match the REST API):

```bash
curl -s -H "Authorization: Bearer $TOKEN" -d '{"query":"{ function(qualifiedName: \"Config.load\") { callees { qualifiedName } } }"}' http://127.0.0.1:8765/v1/graphql
```

Queries nested deeper than 8 fields are rejected and relationship lists are capped at 1000 rows. The graph libraries are vendored under `src/indexedcode/web/` (see `VENDORED.md` there).

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
  https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a9/indexedcode-1.0.0a9-py3-none-any.whl
# Or:
pipx install --force \
  https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a9/indexedcode-1.0.0a9-py3-none-any.whl
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
