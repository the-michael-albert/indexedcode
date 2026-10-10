# Local macOS deployment and MLX integration boundary

## Supported today: the core CLI

The released CLI runs on macOS with Python 3.11 or newer. It indexes source and serves symbol/lexical retrieval without a GPU, MLX, model downloads, or inference dependencies. Install a release using the [README instructions](../../README.md).

Use absolute paths and keep the database outside the watched repository:

```bash
ROOT="/Users/you/work/myrepo"
DB="/Users/you/.local/share/indexedcode/myrepo/index.sqlite"
mkdir -p "$(dirname "$DB")"
indexedcode orchestrator serve --root "$ROOT" --db "$DB" --http-port 8765
```

This starts the indexer, watcher, and loopback retrieval API. It does **not** start MLX workers. Stop it with Ctrl-C or SIGTERM. See [service-manager examples](service-units.md) for launchd.

The CLI does not load TOML configuration. Creating an `indexedcode.toml` file or setting a managed MLX backend in one has no effect on this command. Default hybrid search reports `semantic_unavailable` and uses symbol/lexical retrieval; explicit semantic search is unavailable.

## Experimental components, not a deployment recipe

The source tree contains MLX worker and supervision components for future integration. Their presence does not mean the installed CLI supervises inference, performs model-readiness checks, or enforces an inference memory budget.

There is no supported end-to-end managed MLX deployment in this release. Do not install GPU dependencies or configure models to follow the core quick start. Real-hardware behavior and memory usage require separate validation; the earlier 1–2.5 GB inference memory target is not a measured guarantee for the shipped CLI.

Before an inference deployment can be documented as supported, it needs explicit configuration loading, worker-to-pipeline and retrieval wiring, compatible model/revision handling, lifecycle and failure testing, and hardware validation. None of those are enabled by the core CLI installation instructions.
