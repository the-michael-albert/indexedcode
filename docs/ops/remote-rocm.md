# Linux deployment and remote ROCm integration boundary

## Supported today: the core CLI

The released CLI runs on Linux with Python 3.11 or newer. Core indexing, symbol/lexical search, context, and reference retrieval require no AMD GPU, HIP runtime, ROCm, or inference dependencies. Install a GitHub release using the [README instructions](../../README.md).

For a Linux host, use absolute paths and keep the database outside the repository:

```bash
ROOT="/home/you/work/myrepo"
DB="/home/you/.local/share/indexedcode/myrepo/index.sqlite"
mkdir -p "$(dirname "$DB")"
indexedcode orchestrator serve --root "$ROOT" --db "$DB" --http-port 8765
```

This starts the local indexer, watcher, and loopback HTTP retrieval API, not remote inference. See [service-manager examples](service-units.md) for a core systemd service.

## Remote inference is not integrated

Worker and authenticated client components exist in the source tree, but the packaged CLI does **not** load TOML configuration, start ROCm workers, or connect remote workers to indexing or search. Setting `mode = "remote"` in an `indexedcode.toml` file does not change CLI behavior. Default hybrid search reports `semantic_unavailable` and falls back to symbol/lexical retrieval. Explicit semantic search is unavailable.

There is no supported remote ROCm deployment recipe or mixed MLX/ROCm CLI profile in this release. Manually starting a worker does not enable an end-to-end inference path. The core release CI does not validate GPU runtime compatibility, model readiness, remote fingerprints, or inference memory usage.

Any future cross-machine integration must separately validate ROCm hardware support, model compatibility, configuration loading, pipeline/retrieval wiring, authentication, encrypted transport, and failure handling. Source text and credentials crossing a network require protection: the configuration loader already rejects any `remote_url` that is not `https://`, and workers must stay behind bearer authentication and a TLS-terminating proxy or private tunnel. Do not expose experimental worker ports publicly. A remote host would need to manage its own worker lifecycle rather than expect the local CLI to launch processes on another machine.
