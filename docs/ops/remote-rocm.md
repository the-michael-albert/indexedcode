# Remote deployment: ROCm inference host (AMD/Linux)

Cross-machine inference: the orchestrator never spawns workers on
another host. The remote host runs its own workers under its own process
manager; the local orchestrator only holds an authenticated HTTPS
client.

## Remote host (Linux + HIP)

Verify the ROCm runtime before configuring:

```bash
command -v rocm-smi   # or hipcc; the orchestrator refuses CPU fallback
```

Run the worker entrypoints (no repository database, no watcher):

```bash
indexedcode orchestrator serve ...   # not used here; use worker commands:
python -m indexedcode.workers.serve --role embed --backend rocm \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --port 8901
python -m indexedcode.workers.serve --role summary --backend rocm \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --port 8902
```

Expose them only through a TLS-terminating reverse proxy or private
tunnel with bearer authentication. `http://` remote URLs are rejected at
configuration load — encrypted transport is mandatory because the
bearer token and repository source cross the network.

## Local orchestrator

```toml
[embedding]
mode = "remote"
backend = "rocm"
remote_url = "https://amd-index-worker.internal"
remote_token = "<secret from your secret store>"

[summarization]
mode = "managed"   # mixed deployments are supported
backend = "mlx"
```

Remote sections are exempt from the managed-local memory budget gate:
over-budget settings there need no `allow_over_budget` opt-in. Fingerprint
mismatches are fatal at first use — pin revisions and rebuild when they
change.
