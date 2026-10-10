# Local deployment: managed MLX (Apple Silicon)

All-in-one local profile: the orchestrator supervises both inference
workers on the same machine.

## Configuration (`indexedcode.toml`)

```toml
[embedding]
mode = "managed"
backend = "mlx"
model = "sentence-transformers/all-MiniLM-L6-v2"

[summarization]
mode = "managed"
backend = "mlx"
model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

[memory]
managed_soft_limit_mb = 2300
managed_hard_limit_mb = 2560
```

## Run

```bash
indexedcode orchestrator serve \
  --root /path/to/repo \
  --db /path/to/index.sqlite \
  --http-port 8765
```

The orchestrator verifies Apple Silicon (`darwin`/`arm64`) before
spawning MLX workers, refuses startup otherwise, and never falls back to
CPU. Workers bind only to loopback ports with per-run bearer secrets
passed via environment — never in argv.

## Memory profile

The managed-local target is **approximately 1–2.5 GB total process
memory** under the documented default workload (quantized Qwen, a
compact 384-dimension embedder, 4,096 input / 256 output tokens, one
active request per worker, WAL-mode SQLite with a 32 MiB cache). This is
a measured profile, not a guarantee: exceeding it (FP16 weights, larger
contexts, concurrency > 1) requires `allow_over_budget = true` or
remote inference. At the soft limit new inference is deferred; at the
hard limit it is rejected and `memory_pressure` is reported — a worker
is never killed from the admission check alone. Measure your deployment
with `indexedcode status` RSS reporting and the benchmark harness
(`python -m indexedcode.bench` via `run_benchmarks`).

## Model pinning

Pin model revisions in configuration. The orchestrator validates each
worker's model fingerprint at readiness and on every response; a changed
embedding fingerprint requires a rebuild into a fresh database (see
[backup-rebuild-rollback.md](backup-rebuild-rollback.md)).
