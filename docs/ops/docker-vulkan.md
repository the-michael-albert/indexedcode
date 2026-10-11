# Worker pools in Docker (Vulkan, Linux)

`deploy/docker/` runs the **summarization** and **embedding** services as pools of identical worker containers (llama.cpp, Vulkan backend: AMD, or Intel via Mesa) behind a HAProxy gateway. Indexing sends many independent requests, so more workers means more throughput; each worker handles one request at a time, and the gateway balances by least-connections and routes only to workers that have finished loading their model.

```
indexedcode ──► gateway :8801 ──► summary-1 … summary-N   (GPU)
 (host)         gateway :8802 ──► embedding-1 … embedding-M
```

Entity source is sent to these workers, so they are as sensitive as the index itself. The gateway binds loopback by default and every worker requires the bearer token.

## Requirements

- Linux with Docker Engine and Compose v2, and `/dev/dri` (a working Vulkan GPU driver on the host; verify with `vulkaninfo --summary`).
- Enough GPU memory for `workers × model size`. The default models are small (about 0.6 GB for the summarizer, under 0.1 GB for the embedder); every replica loads its own copy.

## Run it

```bash
cd deploy/docker
cp .env.example .env
# edit .env: set INDEXEDCODE_WORKER_SECRET (openssl rand -hex 32) and, from
# `getent group video render`, VIDEO_GID / RENDER_GID
docker compose up -d --build
docker compose ps          # wait until summary/embedding are "healthy"
```

The first start builds llama.cpp with Vulkan (a few minutes) and downloads the models once into the `models` volume; each worker then loads its model before reporting healthy.

Pool size is `SUMMARY_WORKERS` / `EMBEDDING_WORKERS` in `.env`. Resize live (the gateway discovers workers through Docker DNS, up to 16 per pool):

```bash
docker compose up -d --scale summary=6 --scale embedding=3
```

Check the gateway: `curl -H "Authorization: Bearer $SECRET" http://127.0.0.1:8801/v1/model` returns the model identity.

## Point IndexedCode at it

Easiest: `indexedcode setup`, then for **Summarization** choose *Remote worker pool (Docker)* with URL `http://127.0.0.1:8801` and the token from `.env`; for **Embedding** the same with `http://127.0.0.1:8802`. Set *Parallel requests* to the pool size or a little above (default for the remote mode is 1). Setup saves these to `indexedcode.toml` next to the index database (owner-only permissions, since it holds the token) and reuses them next time. The *Semantic search* switch turns summaries and embeddings off entirely.

Or write the file yourself and pass `--config indexedcode.toml`:

```toml
[summarization]
mode = "remote"
backend = "vulkan"
remote_url = "http://127.0.0.1:8801"
remote_token = "<INDEXEDCODE_WORKER_SECRET>"
max_concurrent_requests = 6

[embedding]
mode = "remote"
backend = "vulkan"
remote_url = "http://127.0.0.1:8802"
remote_token = "<INDEXEDCODE_WORKER_SECRET>"
max_concurrent_requests = 4
```

If you changed `SUMMARY_MODEL` / `EMBEDDING_MODEL` in `.env`, set the same `model = "..."` for that stage: IndexedCode verifies the worker's model fingerprint and refuses a mismatch.

## Another machine

`remote_url` must be `https://` unless it is localhost. To serve a different host, set `BIND_ADDRESS` and put a TLS-terminating proxy (Caddy, nginx, a tunnel) in front of the gateway ports; do not publish them in plain text. The bearer token travels with every request.

## Operations

- Logs: `docker compose logs -f summary gateway`. Gateway health: workers appear/disappear in the HAProxy log as they pass or fail `/healthz`.
- Switching models changes the fingerprint, so IndexedCode regenerates affected summaries and vectors.
- `docker compose down` stops the pools; `docker compose down -v` also deletes the downloaded models.
- A worker that crashes is restarted by Docker; in-flight requests are retried by IndexedCode.
