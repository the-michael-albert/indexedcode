#!/bin/sh
# Build and start the IndexedCode Vulkan worker pools.
# Usage: sh start-workers.sh   (needs Docker; run via sudo if you are not in
# the `docker` group). Waits until every worker reports healthy.
set -e
cd "$(dirname "$0")"

# expected workers: .env overrides compose's defaults (3 summary / 2 embedding)
get_env() {
    value=$(grep -m1 "^$1=" .env 2>/dev/null | cut -d= -f2-)
    echo "${value:-$2}"
}
expected=$(($(get_env SUMMARY_WORKERS 3) + $(get_env EMBEDDING_WORKERS 2)))

docker compose up -d --build

echo "Waiting for $expected workers to report healthy (first build compiles"
echo "llama.cpp and downloads the models; later starts are quick)..."
elapsed=0
while :; do
    healthy=$(docker ps --filter "name=indexedcode-inference" \
                       --filter "health=healthy" \
                       --format '{{.Names}}' | grep -cE 'summary-[0-9]|embedding-[0-9]' || true)
    [ "$healthy" -ge "$expected" ] && break
    if [ "$elapsed" -ge 1800 ]; then
        echo "Timed out after 30 min ($healthy/$expected healthy)." >&2
        docker compose ps -a
        docker compose logs --tail 20 models summary embedding >&2
        exit 1
    fi
    if docker ps -a --filter "name=indexedcode-inference" \
              --filter "status=restarting" --format '{{.Names}}' | grep -q .; then
        echo "A worker is crash-looping; check: docker compose logs summary" >&2
        exit 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
done

echo
echo "Pools are up.  Summary:   http://127.0.0.1:$(get_env SUMMARY_PORT 8801)"
echo "               Embedding: http://127.0.0.1:$(get_env EMBEDDING_PORT 8802)"
echo "Bearer token (INDEXEDCODE_WORKER_SECRET): see .env"
