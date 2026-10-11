"""Managed worker HTTP entrypoint: ``python -m indexedcode.workers.serve``.

Serves the plan §4.2–4.4 contract on a private loopback port with bearer
authentication (both delivered via environment by the supervisor). Model
inference is delegated to the hardware adapters; weights load once at
startup, before the server accepts requests.

Container deployments (``deploy/docker``) bind ``--host 0.0.0.0``; that
requires a non-empty ``INDEXEDCODE_WORKER_SECRET``. Inference is serialized
within one process (llama.cpp contexts are not thread-safe): scale out with
more worker processes behind a load balancer instead.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler

from indexedcode.httpserver import LoopbackHTTPServer
from indexedcode.workers import hardwares
from indexedcode.workers.adapters import backend_available
from indexedcode.workers.fingerprint import model_fingerprint

DEFAULT_DIMENSION = 384


_EMBED_RUNNERS = {"mlx": hardwares.mlx_embed, "rocm": hardwares.rocm_embed,
                  "cuda": hardwares.cuda_embed,
                  "vulkan": hardwares.vulkan_embed,
                  "cpu": hardwares.cpu_embed}
_SUMMARY_RUNNERS = {"mlx": hardwares.mlx_summary,
                    "rocm": hardwares.rocm_summary,
                    "cuda": hardwares.cuda_summary,
                    "vulkan": hardwares.vulkan_summary,
                    "cpu": hardwares.cpu_summary}


def build_model_info(role, backend, model) -> dict:
    """The model identity a worker advertises (minus its fingerprint).

    Shared with the orchestrator so the expected fingerprint of a managed
    worker is derived from exactly what the worker will report.
    """
    info = {"model": model, "revision": "unpinned", "backend": backend}
    if role == "summary":
        info["quantization"] = "default-quantized"
    else:
        info.update(dimension=DEFAULT_DIMENSION, pooling="mean",
                    normalized=True)
    return info


def build_handler(secret, model_info, summary_fn, embed_fn):
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            return self.headers.get("Authorization") == f"Bearer {secret}"

        def do_GET(self):
            if self.path == "/healthz":  # liveness only; reveals nothing
                return self._send(200, {"ok": True})
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            if self.path == "/v1/model":
                return self._send(200, model_info)
            if self.path == "/v1/status":
                return self._send(200, {
                    "state": "ready",
                    "rss_mb": _rss_mb(),
                })
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/v1/embed" and embed_fn is not None:
                with inference_lock:
                    vectors = embed_fn(request.get("texts", []))
                return self._send(200, {
                    "vectors": vectors,
                    "dimension": len(vectors[0]) if vectors else 0,
                    "fingerprint": model_info["fingerprint"],
                })
            if self.path == "/v1/summarize" and summary_fn is not None:
                with inference_lock:
                    summary = summary_fn(
                        request.get("prompt", ""),
                        request.get("max_output_tokens", 256),
                    )
                return self._send(200, {
                    "summary": summary,
                    "fingerprint": model_info["fingerprint"],
                })
            return self._send(404, {"error": "not found"})

    return Handler


def _rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover - platform dependent
        return 0.0


def prefetch(model_path: str) -> None:
    """Download a ``repo_id:filename-glob`` GGUF model into the HF cache
    (a local ``.gguf`` path needs nothing). Lets a container fleet share one
    download instead of racing."""
    if hardwares.is_local_gguf(model_path):
        return
    from huggingface_hub import snapshot_download

    repo_id, _, pattern = model_path.partition(":")
    snapshot_download(repo_id, allow_patterns=[pattern or "*.gguf"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("embedding", "summary"),
                        required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get(
                            "INDEXEDCODE_WORKER_PORT", "0")))
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default=os.environ.get(
        "INDEXEDCODE_WORKER_HOST", "127.0.0.1"))
    parser.add_argument("--warmup", action="store_true",
                        help="load the model before accepting requests"
                             " (so a pool never routes to a cold worker)")
    parser.add_argument("--prefetch", action="store_true",
                        help="download the model and exit")
    args = parser.parse_args(argv)

    if args.prefetch:
        prefetch(args.model)
        return 0

    # refuse startup on an incompatible backend before loading anything
    backend_available(args.backend)

    secret = os.environ.get("INDEXEDCODE_WORKER_SECRET", "")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not secret:
        parser.error("binding a non-loopback --host requires"
                     " INDEXEDCODE_WORKER_SECRET")
    summary_fn = embed_fn = None
    info = build_model_info(args.role, args.backend, args.model)
    if args.role == "summary":
        summary_fn = functools.partial(
            _SUMMARY_RUNNERS[args.backend], model_path=args.model)
    else:
        embed_fn = functools.partial(
            _EMBED_RUNNERS[args.backend], model_path=args.model)
    info["fingerprint"] = model_fingerprint(**info)
    if args.warmup:
        if summary_fn is not None:
            summary_fn("def warmup(): pass", 1)
        else:
            embed_fn(["warmup"])

    server = LoopbackHTTPServer(
        (args.host, args.port),
        build_handler(secret, info, summary_fn, embed_fn),
    )
    print("READY", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
