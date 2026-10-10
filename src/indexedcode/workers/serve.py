"""Managed worker HTTP entrypoint: ``python -m indexedcode.workers.serve``.

Serves the plan §4.2–4.4 contract on a private loopback port with bearer
authentication (both delivered via environment by the supervisor). Model
inference is delegated to the hardware adapters; weights load once at
startup, before the server accepts requests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

from indexedcode.httpserver import LoopbackHTTPServer
from indexedcode.workers import hardwares
from indexedcode.workers.adapters import backend_available
from indexedcode.workers.fingerprint import model_fingerprint

DEFAULT_DIMENSION = 384


def build_handler(secret, model_info, summary_fn, embed_fn):
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
                vectors = embed_fn(request.get("texts", []))
                return self._send(200, {
                    "vectors": vectors,
                    "dimension": len(vectors[0]) if vectors else 0,
                    "fingerprint": model_info["fingerprint"],
                })
            if self.path == "/v1/summarize" and summary_fn is not None:
                return self._send(200, {
                    "summary": summary_fn(
                        request.get("prompt", ""),
                        request.get("max_output_tokens", 256),
                    ),
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("embedding", "summary"),
                        required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get(
                            "INDEXEDCODE_WORKER_PORT", "0")))
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    # refuse startup on an incompatible backend before loading anything
    backend_available(args.backend)

    secret = os.environ.get("INDEXEDCODE_WORKER_SECRET", "")
    summary_fn = embed_fn = None
    info = {"model": args.model, "revision": "unpinned",
            "backend": args.backend}
    if args.role == "summary":
        runner = hardwares.mlx_summary if args.backend == "mlx" \
            else hardwares.rocm_summary
        summary_fn = runner
        info["quantization"] = "default-quantized"
    else:
        runner = hardwares.mlx_embed if args.backend == "mlx" \
            else hardwares.rocm_embed
        embed_fn = runner
        info.update(dimension=DEFAULT_DIMENSION, pooling="mean",
                    normalized=True)
    info["fingerprint"] = model_fingerprint(**info)

    server = LoopbackHTTPServer(
        ("127.0.0.1", args.port),
        build_handler(secret, info, summary_fn, embed_fn),
    )
    print("READY", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
