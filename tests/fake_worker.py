"""Deterministic fake inference worker (test-only infrastructure).

Implements the plan §4.2–4.4 worker HTTP contract using only the stdlib so
supervisor/client tests never need real MLX/ROCm hardware. Behavior is
selected with --mode:

  ready          normal worker; binds, serves authenticated endpoints
  hang           never binds (readiness must time out)
  mismatch       serves a wrong model fingerprint
  exit           exits nonzero immediately (spawn failure)
  ignore_sigterm ignores SIGTERM (supervisor must SIGKILL after grace)

Env contract (set by the supervisor):
  INDEXEDCODE_WORKER_PORT          TCP port to bind on 127.0.0.1
  INDEXEDCODE_WORKER_SECRET        bearer token (never in argv)
  INDEXEDCODE_WORKER_TEST_FP       fingerprint to advertise
  INDEXEDCODE_WORKER_EMBED_STATUS  HTTP status to return from /v1/embed
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# this script runs as a bare child process without pytest's pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indexedcode.httpserver import LoopbackHTTPServer  # noqa: E402


def build_handler(secret, fingerprint, embed_status):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self):
            return self.headers.get("Authorization") == f"Bearer {secret}"

        def do_GET(self):
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            if self.path == "/v1/model":
                return self._send(200, {
                    "model": "fake-embed",
                    "revision": "r1",
                    "backend": "mlx",
                    "dimension": 3,
                    "pooling": "mean",
                    "normalized": True,
                    "fingerprint": fingerprint,
                })
            if self.path == "/v1/status":
                return self._send(200, {
                    "state": "ready",
                    "rss_mb": 123.0,
                    "accelerator_allocated_mb": 45.0,
                })
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/v1/embed":
                if embed_status != 200:
                    return self._send(embed_status, {"error": "overloaded"})
                texts = req.get("texts", [])
                return self._send(200, {
                    "vectors": [[0.5, 0.5, 0.7071067811865476]] * len(texts),
                    "dimension": 3,
                    "fingerprint": fingerprint,
                })
            if self.path == "/v1/summarize":
                return self._send(200, {
                    "summary": "fake summary",
                    "fingerprint": fingerprint,
                })
            return self._send(404, {"error": "not found"})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="ready")
    args = parser.parse_args()

    port = int(os.environ["INDEXEDCODE_WORKER_PORT"])
    secret = os.environ.get("INDEXEDCODE_WORKER_SECRET", "")
    from indexedcode.workers.fingerprint import model_fingerprint

    # advertise the true fingerprint of the model this fake serves, so a
    # supervisor expecting that model validates successfully
    fingerprint = os.environ.get("INDEXEDCODE_WORKER_TEST_FP") or (
        model_fingerprint(
            model="fake-embed", revision="r1", backend="mlx", dimension=3,
        )
    )
    embed_status = int(os.environ.get("INDEXEDCODE_WORKER_EMBED_STATUS", 200))
    if args.mode == "mismatch":
        # a syntactically valid fingerprint for a *different* model
        fingerprint = "sha256:" + "f" * 64

    if args.mode == "exit":
        sys.exit(3)
    if args.mode == "hang":
        time.sleep(3600)
        return

    server = LoopbackHTTPServer(("127.0.0.1", port), build_handler(
        secret, fingerprint, embed_status))
    if args.mode == "ignore_sigterm":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("READY", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
