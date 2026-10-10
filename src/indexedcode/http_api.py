"""HTTP adapter (plan §8, §4.1).

Loopback-only by default; bearer authentication when a token is
configured (required for any non-loopback bind, enforced by callers).
Requests are capped and responses bounded with explicit truncation.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

from indexedcode import retrieval
from indexedcode.httpserver import LoopbackHTTPServer

MAX_REQUEST_BYTES = 32_768
MAX_RESPONSE_CHARS = retrieval.MAX_CHARS_CAP


def create_server(conn, root, token=None, host="127.0.0.1", port=0,
                  max_request_bytes=MAX_REQUEST_BYTES,
                  max_response_chars=retrieval.DEFAULT_MAX_CHARS):
    """Build a ready-to-run HTTP server bound to a loopback port."""
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise ValueError(
            "non-loopback binds require a bearer token (encrypted"
            " transport is the operator's responsibility)"
        )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            if token is None:
                return True
            return self.headers.get("Authorization") == f"Bearer {token}"

        def _dispatch(self, method):
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            try:
                if method == "GET" and self.path == "/v1/status":
                    return self._send(200, retrieval.get_status(conn, root))
                if method != "POST":
                    return self._send(404, {"error": "not found"})

                length = int(self.headers.get("Content-Length", 0))
                if length > max_request_bytes:
                    # drain nothing; reject before reading the body
                    return self._send(413, {"error": "payload too large"})
                raw = self.rfile.read(length) if length else b"{}"
                request = json.loads(raw)

                if self.path == "/v1/search":
                    result = retrieval.search(
                        conn, root, request.get("query", ""),
                        mode=request.get("mode", "hybrid"),
                        limit=request.get("limit",
                                          retrieval.DEFAULT_RESULTS),
                        cursor=request.get("cursor"),
                        max_chars=max_response_chars,
                    )
                elif self.path == "/v1/context":
                    result = retrieval.get_context(
                        conn, root,
                        entity_type=request.get("entity_type"),
                        entity_id=request.get("entity_id"),
                        path=request.get("path"),
                        start_line=request.get("start_line"),
                        end_line=request.get("end_line"),
                        cursor=request.get("cursor"),
                        max_chars=max_response_chars,
                    )
                elif self.path == "/v1/references":
                    result = retrieval.get_references(
                        conn, root, request.get("symbol", ""),
                        limit=request.get("limit",
                                          retrieval.DEFAULT_REFERENCES),
                        cursor=request.get("cursor"),
                    )
                else:
                    return self._send(404, {"error": "not found"})
                return self._send(200, result)
            except retrieval.SearchError as exc:
                return self._send(400, {"error": str(exc)})
            except retrieval.ContextError as exc:
                return self._send(400, {"error": str(exc)})
            except (ValueError, KeyError) as exc:
                return self._send(400, {"error": f"bad request: {exc}"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    server = LoopbackHTTPServer((host, port), Handler)
    server.base_url = f"http://{host}:{server.server_address[1]}"
    return server
