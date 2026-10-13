"""HTTP adapter (plan §8, §4.1).

Loopback-only by default; bearer authentication when a token is
configured (required for any non-loopback bind, enforced by callers).
Requests are capped and responses bounded with explicit truncation.
"""

from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler
from importlib import resources
from urllib.parse import urlsplit

from indexedcode import graphql_api, retrieval
from indexedcode.httpserver import LoopbackHTTPServer

logger = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 32_768
MAX_RESPONSE_CHARS = retrieval.MAX_CHARS_CAP
_VENDORED_JS = frozenset({"graphology.min.js", "sigma.min.js", "fa2.min.js"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})


def _load_web(name: str) -> bytes:
    return (resources.files("indexedcode") / "web" / name).read_bytes()


def create_server(conn, root, token=None, host="127.0.0.1", port=0,
                  max_request_bytes=MAX_REQUEST_BYTES,
                  max_response_chars=retrieval.DEFAULT_MAX_CHARS,
                  embedder=None, model_fingerprint=None,
                  inference_status=None):
    """Build a ready-to-run HTTP server bound to a loopback port.

    ``embedder`` (texts -> vectors) enables semantic and hybrid search;
    ``model_fingerprint`` limits semantic matching to that model's vectors.
    """
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise ValueError(
            "non-loopback binds require a bearer token (encrypted"
            " transport is the operator's responsibility)"
        )

    ui = _load_web("index.html")
    llms_template = _load_web("llms.txt").decode("utf-8")
    auth_note = (
        "Every `/v1/` request needs the header `Authorization: Bearer <token>`;"
        " ask the operator for the token."
        if token is not None else
        "No token is configured; access is limited to this machine."
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.debug("http %s", format % args)

        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body, content_type="text/html; charset=utf-8"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'unsafe-inline' 'self';"
                             " style-src 'unsafe-inline';"
                             " connect-src 'self'; base-uri 'none';"
                             " form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _host_allowed(self):
            # Without a token, only a loopback Host may read the index; this
            # stops DNS-rebinding pages from reading local source.
            if token is not None or host not in _LOOPBACK_HOSTS:
                return True
            name = urlsplit("//" + (self.headers.get("Host") or "")).hostname
            return name in ("127.0.0.1", "localhost", "::1")

        def _authorized(self):
            if token is None:
                return True
            return self.headers.get("Authorization") == f"Bearer {token}"

        def _dispatch(self, method):
            if not self._host_allowed():
                return self._send(403, {"error": "forbidden host"})
            target = urlsplit(self.path)
            if method == "GET" and target.path in ("/", "/index.html"):
                # the page holds no index data; it asks for a token itself
                return self._send_html(ui)
            if method == "GET" and target.path == "/llms.txt":
                # static usage docs for agents: no index data, so no token
                return self._send_html(
                    llms_template.replace("{base}", server.base_url)
                    .replace("{auth}", auth_note).encode("utf-8"),
                    "text/plain; charset=utf-8",
                )
            if method == "GET" and target.path.startswith("/web/"):
                # vendored UI libraries: static code, no index data
                name = target.path[len("/web/"):]
                if name not in _VENDORED_JS:
                    return self._send(404, {"error": "not found"})
                return self._send_html(
                    _load_web(name), "application/javascript; charset=utf-8")
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            try:
                if method == "GET" and target.path == "/v1/status":
                    return self._send(200, retrieval.get_status(
                        conn, root,
                        inference_status=(inference_status()
                                          if inference_status else None)))
                if method == "GET" and target.path == "/v1/queue":
                    return self._send(200, retrieval.get_queue(
                        conn, now_ms=int(time.time() * 1000)))
                if method != "POST":
                    return self._send(404, {"error": "not found"})

                length = int(self.headers.get("Content-Length", 0))
                if length > max_request_bytes:
                    # drain nothing; reject before reading the body
                    return self._send(413, {"error": "payload too large"})
                raw = self.rfile.read(length) if length else b"{}"
                request = json.loads(raw)

                if self.path == "/v1/graphql":
                    query = request.get("query")
                    variables = request.get("variables")
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError("query must be a non-empty string")
                    if variables is not None and not isinstance(variables, dict):
                        raise ValueError("variables must be an object")
                    gql = graphql_api.execute(
                        conn, root, query,
                        variables=variables,
                        operation_name=request.get("operationName"),
                        embedder=embedder,
                        model_fingerprint=model_fingerprint,
                        max_chars=max_response_chars,
                    )
                    payload = {"data": gql.data}
                    if gql.errors:
                        payload["errors"] = [e.formatted for e in gql.errors]
                    return self._send(200, payload)
                if self.path == "/v1/search":
                    result = retrieval.search(
                        conn, root, request.get("query", ""),
                        mode=request.get("mode", "hybrid"),
                        limit=request.get("limit",
                                          retrieval.DEFAULT_RESULTS),
                        embedder=embedder,
                        model_fingerprint=model_fingerprint,
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
