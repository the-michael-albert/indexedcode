"""Threaded HTTP server that never performs reverse-DNS lookups.

``http.server.HTTPServer.server_bind`` calls ``socket.getfqdn`` to fill in
``server_name``. On hosts with a slow resolver that lookup can take tens of
seconds (35s on GitHub's macOS runners), and the socket does not listen until
it returns, so clients time out waiting for a server that is merely starting.
Nothing here uses ``server_name``, so the lookup is skipped.
"""

from __future__ import annotations

import socketserver
from http.server import ThreadingHTTPServer


class LoopbackHTTPServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port
