import socket

from indexedcode.httpserver import LoopbackHTTPServer
from http.server import BaseHTTPRequestHandler


def test_bind_does_not_reverse_resolve(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("socket.getfqdn must not be called while binding")

    monkeypatch.setattr(socket, "getfqdn", forbidden)
    server = LoopbackHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    try:
        assert server.server_name == "127.0.0.1"
        assert server.server_port == server.server_address[1]
    finally:
        server.server_close()
