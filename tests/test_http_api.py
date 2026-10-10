"""HTTP adapter (plan §8, §4.1): loopback bind, bearer auth when
configured, payload limits, and bounded responses."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from indexedcode.http_api import create_server


@pytest.fixture()
def server(indexed_repo):
    conn, root = indexed_repo
    httpd = create_server(conn, str(root), token="secret-token")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def request(server, path, payload=None, token="secret-token"):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(server.base_url + path, data=data,
                                 method="POST" if data else "GET")
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status, json.loads(response.read())


def test_binds_loopback_only(server):
    assert server.base_url.startswith("http://127.0.0.1:")


def test_missing_token_is_unauthorized(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(server, "/v1/status", token=None)
    assert excinfo.value.code == 401


def test_wrong_token_is_unauthorized(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(server, "/v1/status", token="wrong")
    assert excinfo.value.code == 401


def test_status_endpoint_with_valid_token(server):
    status, body = request(server, "/v1/status")
    assert status == 200
    assert body["coverage"]["ready_files"] == 3


def test_search_endpoint(server):
    status, body = request(server, "/v1/search", {"query": "parse_config"})
    assert status == 200
    assert body["results"]
    assert body["results"][0]["qualified_name"] == "parse_config"


def test_context_endpoint(server):
    status, body = request(
        server, "/v1/context", {"path": "src/config_loader.py"},
    )
    assert status == 200
    assert "parse_config" in body["source"]


def test_references_endpoint(server):
    status, body = request(
        server, "/v1/references", {"symbol": "parse_config"},
    )
    assert status == 200
    assert body["references"]


def test_oversized_payload_rejected(server):
    big = {"query": "x" * 40_000}
    req = urllib.request.Request(
        server.base_url + "/v1/search", data=json.dumps(big).encode(),
        method="POST",
    )
    req.add_header("Authorization", "Bearer secret-token")
    req.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 413


def test_response_is_bounded_with_truncation_flag(indexed_repo):
    conn, root = indexed_repo
    big = "\n".join(f"line_{i} = {i}" for i in range(5_000))
    (root / "src" / "huge.py").write_text(big + "\n")
    httpd = create_server(conn, str(root), token=None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(
            httpd.base_url + "/v1/context",
            data=json.dumps({"path": "src/huge.py"}).encode(), method="POST",
        )
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read()
            parsed = json.loads(body)
        assert len(body) <= 32_768
        assert parsed["truncated"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_unknown_endpoint_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(server, "/v1/nope")
    assert excinfo.value.code == 404


def test_no_token_configured_allows_local_access(indexed_repo):
    conn, root = indexed_repo
    httpd = create_server(conn, str(root), token=None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(httpd.base_url + "/v1/status",
                                    timeout=10) as response:
            assert response.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
