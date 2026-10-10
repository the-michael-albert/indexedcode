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


def test_non_loopback_bind_requires_a_token(indexed_repo):
    from indexedcode.http_api import create_server as build

    conn, root = indexed_repo
    with pytest.raises(ValueError, match="token"):
        build(conn, str(root), token=None, host="0.0.0.0")


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


# --- web UI and queue ---------------------------------------------------------


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status, response.headers, response.read()


def test_root_serves_ui_page_without_a_token(server):
    # the page itself holds no index data; its API calls still need the token
    status, headers, body = _get(server.base_url + "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"IndexedCode" in body
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_ui_is_packaged_as_package_data():
    from importlib import resources

    page = resources.files("indexedcode") / "web" / "index.html"
    assert page.is_file()


def test_queue_requires_token(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        request(server, "/v1/queue", token=None)
    assert excinfo.value.code == 401


def test_queue_empty_when_fully_indexed(server):
    status, body = request(server, "/v1/queue")
    assert status == 200
    assert body["total"] == 0 and body["jobs"] == []
    assert body["last_updated_ms"] is not None
    assert body["recent_functions"]
    assert {"qualified_name", "path", "start_line", "updated_at_ms"} <= set(
        body["recent_functions"][0]
    )


def test_queue_lists_pending_file_with_its_functions(indexed_repo):
    from indexedcode.reconcile import reconcile

    conn, root = indexed_repo
    target = root / "src" / "config_loader.py"
    target.write_text(target.read_text() + "\n# changed\n")
    reconcile(conn, str(root), now_ms=5_000)

    httpd = create_server(conn, str(root), token=None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        _, _, raw = _get(httpd.base_url + "/v1/queue")
        body = json.loads(raw)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert body["total"] == 1
    job = body["jobs"][0]
    assert job["status"] == "pending"
    assert job["path"] == "src/config_loader.py"
    assert job["queued_at_ms"] == 5_000
    assert "parse_config" in [f["qualified_name"] for f in job["functions"]]
    assert job["function_count"] == len(job["functions"])


def test_queue_orders_running_before_pending_and_reports_errors(indexed_repo):
    from indexedcode.dal import enqueue_job

    conn, _ = indexed_repo
    enqueue_job(conn, command="refresh_file", folder="src", file="a.py",
                now_ms=10)
    enqueue_job(conn, command="refresh_file", folder="src", file="b.py",
                now_ms=20)
    conn.execute("UPDATE jobs SET status='failed', last_error='boom',"
                 " attempt_count=3 WHERE file='a.py'")
    from indexedcode.retrieval import get_queue

    body = get_queue(conn, now_ms=99)
    assert [j["status"] for j in body["jobs"]] == ["pending", "failed"]
    failed = body["jobs"][1]
    assert failed["last_error"] == "boom" and failed["attempts"] == 3
    assert failed["function_count"] == 0 and failed["functions"] == []


def test_queue_limit_marks_truncation(indexed_repo):
    from indexedcode.dal import enqueue_job
    from indexedcode.retrieval import get_queue

    conn, _ = indexed_repo
    for i in range(5):
        enqueue_job(conn, command="refresh_file", folder="src",
                    file=f"f{i}.py", now_ms=i + 1)
    body = get_queue(conn, now_ms=100, limit=2)
    assert body["total"] == 5 and body["truncated"] is True
    assert len(body["jobs"]) == 2


def test_foreign_host_header_is_rejected_without_a_token(indexed_repo):
    conn, root = indexed_repo
    httpd = create_server(conn, str(root), token=None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        for path in ("/", "/v1/status", "/v1/queue"):
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                _get(httpd.base_url + path, {"Host": "evil.example:8765"})
            assert excinfo.value.code == 403
        # loopback host names remain accepted
        for host in ("localhost:8765", "127.0.0.1:8765"):
            status, _, _ = _get(httpd.base_url + "/v1/status", {"Host": host})
            assert status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- llms.txt -------------------------------------------------------------------


def test_llms_txt_is_served_without_a_token_and_uses_the_real_base_url(server):
    status, headers, body = _get(server.base_url + "/llms.txt")
    text = body.decode()
    assert status == 200
    assert headers["Content-Type"].startswith("text/plain")
    assert text.startswith("# IndexedCode\n")
    assert f"{server.base_url}/v1/search" in text
    assert "{base}" not in text and "{auth}" not in text
    # a token is configured on this fixture: the doc must say how to send it
    assert "Authorization: Bearer" in text


def test_llms_txt_without_token_says_access_is_local(indexed_repo):
    conn, root = indexed_repo
    httpd = create_server(conn, str(root), token=None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        _, _, body = _get(httpd.base_url + "/llms.txt")
        assert "No token is configured" in body.decode()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_llms_txt_follows_llmstxt_format_and_documents_every_route(server):
    _, _, body = _get(server.base_url + "/llms.txt")
    lines = body.decode().splitlines()
    assert lines[0].startswith("# ")                       # H1 title first
    assert lines[2].startswith("> ")                       # blockquote summary
    assert [l for l in lines if l.startswith("## ")]       # H2 sections
    text = "\n".join(lines)
    for route in ("/v1/status", "/v1/queue", "/v1/search", "/v1/context",
                  "/v1/references"):
        assert route in text


def test_llms_txt_is_packaged():
    from importlib import resources

    assert (resources.files("indexedcode") / "web" / "llms.txt").is_file()


def test_llms_txt_rejects_foreign_host_without_token(indexed_repo):
    conn, root = indexed_repo
    httpd = create_server(conn, str(root), token=None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(httpd.base_url + "/llms.txt", {"Host": "evil.example"})
        assert excinfo.value.code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
