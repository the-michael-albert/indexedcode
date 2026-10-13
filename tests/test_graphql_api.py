"""GraphQL adapter: read-only node queries, traversal, semantic search."""

import json
import urllib.error
import urllib.request

import pytest

from indexedcode.http_api import create_server
from tests.conftest import FakeEmbedder


@pytest.fixture()
def server(indexed_repo):
    conn, root = indexed_repo
    httpd = create_server(
        conn, str(root), token="secret-token", embedder=FakeEmbedder()
    )
    import threading

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def gql_request(server, payload, token="secret-token"):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        server.base_url + "/v1/graphql", data=data, method="POST"
    )
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as response:
        return response.status, json.loads(response.read())


def gql(indexed_repo, query, variables=None, **kwargs):
    from indexedcode import graphql_api

    conn, root = indexed_repo
    return graphql_api.execute(conn, str(root), query, variables=variables, **kwargs)


def test_file_lookup_by_path(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { file(path: "src/config_loader.py") { id name language isTest } }
    """,
    )
    assert result.errors is None
    node = result.data["file"]
    assert node["language"] == "python"
    assert node["isTest"] is False
    assert node["id"] and node["name"] == "config_loader.py"


def test_file_lookup_unknown_path_returns_null(indexed_repo):
    result = gql(indexed_repo, '{ file(path: "src/nope.py") { id } }')
    assert result.errors is None
    assert result.data["file"] is None


def test_file_children_list_classes_and_functions(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { file(path: "src/config_loader.py") {
            classes { id name qualifiedName kind }
            functions { id name qualifiedName signature }
        } }
    """,
    )
    assert result.errors is None
    data = result.data["file"]
    assert [c["name"] for c in data["classes"]] == ["Config"]
    assert data["classes"][0]["kind"] == "class"
    assert [f["name"] for f in data["functions"]] == ["parse_config"]


def test_function_lookup_by_qualified_name(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { function(qualifiedName: "parse_config") {
            id name qualifiedName signature startLine endLine source
        } }
    """,
    )
    assert result.errors is None
    node = result.data["function"]
    assert node["name"] == "parse_config"
    assert node["signature"] and "parse_config" in node["signature"]
    assert node["source"] and node["startLine"] >= 1


def test_class_lookup_by_qualified_name(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { class(qualifiedName: "Config") {
            id name qualifiedName kind startLine endLine
        } }
    """,
    )
    assert result.errors is None
    node = result.data["class"]
    assert node["kind"] == "class"
    assert node["name"] == "Config"


def test_module_lookup_by_path_and_children(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { module(path: "src") { id name children { name path } files { path } } }
    """,
    )
    assert result.errors is None
    node = result.data["module"]
    assert node["name"] == "src"
    assert [f["path"] for f in node["files"]] == [
        "src/config_loader.py",
        "src/reporting.py",
    ]


def test_class_methods(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { class(qualifiedName: "Config") { methods { name qualifiedName signature } } }
    """,
    )
    assert result.errors is None
    methods = result.data["class"]["methods"]
    assert [m["name"] for m in methods] == ["load"]
    assert methods[0]["qualifiedName"] == "Config.load"


def test_class_nested_classes_empty(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { class(qualifiedName: "Config") { nestedClasses { name } } }
    """,
    )
    assert result.errors is None
    assert result.data["class"]["nestedClasses"] == []


def test_callers_of_a_function(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { function(qualifiedName: "parse_config") {
            callers { qualifiedName }
        } }
    """,
    )
    assert result.errors is None
    assert [c["qualifiedName"] for c in result.data["function"]["callers"]] == [
        "Config.load",
        "test_parse_config",
    ]


def test_callees_of_a_function(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { function(qualifiedName: "Config.load") {
            callees { qualifiedName }
        } }
    """,
    )
    assert result.errors is None
    assert [c["qualifiedName"] for c in result.data["function"]["callees"]] == [
        "parse_config",
    ]


def test_related_edges_filtered_by_relation(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { function(qualifiedName: "Config.load") {
            related(relation: "calls") { relation targetType targetName }
        } }
    """,
    )
    assert result.errors is None
    edges = result.data["function"]["related"]
    assert edges, "expected at least one related edge"
    assert all(e["relation"] == "calls" for e in edges)
    assert "parse_config" in [e["targetName"] for e in edges]


def test_related_edges_on_file_node(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { file(path: "tests/test_config_loader.py") {
            related { relation targetType targetName }
        } }
    """,
    )
    assert result.errors is None
    relations = {e["relation"] for e in result.data["file"]["related"]}
    assert "imports" in relations


# --- semantic search -------------------------------------------------------------


def test_semantic_search_returns_entries_with_nodes(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { semanticSearch(query: "parse json configuration", limit: 3) {
            score evidence
            node { __typename
                   ... on FunctionNode { qualifiedName }
                   ... on ClassNode { qualifiedName } }
        } }
    """,
        embedder=FakeEmbedder(),
    )
    assert result.errors is None
    entries = result.data["semanticSearch"]
    assert entries
    qnames = [e["node"]["qualifiedName"] for e in entries if e["node"]]
    assert qnames, "at least one entry must resolve to a node"


def test_semantic_search_ranks_identical_source_first(indexed_repo):
    result = gql(
        indexed_repo,
        """
        query($q: String!) {
            semanticSearch(query: $q, limit: 5) {
                node { ... on FunctionNode { qualifiedName } }
            }
        }
    """,
        variables={"q": "def parse_config(text):\n    return json.loads(text)"},
        embedder=FakeEmbedder(),
    )
    assert result.errors is None
    entries = result.data["semanticSearch"]
    qnames = [e["node"]["qualifiedName"] for e in entries if e["node"]]
    assert qnames[0] == "parse_config"


def test_semantic_search_rejects_empty_query(indexed_repo):
    result = gql(
        indexed_repo,
        '{ semanticSearch(query: "   ") { score } }',
        embedder=FakeEmbedder(),
    )
    assert result.errors
    assert "non-empty" in result.errors[0].message


def test_semantic_search_without_embedder_reports_error(indexed_repo):
    result = gql(indexed_repo, '{ semanticSearch(query: "parse json") { score } }')
    assert result.errors
    assert "embedding model" in result.errors[0].message


def test_semantic_search_rejects_unknown_mode(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { semanticSearch(query: "x", mode: BOGUS) { score } }
    """,
    )
    assert result.errors


def test_semantic_search_hybrid_mode_works(indexed_repo):
    result = gql(
        indexed_repo,
        """
        { semanticSearch(query: "parse_config", mode: HYBRID) {
            node { ... on FunctionNode { qualifiedName } }
        } }
    """,
        embedder=FakeEmbedder(),
    )
    assert result.errors is None
    qnames = [
        e["node"]["qualifiedName"] for e in result.data["semanticSearch"] if e["node"]
    ]
    assert qnames[0] == "parse_config"


# --- HTTP endpoint -----------------------------------------------------------------


def test_graphql_endpoint_serves_file_query(server):
    status, body = gql_request(
        server,
        {
            "query": '{ file(path: "src/config_loader.py") { language isTest } }',
        },
    )
    assert status == 200
    assert body["data"]["file"]["language"] == "python"
    assert body["data"]["file"]["isTest"] is False
    assert "errors" not in body


def test_graphql_endpoint_runs_semantic_search(server):
    status, body = gql_request(
        server,
        {
            "query": '{ semanticSearch(query: "parse json", limit: 2) {'
            " node { ... on FunctionNode { qualifiedName } } } }",
        },
    )
    assert status == 200
    assert body["data"]["semanticSearch"], body


def test_graphql_endpoint_requires_token(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        gql_request(server, {"query": '{ file(path: "x") { id } }'}, token=None)
    assert excinfo.value.code == 401


def test_graphql_syntax_error_returns_200_with_errors(server):
    status, body = gql_request(server, {"query": "{"})
    assert status == 200
    assert body.get("errors")


def test_graphql_unknown_field_returns_200_with_errors(server):
    status, body = gql_request(server, {"query": "{ nope }"})
    assert status == 200
    assert body.get("errors")


def test_graphql_variables_round_trip(server):
    status, body = gql_request(
        server,
        {
            "query": "query($p: String!) { file(path: $p) { language } }",
            "variables": {"p": "src/reporting.py"},
        },
    )
    assert status == 200
    assert body["data"]["file"]["language"] == "python"


def test_graphql_oversized_payload_rejected(server):
    big = {"query": "x" * 40_000}
    req = urllib.request.Request(
        server.base_url + "/v1/graphql",
        data=json.dumps(big).encode(),
        method="POST",
    )
    req.add_header("Authorization", "Bearer secret-token")
    req.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 413


def test_graphql_has_no_mutation_type(server):
    status, body = gql_request(server, {"query": "mutation { x }"})
    assert status == 200
    assert body.get("errors")


# --- bounding, introspection, docs ---------------------------------------------------


def test_source_field_is_truncated_to_max_chars(indexed_repo):
    conn, _ = indexed_repo
    conn.execute(
        "UPDATE functions SET source = ? WHERE qualified_name = 'parse_config'",
        ("x" * 40_000,),
    )
    result = gql(
        indexed_repo,
        '{ function(qualifiedName: "parse_config") { source } }',
        max_chars=2_000,
    )
    assert result.errors is None
    assert len(result.data["function"]["source"]) <= 2_000


def test_introspection_is_enabled(server):
    status, body = gql_request(
        server,
        {
            "query": "{ __schema { queryType { name } } }",
        },
    )
    assert status == 200
    assert body["data"]["__schema"]["queryType"]["name"] == "Query"


def test_llms_txt_documents_graphql_endpoint(server):
    req = urllib.request.Request(server.base_url + "/llms.txt")
    with urllib.request.urlopen(req, timeout=10) as response:
        text = response.read().decode()
    assert "/v1/graphql" in text


# --- graph UI -----------------------------------------------------------------


def _packaged(name):
    from importlib import resources

    return resources.files("indexedcode") / "web" / name


def test_vendored_graph_libraries_are_packaged():
    for name in ("graphology.min.js", "sigma.min.js"):
        assert _packaged(name).is_file(), name


def test_index_html_uses_only_local_scripts():
    html = _packaged("index.html").read_text("utf-8")
    import re

    srcs = re.findall(r'<script[^>]+src="([^"]+)"', html)
    assert srcs, "expected vendored script tags"
    for src in srcs:
        assert src.startswith("/web/"), f"external script not allowed: {src}"
        assert _packaged(src.split("/")[-1]).is_file(), src


def test_graph_assets_are_served_without_a_token(server):
    req = urllib.request.Request(server.base_url + "/web/graphology.min.js")
    with urllib.request.urlopen(req, timeout=10) as response:
        assert response.status == 200
        assert "javascript" in response.headers["Content-Type"]
        assert b"graphology" in response.read().lower()


def test_csp_allows_same_origin_scripts(server):
    with urllib.request.urlopen(server.base_url + "/", timeout=10) as response:
        csp = response.headers["Content-Security-Policy"]
    script_src = next(d for d in csp.split(";") if "script-src" in d)
    assert "'self'" in script_src


def test_ui_has_graph_view():
    html = _packaged("index.html").read_text("utf-8")
    assert 'id="graph"' in html
    assert "graphology" in html
    assert "semanticSearch" in html


def test_modules_root_field_lists_every_module(indexed_repo):
    result = gql(indexed_repo, '{ modules { name path } }')
    assert result.errors is None
    paths = [m["path"] for m in result.data["modules"]]
    assert {"src", "tests"} <= set(paths)


def test_module_files_expose_function_callees(indexed_repo):
    result = gql(indexed_repo, '''
        { module(path: "src") { files { path
            functions { qualifiedName callees { qualifiedName } }
            classes { methods { qualifiedName callees { qualifiedName } } } } } }
    ''')
    assert result.errors is None
    files = result.data["module"]["files"]
    config = next(f for f in files if f["path"].endswith("config_loader.py"))
    load = next(fn for cls in config["classes"] for fn in cls["methods"]
                if fn["qualifiedName"] == "Config.load")
    assert "parse_config" in [c["qualifiedName"] for c in load["callees"]]


def test_module_files_include_descendant_modules(db, tmp_path):
    from tests.conftest import insert_file, insert_module

    root = insert_module(db, name="app", relative_path="app")
    child = insert_module(db, name="models", relative_path="app/models",
                          parent_module_id=root)
    insert_file(db, root, folder="app", file="main.py")
    insert_file(db, child, folder="app/models", file="user.py")

    result = gql((db, tmp_path), '{ module(path: "app") { files { path } } }')
    assert result.errors is None
    assert [f["path"] for f in result.data["module"]["files"]] == [
        "app/main.py", "app/models/user.py",
    ]


def _http_status(server, path, payload=None, token="secret-token"):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        server.base_url + path, data=data, method="GET" if data is None else "POST"
    )
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def test_depth_limit_rejects_deeply_nested_queries(indexed_repo):
    from indexedcode import graphql_api

    nested = "{ qualifiedName callers " * (graphql_api.MAX_DEPTH + 1)
    query = '{ function(qualifiedName: "parse_config") ' + nested + "}" * (
        graphql_api.MAX_DEPTH + 2
    )
    result = gql(indexed_repo, query)
    assert result.errors and "depth" in result.errors[0].message


def test_depth_limit_counts_fragments(indexed_repo):
    from indexedcode import graphql_api

    fragments = "".join(
        f"fragment F{i} on FunctionNode {{ callers {{ ...F{i + 1} }} }} "
        for i in range(graphql_api.MAX_DEPTH + 1)
    )
    fragments += "fragment F%d on FunctionNode { name }" % (graphql_api.MAX_DEPTH + 1)
    result = gql(indexed_repo, '{ function(qualifiedName: "x") { ...F0 } } ' + fragments)
    assert result.errors and "depth" in result.errors[0].message


def test_shallow_query_still_passes(indexed_repo):
    result = gql(
        indexed_repo,
        '{ function(qualifiedName: "parse_config") { callers { callers { name } } } }',
    )
    assert result.errors is None


def test_syntax_error_is_reported_not_raised(indexed_repo):
    result = gql(indexed_repo, "{ file(")
    assert result.data is None and result.errors


def test_shared_schema_holds_no_per_request_state(db, tmp_path):
    from tests.conftest import insert_module

    insert_module(db, name="one", relative_path="one")
    first = gql((db, tmp_path), "{ modules { path } }")
    insert_module(db, name="two", relative_path="two")
    second = gql((db, tmp_path), "{ modules { path } }")
    assert [m["path"] for m in first.data["modules"]] == ["one"]
    assert [m["path"] for m in second.data["modules"]] == ["one", "two"]


def test_bad_graphql_requests_are_400(server):
    for payload in ({}, {"query": 5}, {"query": "{ modules { path } }", "variables": []}):
        status, _, body = _http_status(server, "/v1/graphql", payload)
        assert status == 400, payload
        assert "error" in json.loads(body)


def test_graphql_requires_token(server):
    status, _, _ = _http_status(
        server, "/v1/graphql", {"query": "{ modules { path } }"}, token=None
    )
    assert status == 401


def test_vendored_js_served_without_token_and_allowlisted(server):
    status, headers, body = _http_status(server, "/web/sigma.min.js", token=None)
    assert status == 200 and body
    assert headers["Content-Type"].startswith("application/javascript")
    assert _http_status(server, "/web/index.html", token=None)[0] == 404
    assert _http_status(server, "/web/..%2Fhttp_api.py", token=None)[0] == 404


def test_callee_resolution_prefers_same_file(db, tmp_path):
    from tests.conftest import insert_file, insert_function, insert_module

    mod = insert_module(db, name="app", relative_path="app")
    a = insert_file(db, mod, folder="app", file="a.py")
    b = insert_file(db, mod, folder="app", file="b.py")
    insert_function(db, b, name="helper", qualified_name="b.helper")
    insert_function(db, a, name="helper", qualified_name="a.helper")
    caller = insert_function(db, a, name="run", qualified_name="a.run")
    db.execute(
        "INSERT INTO relationships (source_type, source_id, relation, target_name,"
        " line_number, origin_file_id, resolution, snippet)"
        " VALUES ('function', ?, 'calls', 'helper', 1, ?, 'unresolved', 'helper()')",
        (caller, a),
    )
    result = gql(
        (db, tmp_path),
        '{ function(qualifiedName: "a.run") { callees { id qualifiedName } } }',
    )
    assert result.errors is None
    assert [c["qualifiedName"] for c in result.data["function"]["callees"]] == [
        "a.helper"
    ]
