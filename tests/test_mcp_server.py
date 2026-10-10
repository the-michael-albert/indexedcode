"""MCP server: a real MCP client connects and calls all four tools
(plan Phase 5 gate)."""

import asyncio
import json
import sys

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def run_mcp(db_path, root, script):
    """Spawn the MCP server over stdio and run an async client script."""

    async def session():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "indexedcode.mcp", "--db", str(db_path),
                  "--root", str(root)],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                return await script(client)

    return asyncio.run(session())


@pytest.fixture(scope="module")
def mcp_repo(tmp_path_factory):
    """One indexed repo per module; MCP servers spawn per session."""
    from indexedcode.db import connect, initialize
    from tests.conftest import index_repo, write_repo_sources

    root = tmp_path_factory.mktemp("mcp_repo")
    write_repo_sources(root)
    db_path = root / "index.sqlite"
    conn = connect(db_path, writer=True)
    initialize(conn)
    index_repo(conn, root)
    conn.close()
    return db_path, root


def test_lists_all_four_tools(mcp_repo):
    db_path, root = mcp_repo

    async def script(client):
        tools = await client.list_tools()
        return [tool.name for tool in tools.tools]

    names = run_mcp(db_path, root, script)
    assert set(names) >= {
        "code_search", "code_context", "code_references", "code_status",
    }


def test_code_search_tool(mcp_repo):
    db_path, root = mcp_repo

    async def script(client):
        return await client.call_tool(
            "code_search", {"query": "parse_config"},
        )

    result = run_mcp(db_path, root, script)
    payload = json.loads(result.content[0].text)
    assert payload["results"]
    assert payload["results"][0]["qualified_name"] == "parse_config"
    assert len(result.content[0].text) <= 32_768


def test_code_context_and_references_and_status_tools(mcp_repo):
    db_path, root = mcp_repo

    async def script(client):
        context = await client.call_tool(
            "code_context", {"path": "src/config_loader.py"},
        )
        references = await client.call_tool(
            "code_references", {"symbol": "parse_config"},
        )
        status = await client.call_tool("code_status", {})
        return context, references, status

    context, references, status = run_mcp(db_path, root, script)
    assert "parse_config" in json.loads(context.content[0].text)["source"]
    assert json.loads(references.content[0].text)["references"]
    assert json.loads(status.content[0].text)["coverage"]["ready_files"] == 3


def test_oversized_query_is_a_tool_error(mcp_repo):
    db_path, root = mcp_repo

    async def script(client):
        return await client.call_tool(
            "code_search", {"query": "x" * 40_000},
        )

    result = run_mcp(db_path, root, script)
    assert result.isError
