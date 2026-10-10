"""MCP server: read-only retrieval tools over stdio (plan §8.1).

``python -m indexedcode.mcp --db PATH --root PATH`` exposes code_search,
code_context, code_references, and code_status. Tools never execute
anything, never write, and return bounded JSON.
"""

from __future__ import annotations

import argparse
import json
import sys

from mcp.server.fastmcp import FastMCP

from indexedcode import retrieval

mcp = FastMCP("indexedcode")


@mcp.tool()
def code_search(query: str, mode: str = "hybrid", limit: int = 5) -> str:
    """Find entities by behavior, symbol, or text.

    Returns up to `limit` ranked module/file/class/function results with
    evidence, freshness, and IDs. `mode`: lexical | semantic | hybrid |
    symbol. Hybrid degrades to lexical when no embedding model is
    available (flagged explicitly in the result).
    """
    result = retrieval.search(_conn(), _root(), query, mode=mode,
                              limit=limit)
    return json.dumps(result)


@mcp.tool()
def code_context(path: str = "", entity_type: str = "",
                 entity_id: int = 0, start_line: int = 0,
                 end_line: int = 0, cursor: str = "") -> str:
    """Fetch exact current source for an entity or bounded path/line range.

    Indexed snapshots are marked current or stale by comparing the
    on-disk hash; explicit path reads return current disk source. Pages
    default to 120 lines with an opaque continuation cursor.
    """
    result = retrieval.get_context(
        _conn(), _root(),
        entity_type=entity_type or None,
        entity_id=entity_id or None,
        path=path or None,
        start_line=start_line or None,
        end_line=end_line or None,
        cursor=cursor or None,
    )
    return json.dumps(result)


@mcp.tool()
def code_references(symbol: str, limit: int = 10) -> str:
    """Find syntactic candidate imports/calls/inheritance/usages/tests.

    Returns up to ten source-grounded relationship records with snippets
    and resolution state.
    """
    result = retrieval.get_references(_conn(), _root(), symbol, limit=limit)
    return json.dumps(result)


@mcp.tool()
def code_status() -> str:
    """Report index coverage, pending jobs, stale paths, failures, and
    the active embedding fingerprint."""
    return json.dumps(retrieval.get_status(_conn(), _root()))


_STATE = {"conn": None, "root": None}


def _conn():
    if _STATE["conn"] is None:
        from indexedcode.db import connect

        _STATE["conn"] = connect(_STATE["db"], threaded=True)
    return _STATE["conn"]


def _root():
    return _STATE["root"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="indexedcode-mcp")
    parser.add_argument("--db", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)

    _STATE["db"] = args.db
    _STATE["root"] = args.root
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
