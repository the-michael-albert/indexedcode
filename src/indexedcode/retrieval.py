"""Read-only retrieval core (plan §8.1).

Four operations shared by the CLI, HTTP, and MCP adapters: search,
context, references, and status. Responses are bounded by default with
opaque continuation cursors that are invalidated by index changes.
Indexed source is only reported "current" after comparing the on-disk
file hash with ``files.indexed_hash``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from pathlib import Path, PurePosixPath

DEFAULT_RESULTS = 5
MAX_RESULTS = 50
DEFAULT_CONTEXT_LINES = 120
DEFAULT_REFERENCES = 10
MAX_REFERENCES = 10
DEFAULT_MAX_CHARS = 12_000
MAX_CHARS_CAP = 32_000

_EVIDENCE_CHARS = 240


class SearchError(ValueError):
    """Invalid or unavailable search request."""


class StaleCursorError(SearchError):
    """The cursor predates the current index state."""


class ContextError(ValueError):
    """Invalid context request (unknown entity or forbidden path)."""


# --- cursors -----------------------------------------------------------------


def _index_version(conn):
    # sum-of-timestamps detects in-place rewrites that count alone misses
    row = conn.execute(
        "SELECT COUNT(*) AS c, COALESCE(SUM(updated_at_ms), 0) AS s"
        " FROM entity_index"
    ).fetchone()
    return f"{row['c']}:{row['s']}"


def _encode_cursor(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> dict:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        assert isinstance(payload, dict) and {"k", "o", "v"} <= set(payload)
        return payload
    except Exception as exc:
        raise SearchError("malformed cursor") from exc


# --- search ------------------------------------------------------------------


def search(conn, root, query, *, mode="hybrid", limit=DEFAULT_RESULTS,
           embedder=None, cursor=None, max_chars=DEFAULT_MAX_CHARS):
    """Search entities by symbol, text, and/or vector similarity."""
    if not isinstance(query, str) or not query.strip():
        raise SearchError("query must be a non-empty string")
    if len(query) > MAX_CHARS_CAP:
        raise SearchError(f"query exceeds {MAX_CHARS_CAP} characters")
    if mode not in ("lexical", "semantic", "hybrid", "symbol"):
        raise SearchError(f"unknown mode {mode!r}")

    max_chars = min(max_chars, MAX_CHARS_CAP)
    limit = max(1, min(limit, MAX_RESULTS))
    version = _index_version(conn)

    offset = 0
    if cursor is not None:
        payload = _decode_cursor(cursor)
        if payload["v"] != version:
            raise StaleCursorError("index changed since this cursor")
        if payload["k"] != _cursor_key("search", query, mode):
            raise StaleCursorError("cursor belongs to a different query")
        offset = payload["o"]

    degraded = None
    use_semantic = mode in ("semantic", "hybrid")
    if use_semantic and embedder is None:
        if mode == "semantic":
            raise SearchError(
                "semantic search requires an embedding model; none is"
                " available"
            )
        degraded = "semantic_unavailable"
        use_semantic = False

    scored: dict[tuple[str, int], dict] = {}

    # symbol tier: exact name / qualified-name matches rank first
    if mode in ("symbol", "hybrid", "lexical"):
        for row in conn.execute(
            "SELECT 'function' AS entity_type, id AS entity_id,"
            " qualified_name, name,"
            " signature AS evidence FROM functions"
            " WHERE name = :q OR qualified_name = :q"
            " UNION ALL"
            " SELECT 'class', id, qualified_name, name,"
            " substr(source, 1, :ev) FROM classes"
            " WHERE name = :q OR qualified_name = :q",
            {"q": query, "ev": _EVIDENCE_CHARS},
        ):
            scored[(row["entity_type"], row["entity_id"])] = {
                "row": row, "score": 3.0,
            }

    # lexical tier over indexed search text
    if mode in ("lexical", "hybrid"):
        pattern = f"%{_escape_like(query)}%"
        for row in conn.execute(
            "SELECT ei.entity_type, ei.entity_id, ei.search_text FROM"
            " entity_index ei WHERE ei.search_text LIKE :p LIMIT 500",
            {"p": pattern},
        ):
            entry = scored.get((row["entity_type"], row["entity_id"]))
            evidence = _evidence_window(row["search_text"], query)
            if entry is not None:
                entry["score"] += 1.0
                entry.setdefault("evidence", evidence)
            else:
                scored[(row["entity_type"], row["entity_id"])] = {
                    "row": row, "score": 2.0, "evidence": evidence,
                }

    # semantic tier: cosine similarity over stored embeddings
    if use_semantic:
        query_vector = embedder([query])[0]
        for row in conn.execute(
            "SELECT entity_type, entity_id, embedding, embedding_dimensions,"
            " search_text FROM entity_index"
        ):
            stored = struct.unpack(
                f"<{row['embedding_dimensions']}f", row["embedding"],
            )
            similarity = _cosine(query_vector, stored)
            key = (row["entity_type"], row["entity_id"])
            entry = scored.get(key)
            if entry is not None:
                entry["score"] = max(entry["score"], 1.0 + similarity)
            else:
                scored[key] = {
                    "row": row, "score": 1.0 + similarity,
                    "evidence": row["search_text"][:_EVIDENCE_CHARS],
                }

    ranked = sorted(
        scored.values(), key=lambda entry: -entry["score"],
    )
    total = len(ranked)
    page = ranked[offset:offset + limit]
    truncated = offset + limit < total
    next_cursor = None
    if truncated:
        next_cursor = _encode_cursor({
            "k": _cursor_key("search", query, mode),
            "o": offset + limit,
            "v": version,
        })

    return {
        "results": [_result_entry(conn, entry) for entry in page],
        "degraded": degraded,
        "truncated": truncated,
        "next_cursor": next_cursor,
        "total": total,
    }


def _result_entry(conn, entry):
    row = entry["row"]
    entity_type, entity_id = row["entity_type"], row["entity_id"]
    if entity_type in ("function", "class"):
        table = "functions" if entity_type == "function" else "classes"
        sql = (
            f"SELECT qualified_name, name, start_line, end_line, folder,"
            f" file FROM {table} t JOIN files fl ON t.file_id = fl.id"
            f" WHERE t.id = ?"
        )
        detail = conn.execute(sql, (entity_id,)).fetchone()
    else:
        detail = conn.execute(
            "SELECT name AS qualified_name, name, 1 AS start_line,"
            " 1 AS end_line, '' AS folder, relative_path AS file FROM"
            " modules WHERE id = ?",
            (entity_id,),
        ).fetchone()
    evidence = entry.get("evidence") or (
        row["evidence"] if "evidence" in row.keys() else ""
    )
    result = {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "score": round(entry["score"], 6),
        "evidence": (evidence or "")[:_EVIDENCE_CHARS],
    }
    if detail is not None:
        result.update({
            "qualified_name": detail["qualified_name"],
            "name": detail["name"],
            "path": (f"{detail['folder']}/{detail['file']}"
                     if detail["folder"] else detail["file"]),
            "start_line": detail["start_line"],
            "end_line": detail["end_line"],
        })
    return result


# --- context -----------------------------------------------------------------


def get_context(conn, root, *, entity_type=None, entity_id=None, path=None,
                start_line=None, end_line=None, cursor=None,
                max_chars=DEFAULT_MAX_CHARS,
                lines=DEFAULT_CONTEXT_LINES):
    """Exact source for an indexed entity or an explicit path/line range."""
    max_chars = min(max_chars, MAX_CHARS_CAP)
    if entity_type is not None or entity_id is not None:
        if entity_type not in ("function", "class") or entity_id is None:
            raise ContextError("entity context needs function/class + id")
        return _entity_context(conn, root, entity_type, entity_id, max_chars)

    if path is None:
        raise ContextError("provide an entity or a path")
    return _path_context(
        conn, root, path, start_line, end_line, cursor, max_chars, lines,
    )


def _entity_context(conn, root, entity_type, entity_id, max_chars):
    table = "functions" if entity_type == "function" else "classes"
    row = conn.execute(
        f"SELECT t.*, fl.folder, fl.file, fl.indexed_hash FROM {table} t"
        " JOIN files fl ON t.file_id = fl.id WHERE t.id = ?",
        (entity_id,),
    ).fetchone()
    if row is None:
        raise ContextError(f"unknown {entity_type} {entity_id}")

    path = f"{row['folder']}/{row['file']}" if row["folder"] else row["file"]
    freshness = _freshness(Path(root) / path, row["indexed_hash"])

    enclosing = None
    related = []
    if entity_type == "function":
        if row["class_id"] is not None:
            parent = conn.execute(
                "SELECT qualified_name FROM classes WHERE id = ?",
                (row["class_id"],),
            ).fetchone()
            if parent is not None:
                enclosing = {"entity_type": "class",
                             "qualified_name": parent["qualified_name"]}
        related = _related_edges(conn, entity_id)

    return {
        "origin": "index",
        "freshness": freshness,
        "source": row["source"][:max_chars],
        "path": path,
        "start_line": row["start_line"],
        "end_line": row["end_line"],
        "enclosing": enclosing,
        "related": related,
        "truncated": len(row["source"]) > max_chars,
        "next_cursor": None,
    }


def _path_context(conn, root, path, start_line, end_line, cursor,
                  max_chars, lines):
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ContextError("path must be relative to the repository root")
    absolute = Path(root) / Path(*pure.parts)
    if absolute.is_symlink():
        raise ContextError("symlinked paths are rejected")
    try:
        resolved = absolute.resolve()
        contained = resolved.is_relative_to(Path(root).resolve())
    except OSError as exc:
        raise ContextError("path cannot be resolved") from exc
    if not contained:
        raise ContextError("path escapes the repository root")
    if not resolved.is_file():
        raise ContextError(f"no such file: {path}")

    version = _index_version(conn)
    offset_line = max(1, start_line or 1)
    if cursor is not None:
        payload = _decode_cursor(cursor)
        if payload["v"] != version:
            raise StaleCursorError("index changed since this cursor")
        offset_line = payload["o"]

    file_row = conn.execute(
        "SELECT indexed_hash FROM files WHERE folder = ? AND file = ?",
        _split_path(pure),
    ).fetchone()
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ContextError("file cannot be read") from exc

    all_lines = text.splitlines()
    if end_line is not None:
        page = all_lines[offset_line - 1:end_line]
        next_start = None
        truncated = False
    else:
        page = all_lines[offset_line - 1:offset_line - 1 + lines]
        next_start = offset_line + len(page)
        truncated = next_start <= len(all_lines)
    body = "\n".join(page)[:max_chars]

    freshness = "unindexed"
    if file_row is not None and file_row["indexed_hash"] is not None:
        freshness = (
            "current"
            if hashlib.sha256(text.encode()).hexdigest()
            == file_row["indexed_hash"]
            else "stale"
        )
    next_cursor = None
    if truncated and next_start is not None:
        next_cursor = _encode_cursor({
            "k": _cursor_key("context", str(pure), ""),
            "o": next_start,
            "v": version,
        })

    return {
        "origin": "disk",
        "freshness": freshness,
        "source": body,
        "path": str(pure),
        "start_line": offset_line,
        "end_line": offset_line + len(page) - 1,
        "enclosing": None,
        "related": [],
        "truncated": truncated,
        "next_cursor": next_cursor,
    }


def _related_edges(conn, entity_id):
    edges = []
    for row in conn.execute(
        "SELECT r.relation, r.target_name, r.line_number, r.resolution,"
        " r.snippet, fl.folder, fl.file, f.qualified_name AS"
        " source_qualified_name FROM relationships r"
        " JOIN files fl ON r.origin_file_id = fl.id"
        " LEFT JOIN functions f ON r.source_type = 'function'"
        " AND r.source_id = f.id"
        " WHERE r.source_id = ? OR r.target_id = ? LIMIT 10",
        (entity_id, entity_id),
    ):
        edges.append({
            "relation": row["relation"],
            "target_name": row["target_name"],
            "source_qualified_name": row["source_qualified_name"],
            "line_number": row["line_number"],
            "resolution": row["resolution"],
            "snippet": row["snippet"][:_EVIDENCE_CHARS],
            "path": (f"{row['folder']}/{row['file']}"
                     if row["folder"] else row["file"]),
        })
    return edges


# --- references ----------------------------------------------------------------


def get_references(conn, root, symbol, *, limit=DEFAULT_REFERENCES,
                   cursor=None, max_chars=DEFAULT_MAX_CHARS):
    """Source-grounded import/call/inheritance candidates for a symbol."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise SearchError("symbol must be a non-empty string")
    if len(symbol) > MAX_CHARS_CAP:
        raise SearchError(f"symbol exceeds {MAX_CHARS_CAP} characters")
    limit = max(1, min(limit, MAX_REFERENCES))
    version = _index_version(conn)

    offset = 0
    if cursor is not None:
        payload = _decode_cursor(cursor)
        if payload["v"] != version:
            raise StaleCursorError("index changed since this cursor")
        offset = payload["o"]

    rows = conn.execute(
        "SELECT r.relation, r.target_name, r.line_number, r.resolution,"
        " r.snippet, fl.folder, fl.file, f.qualified_name AS"
        " source_qualified_name FROM relationships r"
        " JOIN files fl ON r.origin_file_id = fl.id"
        " LEFT JOIN functions f ON r.source_type = 'function'"
        " AND r.source_id = f.id"
        " WHERE r.target_name LIKE :like OR r.target_name = :exact"
        " OR f.qualified_name = :exact"
        " ORDER BY r.line_number LIMIT 200",
        {"like": f"%{_escape_like(symbol)}%", "exact": symbol},
    ).fetchall()

    page = rows[offset:offset + limit]
    truncated = offset + limit < len(rows)
    references = [
        {
            "relation": row["relation"],
            "target_name": row["target_name"],
            "source_qualified_name": row["source_qualified_name"],
            "line_number": row["line_number"],
            "resolution": row["resolution"],
            "snippet": row["snippet"][:_EVIDENCE_CHARS],
            "path": (f"{row['folder']}/{row['file']}"
                     if row["folder"] else row["file"]),
        }
        for row in page
    ]
    return {
        "references": references,
        "truncated": truncated,
        "next_cursor": (
            _encode_cursor({
                "k": _cursor_key("references", symbol, ""),
                "o": offset + limit,
                "v": version,
            }) if truncated else None
        ),
    }


# --- status --------------------------------------------------------------------


def get_status(conn, root):
    """Index health: coverage, queue state, staleness, failures."""
    root_path = Path(root)
    coverage = {
        "ready_files": conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE status = 'ready'"
        ).fetchone()["c"],
        "known_files": conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE status != 'deleted'"
        ).fetchone()["c"],
        "indexed_entities": conn.execute(
            "SELECT COUNT(*) AS c FROM entity_index"
        ).fetchone()["c"],
    }
    stale_paths = []
    for row in conn.execute(
        "SELECT folder, file, indexed_hash FROM files WHERE status = 'ready'"
        " AND indexed_hash IS NOT NULL"
    ):
        rel = f"{row['folder']}/{row['file']}" if row["folder"] else row["file"]
        if _freshness(root_path / rel, row["indexed_hash"]) == "stale":
            stale_paths.append(rel)
    fingerprint = conn.execute(
        "SELECT model_fingerprint FROM entity_index LIMIT 1"
    ).fetchone()
    return {
        "coverage": coverage,
        "pending_jobs": conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE status IN"
            " ('pending', 'retry')"
        ).fetchone()["c"],
        "stale_paths": sorted(stale_paths),
        "failures": {
            "failed_jobs": conn.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE status = 'failed'"
            ).fetchone()["c"],
            "error_files": conn.execute(
                "SELECT COUNT(*) AS c FROM files WHERE status = 'error'"
            ).fetchone()["c"],
        },
        "model_fingerprint": (
            fingerprint["model_fingerprint"] if fingerprint else None
        ),
    }


# --- helpers -------------------------------------------------------------------


def _cursor_key(kind, query, mode):
    return hashlib.sha256(f"{kind}|{mode}|{query}".encode()).hexdigest()


def _escape_like(text):
    return text.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def _evidence_window(text, query, window=_EVIDENCE_CHARS):
    lower = text.lower()
    index = lower.find(query.lower())
    if index < 0:
        return text[:window]
    half = window // 2
    start = max(0, index - half)
    return text[start:start + window]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _freshness(path, indexed_hash):
    try:
        data = path.read_bytes()
    except OSError:
        return "stale"
    if hashlib.sha256(data).hexdigest() != indexed_hash:
        return "stale"
    return "current"


def _split_path(pure):
    parts = pure.parts
    if len(parts) == 1:
        return "", parts[0]
    return "/".join(parts[:-1]), parts[-1]
