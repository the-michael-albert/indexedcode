"""GraphQL adapter: read-only node queries over the indexed store.

SDL-first on graphql-core (no extra frameworks). Resolvers read via
the same SQLite connection and retrieval core the REST endpoints use;
there is no Mutation type.
"""

from __future__ import annotations

from graphql import (
    ExecutionResult,
    GraphQLError,
    build_schema,
    execute as gql_execute,
    parse,
    specified_rules,
    validate,
)
from graphql.language import FieldNode, FragmentSpreadNode
from graphql.error import GraphQLSyntaxError
from graphql.validation import ValidationRule

from indexedcode import retrieval

_SDL = """
type File {
  id: ID!
  name: String!
  path: String!
  language: String
  isTest: Boolean!
  classes: [ClassNode!]!
  functions: [FunctionNode!]!
  related(relation: String): [RelatedEdge!]!
}

type ClassNode {
  id: ID!
  name: String!
  qualifiedName: String!
  kind: String
  startLine: Int
  endLine: Int
  methods: [FunctionNode!]!
  nestedClasses: [ClassNode!]!
  related(relation: String): [RelatedEdge!]!
}

type FunctionNode {
  id: ID!
  name: String!
  qualifiedName: String!
  signature: String
  startLine: Int
  endLine: Int
  source: String
  callers: [FunctionNode!]!
  callees: [FunctionNode!]!
  related(relation: String): [RelatedEdge!]!
}

type Module {
  id: ID!
  name: String!
  path: String!
  children: [Module!]!
  files: [File!]!
  related(relation: String): [RelatedEdge!]!
}

union Node = File | ClassNode | FunctionNode | Module

type RelatedEdge {
  relation: String!
  sourceType: String!
  targetType: String
  targetId: Int
  targetName: String
  node: Node
}

enum SearchMode { LEXICAL SYMBOL HYBRID SEMANTIC }

type SearchEntry {
  score: Float!
  evidence: String
  node: Node
}

type Query {
  modules: [Module!]!
  file(path: String!): File
  module(path: String!): Module
  class(qualifiedName: String!): ClassNode
  function(qualifiedName: String!): FunctionNode
  semanticSearch(query: String!, limit: Int, mode: SearchMode): [SearchEntry!]!
}
"""

MAX_DEPTH = 8  # selection-set nesting allowed in one query
MAX_LIST = 1000  # rows any one relationship list may return

_TABLES = {
    "file": "files",
    "class": "classes",
    "function": "functions",
    "module": "modules",
}


def _file_node(row):
    return {
        "_kind": "file",
        "id": row["id"],
        "name": row["file"],
        "path": f"{row['folder']}/{row['file']}",
        "language": row["language"],
        "isTest": bool(row["is_test"]),
    }


def _class_node(row):
    return {
        "_kind": "class",
        "id": row["id"],
        "name": row["name"],
        "qualifiedName": row["qualified_name"],
        "kind": row["kind"],
        "startLine": row["start_line"],
        "endLine": row["end_line"],
    }


def _function_node(row, max_chars=retrieval.DEFAULT_MAX_CHARS):
    return {
        "_kind": "function",
        "id": row["id"],
        "name": row["name"],
        "qualifiedName": row["qualified_name"],
        "signature": row["signature"],
        "startLine": row["start_line"],
        "endLine": row["end_line"],
        "source": (row["source"] or "")[:max_chars],
    }


def _module_node(row):
    return {
        "_kind": "module",
        "id": row["id"],
        "name": row["name"],
        "path": row["relative_path"],
    }


_NODE_BUILDERS = {
    "file": _file_node,
    "class": _class_node,
    "function": _function_node,
    "module": _module_node,
}



class _Context:
    """Per-request state the shared schema's resolvers read."""

    def __init__(self, conn, root, embedder, model_fingerprint, max_chars):
        self.conn = conn
        self.root = root
        self.embedder = embedder
        self.model_fingerprint = model_fingerprint
        self.max_chars = max_chars

    def fetch_one(self, sql, params):
        return self.conn.execute(sql, params).fetchone()

    def resolve_node(self, entity_type, entity_id):
        if entity_type not in _TABLES or entity_id is None:
            return None
        row = self.fetch_one(
            f"SELECT * FROM {_TABLES[entity_type]} WHERE id = ?", (entity_id,)
        )
        if row is None:
            return None
        if entity_type == "function":  # only function nodes carry source
            return _function_node(row, self.max_chars)
        return _NODE_BUILDERS[entity_type](row)

    def resolve_by_name(self, name, prefer_file_id=None):
        """Resolve a call-target name to one function, deterministically.

        Relationships store targets by name. An exact qualified-name match
        wins; otherwise the same-file function, then the lowest id.
        """
        row = self.fetch_one(
            "SELECT * FROM functions WHERE qualified_name = ?", (name,)
        )
        if row is None:
            row = self.fetch_one(
                "SELECT * FROM functions WHERE name = ?"
                " ORDER BY (file_id IS ?) DESC, id LIMIT 1",
                (name, prefer_file_id),
            )
        return _function_node(row, self.max_chars) if row else None


class _DepthLimit(ValidationRule):
    """Reject queries nested deeper than MAX_DEPTH (fragments included)."""

    def enter_operation_definition(self, node, *_):
        depth = self._depth(node.selection_set, set())
        if depth > MAX_DEPTH:
            self.report_error(
                GraphQLError(
                    f"query depth {depth} exceeds the limit of {MAX_DEPTH}", node
                )
            )

    def _depth(self, selection_set, seen):
        """Field nesting below ``selection_set``; fragments add no level."""
        if selection_set is None:
            return 0
        deepest = 0
        for sel in selection_set.selections:
            if isinstance(sel, FragmentSpreadNode):
                name = sel.name.value
                frag = self.context.get_fragment(name)
                if frag is None or name in seen:
                    continue  # cycles/unknown names are reported elsewhere
                level = self._depth(frag.selection_set, seen | {name})
            elif isinstance(sel, FieldNode):
                level = 1 + self._depth(sel.selection_set, seen)
            else:  # inline fragment
                level = self._depth(sel.selection_set, seen)
            deepest = max(deepest, level)
        return deepest


def _ctx(info):
    return info.context


def _build():
    schema = build_schema(_SDL)

    def resolve_file(_root, info, path):
        folder, _, name = path.rpartition("/")
        row = _ctx(info).fetch_one(
            "SELECT * FROM files WHERE folder = ? AND file = ?", (folder, name)
        )
        return _file_node(row) if row else None

    def resolve_module(_root, info, path):
        row = _ctx(info).fetch_one(
            "SELECT * FROM modules WHERE relative_path = ?", (path,)
        )
        return _module_node(row) if row else None

    def resolve_class(_root, info, qualifiedName):
        row = _ctx(info).fetch_one(
            "SELECT * FROM classes WHERE qualified_name = ?", (qualifiedName,)
        )
        return _class_node(row) if row else None

    def resolve_function(_root, info, qualifiedName):
        c = _ctx(info)
        row = c.fetch_one(
            "SELECT * FROM functions WHERE qualified_name = ?", (qualifiedName,)
        )
        return _function_node(row, c.max_chars) if row else None

    def resolve_modules(_root, info):
        return [
            _module_node(r)
            for r in _ctx(info).conn.execute(
                "SELECT * FROM modules ORDER BY relative_path LIMIT ?", (MAX_LIST,)
            )
        ]

    def file_classes(file_node, info):
        return [
            _class_node(r)
            for r in _ctx(info).conn.execute(
                "SELECT * FROM classes WHERE file_id = ? ORDER BY start_line",
                (file_node["id"],),
            )
        ]

    def file_functions(file_node, info):
        c = _ctx(info)
        return [
            _function_node(r, c.max_chars)
            for r in c.conn.execute(
                "SELECT * FROM functions WHERE file_id = ? AND class_id IS NULL"
                " ORDER BY start_line",
                (file_node["id"],),
            )
        ]

    def module_children(module_node, info):
        return [
            _module_node(r)
            for r in _ctx(info).conn.execute(
                "SELECT * FROM modules WHERE parent_module_id = ? ORDER BY name",
                (module_node["id"],),
            )
        ]

    def module_files(module_node, info):
        # a module scope covers its whole subtree, not just direct files
        rows = _ctx(info).conn.execute(
            "WITH RECURSIVE sub(id) AS ("
            "  SELECT ?"
            "  UNION ALL"
            "  SELECT m.id FROM modules m JOIN sub ON m.parent_module_id = sub.id"
            ") SELECT f.* FROM files f JOIN sub ON f.module_id = sub.id"
            " ORDER BY f.file LIMIT ?",
            (module_node["id"], MAX_LIST),
        )
        return [_file_node(r) for r in rows]

    def class_methods(class_node, info):
        c = _ctx(info)
        return [
            _function_node(r, c.max_chars)
            for r in c.conn.execute(
                "SELECT * FROM functions WHERE class_id = ? ORDER BY start_line",
                (class_node["id"],),
            )
        ]

    def class_nested(class_node, info):
        return [
            _class_node(r)
            for r in _ctx(info).conn.execute(
                "SELECT * FROM classes WHERE parent_class_id = ? ORDER BY start_line",
                (class_node["id"],),
            )
        ]

    def function_callers(fn_node, info):
        c = _ctx(info)
        rows = c.conn.execute(
            "SELECT source_id FROM relationships"
            " WHERE relation = 'calls' AND source_type = 'function'"
            " AND (target_name = ? OR target_name = ?)"
            " ORDER BY source_id LIMIT ?",
            (fn_node["qualifiedName"], fn_node["name"], MAX_LIST),
        )
        return [
            n for n in (c.resolve_node("function", r["source_id"]) for r in rows) if n
        ]

    def function_callees(fn_node, info):
        c = _ctx(info)
        own = c.fetch_one(
            "SELECT file_id FROM functions WHERE id = ?", (fn_node["id"],)
        )
        prefer = own["file_id"] if own else None
        rows = c.conn.execute(
            "SELECT target_name FROM relationships"
            " WHERE relation = 'calls' AND source_type = 'function'"
            " AND source_id = ? ORDER BY line_number LIMIT ?",
            (fn_node["id"], MAX_LIST),
        )
        return [
            n for n in (c.resolve_by_name(r["target_name"], prefer) for r in rows) if n
        ]

    def make_related(entity_type):
        def related(node, info, relation=None):
            sql = "SELECT * FROM relationships WHERE source_type = ? AND source_id = ?"
            params = [entity_type, node["id"]]
            if relation is not None:
                sql += " AND relation = ?"
                params.append(relation)
            return [
                {
                    "relation": r["relation"],
                    "sourceType": r["source_type"],
                    "targetType": r["target_type"],
                    "targetId": r["target_id"],
                    "targetName": r["target_name"],
                    "_target": (r["target_type"], r["target_id"]),
                }
                for r in _ctx(info).conn.execute(
                    sql + " ORDER BY line_number LIMIT ?", [*params, MAX_LIST]
                )
            ]

        return related

    def edge_node(edge, info):
        return _ctx(info).resolve_node(*edge["_target"])

    def semantic_search(_root, info, query, limit=None, mode=None):
        c = _ctx(info)
        payload = retrieval.search(
            c.conn,
            c.root,
            query,
            mode=mode.lower() if mode else "semantic",
            limit=limit if limit is not None else retrieval.DEFAULT_RESULTS,
            embedder=c.embedder,
            model_fingerprint=c.model_fingerprint,
        )
        return [
            {
                "score": r["score"],
                "evidence": r["evidence"],
                "node": c.resolve_node(r["entity_type"], r["entity_id"]),
            }
            for r in payload["results"]
        ]

    fields = schema.query_type.fields
    fields["file"].resolve = resolve_file
    fields["module"].resolve = resolve_module
    fields["class"].resolve = resolve_class
    fields["function"].resolve = resolve_function
    fields["semanticSearch"].resolve = semantic_search
    fields["modules"].resolve = resolve_modules

    file_fields = schema.get_type("File").fields
    file_fields["classes"].resolve = file_classes
    file_fields["functions"].resolve = file_functions
    file_fields["related"].resolve = make_related("file")

    module_fields = schema.get_type("Module").fields
    module_fields["children"].resolve = module_children
    module_fields["files"].resolve = module_files
    module_fields["related"].resolve = make_related("module")

    class_fields = schema.get_type("ClassNode").fields
    class_fields["methods"].resolve = class_methods
    class_fields["nestedClasses"].resolve = class_nested
    class_fields["related"].resolve = make_related("class")

    function_fields = schema.get_type("FunctionNode").fields
    function_fields["callers"].resolve = function_callers
    function_fields["callees"].resolve = function_callees
    function_fields["related"].resolve = make_related("function")

    schema.get_type("RelatedEdge").fields["node"].resolve = edge_node
    # graphql-core 3.3 unions resolve by type-name string
    schema.get_type("Node").resolve_type = lambda value, _info, _type: {
        "file": "File",
        "class": "ClassNode",
        "function": "FunctionNode",
        "module": "Module",
    }[value["_kind"]]
    return schema


_SCHEMA = _build()
_RULES = [*specified_rules, _DepthLimit]


def execute(
    conn,
    root,
    query,
    *,
    variables=None,
    operation_name=None,
    embedder=None,
    model_fingerprint=None,
    max_chars=retrieval.DEFAULT_MAX_CHARS,
):
    """Run one GraphQL query; returns a graphql ExecutionResult."""
    try:
        document = parse(query)
    except GraphQLSyntaxError as exc:
        return ExecutionResult(data=None, errors=[exc])
    errors = validate(_SCHEMA, document, _RULES)
    if errors:
        return ExecutionResult(data=None, errors=errors)
    result = gql_execute(
        _SCHEMA,
        document,
        variable_values=variables,
        operation_name=operation_name,
        context_value=_Context(conn, root, embedder, model_fingerprint, max_chars),
    )
    return result
