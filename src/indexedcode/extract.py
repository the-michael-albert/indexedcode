"""Tree-sitter extraction for the eight supported languages (plan §9, Phase 2).

Extraction produces entity shapes with parent *qualified names* — database
ids are resolved later by the Phase 4 publication layer. Coordinates match
the plan conventions: 1-based inclusive lines, 0-based byte offsets with
exclusive ends, and `source` is always the exact source slice.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from pathlib import PurePosixPath

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_rust
import tree_sitter_typescript
from tree_sitter import Language, Parser

from indexedcode.models import ClassKind, EntityType, Relation, Resolution, RowModel

_EXTENSION_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

_LANGUAGES: dict[str, Language] = {
    "python": Language(tree_sitter_python.language()),
    "javascript": Language(tree_sitter_javascript.language()),
    "typescript": Language(tree_sitter_typescript.language_typescript()),
    "tsx": Language(tree_sitter_typescript.language_tsx()),
    "go": Language(tree_sitter_go.language()),
    "rust": Language(tree_sitter_rust.language()),
    "java": Language(tree_sitter_java.language()),
    "c": Language(tree_sitter_c.language()),
    "cpp": Language(tree_sitter_cpp.language()),
}


class ExtractedFunction(RowModel):
    name: str
    qualified_name: str
    signature: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    source: str
    source_hash: str
    parent_class_qname: str | None = None
    parent_function_qname: str | None = None


class ExtractedClass(RowModel):
    name: str
    qualified_name: str
    kind: ClassKind
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    source: str
    source_hash: str
    parent_class_qname: str | None = None


class RelationshipCandidate(RowModel):
    relation: Relation
    source_qname: str
    source_type: EntityType = EntityType.FILE
    target_name: str | None
    line_number: int
    resolution: Resolution
    snippet: str


class ExtractedFile(RowModel):
    path: str
    language: str
    parse_ok: bool
    functions: list[ExtractedFunction] = []
    classes: list[ExtractedClass] = []
    relationships: list[RelationshipCandidate] = []


def detect_language(path) -> str | None:
    """Map a repository-relative POSIX path to its language, or None."""
    return _EXTENSION_LANGUAGE.get(PurePosixPath(str(path)).suffix.lower())


def _sha(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _lines_index(source: bytes):
    starts = [0]
    for i, b in enumerate(source):
        if b == 0x0A:
            starts.append(i + 1)
    return starts


def _line_snippet(source: bytes, starts, byte_offset: int) -> str:
    line_no = bisect_right(starts, byte_offset)
    begin = starts[line_no - 1]
    end = starts[line_no] - 1 if line_no < len(starts) else len(source)
    return source[begin:end].decode("utf-8", "replace").strip()[:200]


def _header(source: bytes, node, body) -> str:
    end = body.start_byte if body is not None else node.end_byte
    return (
        source[node.start_byte:end]
        .decode("utf-8", "replace")
        .strip()
        .rstrip(":")
        .strip()
    )


def _node_name(node) -> str | None:
    name = node.child_by_field_name("name")
    return None if name is None else name.text.decode("utf-8", "replace")


class _Extractor:
    """One pass over a parse tree, collecting entities and relationships."""

    def __init__(self, source: bytes, path: str, language: str):
        self.source = source
        self.path = path
        self.language = language
        self.starts = _lines_index(source)
        self.functions: list[ExtractedFunction] = []
        self.classes: list[ExtractedClass] = []
        self.relationships: list[RelationshipCandidate] = []
        self._seen_rels: set = set()

    # -- helpers ------------------------------------------------------------

    def _join(self, parent: str | None, name: str) -> str:
        return f"{parent}.{name}" if parent else name

    def _add_rel(self, relation, source_qname, target_name, node, resolution):
        line = node.start_point[0] + 1
        key = (relation, source_qname, target_name, line)
        if key in self._seen_rels:
            return
        self._seen_rels.add(key)
        self.relationships.append(
            RelationshipCandidate(
                relation=relation,
                source_qname=source_qname,
                target_name=target_name,
                line_number=line,
                resolution=resolution,
                snippet=_line_snippet(self.source, self.starts, node.start_byte),
            )
        )

    def _call_target(self, call_node) -> str | None:
        fn = call_node.child_by_field_name("function")
        if fn is None:
            return None
        if fn.type in (
            "identifier",
            "field_identifier",
            "attribute",
            "member_expression",
            "selector_expression",
            "scoped_identifier",
            "qualified_identifier",
        ):
            return fn.text.decode("utf-8", "replace")
        return None

    # -- walkers --------------------------------------------------------------

    def walk(self, node, fn_qname=None, cls_qname=None):
        handler = getattr(self, f"_on_{self.language}_{node.type}", None)
        if handler is not None:
            handler(node, fn_qname, cls_qname)
        else:
            self._recurse(node, fn_qname, cls_qname)

    def _recurse(self, node, fn_qname, cls_qname):
        for child in node.children:
            self.walk(child, fn_qname, cls_qname)

    def _context(self, fn_qname, cls_qname) -> str:
        return fn_qname or cls_qname or "file"

    # -- python ----------------------------------------------------------------

    def _on_python_function_definition(self, node, fn_qname, cls_qname):
        name = _node_name(node)
        qname = fn_qname
        if name and not node.has_error:
            qname = self._join(fn_qname or cls_qname, name)
            self.functions.append(
                ExtractedFunction(
                    name=name,
                    qualified_name=qname,
                    signature=_header(self.source, node, node.child_by_field_name("body")),
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    source=self.source[node.start_byte:node.end_byte].decode(
                        "utf-8", "replace"
                    ),
                    source_hash=_sha(
                        self.source[node.start_byte:node.end_byte]
                    ),
                    parent_class_qname=cls_qname,
                    parent_function_qname=fn_qname,
                )
            )
        self._recurse(node, qname if name else fn_qname, cls_qname)

    def _on_python_class_definition(self, node, fn_qname, cls_qname):
        name = _node_name(node)
        qname = self._join(cls_qname, name) if name else cls_qname
        if name:
            self.classes.append(
                ExtractedClass(
                    name=name,
                    qualified_name=qname,
                    kind="class",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    source=self.source[node.start_byte:node.end_byte].decode(
                        "utf-8", "replace"
                    ),
                    source_hash=_sha(self.source[node.start_byte:node.end_byte]),
                    parent_class_qname=cls_qname,
                )
            )
            supers = node.child_by_field_name("superclasses")
            if supers is not None:
                for child in supers.children:
                    if child.type in ("identifier", "attribute"):
                        self._add_rel(
                            "inherits", qname,
                            child.text.decode("utf-8", "replace"), child,
                            "syntactic_candidate",
                        )
        self._recurse(node, fn_qname, qname)

    def _on_python_import_statement(self, node, fn_qname, cls_qname):
        ctx = self._context(fn_qname, cls_qname)
        for child in node.children:
            if child.type == "dotted_name":
                self._add_rel(
                    "imports", ctx,
                    child.text.decode("utf-8", "replace"), child, "unresolved",
                )

    def _on_python_import_from_statement(self, node, fn_qname, cls_qname):
        ctx = self._context(fn_qname, cls_qname)
        module = node.child_by_field_name("module_name")
        base = module.text.decode("utf-8", "replace") if module is not None else ""
        for child in node.children_by_field_name("name"):
            leaf = child.text.decode("utf-8", "replace")
            target = f"{base}.{leaf}" if base else leaf
            self._add_rel("imports", ctx, target, child, "unresolved")

    def _on_python_call(self, node, fn_qname, cls_qname):
        target = self._call_target(node)
        if target:
            self._add_rel(
                "calls", self._context(fn_qname, cls_qname), target, node,
                "syntactic_candidate",
            )
        self._recurse(node, fn_qname, cls_qname)

    # -- javascript / typescript / tsx --------------------------------------------

    def _emit_function(self, node, name, fn_qname, cls_qname, body_field="body"):
        if node.has_error:
            return None  # never publish a partially recovered definition
        qname = self._join(fn_qname or cls_qname, name)
        self.functions.append(
            ExtractedFunction(
                name=name,
                qualified_name=qname,
                signature=_header(
                    self.source, node, node.child_by_field_name(body_field)
                ),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                source=self.source[node.start_byte:node.end_byte].decode(
                    "utf-8", "replace"
                ),
                source_hash=_sha(self.source[node.start_byte:node.end_byte]),
                parent_class_qname=cls_qname,
                parent_function_qname=fn_qname,
            )
        )
        return qname

    def _on_javascript_function_declaration(self, node, fn_qname, cls_qname):
        name = _node_name(node)
        if name:
            self._emit_function(node, name, fn_qname, cls_qname)
        self._recurse(node, fn_qname, cls_qname)

    _on_typescript_function_declaration = _on_javascript_function_declaration
    _on_tsx_function_declaration = _on_javascript_function_declaration

    def _on_javascript_lexical_declaration(self, node, fn_qname, cls_qname):
        self._js_variable_functions(node, fn_qname, cls_qname)

    _on_javascript_variable_declaration = _on_javascript_lexical_declaration
    _on_typescript_lexical_declaration = _on_javascript_lexical_declaration
    _on_typescript_variable_declaration = _on_javascript_lexical_declaration
    _on_tsx_lexical_declaration = _on_javascript_lexical_declaration
    _on_tsx_variable_declaration = _on_javascript_lexical_declaration

    def _js_variable_functions(self, node, fn_qname, cls_qname):
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name = _node_name(child)
            value = child.child_by_field_name("value")
            if name and value is not None and value.type in (
                "arrow_function", "function", "function_expression",
            ):
                self._emit_function(child, name, fn_qname, cls_qname)
        self._recurse(node, fn_qname, cls_qname)

    def _on_javascript_class_declaration(self, node, fn_qname, cls_qname):
        self._emit_class(node, cls_qname, "class")
        name = _node_name(node)
        self._recurse(node, fn_qname, self._join(cls_qname, name) if name else cls_qname)

    _on_javascript_abstract_class_declaration = _on_javascript_class_declaration
    _on_typescript_class_declaration = _on_javascript_class_declaration
    _on_typescript_abstract_class_declaration = _on_javascript_class_declaration
    _on_tsx_class_declaration = _on_javascript_class_declaration

    def _on_typescript_interface_declaration(self, node, fn_qname, cls_qname):
        self._emit_class(node, cls_qname, "interface")
        self._recurse(node, fn_qname, cls_qname)

    _on_tsx_interface_declaration = _on_typescript_interface_declaration

    def _emit_class(self, node, cls_qname, kind):
        name = _node_name(node)
        if not name:
            return None
        qname = self._join(cls_qname, name)
        self.classes.append(
            ExtractedClass(
                name=name,
                qualified_name=qname,
                kind=kind,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                source=self.source[node.start_byte:node.end_byte].decode(
                    "utf-8", "replace"
                ),
                source_hash=_sha(self.source[node.start_byte:node.end_byte]),
                parent_class_qname=cls_qname,
            )
        )
        return qname

    def _on_javascript_method_definition(self, node, fn_qname, cls_qname):
        name = _node_name(node)
        if name and cls_qname:
            self._emit_function(node, name, fn_qname, cls_qname)
        self._recurse(node, fn_qname, cls_qname)

    _on_typescript_method_definition = _on_javascript_method_definition
    _on_tsx_method_definition = _on_javascript_method_definition

    def _on_javascript_import_statement(self, node, fn_qname, cls_qname):
        source = node.child_by_field_name("source")
        if source is not None:
            self._add_rel(
                "imports", self._context(fn_qname, cls_qname),
                source.text.decode("utf-8", "replace").strip("'\""),
                source, "unresolved",
            )

    _on_typescript_import_statement = _on_javascript_import_statement
    _on_tsx_import_statement = _on_javascript_import_statement

    def _on_javascript_call_expression(self, node, fn_qname, cls_qname):
        target = self._call_target(node)
        if target:
            self._add_rel(
                "calls", self._context(fn_qname, cls_qname), target, node,
                "syntactic_candidate",
            )
        self._recurse(node, fn_qname, cls_qname)

    _on_typescript_call_expression = _on_javascript_call_expression
    _on_tsx_call_expression = _on_javascript_call_expression

    # -- go -----------------------------------------------------------------------

    def _on_go_function_declaration(self, node, fn_qname, cls_qname):
        name = _node_name(node)
        if name:
            self._emit_function(node, name, fn_qname, cls_qname)
        self._recurse(node, fn_qname, cls_qname)

    def _on_go_method_declaration(self, node, fn_qname, cls_qname):
        receiver = node.child_by_field_name("receiver")
        type_name = None
        if receiver is not None:
            for descendant in receiver.children:
                stack = list(descendant.children)
                while stack:
                    current = stack.pop()
                    if current.type == "type_identifier":
                        type_name = current.text.decode("utf-8", "replace")
                    stack.extend(current.children)
        name = _node_name(node)
        if name and type_name:
            self._emit_function(node, name, fn_qname, type_name)
        elif name:
            self._emit_function(node, name, fn_qname, cls_qname)
        self._recurse(node, fn_qname, cls_qname)

    def _on_go_type_declaration(self, node, fn_qname, cls_qname):
        for child in node.children:
            if child.type == "type_spec":
                name = _node_name(child)
                type_value = child.child_by_field_name("type")
                if name and type_value is not None and type_value.type == "struct_type":
                    self._emit_class(child, cls_qname, "struct")
        self._recurse(node, fn_qname, cls_qname)

    def _on_go_import_declaration(self, node, fn_qname, cls_qname):
        ctx = self._context(fn_qname, cls_qname)
        for child in node.children:
            if child.type == "import_spec":
                path = child.child_by_field_name("path")
                if path is not None:
                    self._add_rel(
                        "imports", ctx,
                        path.text.decode("utf-8", "replace").strip('"'),
                        path, "unresolved",
                    )

    _on_go_call_expression = _on_javascript_call_expression

    # -- rust -----------------------------------------------------------------------

    def _on_rust_function_item(self, node, fn_qname, cls_qname):
        name = _node_name(node)
        if name:
            self._emit_function(node, name, fn_qname, cls_qname)
        self._recurse(node, fn_qname, cls_qname)

    def _on_rust_struct_item(self, node, fn_qname, cls_qname):
        self._emit_class(node, cls_qname, "struct")
        self._recurse(node, fn_qname, cls_qname)

    def _on_rust_impl_item(self, node, fn_qname, cls_qname):
        type_field = node.child_by_field_name("type")
        impl_qname = cls_qname
        if type_field is not None:
            impl_qname = type_field.text.decode("utf-8", "replace")
        self._recurse(node, fn_qname, impl_qname)

    # -- java -------------------------------------------------------------------------

    def _on_java_class_declaration(self, node, fn_qname, cls_qname):
        self._emit_class(node, cls_qname, "class")
        name = _node_name(node)
        supers = node.child_by_field_name("superclass")
        qname = self._join(cls_qname, name) if name else cls_qname
        if supers is not None:
            self._add_rel(
                "inherits", qname, supers.text.decode("utf-8", "replace"),
                supers, "syntactic_candidate",
            )
        self._recurse(node, fn_qname, qname)

    def _on_java_interface_declaration(self, node, fn_qname, cls_qname):
        self._emit_class(node, cls_qname, "interface")
        name = _node_name(node)
        self._recurse(node, fn_qname, self._join(cls_qname, name) if name else cls_qname)

    _on_java_enum_declaration = _on_java_class_declaration

    def _on_java_method_declaration(self, node, fn_qname, cls_qname):
        name = _node_name(node)
        if name and cls_qname:
            self._emit_function(node, name, fn_qname, cls_qname)
        self._recurse(node, fn_qname, cls_qname)

    def _on_java_method_invocation(self, node, fn_qname, cls_qname):
        target = self._call_target(node)
        if target:
            self._add_rel(
                "calls", self._context(fn_qname, cls_qname), target, node,
                "syntactic_candidate",
            )
        self._recurse(node, fn_qname, cls_qname)

    # -- c / cpp ------------------------------------------------------------------------

    def _c_function(self, node, fn_qname, cls_qname):
        declarator = node.child_by_field_name("declarator")
        name = None
        if declarator is not None:
            inner = declarator.child_by_field_name("declarator")
            if inner is not None:
                raw = inner.text.decode("utf-8", "replace")
                name = raw.split("::")[-1]
                if "::" in raw:
                    cls_qname = raw.split("::")[0]
        if name:
            self._emit_function(node, name, fn_qname, cls_qname)
        self._recurse(node, fn_qname, cls_qname)

    _on_c_function_definition = _c_function
    _on_cpp_function_definition = _c_function

    def _on_cpp_class_specifier(self, node, fn_qname, cls_qname):
        self._emit_class(node, cls_qname, "class")
        name = _node_name(node)
        self._recurse(node, fn_qname, self._join(cls_qname, name) if name else cls_qname)

    def _on_cpp_struct_specifier(self, node, fn_qname, cls_qname):
        self._emit_class(node, cls_qname, "struct")
        name = _node_name(node)
        self._recurse(node, fn_qname, self._join(cls_qname, name) if name else cls_qname)

    _on_c_call_expression = _on_javascript_call_expression
    _on_cpp_call_expression = _on_javascript_call_expression


_PARSERS: dict[str, Parser] = {}


def _parser(language: str) -> Parser:
    parser = _PARSERS.get(language)
    if parser is None:
        parser = Parser(_LANGUAGES[language])
        _PARSERS[language] = parser
    return parser


def extract(source: bytes, language: str, path: str) -> ExtractedFile:
    """Parse one source file into entities and relationship candidates."""
    if language not in _LANGUAGES:
        raise ValueError(f"unsupported language: {language!r}")
    tree = _parser(language).parse(source)
    ex = _Extractor(source, str(path), language)
    ex.walk(tree.root_node)
    return ExtractedFile(
        path=str(path),
        language=language,
        parse_ok=not tree.root_node.has_error,
        functions=ex.functions,
        classes=ex.classes,
        relationships=ex.relationships,
    )
