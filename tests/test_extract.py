"""Tree-sitter extraction contract tests (plan §9 Phase 2 gate).

Every supported language gets real source parsed by its pinned grammar.
Byte coordinates must slice the exact source text back out.
"""

from __future__ import annotations

import pytest

from indexedcode.extract import detect_language, extract


def src(text):
    return text.encode("utf-8")


def fn_by_qname(result, qname):
    matches = [f for f in result.functions if f.qualified_name == qname]
    assert matches, f"no function {qname!r} in {[f.qualified_name for f in result.functions]}"
    return matches[0]


def cls_by_qname(result, qname):
    matches = [c for c in result.classes if c.qualified_name == qname]
    assert matches, f"no class {qname!r} in {[c.qualified_name for c in result.classes]}"
    return matches[0]


# --- language detection -------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("a.py", "python"),
        ("a.js", "javascript"),
        ("a.jsx", "javascript"),
        ("a.ts", "typescript"),
        ("a.tsx", "tsx"),
        ("a.go", "go"),
        ("a.rs", "rust"),
        ("A.java", "java"),
        ("a.c", "c"),
        ("a.h", "c"),
        ("a.cpp", "cpp"),
        ("a.cc", "cpp"),
    ],
)
def test_detect_language(path, language):
    assert detect_language(path) == language


def test_detect_language_unknown_returns_none():
    assert detect_language("a.txt") is None
    assert detect_language("a.md") is None


# --- python -------------------------------------------------------------------


PY = """
import os
from helpers.util import deep

class Base:
    pass

class Greeter(Base):
    def greet(self, name="world"):
        return format(name)

def outer(a, b=1):
    def inner(x):
        return helper(x)
    return inner
"""


def test_python_functions_and_nesting():
    r = extract(src(PY), "python", "pkg/greet.py")
    assert r.parse_ok
    outer = fn_by_qname(r, "outer")
    assert outer.parent_function_qname is None
    inner = fn_by_qname(r, "outer.inner")
    assert inner.parent_function_qname == "outer"
    assert outer.signature.startswith("def outer")
    assert "a, b=1" in outer.signature


def test_python_methods_attach_to_class():
    r = extract(src(PY), "python", "pkg/greet.py")
    greet = fn_by_qname(r, "Greeter.greet")
    assert greet.parent_class_qname == "Greeter"
    assert greet.name == "greet"
    greeter = cls_by_qname(r, "Greeter")
    assert greeter.kind == "class"
    assert greeter.name == "Greeter"


def test_python_byte_coordinates_slice_exact_source():
    r = extract(src(PY), "python", "pkg/greet.py")
    text = PY.encode("utf-8")
    for f in r.functions:
        assert text[f.start_byte:f.end_byte].decode() == f.source
    for c in r.classes:
        assert text[c.start_byte:c.end_byte].decode() == c.source


def test_python_line_numbers_one_based():
    r = extract(src(PY), "python", "pkg/greet.py")
    outer = fn_by_qname(r, "outer")
    assert outer.start_line == PY[: PY.index("def outer")].count("\n") + 1
    assert outer.end_line >= outer.start_line


def test_python_imports_and_calls_and_inheritance():
    r = extract(src(PY), "python", "pkg/greet.py")
    imports = [x for x in r.relationships if x.relation == "imports"]
    assert {x.target_name for x in imports} == {"os", "helpers.util.deep"}
    assert all(x.resolution == "unresolved" for x in imports)
    calls = [x for x in r.relationships if x.relation == "calls"]
    assert "helper" in {x.target_name for x in calls}
    helper_call = next(x for x in calls if x.target_name == "helper")
    assert helper_call.source_qname == "outer.inner"
    assert helper_call.resolution == "syntactic_candidate"
    assert helper_call.snippet  # non-empty source-grounded snippet
    inherits = [x for x in r.relationships if x.relation == "inherits"]
    assert {x.target_name for x in inherits} == {"Base"}
    assert inherits[0].source_qname == "Greeter"


def test_python_syntax_error_still_extracts_valid_nodes():
    bad = "def good():\n    pass\n\ndef broken(:\n"
    r = extract(src(bad), "python", "pkg/bad.py")
    assert r.parse_ok is False
    assert [f.qualified_name for f in r.functions] == ["good"]


# --- javascript ---------------------------------------------------------------


JS = """
import { read } from './reader';

const arrow = (x) => x + 1;

function plain(a, b) {
    return compute(a);
}

class Widget {
    render() {
        return plain(1, 2);
    }
}
"""


def test_javascript_functions_arrow_and_class():
    r = extract(src(JS), "javascript", "src/w.js")
    assert r.parse_ok
    assert {"arrow", "plain"} <= {f.qualified_name for f in r.functions}
    render = fn_by_qname(r, "Widget.render")
    assert render.parent_class_qname == "Widget"
    widget = cls_by_qname(r, "Widget")
    assert widget.kind == "class"


def test_javascript_imports_and_calls():
    r = extract(src(JS), "javascript", "src/w.js")
    imports = [x for x in r.relationships if x.relation == "imports"]
    assert any("./reader" in x.target_name for x in imports)
    calls = [x for x in r.relationships if x.relation == "calls"]
    assert "compute" in {x.target_name for x in calls}
    assert "plain" in {x.target_name for x in calls}


# --- typescript / tsx ---------------------------------------------------------


TS = """
import type { Shape } from './types';

export interface Greets {
    hello(name: string): string;
}

export function greet(n: number): string {
    return String(n);
}
"""


def test_typescript_interface_and_function():
    r = extract(src(TS), "typescript", "src/g.ts")
    assert r.parse_ok
    iface = cls_by_qname(r, "Greets")
    assert iface.kind == "interface"
    fn_by_qname(r, "greet")
    assert any(x.relation == "imports" for x in r.relationships)


TSX = """
export function App({ label }: { label: string }) {
    return <div>{label}</div>;
}
"""


def test_tsx_component_function():
    r = extract(src(TSX), "tsx", "src/App.tsx")
    assert r.parse_ok
    fn_by_qname(r, "App")


# --- go ------------------------------------------------------------------------


GO = """
package repo

import "fmt"

type Reader struct {
    buf []byte
}

func (r *Reader) Read(p []byte) (int, error) {
    return len(p), nil
}

func Add(a, b int) int {
    fmt.Println("adding")
    return a + b
}
"""


def test_go_functions_methods_and_struct():
    r = extract(src(GO), "go", "repo/repo.go")
    assert r.parse_ok
    add = fn_by_qname(r, "Add")
    assert add.parent_class_qname is None
    read = fn_by_qname(r, "Reader.Read")
    assert read.parent_class_qname == "Reader"
    reader = cls_by_qname(r, "Reader")
    assert reader.kind == "struct"


def test_go_calls():
    r = extract(src(GO), "go", "repo/repo.go")
    calls = [x for x in r.relationships if x.relation == "calls"]
    assert "fmt.Println" in {x.target_name for x in calls}


# --- rust ------------------------------------------------------------------------


RS = """
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn norm(&self) -> f64 {
        self.x * self.x
    }
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""


def test_rust_impl_methods_attach_to_struct():
    r = extract(src(RS), "rust", "src/geom.rs")
    assert r.parse_ok
    point = cls_by_qname(r, "Point")
    assert point.kind == "struct"
    norm = fn_by_qname(r, "Point.norm")
    assert norm.parent_class_qname == "Point"
    fn_by_qname(r, "add")


# --- java ------------------------------------------------------------------------


JAVA = """
package pkg;

public class Hello {
    public static void main(String[] args) {
        System.out.println("hi");
    }
}
"""


def test_java_class_and_method():
    r = extract(src(JAVA), "java", "pkg/Hello.java")
    assert r.parse_ok
    hello = cls_by_qname(r, "Hello")
    assert hello.kind == "class"
    main = fn_by_qname(r, "Hello.main")
    assert main.parent_class_qname == "Hello"


# --- c / cpp ----------------------------------------------------------------------


C = """
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}
"""


def test_c_function():
    r = extract(src(C), "c", "src/add.c")
    assert r.parse_ok
    add = fn_by_qname(r, "add")
    assert "int a, int b" in add.signature


CPP = """
class Engine {
public:
    int start(int mode);
};

int Engine::start(int mode) {
    return mode;
}

void freefn() {}
"""


def test_cpp_class_and_methods():
    r = extract(src(CPP), "cpp", "src/engine.cpp")
    assert r.parse_ok
    engine = cls_by_qname(r, "Engine")
    assert engine.kind == "class"
    start = fn_by_qname(r, "Engine.start")
    assert start.parent_class_qname == "Engine"
    fn_by_qname(r, "freefn")


# --- unicode -------------------------------------------------------------------------


def test_unicode_source_coordinates():
    uni = 'def héllo():\n    msg = "héllo wörld 🎉"\n    return msg\n'
    r = extract(src(uni), "python", "pkg/uni.py")
    assert r.parse_ok
    f = fn_by_qname(r, "héllo")
    text = uni.encode("utf-8")
    assert text[f.start_byte:f.end_byte].decode("utf-8") == f.source
    assert f.start_line == 1


# --- unknown language ------------------------------------------------------------------


def test_extract_rejects_unknown_language():
    with pytest.raises(ValueError):
        extract(src("x = 1"), "cobol", "a.cbl")
