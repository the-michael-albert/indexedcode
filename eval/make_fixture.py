"""Build the indexing-test fixture corpus and zip it.

Deterministic multi-language corpus exercising abstract classes,
inheritance chains, interfaces/traits, methods, nested and arrow
functions, imports, and cross-file calls — the shapes the Phase 2
extractor must handle at scale.

Usage:
    python eval/make_fixture.py [eval/indexing-fixture.zip]
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PY_MODULE = '''import json
import logging
from abc import ABC, abstractmethod

LOGGER = logging.getLogger(__name__)


class AbstractService{n}(ABC):
    """Abstract service contract #{n}."""

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def handle(self, payload: dict) -> dict:
        """Process one payload."""

    @abstractmethod
    def name(self) -> str:
        """Return the service name."""


class ConcreteService{n}A(AbstractService{n}):
    def handle(self, payload: dict) -> dict:
        LOGGER.debug("handling %s", payload)
        return dict(sorted(payload.items()))

    def name(self) -> str:
        return f"{{self._name}}-a"


class ConcreteService{n}B(AbstractService{n}):
    def handle(self, payload: dict) -> dict:
        merged = {{**payload, "service": self.name()}}
        return merged

    def name(self) -> str:
        return f"{{self._name}}-b"


def dispatch{n}(service: AbstractService{n}, events: list) -> list:
    results = []
    for event in events:
        results.append(service.handle(json.loads(json.dumps(event))))
    return results


def summarize{n}(results: list) -> dict:
    return {{"count": len(results), "ok": all(r is not None for r in results)}}
'''

PY_TEST = '''from services.abstract_service_0 import (
    AbstractService0,
    ConcreteService0A,
    dispatch0,
    summarize0,
)


def test_dispatch_round_trip():
    service = ConcreteService0A("unit")
    results = dispatch0(service, [{{"a": 1}}, {{"b": 2}}])
    assert summarize0(results)["count"] == 2


def test_abstract_cannot_instantiate():
    try:
        AbstractService0("nope")
    except TypeError:
        return
    raise AssertionError("abstract class was instantiated")
'''

TS_MODULE = '''export interface Repository{n} {{
  find(id: string): Promise<Record<string, unknown>>;
  save(entity: Record<string, unknown>): Promise<void>;
}}

export abstract class BaseRepository{n} implements Repository{n} {{
  protected prefix = "repo{n}";

  abstract find(id: string): Promise<Record<string, unknown>>;

  async save(entity: Record<string, unknown>): Promise<void> {{
    await this.find(String(entity.id ?? "new"));
  }}

  describe(): string {{
    return `${{this.prefix}} repository`;
  }}
}}

export class SqlRepository{n} extends BaseRepository{n} {{
  async find(id: string): Promise<Record<string, unknown>> {{
    return {{ id, prefix: this.prefix }};
  }}
}}

export function createRepository{n}(): Repository{n} {{
  return new SqlRepository{n}();
}}

export const arrowHandler{n} = async (input: string) => input.toUpperCase();
'''

JAVA_MODULE = '''package demo{n};

public abstract class Device{n} {{
    protected final String id;

    public Device{n}(String id) {{
        this.id = id;
    }}

    public abstract String read();

    public String describe() {{
        return id + ":" + read();
    }}
}}

class Sensor{n} extends Device{n} {{
    public Sensor{n}(String id) {{
        super(id);
    }}

    @Override
    public String read() {{
        return "value";
    }}
}}
'''

GO_MODULE = '''package store{n}

type Store{n} struct {{
	items map[string]string
}}

func NewStore{n}() *Store{n} {{
	return &Store{n}{{items: make(map[string]string)}}
}}

func (s *Store{n}) Put(key string, value string) {{
	s.items[key] = value
}}

func (s *Store{n}) Get(key string) string {{
	return s.items[key]
}}

type Loader{n} interface {{
	Load(key string) string
}}
'''

RUST_MODULE = '''pub trait Shape{n} {{
    fn area(&self) -> f64;

    fn describe(&self) -> String {{
        format!("shape-{{}}", self.area())
    }}
}}

pub struct Circle{n} {{
    pub radius: f64,
}}

impl Shape{n} for Circle{n} {{
    fn area(&self) -> f64 {{
        3.14159 * self.radius * self.radius
    }}
}}

pub fn make_circle{n}(radius: f64) -> Circle{n} {{
    Circle{n} {{ radius }}
}}
'''

CPP_MODULE = '''class Node{n} {{
public:
    virtual ~Node{n}() = default;
    virtual int evaluate() const = 0;
}};

class Constant{n} : public Node{n} {{
    int value;

public:
    explicit Constant{n}(int v) : value(v) {{}}

    int evaluate() const override {{ return value; }}
}};

inline int compute{n}(const Node{n}& node) {{ return node.evaluate(); }}
'''

JS_MODULE = '''export class Animal{n} {{
  constructor(name) {{
    this.name = name;
  }}

  speak() {{
    return `${{this.name}} makes a sound`;
  }}
}}

export class Dog{n} extends Animal{n} {{
  speak() {{
    return `${{this.name}} barks`;
  }}
}}

export function adopt{n}(animal) {{
  return animal.speak();
}}

export const feed{n} = (animal) => animal.name;
'''

README = """# Indexing fixture corpus

Deterministic generated corpus for testing IndexedCode end-to-end.

- Python: ABC-based abstract services, two concrete subclasses each,
  module functions, a test file importing them.
- TypeScript: interfaces, abstract base classes, `implements`/`extends`,
  async methods, arrow functions.
- Java: abstract classes with abstract methods and subclasses.
- Go: structs with receiver methods and an interface.
- Rust: traits with default methods and `impl` blocks.
- C++: virtual abstract classes with inheritance.
- JavaScript: `class extends` hierarchies and arrow functions.
"""


def build_corpus(dest: Path, *, python_modules: int = 12, ts_modules: int = 4,
                 java_modules: int = 3, go_modules: int = 3,
                 rust_modules: int = 3, cpp_modules: int = 3,
                 js_modules: int = 3) -> dict:
    """Write the corpus under ``dest``; returns a file-count summary."""
    dest.mkdir(parents=True, exist_ok=True)
    counts = {}

    for n in range(python_modules):
        _write(dest / "src" / "services" / f"abstract_service_{n}.py",
               PY_MODULE.format(n=n))
        counts["python"] = counts.get("python", 0) + 1
    _write(dest / "tests" / "test_services.py", PY_TEST)
    counts["python"] += 1

    for name, template, count, folder in (
        ("typescript", TS_MODULE, ts_modules, "ts"),
        ("java", JAVA_MODULE, java_modules, "java"),
        ("go", GO_MODULE, go_modules, "go"),
        ("rust", RUST_MODULE, rust_modules, "rust"),
        ("cpp", CPP_MODULE, cpp_modules, "cpp"),
        ("javascript", JS_MODULE, js_modules, "js"),
    ):
        for n in range(count):
            _write(dest / folder / f"module_{n}{EXTENSIONS[name]}",
                   template.format(n=n))
        counts[name] = count

    _write(dest / "README.md", README)
    return counts


EXTENSIONS = {
    "typescript": ".ts", "java": ".java", "go": ".go", "rust": ".rs",
    "cpp": ".cc", "javascript": ".js",
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_zip(zip_path: Path) -> Path:
    """Build the corpus in a temp dir and zip it deterministically."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        build_corpus(corpus)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(corpus.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(corpus.parent))
    return zip_path


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1
                  else Path(__file__).parent / "indexing-fixture.zip")
    result = write_zip(target)
    print(f"wrote {result}")
