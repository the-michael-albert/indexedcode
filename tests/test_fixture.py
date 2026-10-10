"""The committed indexing-fixture corpus as a reusable module."""

import zipfile

from indexedcode.fixture import build_corpus, write_zip


def test_corpus_extracts_abstract_and_inheritance_shapes(tmp_path):
    from indexedcode.extract import detect_language, extract

    build_corpus(tmp_path)
    python_files = sorted(tmp_path.glob("src/services/*.py"))
    assert len(python_files) >= 8

    classes = functions = inherits = abstract_bases = 0
    for path in python_files:
        extracted = extract(path.read_bytes(), "python", path=str(path))
        functions += len(extracted.functions)
        classes += len(extracted.classes)
        inherits += sum(
            1 for r in extracted.relationships if r.relation == "inherits"
        )
        abstract_bases += sum(
            1 for c in extracted.classes
            if c.name.startswith("AbstractService")
        )
    assert abstract_bases >= 8      # one ABC per module
    assert classes >= abstract_bases * 3  # ABC + two concrete subclasses
    assert inherits >= abstract_bases * 2  # each subclass inherits the ABC
    assert functions >= classes * 2  # methods plus module functions


def test_corpus_spans_all_supported_languages(tmp_path):
    from indexedcode.extract import detect_language
    from indexedcode.scan import scan_repository

    build_corpus(tmp_path)
    result = scan_repository(tmp_path)
    languages = {detect_language(f.relative_path) for f in result.files}
    assert languages == {
        "python", "typescript", "java", "go", "rust", "cpp", "javascript",
    }


def test_corpus_is_deterministic(tmp_path):
    import hashlib

    def tree_digest(root):
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    a, b = tmp_path / "a", tmp_path / "b"
    build_corpus(a)
    build_corpus(b)
    assert tree_digest(a) == tree_digest(b)


def test_write_zip_round_trips(tmp_path):
    zip_path = write_zip(tmp_path / "fixture.zip")
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert any("corpus/src/services/abstract_service_0.py" in n
               for n in names)
    assert any(n.endswith("README.md") for n in names)


def test_scale_parameter_controls_corpus_size(tmp_path):
    build_corpus(tmp_path, python_modules=2)
    assert len(list(tmp_path.glob("src/services/*.py"))) == 2
