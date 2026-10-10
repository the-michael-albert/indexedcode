"""The shipped skill document exists and carries the key safety rules."""

from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / "skills" / "indexedcode" / "SKILL.md"


def test_skill_document_exists():
    assert SKILL.is_file()


def test_skill_names_all_four_tools():
    text = SKILL.read_text()
    for tool in ("code_search", "code_context", "code_references",
                 "code_status"):
        assert tool in text


def test_skill_marks_retrieved_content_untrusted():
    text = SKILL.read_text().lower()
    assert "untrusted" in text
    assert "current" in text  # use current exact source as evidence
