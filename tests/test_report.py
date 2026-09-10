"""Markdown section upsert used by every results-producing script."""

from __future__ import annotations

from ml.report import markdown_table, upsert_section


def test_section_is_created_when_absent(tmp_path):
    path = tmp_path / "results.md"
    upsert_section("Alpha", "first body", path)
    assert "## Alpha" in path.read_text(encoding="utf-8")


def test_rerunning_replaces_rather_than_appends(tmp_path):
    """Scripts are re-run constantly; stacking stale tables would mislead."""
    path = tmp_path / "results.md"
    upsert_section("Alpha", "old numbers", path)
    upsert_section("Alpha", "new numbers", path)
    text = path.read_text(encoding="utf-8")
    assert text.count("## Alpha") == 1
    assert "old numbers" not in text
    assert "new numbers" in text


def test_other_sections_survive_an_update(tmp_path):
    path = tmp_path / "results.md"
    upsert_section("Alpha", "a", path)
    upsert_section("Beta", "b", path)
    upsert_section("Alpha", "a2", path)
    text = path.read_text(encoding="utf-8")
    assert "## Beta" in text and "b" in text
    assert "a2" in text


def test_markdown_table_shape():
    table = markdown_table(["x", "y"], [[1, 2], [3, 4]])
    lines = table.splitlines()
    assert lines[0] == "| x | y |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1 | 2 |"


def test_body_leading_blank_lines_are_trimmed(tmp_path):
    """Callers pass triple-quoted blocks that open with a newline."""
    path = tmp_path / "results.md"
    upsert_section("Alpha", "\n\nbody text\n", path)
    assert "## Alpha\n\nbody text\n" in path.read_text(encoding="utf-8")
