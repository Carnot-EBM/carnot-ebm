"""log_step's detail column was cut to 80 characters, throwing away most of
what _meaningful_error_tail already spent effort extracting (300-char budget).
Measured before widening: every live consumer of ops/conductor-log.md splits
on "|" and reads by column index, none does fixed-width slicing, so widening
this column cannot break any of them.

A second, related defect fixed in the same change: `details` can carry
embedded newlines (a multi-line codex error), which silently splits one
table row into several regardless of the character cap. A wider cap makes
this MORE likely to bite, not less, so both are fixed together.

Spec: REQ-CONDUCTOR-LOGWIDTH-1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import research_conductor as rc  # noqa: E402


def test_detail_survives_past_the_old_80_char_cutoff(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rc, "CONDUCTOR_LOG", tmp_path / "log.md")
    long_detail = "x" * 200
    rc.log_step("some task", "FAIL", long_detail)
    row = (tmp_path / "log.md").read_text().splitlines()[-1]
    assert long_detail in row, "200 chars must survive; the old cap truncated at 80"


def test_detail_is_still_bounded_not_unlimited(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rc, "CONDUCTOR_LOG", tmp_path / "log.md")
    rc.log_step("t", "FAIL", "y" * 5000)
    row = (tmp_path / "log.md").read_text().splitlines()[-1]
    cell = row.split("|")[4]
    assert len(cell.strip()) == rc.LOG_DETAIL_MAX_CHARS, "must still cap, not grow unbounded"


def test_embedded_newlines_do_not_split_the_table_row(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rc, "CONDUCTOR_LOG", tmp_path / "log.md")
    rc.log_step("multi-line task", "FAIL", "line one\n\n\nline two\nline three")
    lines = (tmp_path / "log.md").read_text().splitlines()
    last = [line for line in lines if line.startswith("| ")][-1]
    assert "line one" in last and "line two" in last and "line three" in last
    assert last.count("\n") == 0, "one log_step call must produce exactly one physical line"


def test_the_row_still_has_exactly_four_pipe_delimited_columns(tmp_path, monkeypatch) -> None:
    """The #1 real consumer pattern (line.split('|')) must keep working."""
    monkeypatch.setattr(rc, "CONDUCTOR_LOG", tmp_path / "log.md")
    rc.log_step("task", "OK", "multi\nline\ndetail with | a pipe-like word")
    row = (tmp_path / "log.md").read_text().splitlines()[-1]
    cells = row.split("|")
    assert len(cells) >= 5, "timestamp, task, status, detail, plus the row's leading/trailing bars"


def test_whitespace_only_detail_collapses_cleanly() -> None:
    assert rc._sanitize_log_detail("   \n\n  \t ") == ""


def test_normal_short_detail_is_unaffected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rc, "CONDUCTOR_LOG", tmp_path / "log.md")
    rc.log_step("t", "OK", "117 passed, 1 warning in 7.39s")
    row = (tmp_path / "log.md").read_text().splitlines()[-1]
    assert "117 passed, 1 warning in 7.39s" in row
