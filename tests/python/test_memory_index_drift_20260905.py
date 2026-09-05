"""Spec: REQ-INFRA-6975, SCENARIO-INFRA-6975-A..D

A memory file whose body grew while its `description:` and `MEMORY.md` line stayed the same
is flagged. The check keeps its own baseline and updates it by observation, so no discipline
is required to maintain it.

INCIDENT 2026-09-05, twice. `project_arc_eval_is_unobservable_until_it_ends.md` carried a
correction for three days while its index line asserted the pre-fix claim; a session read the
line, believed it, and briefed a subagent with it. Seven hours after that lesson was written
down, `feedback_measure_the_working_process.md` was appended to four times with its index line
untouched. Prose failed; this is the check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import memory_index_drift as mid  # noqa: E402
import outer_loop_dashboard as dash  # noqa: E402

_FILE = """---
name: a-fact
description: {desc}
metadata:
  type: feedback
---

{body}
"""


def _write(mem: Path, name: str, desc: str, body: str) -> None:
    (mem / name).write_text(_FILE.format(desc=desc, body=body))


def _index(mem: Path, entries: dict[str, str]) -> None:
    (mem / "MEMORY.md").write_text(
        "".join(f"- [{n}]({n}) — {hook}\n" for n, hook in entries.items())
    )


@pytest.fixture()
def mem(tmp_path: Path) -> Path:
    """A memory directory with one indexed file and a baseline already taken."""

    _write(tmp_path, "a.md", "one fact", "The fact.\n")
    _index(tmp_path, {"a.md": "the fact"})
    assert "baseline created" in mid.memory_lines(tmp_path)[0]
    return tmp_path


# --- SCENARIO-A: body grows, summary untouched -> DRIFTED; persists; both halves must move ---


def test_body_growth_with_summary_untouched_is_flagged(mem: Path) -> None:
    _write(mem, "a.md", "one fact", "The fact.\n\nA new fact.\nWith a second line.\n")
    line = mid.memory_lines(mem)[0]
    assert "1 DRIFTED" in line
    assert "a.md(+2)" in line


def test_flag_persists_and_growth_accumulates_until_summary_moves(mem: Path) -> None:
    """Re-baselining on a flag would forgive the drift after one hour. The flag must hold."""
    _write(mem, "a.md", "one fact", "The fact.\n\nA new fact.\nWith a second line.\n")
    assert "a.md(+2)" in mid.memory_lines(mem)[0]
    _write(mem, "a.md", "one fact", "The fact.\n\nA new fact.\nWith a second line.\nAnd a third.\n")
    assert "a.md(+3)" in mid.memory_lines(mem)[0]


def test_description_alone_does_not_clear_the_flag(mem: Path) -> None:
    """The first incident's residual state: description fixed, index line still stale."""
    _write(mem, "a.md", "two facts now", "The fact.\n\nA new fact.\nWith a second line.\n")
    assert "1 DRIFTED" in mid.memory_lines(mem)[0]


def test_index_line_alone_does_not_clear_the_flag(mem: Path) -> None:
    _write(mem, "a.md", "one fact", "The fact.\n\nA new fact.\nWith a second line.\n")
    _index(mem, {"a.md": "the fact, and a new fact"})
    assert "1 DRIFTED" in mid.memory_lines(mem)[0]


def test_both_halves_moving_clears_the_flag(mem: Path) -> None:
    _write(mem, "a.md", "two facts now", "The fact.\n\nA new fact.\nWith a second line.\n")
    _index(mem, {"a.md": "the fact, and a new fact"})
    assert mid.memory_lines(mem)[0].endswith("1 files, 0 drifted")


def test_a_one_line_touch_up_is_not_drift(mem: Path) -> None:
    """A typo fix or a one-line clarification re-baselines silently. A fact is never one line."""
    _write(mem, "a.md", "one fact", "The fact, stated better.\n")
    assert mid.memory_lines(mem)[0].endswith("1 files, 0 drifted")
    _write(mem, "a.md", "one fact", "The fact, stated better.\nOne more line.\n")
    assert mid.memory_lines(mem)[0].endswith("1 files, 0 drifted")


def test_an_unindexed_file_is_judged_on_its_description_alone(tmp_path: Path) -> None:
    _write(tmp_path, "u.md", "one fact", "The fact.\n")
    _index(tmp_path, {})
    mid.memory_lines(tmp_path)
    _write(tmp_path, "u.md", "one fact", "The fact.\n\nA new fact.\nWith a second line.\n")
    assert "u.md(+2)" in mid.memory_lines(tmp_path)[0]
    _write(tmp_path, "u.md", "two facts", "The fact.\n\nA new fact.\nWith a second line.\n")
    assert mid.memory_lines(tmp_path)[0].endswith("1 files, 0 drifted")


def test_tooling_requoting_the_description_is_not_a_summary_move(mem: Path) -> None:
    """Found live 2026-09-05: the Edit tool rewrote `description: a "b"` as
    `description: "a \\"b\\""` on the first append. That must not count as the author
    moving the description, or the reminder goes quiet on exactly the append that caused it."""
    _write(mem, "a.md", 'the "quoted" fact', "The fact.\n")
    _index(mem, {"a.md": "the fact"})
    mid.memory_lines(mem)
    _write(mem, "a.md", '"the \\"quoted\\" fact"', "The fact.\n\nNew.\nMore.\n")
    payload = _edit(mem, "a.md", "The fact.\n", "The fact.\n\nNew.\nMore.\n")
    assert "has not moved" in mid.hook_reminder(payload, mem)
    # The author moved ONLY the index line. Hashed raw, the re-quote would count as the
    # description moving too, and the flag would clear with the description still stale.
    _index(mem, {"a.md": "the fact, and a new fact"})
    assert "1 DRIFTED" in mid.memory_lines(mem)[0]


def test_the_baseline_is_maintained_by_the_check_not_by_anyone(mem: Path) -> None:
    """The sidecar exists after a run and is written without any caller touching it."""
    assert (mem / mid.BASELINE_NAME).exists()
    data = json.loads((mem / mid.BASELINE_NAME).read_text())
    assert set(data["files"]) == {"a.md"}


# --- SCENARIO-B: fail closed and loud ---


def test_first_run_says_so_and_never_reads_as_clean(tmp_path: Path) -> None:
    _write(tmp_path, "a.md", "one fact", "The fact.\n")
    _index(tmp_path, {"a.md": "the fact"})
    line = mid.memory_lines(tmp_path)[0]
    assert "baseline created for 1 files" in line
    assert "0 drifted" not in line


def test_a_corrupt_baseline_is_named_and_reset(mem: Path) -> None:
    (mem / mid.BASELINE_NAME).write_text("{not json")
    line = mid.memory_lines(mem)[0]
    assert "RESET" in line
    assert "0 drifted" not in line
    assert json.loads((mem / mid.BASELINE_NAME).read_text())["files"]


def test_a_missing_directory_is_named_not_silent(tmp_path: Path) -> None:
    line = mid.memory_lines(tmp_path / "nope")[0]
    assert line.startswith(mid.PREFIX + "UNREADABLE")


def test_a_clean_run_states_the_population_it_scanned(mem: Path) -> None:
    """A count of zero over an unstated population is not a result."""
    assert mid.memory_lines(mem)[0] == mid.PREFIX + "1 files, 0 drifted"


def test_dry_run_does_not_write_the_baseline(mem: Path) -> None:
    _write(mem, "a.md", "one fact", "The fact.\n\nA new fact.\nWith a second line.\n")
    before = (mem / mid.BASELINE_NAME).read_bytes()
    mid.memory_lines(mem, write=False)
    assert (mem / mid.BASELINE_NAME).read_bytes() == before


# --- SCENARIO-C: the point-of-append reminder, from the payload alone ---


def _edit(mem: Path, name: str, old: str, new: str) -> dict:
    return {
        "tool_name": "Edit",
        "tool_input": {"file_path": str(mem / name), "old_string": old, "new_string": new},
    }


def test_an_append_that_does_not_touch_the_description_gets_a_reminder(mem: Path) -> None:
    text = mid.hook_reminder(_edit(mem, "a.md", "The fact.\n", "The fact.\n\nNew.\nMore.\n"), mem)
    assert "a.md" in text
    assert "`description:` has not moved" in text


def test_an_edit_that_touches_the_description_gets_no_reminder(mem: Path) -> None:
    payload = _edit(
        mem,
        "a.md",
        "description: one fact\n",
        "description: two facts\n\nNew.\nMore.\nStill more.\n",
    )
    assert mid.hook_reminder(payload, mem) == ""


def test_no_repeat_once_the_description_has_moved_since_baseline(mem: Path) -> None:
    """Sixteen appends to one file in five hours (2026-07-24) must not mean sixteen reminders
    once the author has already updated the description."""
    _write(mem, "a.md", "two facts now", "The fact.\n")
    payload = _edit(mem, "a.md", "The fact.\n", "The fact.\n\nNew.\nMore.\n")
    assert mid.hook_reminder(payload, mem) == ""


def test_a_one_line_edit_gets_no_reminder(mem: Path) -> None:
    assert mid.hook_reminder(_edit(mem, "a.md", "The fact.", "The fact, better.\n"), mem) == ""


def test_a_new_unindexed_file_gets_a_reminder_on_write(mem: Path) -> None:
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(mem / "new.md"), "content": "x"},
    }
    assert "`MEMORY.md` has no line for it" in mid.hook_reminder(payload, mem)


def test_the_hook_ignores_files_outside_the_memory_directory(mem: Path, tmp_path: Path) -> None:
    other = tmp_path.parent / f"{tmp_path.name}_other"
    other.mkdir()
    (other / "x.md").write_text("x")
    payload = _edit(other, "x.md", "x", "a\nb\nc\n")
    assert mid.hook_reminder(payload, mem) == ""


def test_the_hook_ignores_memory_md_itself(mem: Path) -> None:
    assert mid.hook_reminder(_edit(mem, "MEMORY.md", "a", "a\nb\nc\n"), mem) == ""


def test_the_hook_reads_the_baseline_for_a_rewrite_of_an_indexed_file(mem: Path) -> None:
    """A Write payload carries no old content, so growth comes from the baseline."""
    _write(mem, "a.md", "one fact", "The fact.\n\nNew.\nMore.\n")
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(mem / "a.md"), "content": "ignored"},
    }
    assert "body grew by 2" in mid.hook_reminder(payload, mem)


def test_the_hook_entrypoint_never_fails_an_edit(
    mem: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed payload (file_path is not a string) must exit 0 and print nothing."""
    import io

    bad = io.StringIO(json.dumps({"tool_name": "Edit", "tool_input": {"file_path": {"x": 1}}}))
    sys.stdin = bad  # type: ignore[assignment]
    try:
        assert mid.main(["--hook", "--memory-dir", str(mem)]) == 0
    finally:
        sys.stdin = sys.__stdin__
    assert capsys.readouterr().out == ""


# --- SCENARIO-D: the dashboard prints the line (the call site, not only the function) ---


def test_the_dashboard_render_carries_the_memory_line(
    mem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(mem))
    _write(mem, "a.md", "one fact", "The fact.\n\nA new fact.\nWith a second line.\n")
    monkeypatch.setattr(dash, "_run", lambda *a: "")
    monkeypatch.setattr(dash, "gpu_rows", lambda: [])
    monkeypatch.setattr(dash, "public_set_efficiency", lambda: {"mean_ratio": None, "missing": 0})
    monkeypatch.setattr(dash, "generalization_levels", lambda: {"measured": False})
    monkeypatch.setattr(dash, "gate_cascade_lines", lambda: [])
    monkeypatch.setattr(dash, "flag_states", lambda names: {})
    monkeypatch.setattr(
        dash, "conductor_state", lambda: {"active": "?", "pid": 0, "milestone": "?", "children": 0}
    )
    out = dash.render([])
    assert any(line.startswith(mid.PREFIX) and "a.md(+2)" in line for line in out.splitlines())
