"""Spec: REQ-INFRA-6975, SCENARIO-INFRA-6975-A..D

A memory file whose body grew while its `description:` and `MEMORY.md` line stayed the same
is flagged. The check keeps its own baseline and updates it by observation, so no discipline
is required to maintain it.

INCIDENT 2026-09-05, twice. `project_arc_eval_is_unobservable_until_it_ends.md` carried a
correction for three days while its index line asserted the pre-fix claim; a session read the
line, believed it, and briefed a subagent with it. Seven hours after that lesson was written
down, `feedback_measure_the_working_process.md` was appended to four times with its index line
untouched. Prose failed; this is the check.

Several tests below came from the same-day adversarial review: the pre-existing dashboard tests
were rewriting the LIVE baseline on every pytest run; an Edit that used the description line as
unchanged anchor context got no reminder; a missing MEMORY.md read as clean.
"""

from __future__ import annotations

import io
import json
import os
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


def _edit(mem: Path, name: str, old: str, new: str) -> dict:
    return {
        "tool_name": "Edit",
        "tool_input": {"file_path": str(mem / name), "old_string": old, "new_string": new},
    }


def _sidecar(mem: Path) -> dict:
    return json.loads((mem / mid.BASELINE_NAME).read_text())["files"]


@pytest.fixture()
def mem(tmp_path: Path) -> Path:
    """A memory directory with one indexed file and a baseline already taken."""

    _write(tmp_path, "a.md", "one fact", "The fact.\n")
    _index(tmp_path, {"a.md": "the fact"})
    assert "baseline created" in mid.memory_lines(tmp_path)[0]
    return tmp_path


GROWN = "The fact.\n\nA new fact.\nWith a second line.\n"


# --- SCENARIO-A: body grows, summary untouched -> DRIFTED; persists; both halves must move ---


def test_body_growth_with_summary_untouched_is_flagged(mem: Path) -> None:
    _write(mem, "a.md", "one fact", GROWN)
    line = mid.memory_lines(mem)[0]
    assert "1 DRIFTED" in line
    assert "a.md(+2)" in line


def test_flag_persists_and_growth_accumulates_until_summary_moves(mem: Path) -> None:
    """Re-baselining on a flag would forgive the drift after one hour. The flag must hold."""
    _write(mem, "a.md", "one fact", GROWN)
    assert "a.md(+2)" in mid.memory_lines(mem)[0]
    _write(mem, "a.md", "one fact", GROWN + "And a third.\n")
    assert "a.md(+3)" in mid.memory_lines(mem)[0]


def test_description_alone_does_not_clear_the_flag(mem: Path) -> None:
    """The first incident's residual state: description fixed, index line still stale."""
    _write(mem, "a.md", "two facts now", GROWN)
    assert "1 DRIFTED" in mid.memory_lines(mem)[0]


def test_index_line_alone_does_not_clear_the_flag(mem: Path) -> None:
    _write(mem, "a.md", "one fact", GROWN)
    _index(mem, {"a.md": "the fact, and a new fact"})
    assert "1 DRIFTED" in mid.memory_lines(mem)[0]


def test_both_halves_moving_clears_the_flag(mem: Path) -> None:
    _write(mem, "a.md", "two facts now", GROWN)
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
    _write(tmp_path, "u.md", "one fact", GROWN)
    assert "u.md(+2)" in mid.memory_lines(tmp_path)[0]
    _write(tmp_path, "u.md", "two facts", GROWN)
    assert mid.memory_lines(tmp_path)[0].endswith("1 files, 0 drifted")


def test_the_baseline_is_maintained_by_the_check_not_by_anyone(mem: Path) -> None:
    assert set(_sidecar(mem)) == {"a.md"}


def test_a_deleted_file_leaves_the_baseline(mem: Path) -> None:
    _write(mem, "b.md", "b", "B.\n")
    mid.memory_lines(mem)
    assert set(_sidecar(mem)) == {"a.md", "b.md"}
    (mem / "b.md").unlink()
    mid.memory_lines(mem)
    assert set(_sidecar(mem)) == {"a.md"}


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


def test_single_quoted_descriptions_normalize_too() -> None:
    assert mid.normalize_description("'it''s a fact'") == "it's a fact"
    assert mid.normalize_description('"a \\"b\\" c"') == 'a "b" c'
    assert mid.normalize_description("plain") == "plain"


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
    assert _sidecar(mem)


def test_a_malformed_baseline_entry_is_reset_not_crashed(mem: Path) -> None:
    """An entry `{}` raised KeyError inside compare and killed the dashboard (review finding)."""
    (mem / mid.BASELINE_NAME).write_text(json.dumps({"version": 1, "files": {"a.md": {}}}))
    assert "RESET" in mid.memory_lines(mem)[0]
    (mem / mid.BASELINE_NAME).write_text(json.dumps({"version": 1, "files": []}))
    assert "RESET" in mid.memory_lines(mem)[0]


def test_a_missing_directory_is_named_not_silent(tmp_path: Path) -> None:
    line = mid.memory_lines(tmp_path / "nope")[0]
    assert line.startswith(mid.PREFIX + "UNREADABLE")


def test_a_missing_memory_md_is_named_and_nothing_is_written(tmp_path: Path) -> None:
    """With MEMORY.md gone every file looked unindexed and the line read `0 drifted`."""
    _write(tmp_path, "a.md", "one fact", "The fact.\n")
    line = mid.memory_lines(tmp_path)[0]
    assert "MEMORY.md MISSING" in line
    assert "0 drifted" not in line
    assert not (tmp_path / mid.BASELINE_NAME).exists()
    assert mid.main(["--memory-dir", str(tmp_path)]) == 1


def test_an_unreadable_file_is_named_and_nothing_is_written(mem: Path) -> None:
    _write(mem, "b.md", "b", "B.\n")
    (mem / "b.md").chmod(0)
    try:
        before = (mem / mid.BASELINE_NAME).read_bytes()
        line = mid.memory_lines(mem)[0]
        assert line.startswith(mid.PREFIX + "UNREADABLE 1 file(s): b.md")
        assert (mem / mid.BASELINE_NAME).read_bytes() == before
    finally:
        (mem / "b.md").chmod(0o644)


def test_a_clean_run_states_the_population_it_scanned(mem: Path) -> None:
    """A count of zero over an unstated population is not a result."""
    assert mid.memory_lines(mem)[0] == mid.PREFIX + "1 files, 0 drifted"


def test_dry_run_does_not_write_the_baseline(mem: Path) -> None:
    _write(mem, "a.md", "one fact", GROWN)
    before = (mem / mid.BASELINE_NAME).read_bytes()
    assert mid.main(["--dry-run", "--memory-dir", str(mem)]) == 1
    assert (mem / mid.BASELINE_NAME).read_bytes() == before


def test_a_test_that_did_not_opt_in_never_writes_the_live_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: the dashboard's own tests call render() nine times and would have
    rewritten the real sidecar on every pytest run. Under pytest the IMPLICIT directory is read
    but not written unless CLAUDE_MEMORY_DIR opts in."""
    _write(tmp_path, "a.md", "one fact", "The fact.\n")
    _index(tmp_path, {"a.md": "the fact"})
    monkeypatch.delenv("CLAUDE_MEMORY_DIR", raising=False)
    monkeypatch.setattr(mid, "memory_dir", lambda repo=None: tmp_path)
    assert os.environ.get("PYTEST_CURRENT_TEST")
    assert "baseline created" in mid.memory_lines()[0]
    assert not (tmp_path / mid.BASELINE_NAME).exists()
    mid.memory_lines(tmp_path)  # an explicit directory is an opt-in
    assert (tmp_path / mid.BASELINE_NAME).exists()


# --- SCENARIO-C: the point-of-append reminder, from the payload alone ---


def test_an_append_that_does_not_touch_the_description_gets_a_reminder(mem: Path) -> None:
    text = mid.hook_reminder(_edit(mem, "a.md", "The fact.\n", "The fact.\n\nNew.\nMore.\n"), mem)
    assert "a.md" in text
    assert "`description:` has not moved" in text


def test_an_edit_that_changes_the_description_is_reminded_only_about_the_index_line(
    mem: Path,
) -> None:
    payload = _edit(
        mem,
        "a.md",
        "description: one fact\n",
        "description: two facts\n\nNew.\nMore.\nStill more.\n",
    )
    text = mid.hook_reminder(payload, mem)
    assert "`MEMORY.md` line has not" in text
    assert "`description:` has not moved" not in text


def test_a_description_line_as_unchanged_anchor_context_is_not_a_touch(mem: Path) -> None:
    """Review finding: presence of `description:` in old/new was read as a touch, so an Edit
    anchored on the frontmatter got no reminder."""
    old = "description: one fact\nmetadata:\n  type: feedback\n---\n\nThe fact.\n"
    new = old + "\nNew.\nMore.\n"
    assert "`description:` has not moved" in mid.hook_reminder(_edit(mem, "a.md", old, new), mem)


def test_a_one_line_edit_gets_no_reminder(mem: Path) -> None:
    assert mid.hook_reminder(_edit(mem, "a.md", "The fact.", "The fact, better.\n"), mem) == ""


def test_a_same_size_replacement_gets_no_reminder(mem: Path) -> None:
    assert mid.hook_reminder(_edit(mem, "a.md", "a\nb\nc\n", "x\ny\nz\n"), mem) == ""


def test_reminder_names_the_stale_half_and_goes_quiet_when_both_moved(mem: Path) -> None:
    """Sixteen appends to one file in five hours (2026-07-24) must not mean sixteen reminders
    once the author has updated the summary; but a description moved without the index line
    is the first incident's state and stays loud, naming the index line."""
    _write(mem, "a.md", "two facts now", "The fact.\n")
    payload = _edit(mem, "a.md", "The fact.\n", "The fact.\n\nNew.\nMore.\n")
    text = mid.hook_reminder(payload, mem)
    assert "`MEMORY.md` line has not" in text
    _index(mem, {"a.md": "the fact, and a new fact"})
    assert mid.hook_reminder(payload, mem) == ""


def test_a_new_unindexed_file_gets_a_reminder_on_write(mem: Path) -> None:
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(mem / "new.md"), "content": "x"},
    }
    assert "`MEMORY.md` has no line for it" in mid.hook_reminder(payload, mem)


def test_a_typo_fix_in_an_unindexed_file_gets_no_reminder(mem: Path) -> None:
    """Review finding: 51 files are unindexed; a one-word fix to any of them printed a reminder."""
    _write(mem, "u.md", "u", "U.\n")
    assert mid.hook_reminder(_edit(mem, "u.md", "U.", "U, fixed."), mem) == ""
    assert "no line for it" in mid.hook_reminder(_edit(mem, "u.md", "U.\n", "U.\n\nA.\nB.\n"), mem)


def test_the_hook_ignores_files_outside_the_memory_directory(mem: Path, tmp_path: Path) -> None:
    other = tmp_path.parent / f"{tmp_path.name}_other"
    other.mkdir()
    (other / "x.md").write_text("x")
    assert mid.hook_reminder(_edit(other, "x.md", "x", "a\nb\nc\n"), mem) == ""


def test_the_hook_ignores_memory_md_itself_and_non_markdown(mem: Path) -> None:
    assert mid.hook_reminder(_edit(mem, "MEMORY.md", "a", "a\nb\nc\n"), mem) == ""
    assert mid.hook_reminder(_edit(mem, mid.BASELINE_NAME, "a", "a\nb\nc\n"), mem) == ""


def test_the_hook_ignores_other_tools(mem: Path) -> None:
    payload = {"tool_name": "Read", "tool_input": {"file_path": str(mem / "zzz.md")}}
    assert mid.hook_reminder(payload, mem) == ""


def test_the_hook_reads_the_baseline_for_a_rewrite_of_an_indexed_file(mem: Path) -> None:
    """A Write payload carries no old content, so growth comes from the baseline."""
    _write(mem, "a.md", "one fact", "The fact.\n\nNew.\nMore.\n")
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(mem / "a.md"), "content": "ignored"},
    }
    assert "body grew by 2" in mid.hook_reminder(payload, mem)
    _write(mem, "a.md", "two facts", "The fact.\n\nNew.\nMore.\n")
    _index(mem, {"a.md": "the fact, and a new fact"})
    assert mid.hook_reminder(payload, mem) == ""


def test_the_hook_derives_the_directory_from_the_edited_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: a session in a git worktree derives a different project name, so the
    reminder was silent for real memory edits. The edited file's own path decides."""
    real = tmp_path / ".claude" / "projects" / "-some-other-project" / "memory"
    real.mkdir(parents=True)
    _write(real, "a.md", "one fact", "The fact.\n")
    _index(real, {"a.md": "the fact"})
    monkeypatch.delenv("CLAUDE_MEMORY_DIR", raising=False)
    monkeypatch.setattr(mid, "memory_dir", lambda repo=None: tmp_path / "elsewhere")
    text = mid.hook_reminder(_edit(real, "a.md", "The fact.\n", "The fact.\n\nNew.\nMore.\n"))
    assert "`description:` has not moved" in text


def test_the_hook_entrypoint_never_fails_an_edit(
    mem: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed payload, or `--memory-dir` with no value, must exit 0 and print nothing."""
    bad = {"tool_name": "Edit", "tool_input": {"file_path": {"x": 1}}}
    for argv, stdin in (
        (["--hook", "--memory-dir", str(mem)], json.dumps(bad)),
        (["--hook", "--memory-dir"], json.dumps(_edit(mem, "a.md", "a", "a\nb\nc\n"))),
        (["--hook"], "not json"),
    ):
        sys.stdin = io.StringIO(stdin)  # type: ignore[assignment]
        try:
            assert mid.main(argv) == 0
        finally:
            sys.stdin = sys.__stdin__
        assert capsys.readouterr().out == ""


def test_the_hook_entrypoint_emits_additional_context(
    mem: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sys.stdin = io.StringIO(
        json.dumps(_edit(mem, "a.md", "The fact.\n", "The fact.\n\nNew.\nMore.\n"))
    )  # type: ignore[assignment]
    try:
        assert mid.main(["--hook", "--memory-dir", str(mem)]) == 0
    finally:
        sys.stdin = sys.__stdin__
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "a.md" in out["hookSpecificOutput"]["additionalContext"]


# --- SCENARIO-D: the dashboard prints the line (the call site, not only the function) ---


def _quiet_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dash, "_run", lambda *a: "")
    monkeypatch.setattr(dash, "gpu_rows", lambda: [])
    monkeypatch.setattr(dash, "public_set_efficiency", lambda: {"mean_ratio": None, "missing": 0})
    monkeypatch.setattr(dash, "generalization_levels", lambda: {"measured": False})
    monkeypatch.setattr(dash, "gate_cascade_lines", lambda: [])
    monkeypatch.setattr(dash, "flag_states", lambda names: {})
    monkeypatch.setattr(
        dash, "conductor_state", lambda: {"active": "?", "pid": 0, "milestone": "?", "children": 0}
    )


def test_the_dashboard_render_carries_the_memory_line(
    mem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(mem))
    _write(mem, "a.md", "one fact", GROWN)
    _quiet_dashboard(monkeypatch)
    out = dash.render([])
    assert any(line.startswith(mid.PREFIX) and "a.md(+2)" in line for line in out.splitlines())


def test_a_crash_inside_the_check_does_not_take_the_dashboard_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review finding: the call had no guard, so a KeyError in the check killed every line."""
    _quiet_dashboard(monkeypatch)

    def boom() -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(dash, "memory_lines", boom)
    out = dash.render([])
    assert "memory      CHECK FAILED: RuntimeError: boom" in out
    assert "head        " in out
