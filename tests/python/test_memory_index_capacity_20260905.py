"""Spec: REQ-INFRA-6976, SCENARIO-INFRA-6976-A..E

The harness loads `MEMORY.md` with a 200-line cap and a 25,000 UTF-16-unit cap, and drops whole
lines from the tail. The check replicates that cut, names what is invisible, fails loud on the
envelope and tier states, and `--demote` moves a pointer line verbatim to a tier-2 index file.

Established 2026-09-05 from the installed binary and two probes through the real harness:
180 lines at 33,119 units loaded 135 entries (the replica predicted 135); 220 short lines loaded
200. The live index was 152 lines and 25,201 units with one entry, the newest, past the cut.
"""

from __future__ import annotations

import io
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

DASH = "—"  # one UTF-16 unit, three UTF-8 bytes: the unit that matters


def _write(mem: Path, name: str, desc: str = "one fact", body: str = "The fact.\n") -> None:
    (mem / name).write_text(_FILE.format(desc=desc, body=body))


def _line(name: str, units: int) -> str:
    """A pointer line of exactly `units` UTF-16 units, with a non-ASCII separator."""

    head = f"- [T]({name}) {DASH} "
    assert units > mid.utf16_units(head)
    return head + "x" * (units - mid.utf16_units(head))


def _index(mem: Path, lines: list[str], name: str = "MEMORY.md") -> None:
    (mem / name).write_text("\n".join(lines) + "\n")


def _targets(mem: Path) -> set[str]:
    return {
        n for rows in mid.index_entries(mem).values() for n, _ in rows if not mid.is_index_file(n)
    }


@pytest.fixture()
def mem(tmp_path: Path) -> Path:
    """A memory directory with one indexed feedback file, one reference file, and a baseline."""

    _write(tmp_path, "feedback_a.md")
    _write(tmp_path, "reference_b.md")
    _index(tmp_path, [_line("feedback_a.md", 40)])
    assert "baseline created" in mid.memory_lines(tmp_path)[0]
    return tmp_path


# --- SCENARIO-A: the replica of the harness cut ---


def test_units_are_utf16_code_units_not_bytes() -> None:
    assert mid.utf16_units(DASH) == 1
    assert len(DASH.encode()) == 3
    assert mid.utf16_units("\U0001f600") == 2  # an astral code point is two units


def test_cut_falls_at_the_last_newline_at_or_before_the_unit_cap(tmp_path: Path) -> None:
    # Newline after line k sits at unit 250 + 250k; after line 99 it sits at exactly 25,000.
    lines = [_line("f000.md", 250)] + [_line(f"f{i:03d}.md", 249) for i in range(1, 100)]
    lines += [_line(f"f{i:03d}.md", 100) for i in range(100, 105)]
    kept, dropped, first_cut = mid.harness_cut("\n".join(lines) + "\n")
    assert (len(kept), len(dropped), first_cut) == (100, 5, False)
    assert dropped[0].startswith("- [T](f100.md)")


def test_an_index_at_exactly_the_cap_loads_in_full(tmp_path: Path) -> None:
    lines = [_line("f000.md", 250)] + [_line(f"f{i:03d}.md", 249) for i in range(1, 100)]
    assert mid.utf16_units("\n".join(lines)) == mid.HARNESS_MAX_UNITS
    kept, dropped, _ = mid.harness_cut("\n".join(lines) + "\n\n")
    assert (len(kept), dropped) == (100, [])


def test_the_line_cap_applies_before_the_unit_cap(mem: Path) -> None:
    _index(mem, [_line(f"f{i:03d}.md", 40) for i in range(220)])
    cap = mid.index_capacity(mem)
    assert cap["lines"] == 220 and cap["units"] < mid.HARNESS_MAX_UNITS
    assert cap["invisible"] == [f"f{i:03d}.md" for i in range(200, 220)]


def test_a_single_line_over_the_cap_is_reported_as_cut_mid_line(mem: Path) -> None:
    _index(mem, [_line("huge.md", mid.HARNESS_MAX_UNITS + 10)])
    cap = mid.index_capacity(mem)
    assert cap["first_line_cut"] is True
    assert any(p.startswith("OVER_CAP") and "mid-line" in p for p in mid.capacity_problems(cap))


def test_the_dashboard_block_carries_the_capacity_line(mem: Path) -> None:
    lines = mid.memory_lines(mem)
    assert len(lines) == 2 and lines[0].endswith("0 drifted")
    assert "units" in lines[1] and "/200 lines" in lines[1] and lines[1].startswith(mid.PREFIX)


# --- SCENARIO-B: the tokens, their fixes, and the clean line ---


def _over_cap(mem: Path, extra: int = 3) -> list[str]:
    per = 200
    n = mid.HARNESS_MAX_UNITS // (per + 1) + extra
    lines = [_line(f"feedback_{i:03d}.md", per) for i in range(n)]
    for i in range(n):
        _write(mem, f"feedback_{i:03d}.md")
    _index(mem, lines)
    return lines


def test_over_cap_names_the_invisible_entries_and_the_fix(mem: Path) -> None:
    n = len(_over_cap(mem))
    line = mid.capacity_lines(mem)[0]
    assert "OVER_CAP" in line and f"feedback_{n - 1:03d}.md" in line
    assert "--demote" in line and f"/{mid.HARNESS_MAX_UNITS:,} units" in line
    assert mid.main(["--memory-dir", str(mem), "--dry-run"]) == 1


def test_over_budget_fires_before_anything_is_dropped(mem: Path) -> None:
    lines = [_line(f"feedback_{i:03d}.md", 200) for i in range(mid.BUDGET_UNITS // 201 + 2)]
    _index(mem, lines)
    cap = mid.index_capacity(mem)
    assert mid.BUDGET_UNITS < cap["units"] <= mid.HARNESS_MAX_UNITS and cap["invisible"] == []
    line = mid.capacity_lines(mem)[0]
    assert "OVER_BUDGET" in line and "OVER_CAP" not in line and "--demote" in line
    assert mid.main(["--memory-dir", str(mem), "--dry-run"]) == 1


def test_a_tier2_group_pointer_in_memory_md_is_misplaced(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40), _line("reference_b.md", 40)])
    line = mid.capacity_lines(mem)[0]
    assert "MISPLACED" in line and line.endswith("--demote reference_b.md")


def test_a_target_indexed_in_two_files_is_duplicate(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40)], name="_index_feedback.md")
    assert "DUPLICATE indexed twice: feedback_a.md" in mid.capacity_lines(mem)[0]


def test_a_pointer_to_no_file_is_missing_target(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40), _line("feedback_gone.md", 40)])
    assert "MISSING_TARGET no such file: feedback_gone.md" in mid.capacity_lines(mem)[0]


def test_a_group_line_whose_count_is_wrong_is_stale(mem: Path) -> None:
    _index(mem, [_line("reference_b.md", 40)], name="_index_reference.md")
    _index(mem, [mid._group_line("reference", 7), _line("feedback_a.md", 40)])
    assert (
        "GROUP_COUNT_STALE _index_reference.md says 7 entries, holds 1"
        in mid.capacity_lines(mem)[0]
    )


def test_a_clean_index_states_units_lines_tiers_and_reachable_pointers(mem: Path) -> None:
    _index(mem, [_line("reference_b.md", 40)], name="_index_reference.md")
    _index(mem, [mid._group_line("reference", 1), _line("feedback_a.md", 40)])
    line = mid.capacity_lines(mem)[0]
    assert "tier-2 _index_reference.md 1" in line and line.endswith("2 pointers reachable")
    assert mid.main(["--memory-dir", str(mem), "--dry-run"]) == 0


def test_the_exit_code_reads_every_line_not_only_the_first(mem: Path) -> None:
    _over_cap(mem)
    lines = mid.memory_lines(mem, write=False)
    assert "DRIFTED" not in lines[0] and "OVER_CAP" in lines[1]
    assert mid.main(["--memory-dir", str(mem), "--dry-run"]) == 1


# --- SCENARIO-C: --demote moves a line verbatim and never deletes ---


def test_demote_moves_the_line_verbatim_creates_the_tier2_file_and_the_group_line(
    mem: Path,
) -> None:
    ref = _line("reference_b.md", 60)
    _index(mem, [_line("feedback_a.md", 40), ref])
    before = _targets(mem)
    msgs = mid.demote(mem, ["reference_b.md"])
    assert msgs == ["reference_b.md: demoted to _index_reference.md"]
    assert _targets(mem) == before
    assert ref not in (mem / "MEMORY.md").read_text()
    assert ref in (mem / "_index_reference.md").read_text().splitlines()
    top = (mem / "MEMORY.md").read_text().splitlines()[0]
    assert top.startswith("- [Index: reference (1 entries)](_index_reference.md)")
    assert mid.capacity_problems(mid.index_capacity(mem)) == []


def test_demote_recounts_the_group_line_instead_of_adding_a_second(mem: Path) -> None:
    _write(mem, "reference_c.md")
    _index(
        mem, [_line("feedback_a.md", 40), _line("reference_b.md", 40), _line("reference_c.md", 40)]
    )
    mid.demote(mem, ["reference_b.md"])
    mid.demote(mem, ["reference_c.md"])
    text = (mem / "MEMORY.md").read_text()
    assert text.count("(_index_reference.md)") == 1 and "(2 entries)" in text
    assert mid.index_capacity(mem)["tier2"] == {"_index_reference.md": 2}


def test_demote_keeps_group_lines_together_at_the_top(mem: Path) -> None:
    _write(mem, "project_p.md")
    _index(
        mem, [_line("feedback_a.md", 40), _line("reference_b.md", 40), _line("project_p.md", 40)]
    )
    mid.demote(mem, ["reference_b.md"])
    mid.demote(mem, ["project_p.md"])
    lines = (mem / "MEMORY.md").read_text().splitlines()
    assert "(_index_reference.md)" in lines[0] and "(_index_project.md)" in lines[1]
    assert lines[2].startswith("- [T](feedback_a.md)")


def test_demote_reports_and_skips_a_name_it_cannot_move(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40)], name="_index_feedback.md")
    _index(mem, [mid._group_line("feedback", 1)])
    assert mid.demote(mem, ["feedback_a.md"]) == [
        "feedback_a.md: no MEMORY.md line, already in _index_feedback.md"
    ]
    assert mid.demote(mem, ["nothing.md"]) == ["nothing.md: no MEMORY.md line anywhere"]
    assert mid.demote(mem, ["MEMORY.md"]) == ["MEMORY.md: refused, it is an index file"]


def test_a_demotion_is_not_a_summary_move(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40), _line("reference_b.md", 40)])
    assert "0 drifted" in mid.memory_lines(mem)[0]
    before = mid.index_lines(mem)["reference_b.md"]
    mid.demote(mem, ["reference_b.md"])
    assert mid.index_lines(mem)["reference_b.md"] == before
    assert "0 drifted" in mid.memory_lines(mem)[0]


def test_the_demote_entrypoint_moves_the_line(
    mem: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _index(mem, [_line("feedback_a.md", 40), _line("reference_b.md", 40)])
    assert mid.main(["--memory-dir", str(mem), "--demote", "reference_b.md"]) == 0
    assert "demoted to _index_reference.md" in capsys.readouterr().out
    assert "reference_b.md" not in (mem / "MEMORY.md").read_text()
    assert "reference_b.md" in (mem / "_index_reference.md").read_text()


# --- SCENARIO-D: the drift check reads every index file, and tier-2 files are not memories ---


def test_a_demoted_file_is_still_judged_on_both_halves_of_its_summary(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40), _line("reference_b.md", 40)])
    mid.memory_lines(mem)
    mid.demote(mem, ["reference_b.md"])
    _write(
        mem, "reference_b.md", desc="a moved description", body="The fact.\n\nMore.\nAnd more.\n"
    )
    line = mid.memory_lines(mem)[0]
    assert "DRIFTED" in line and "reference_b.md(+2)" in line


def test_a_tier2_file_is_never_a_memory_file(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40), _line("reference_b.md", 40)])
    mid.demote(mem, ["reference_b.md"])
    current, unreadable = mid.snapshot(mem)
    assert "_index_reference.md" not in current and unreadable == []
    mid.memory_lines(mem)
    sidecar = json.loads((mem / mid.BASELINE_NAME).read_text())["files"]
    assert "_index_reference.md" not in sidecar and "reference_b.md" in sidecar
    assert mid.memory_lines(mem)[0].startswith(f"{mid.PREFIX}2 files, 0 drifted")


def test_both_caps_apply_together_and_the_line_cap_comes_first(mem: Path) -> None:
    # 220 lines of 130 units: the line cap keeps 200, then the unit cap cuts inside those 200.
    _index(mem, [_line(f"f{i:03d}.md", 130) for i in range(220)])
    kept, dropped, _ = mid.harness_cut((mem / "MEMORY.md").read_text())
    assert len(kept) == mid.HARNESS_MAX_UNITS // 131 and len(kept) + len(dropped) == 220
    invisible = mid.index_capacity(mem)["invisible"]
    assert invisible == [f"f{i:03d}.md" for i in range(len(kept), 220)]


# --- SCENARIO-E: the editor of an index file is reminded at the moment of the write ---


def _payload(mem: Path, name: str, tool: str = "Write") -> dict:
    return {"tool_name": tool, "tool_input": {"file_path": str(mem / name), "content": ""}}


def test_the_hook_reminds_an_editor_of_memory_md_when_the_cut_bites(mem: Path) -> None:
    n = len(_over_cap(mem))
    text = mid.hook_reminder(_payload(mem, "MEMORY.md"), mem)
    assert text.startswith("memory_index_drift: `MEMORY.md` -- OVER_CAP")
    assert (
        f"feedback_{n - 1:03d}.md" in text and "--demote" in text and "25,000 UTF-16 units" in text
    )


def test_the_hook_reminds_on_a_misplaced_pointer(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40), _line("reference_b.md", 40)])
    text = mid.hook_reminder(_payload(mem, "MEMORY.md", tool="Edit"), mem)
    assert "MISPLACED" in text and "--demote reference_b.md" in text


def test_the_hook_reminds_on_a_tier2_edit_that_duplicates(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40)], name="_index_feedback.md")
    assert "DUPLICATE" in mid.hook_reminder(_payload(mem, "_index_feedback.md", tool="Edit"), mem)


def test_the_hook_is_silent_on_a_clean_index(mem: Path) -> None:
    assert mid.hook_reminder(_payload(mem, "MEMORY.md"), mem) == ""


def test_the_hook_leaves_stale_counts_and_missing_targets_to_the_dashboard(mem: Path) -> None:
    _index(mem, [_line("reference_b.md", 40)], name="_index_reference.md")
    _index(
        mem,
        [
            mid._group_line("reference", 9),
            _line("feedback_a.md", 40),
            _line("feedback_gone.md", 40),
        ],
    )
    assert any("GROUP_COUNT_STALE" in ln for ln in mid.capacity_lines(mem))
    assert mid.hook_reminder(_payload(mem, "MEMORY.md"), mem) == ""


def test_the_hook_entrypoint_emits_the_index_reminder(
    mem: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _over_cap(mem)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(_payload(mem, "MEMORY.md"))))
    assert mid.main(["--hook", "--memory-dir", str(mem)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "OVER_CAP" in out["hookSpecificOutput"]["additionalContext"]


def test_the_dashboard_render_carries_the_capacity_line(
    mem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(mem))
    monkeypatch.setattr(dash, "memory_lines", lambda: mid.memory_lines(mem, write=False))
    text = dash.render()
    assert f"{mid.PREFIX}2 files, 0 drifted" in text
    assert f"{mid.PREFIX}index MEMORY.md" in text and "pointers reachable" in text


# --- SCENARIO-D, the half that matters to an editor: the reminder names the file that HOLDS the line ---


def _grow(mem: Path, name: str) -> dict:
    return {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": str(mem / name),
            "old_string": "The fact.\n",
            "new_string": "The fact.\n\nMore.\nAnd more.\n",
        },
    }


def test_the_reminder_for_a_demoted_file_names_its_tier2_file_not_memory_md(mem: Path) -> None:
    _index(mem, [_line("feedback_a.md", 40), _line("reference_b.md", 40)])
    mid.memory_lines(mem)
    mid.demote(mem, ["reference_b.md"])
    text = mid.hook_reminder(_grow(mem, "reference_b.md"), mem)
    assert "`_index_reference.md` line" in text and "`MEMORY.md`" not in text


def test_an_unindexed_tier2_group_file_is_told_where_its_line_belongs(mem: Path) -> None:
    (mem / "reference_new.md").write_text(_FILE.format(desc="new", body="The fact.\n"))
    assert mid.expected_index_file("reference_new.md") == "_index_reference.md"
    text = mid.hook_reminder(_grow(mem, "reference_new.md"), mem)
    assert "`_index_reference.md` has no line for it" in text
    assert mid.expected_index_file("feedback_a.md") == "MEMORY.md"
    text = mid.hook_reminder(_grow(mem, "feedback_a.md"), mem)
    assert "`MEMORY.md` line" in text and "`_index_" not in text
