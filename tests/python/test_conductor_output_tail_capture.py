"""A killed subagent must leave its whole output on disk, not 19 characters.

Measured 2026-09-09: the conductor extracts a 300-character failure tail and
`log_step` then writes `details[:80]`, so of 386 kill rows 45 kept no tail at
all and the rest kept 10-19 characters. The child's output lived only in
memory, so a kill destroyed its own evidence and post-mortems were guesswork.

Spec: REQ-CONDUCTOR-TAIL-1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import research_conductor as rc  # noqa: E402


def _redirect(tmp_path, monkeypatch):
    """Point the tail directory at tmp_path so no test writes the real ops dir."""
    monkeypatch.setattr(rc, "TASK_OUTPUT_TAILS", tmp_path / "tails")


def test_whole_output_is_written_not_a_truncated_tail(tmp_path, monkeypatch) -> None:
    _redirect(tmp_path, monkeypatch)
    body = "phase 1\n" + ("x" * 5000) + "\nphase 9 done"
    name = rc._persist_output_tail(body, "hard-cap", "results/experiment_7157_foo.json")
    assert name, "a successful capture must return the file name"
    written = (tmp_path / "tails" / name).read_text()
    assert written == body, "the file must hold the WHOLE output, not a slice"
    assert len(written) > 80, "capturing <=80 chars would reproduce the defect"


def test_name_carries_the_reason_and_the_deliverable(tmp_path, monkeypatch) -> None:
    _redirect(tmp_path, monkeypatch)
    name = rc._persist_output_tail("out", "wall-clock-idle", "results/experiment_7157_foo.json")
    assert "wall-clock-idle" in name
    assert "experiment_7157_foo" in name
    assert name.endswith(".txt")
    # The timestamp prefix is what makes a log row's minute globbable.
    assert name[:8].isdigit() and name[8] == "T"


def test_missing_deliverable_still_captures(tmp_path, monkeypatch) -> None:
    _redirect(tmp_path, monkeypatch)
    name = rc._persist_output_tail("out", "stall", None)
    assert "no-deliverable" in name
    assert (tmp_path / "tails" / name).read_text() == "out"


def test_empty_output_is_recorded_as_such(tmp_path, monkeypatch) -> None:
    """An agent that produced nothing is a FINDING, so it must not write an empty file."""
    _redirect(tmp_path, monkeypatch)
    name = rc._persist_output_tail("", "stall", None)
    assert (tmp_path / "tails" / name).read_text() == "(no output captured)"


def test_capture_never_raises_when_the_directory_cannot_be_made(tmp_path, monkeypatch) -> None:
    """Losing a diagnostic must not also lose the kill."""
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, so mkdir under me fails")
    monkeypatch.setattr(rc, "TASK_OUTPUT_TAILS", blocker / "tails")
    assert rc._persist_output_tail("out", "hard-cap", None) == ""


def test_pruning_keeps_the_newest_and_drops_the_rest(tmp_path, monkeypatch) -> None:
    _redirect(tmp_path, monkeypatch)
    monkeypatch.setattr(rc, "TASK_OUTPUT_TAIL_KEEP", 3)
    d = tmp_path / "tails"
    d.mkdir()
    for i in range(6):
        (d / f"2026090{i}T000000Z-stall-x.txt").write_text(str(i))
    rc._prune_output_tails()
    left = sorted(p.name for p in d.glob("*.txt"))
    assert len(left) == 3, f"expected 3 kept, got {left}"
    assert left == [
        "20260903T000000Z-stall-x.txt",
        "20260904T000000Z-stall-x.txt",
        "20260905T000000Z-stall-x.txt",
    ], "pruning must keep the NEWEST by timestamp-prefixed name"


def test_every_kill_path_captures(tmp_path, monkeypatch) -> None:
    """All three kill messages must be paired with a capture call.

    Deleting any one call site must fail this test, or the guard is decorative.
    """
    src = (Path(rc.__file__)).read_text()
    for message, reason in (
        ("Stalled after", "stall"),
        ("Hard wall-clock cap after", "hard-cap"),
        ("Wall-clock+idle timeout after", "wall-clock-idle"),
    ):
        assert message in src, f"kill message {message!r} vanished - update this test"
        assert f'_persist_output_tail(full_output, "{reason}"' in src, (
            f"the {reason} kill path does not persist its output"
        )
