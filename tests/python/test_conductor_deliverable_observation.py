"""A deliverable's status CHANGE must leave a durable trace.

Measured 2026-09-09: two conductor guards deliberately ignore a `blocked`
artifact because the agent may still supersede it. Whether that ever happens
was unmeasurable - 0 same-run supersedes observed against 55 non-superseded
blocked end states, but a floor rather than a rate, because the deliverable
watch parsed the artifact and threw the observation away (a bare `pass`).

Spec: REQ-CONDUCTOR-OBSERVE-1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import research_conductor as rc  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_one_row_is_appended_with_the_fields_that_decide_the_question(
    tmp_path, monkeypatch
) -> None:
    out = tmp_path / "obs.jsonl"
    monkeypatch.setattr(rc, "DELIVERABLE_OBSERVATIONS", out)
    rc._record_deliverable_observation(
        "results/experiment_7157_v630_qwen38_runtime.json",
        "blocked",
        "blocked_idle_rtx_3090",
        1234.56,
    )
    (row,) = _rows(out)
    assert row["deliverable"].endswith("qwen38_runtime.json")
    assert row["status"] == "blocked"
    assert row["honest_verdict"] == "blocked_idle_rtx_3090"
    assert row["elapsed_s"] == 1234.6, "elapsed is what makes a supersede measurable"
    assert row["at"].endswith("Z")


def test_a_supersede_appends_a_second_row_rather_than_replacing(tmp_path, monkeypatch) -> None:
    """blocked then terminal in one run is the event the whole record exists for."""
    out = tmp_path / "obs.jsonl"
    monkeypatch.setattr(rc, "DELIVERABLE_OBSERVATIONS", out)
    rc._record_deliverable_observation("results/x.json", "blocked", "blocked_no_gpu", 10.0)
    rc._record_deliverable_observation("results/x.json", "success", "complete_x", 900.0)
    rows = _rows(out)
    assert [r["status"] for r in rows] == ["blocked", "success"]
    assert rows[1]["elapsed_s"] > rows[0]["elapsed_s"]


def test_non_string_status_is_recorded_as_null_not_coerced(tmp_path, monkeypatch) -> None:
    """A field name can lie; a dict-wrapped status must not become the string 'dict'."""
    out = tmp_path / "obs.jsonl"
    monkeypatch.setattr(rc, "DELIVERABLE_OBSERVATIONS", out)
    rc._record_deliverable_observation("results/x.json", {"value": "blocked"}, None, 1.0)
    (row,) = _rows(out)
    assert row["status"] is None
    assert row["honest_verdict"] is None


def test_recording_never_raises_when_the_path_is_unusable(tmp_path, monkeypatch) -> None:
    """This runs in the polling loop; losing an observation must not lose the run."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, so mkdir under it fails")
    monkeypatch.setattr(rc, "DELIVERABLE_OBSERVATIONS", blocker / "sub" / "obs.jsonl")
    rc._record_deliverable_observation("results/x.json", "blocked", "b", 1.0)


def test_the_watch_records_on_change_and_outside_the_bootstrap_branch() -> None:
    """The supersede is the transition OUT of a bootstrap status.

    Recording inside `if bootstrap_only:` would miss exactly the event this
    exists to capture, which is the bug this test was written against.
    """
    src = Path(rc.__file__).read_text()
    call = "_record_deliverable_observation(\n"
    assert call in src, "the watch no longer records - update this test"
    guard = "if _seen_status != deliverable_last_status:"
    assert guard in src, "recording must be gated on a CHANGE, not fire every poll"
    before_guard = src.index(guard)
    bootstrap_branch = src.index("if bootstrap_only:", before_guard - 4000)
    assert before_guard < bootstrap_branch, (
        "the recording call must precede the bootstrap_only branch, or a supersede "
        "(which leaves that branch) is never recorded"
    )
