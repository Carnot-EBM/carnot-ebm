"""An honest terminal-blocked artifact counts as finished; skeletons still re-run.

Spec: REQ-CONDUCTOR-FINISHED-1, SCENARIO-FINISHED-1-A through SCENARIO-FINISHED-1-F.

Background (2026-09-02, exp6901): `_artifact_is_finished` checked
`status in _BOOTSTRAP_STATUSES` before it read the verdict. The set contains
"blocked", so an artifact with `status: "blocked"` and the honest terminal
verdict `complete_blocked_*` read as an unfinished bootstrap skeleton. The
conductor re-ran the task, the rerun reproduced the same blocked artifact,
and after three FAIL rows the task retired and cascade-blocked three
dependents. Seven tasks repeated this pattern in two weeks (exp6753, 6765,
6773, 6784, 6796, 6837, 6901 — each burned exactly 3 FAILs).

The fix: for status == "blocked" ONLY, consult `_verdict_is_untrustworthy`
first. A trustworthy, non-empty verdict marks the artifact finished. Every
other bootstrap status keeps the strict behavior, so a mid-run skeleton
(status "running") is still re-run even if a verdict was pre-written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import research_conductor  # noqa: E402


def _write_artifact(tmp_path: Path, payload: dict) -> dict:
    """Write a deliverable under tmp_path and return the matching task dict."""
    deliverable = tmp_path / "results" / "exp.json"
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    deliverable.write_text(json.dumps(payload))
    return {"id": "test-exp", "title": "Test task", "deliverable": "results/exp.json"}


def test_terminal_prefixed_blocked_verdict_is_finished(tmp_path, monkeypatch):
    """SCENARIO-FINISHED-1-A: the exp6901 shape — status blocked, terminal verdict.

    `complete_blocked_*` is a concluded run whose finding is "blocked".
    Re-running it reproduces the same artifact; it must count as finished.
    """
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    task = _write_artifact(
        tmp_path,
        {
            "status": "blocked",
            "honest_verdict": "complete_blocked_independent_model_relation_qualification",
        },
    )
    assert research_conductor._artifact_is_finished(task) is True


def test_bare_blocked_resource_verdict_is_finished(tmp_path, monkeypatch):
    """SCENARIO-FINISHED-1-B: bare `blocked_<resource>` is honest-terminal.

    Per the Pre-Launch Preconditions Discipline the task "simply retires";
    the verdict classifier already trusts this shape (the .294 incident:
    "no retry can change that"). The status field must not override it.
    """
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    task = _write_artifact(
        tmp_path,
        {"status": "blocked", "honest_verdict": "blocked_model_not_cached_gemma_4_26B"},
    )
    assert research_conductor._artifact_is_finished(task) is True


def test_blocked_skeleton_without_verdict_still_reruns(tmp_path, monkeypatch):
    """SCENARIO-FINISHED-1-C: status blocked with NO verdict is a skeleton.

    A bootstrap skeleton pre-writes status but no verdict. It must still
    read as unfinished, or a bailed agent's stub would be accepted as done.
    """
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    task = _write_artifact(tmp_path, {"status": "blocked"})
    assert research_conductor._artifact_is_finished(task) is False


def test_running_status_with_terminal_verdict_still_reruns(tmp_path, monkeypatch):
    """SCENARIO-FINISHED-1-D: the override is scoped to status "blocked" only.

    status "running" plus a pre-written terminal verdict is the exp1028
    bail-without-updating shape. Widening the override here would re-open
    the bootstrap-poisoning hole the status check exists to close.
    """
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    task = _write_artifact(
        tmp_path,
        {"status": "running", "honest_verdict": "complete_all_gates_passed"},
    )
    assert research_conductor._artifact_is_finished(task) is False


def test_blocked_with_partial_verdict_still_reruns(tmp_path, monkeypatch):
    """SCENARIO-FINISHED-1-E: an untrustworthy verdict never overrides status.

    A declared `verdict_class: partial` is the one class that may retry;
    status blocked plus a partial verdict keeps the strict re-run path.
    """
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    task = _write_artifact(
        tmp_path,
        {
            "status": "blocked",
            "honest_verdict": "partial_tests_still_failing",
            "verdict_class": "partial",
        },
    )
    assert research_conductor._artifact_is_finished(task) is False


def test_terminal_success_status_unchanged(tmp_path, monkeypatch):
    """SCENARIO-FINISHED-1-F: normal terminal artifacts are untouched."""
    monkeypatch.setattr(research_conductor, "PROJECT_ROOT", tmp_path)
    task = _write_artifact(
        tmp_path,
        {"status": "success", "honest_verdict": "complete_all_gates_passed"},
    )
    assert research_conductor._artifact_is_finished(task) is True
