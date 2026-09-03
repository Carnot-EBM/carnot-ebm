"""Spec: REQ-CONDUCTOR-VERDICT-3, SCENARIO-CONDUCTOR-VERDICT-3-A

A rejected artifact is reported with the reason it was actually rejected.

INCIDENT 2026-09-03. `_artifact_is_finished` has two rejection paths -- a bootstrap
`status`, and a verdict the classifier will not trust -- and `_log_experiment_completion`
reported both as `artifact_not_updated_past_bootstrap`. That message asserts the artifact
was never written. Of exp6952 it was false: the artifact was written, complete, clean, and
rejected for its `partial` verdict_class. An outer-loop session spent an hour chasing a
bootstrap-write bug that did not exist, and only found the real cause by calling
`_verdict_is_untrustworthy` by hand.

The fix splits the reason out so the token names the path taken. The bootstrap token keeps
its exact old spelling because log greps depend on it.
"""

from __future__ import annotations

import json
import sys

import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import research_conductor as rc  # noqa: E402


@pytest.fixture()
def rooted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point PROJECT_ROOT at tmp_path so artifacts never touch the real repo.

    `_artifact_unfinished_reason` resolves `deliverable` against PROJECT_ROOT. Writing a
    fixture artifact into the real tree would be a test mutating tracked state, which this
    project forbids outright; and passing an absolute path or an artifact dict instead makes
    the function return None, which reads as a pass. Both mistakes were made while
    investigating this bug -- the second one twice.
    """
    monkeypatch.setattr(rc, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _task(root: Path, payload: dict) -> dict:
    d = root / "art.json"
    d.write_text(json.dumps(payload))
    return {"title": "t", "deliverable": "art.json"}


def test_a_bootstrap_status_keeps_its_historical_token(rooted: Path) -> None:
    reason = _artifact(rooted, {"status": "running"})
    assert reason is not None
    assert reason[0] == "artifact_not_updated_past_bootstrap"


def test_a_rejected_verdict_gets_its_own_token(rooted: Path) -> None:
    """The exp6952 shape: written, complete status, partial class."""
    reason = _artifact(
        rooted, {"status": "complete", "honest_verdict": "partial_x", "verdict_class": "partial"}
    )
    assert reason is not None
    assert reason[0] == "artifact_verdict_not_terminal"
    assert reason[0] != "artifact_not_updated_past_bootstrap"


def test_the_verdict_detail_names_the_verdict_and_class(rooted: Path) -> None:
    """A reader must be able to act without opening the artifact."""
    reason = _artifact(
        rooted, {"status": "complete", "honest_verdict": "partial_x", "verdict_class": "partial"}
    )
    assert "partial_x" in reason[1]
    assert "partial" in reason[1]


def test_a_terminal_artifact_has_no_reason(rooted: Path) -> None:
    assert _artifact(rooted, {"status": "complete", "honest_verdict": "complete_x"}) is None


def test_terminal_blocked_is_still_trusted(rooted: Path) -> None:
    """REQ-CONDUCTOR-FINISHED-1 must survive the refactor."""
    assert (
        _artifact(rooted, {"status": "blocked", "honest_verdict": "complete_blocked_gpu"}) is None
    )


def test_the_bool_wrapper_still_answers_for_its_three_callers(rooted: Path) -> None:
    assert rc._artifact_is_finished({"id": "plan"}) is True
    t = _task(rooted, {"status": "running"})
    assert rc._artifact_is_finished(t) is False


def _artifact(root: Path, payload: dict):
    return rc._artifact_unfinished_reason(_task(root, payload))
