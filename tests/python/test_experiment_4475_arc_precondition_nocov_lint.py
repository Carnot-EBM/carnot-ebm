"""Tests for Exp 4475 ARC precondition no-cov lint artifact.

Spec refs: REQ-REPORT-4475, SCENARIO-REPORT-4475-SMOKE,
SCENARIO-REPORT-4475-NOCOV-LINT, SCENARIO-REPORT-4475-SC25-COUNT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4475_arc_precondition_nocov_lint as exp4475


def test_req_report_4475_builds_required_artifact_fields(tmp_path: Path) -> None:
    """REQ-REPORT-4475: artifact records the helper, lint, count guard, and tests."""

    artifact = exp4475.build_artifact(
        duration_s=0.001,
        smoke_helper_shipped=True,
        nocov_lint_shipped=True,
        catches_cov_gated_precondition=True,
        count_integrity_extended=True,
        precommit_hook_configured=True,
        nocov_lint_issue_count=0,
        count_integrity_issue_count=0,
    )
    output_path = tmp_path / "experiment_4475_arc_precondition_nocov_lint.json"
    written = exp4475.write_artifact(output_path=output_path, artifact=artifact)
    reloaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert artifact["smoke_helper_shipped"] is True
    assert artifact["nocov_lint_shipped"] is True
    assert artifact["catches_cov_gated_precondition"] is True
    assert artifact["count_integrity_extended"] is True
    assert artifact["tests_pass"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["retroactively_rewrote_past_artifacts"] is False
    assert artifact["production_verifier_edits"] is False
    assert artifact["inference_substrate"] == exp4475.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.001
    assert artifact["field_principles"]["honest_verdict"]["principle"] == "terminal-prefixed"
    for field in (
        "smoke_helper_shipped",
        "nocov_lint_shipped",
        "catches_cov_gated_precondition",
        "count_integrity_extended",
        "tests_pass",
    ):
        assert artifact["field_principles"][field]["principle"].startswith("bare bool")
    assert artifact["field_principles"]["inference_substrate"]["principle"].startswith(
        "aggregation_from_upstream_artifacts"
    )
    assert "REQ-REPORT-4475" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert written == artifact
    assert reloaded == artifact


def test_scenario_report_4475_run_guard_orchestrates_lints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-4475-NOCOV-LINT: run_guard combines all shipped signals."""

    ticks = iter([1.0, 1.25])
    written: dict[str, object] = {}

    monkeypatch.setattr(exp4475.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(exp4475, "smoke_helper_shipped", lambda: True)
    monkeypatch.setattr(exp4475, "lint_current_scripts", lambda: [])
    monkeypatch.setattr(exp4475, "catches_cov_gated_precondition", lambda: True)
    monkeypatch.setattr(exp4475, "count_integrity_extended", lambda: True)
    monkeypatch.setattr(exp4475, "current_count_integrity_issues", lambda: [])
    monkeypatch.setattr(exp4475, "precommit_guard_configured", lambda: True)
    monkeypatch.setattr(exp4475, "write_artifact", lambda *, artifact: written.update(artifact) or artifact)

    artifact = exp4475.run_guard()

    assert artifact["honest_verdict"].startswith("shipped:")
    assert artifact["duration_s"] == 0.25
    assert artifact["nocov_lint_issue_count"] == 0
    assert artifact["count_integrity_issue_count"] == 0
    assert written["smoke_helper_shipped"] is True


def test_req_report_4475_main_returns_nonzero_when_guard_not_shipped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4475: CLI exit status mirrors required shipped booleans."""

    artifact = exp4475.build_artifact(
        duration_s=0.001,
        smoke_helper_shipped=True,
        nocov_lint_shipped=False,
        catches_cov_gated_precondition=True,
        count_integrity_extended=True,
        precommit_hook_configured=False,
        nocov_lint_issue_count=1,
        count_integrity_issue_count=0,
    )
    monkeypatch.setattr(exp4475, "run_guard", lambda: artifact)

    assert exp4475.main() == 1
    assert json.loads(capsys.readouterr().out) == artifact
