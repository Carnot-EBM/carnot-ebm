"""Tests for Exp 4462 count-integrity guard artifact.

Spec refs: REQ-REPORT-4462, SCENARIO-REPORT-4462, SCENARIO-REPORT-4462-SUBMISSION.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4462_provisional_reproduced_count_integrity_lint as exp4462


def test_req_report_4462_builds_required_artifact_fields(tmp_path: Path) -> None:
    """REQ-REPORT-4462: artifact records the shipped guard and test signal."""

    assert exp4462.precommit_guard_configured() is True

    artifact = exp4462.build_artifact(
        duration_s=0.001,
        guard_shipped=True,
        catches_provisional_inflation=True,
        registry_lint_issue_count=0,
        submission_lint_issue_count=0,
        precommit_hook_configured=True,
    )
    output_path = tmp_path / "experiment_4462_provisional_reproduced_count_integrity_lint.json"
    written = exp4462.write_artifact(output_path=output_path, artifact=artifact)
    reloaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert artifact["guard_shipped"] is True
    assert artifact["catches_provisional_inflation"] is True
    assert artifact["tests_pass"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["retroactively_rewrote_past_artifacts"] is False
    assert artifact["inference_substrate"] == exp4462.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.001
    assert artifact["field_principles"]["honest_verdict"]["principle"] == "terminal-prefixed"
    assert artifact["field_principles"]["guard_shipped"]["principle"].startswith("bare bool")
    assert artifact["field_principles"]["catches_provisional_inflation"]["principle"].startswith(
        "bare bool"
    )
    assert artifact["field_principles"]["tests_pass"]["principle"].startswith("bare bool")
    assert artifact["field_principles"]["inference_substrate"]["principle"].startswith(
        "aggregation_from_upstream_artifacts"
    )
    assert "REQ-REPORT-4462" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert written == artifact
    assert reloaded == artifact


def test_req_report_4462_run_guard_orchestrates_lints(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-4462: run_guard combines lint, fixture, and hook outcomes."""

    ticks = iter([1.0, 1.25])
    written: dict[str, object] = {}

    monkeypatch.setattr(exp4462.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(exp4462.arc_count_integrity_lint, "lint_registry_path", lambda *a, **k: [])
    monkeypatch.setattr(
        exp4462.arc_count_integrity_lint,
        "lint_submission_package_path",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(exp4462, "catches_provisional_inflation", lambda: True)
    monkeypatch.setattr(exp4462, "precommit_guard_configured", lambda: True)
    monkeypatch.setattr(exp4462, "write_artifact", lambda *, artifact: written.update(artifact) or artifact)

    artifact = exp4462.run_guard()

    assert artifact["guard_shipped"] is True
    assert artifact["duration_s"] == 0.25
    assert artifact["registry_lint_issues"] == []
    assert artifact["submission_lint_issues"] == []
    assert written["guard_shipped"] is True


def test_scenario_report_4462_artifact_defensive_paths_assert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-4462: defensive branches return explicit outcomes."""

    assert exp4462._duration(1.0, 1.5) == 0.5
    assert exp4462.catches_provisional_inflation() is True

    missing_path = tmp_path / "absent.yaml"
    monkeypatch.setattr(exp4462, "PRE_COMMIT_PATH", missing_path)
    assert exp4462.precommit_guard_configured() is False

    missing_hook = tmp_path / "missing_pre_commit.yaml"
    missing_hook.write_text("repos: []\n", encoding="utf-8")
    monkeypatch.setattr(exp4462, "PRE_COMMIT_PATH", missing_hook)

    assert exp4462.precommit_guard_configured() is False

    missing_files = tmp_path / "missing_files_pre_commit.yaml"
    missing_files.write_text(
        "repos:\n  - repo: local\n    hooks:\n      - id: arc-count-integrity-lint\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exp4462, "PRE_COMMIT_PATH", missing_files)
    assert exp4462.precommit_guard_configured() is False

    artifact = exp4462.build_artifact(
        duration_s=0.00001,
        guard_shipped=False,
        catches_provisional_inflation=False,
        registry_lint_issue_count=1,
        submission_lint_issue_count=2,
        precommit_hook_configured=False,
    )

    assert artifact["duration_s"] == 0.001
    assert artifact["guard_shipped"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["registry_lint_issue_count"] == 1
    assert artifact["submission_lint_issue_count"] == 2

    monkeypatch.setattr(exp4462, "run_guard", lambda: artifact)

    assert exp4462.main() == 1
    assert json.loads(capsys.readouterr().out) == artifact
