"""Tests for the exp4450 ARC inference-substrate lint guard artifact.

Spec refs: REQ-VERIFY-4450, SCENARIO-VERIFY-4450.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4450_inference_substrate_emission_lint_guard as exp4450


def test_req_verify_4450_builds_required_artifact_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4450: the artifact records the shipped guard and test result."""

    artifact = exp4450.build_artifact(duration_s=0.001)
    output_path = tmp_path / "experiment_4450_inference_substrate_emission_lint_guard.json"
    written = exp4450.write_artifact(output_path=output_path, duration_s=0.001)
    reloaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert artifact["guard_shipped"] is True
    assert artifact["catches_exp4433_class"] is True
    assert artifact["tests_pass"] is True
    assert artifact["leaderboard_submission"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["field_principles"]["honest_verdict"] == "terminal-prefixed"
    assert artifact["field_principles"]["guard_shipped"].startswith("bare bool")
    assert artifact["field_principles"]["catches_exp4433_class"].startswith("bare bool")
    assert artifact["field_principles"]["tests_pass"].startswith("bare bool")
    assert artifact["field_principles"]["inference_substrate"].startswith(
        "aggregation_from_upstream_artifacts"
    )
    assert written == artifact
    assert reloaded == artifact


def test_scenario_verify_4450_defensive_paths_assert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-VERIFY-4450: defensive branches return explicit outcomes."""

    bad_payload_path = tmp_path / "bad_exp4433.json"
    bad_payload_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(exp4450, "EXP4433_PATH", bad_payload_path)
    with pytest.raises(ValueError, match="JSON object"):
        exp4450._load_exp4433_payload()

    missing_hook_config = tmp_path / "missing_hook.yaml"
    missing_hook_config.write_text("repos: []\n", encoding="utf-8")
    monkeypatch.setattr(exp4450, "PRE_COMMIT_PATH", missing_hook_config)
    assert exp4450.precommit_guard_configured() is False

    hook_without_files = tmp_path / "hook_without_files.yaml"
    hook_without_files.write_text(
        "repos:\n  - repo: local\n    hooks:\n      - id: arc-artifact-lint\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exp4450, "PRE_COMMIT_PATH", hook_without_files)
    assert exp4450.precommit_guard_configured() is False

    artifact = {
        "guard_shipped": True,
        "catches_exp4433_class": True,
        "honest_verdict": "shipped: stub",
    }
    monkeypatch.setattr(exp4450, "write_artifact", lambda: artifact)

    assert exp4450.main() == 0
    assert json.loads(capsys.readouterr().out) == artifact
