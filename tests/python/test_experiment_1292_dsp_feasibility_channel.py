"""Tests for Exp 1292 DSP feasibility-channel repair diagnostics.

Spec refs: REQ-KONA-030, SCENARIO-KONA-030.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase3.continuous_ebm import (
    FeasibilityChannelCase,
    evaluate_feasibility_channels,
)
from scripts import experiment_1292_dsp_feasibility_channel_diagnostic as experiment


def test_feasibility_channel_distinguishes_continue_from_stop() -> None:
    """REQ-KONA-030: residual violations continue; feasible states stop."""
    cases = [
        FeasibilityChannelCase(
            case_id="raw_to_repair",
            cohort="raw",
            before_violation_energy=0.25,
            before_violation_count=2,
            after_violation_energy=0.0,
            after_violation_count=0,
            distortion_delta=0.30,
        ),
        FeasibilityChannelCase(
            case_id="already_feasible",
            cohort="feasible",
            before_violation_energy=0.0,
            before_violation_count=0,
            after_violation_energy=0.0,
            after_violation_count=0,
            distortion_delta=0.05,
        ),
    ]

    report = evaluate_feasibility_channels(cases)
    by_id = {row["case_id"]: row for row in report["per_case"]}

    assert by_id["raw_to_repair"]["phi_local"] > 0.5
    assert by_id["raw_to_repair"]["Phi_global"] > 0.5
    assert by_id["raw_to_repair"]["predicted_continue"] is True
    assert by_id["raw_to_repair"]["repair_helped"] is True

    assert by_id["already_feasible"]["phi_local"] == pytest.approx(0.0)
    assert by_id["already_feasible"]["Phi_global"] == pytest.approx(0.0)
    assert by_id["already_feasible"]["predicted_continue"] is False
    assert by_id["already_feasible"]["repair_helped"] is False


def test_feasibility_channel_metrics_cover_false_continue_cases() -> None:
    """SCENARIO-KONA-030: helper reports AUROC, accuracy, and wrong distortion."""
    cases = [
        FeasibilityChannelCase(
            case_id="positive_0",
            cohort="positive",
            before_violation_energy=1.0,
            before_violation_count=1,
            after_violation_energy=0.0,
            after_violation_count=0,
            distortion_delta=0.20,
        ),
        FeasibilityChannelCase(
            case_id="positive_1",
            cohort="positive",
            before_violation_energy=1.0,
            before_violation_count=1,
            after_violation_energy=0.0,
            after_violation_count=0,
            distortion_delta=0.25,
        ),
        FeasibilityChannelCase(
            case_id="true_stop",
            cohort="zero",
            before_violation_energy=0.0,
            before_violation_count=0,
            after_violation_energy=0.0,
            after_violation_count=0,
            distortion_delta=0.10,
        ),
        FeasibilityChannelCase(
            case_id="false_continue",
            cohort="misleading",
            before_violation_energy=1.0,
            before_violation_count=1,
            after_violation_energy=1.0,
            after_violation_count=1,
            distortion_delta=0.55,
        ),
    ]

    report = evaluate_feasibility_channels(cases)

    assert report["feasibility_channel_auc"] == pytest.approx(0.75)
    assert report["repair_help_prediction_accuracy"] == pytest.approx(0.75)
    assert report["false_continue_rate"] == pytest.approx(0.5)
    assert report["false_stop_rate"] == pytest.approx(0.0)
    assert report["distortion_when_wrong"] == pytest.approx(0.55)
    assert report["n_cases"] == 4


def test_feasibility_channel_rejects_malformed_cases() -> None:
    """REQ-KONA-030: impossible diagnostic inputs fail before metric reporting."""
    with pytest.raises(ValueError, match="at least one"):
        evaluate_feasibility_channels([])

    with pytest.raises(ValueError, match="violation"):
        evaluate_feasibility_channels(
            [
                FeasibilityChannelCase(
                    case_id="bad",
                    cohort="bad",
                    before_violation_energy=-1.0,
                    before_violation_count=0,
                    after_violation_energy=0.0,
                    after_violation_count=0,
                    distortion_delta=0.0,
                )
            ]
        )


def test_build_artifact_contains_required_exp1292_fields() -> None:
    """SCENARIO-KONA-030: artifact replays linear and nonlinear repair cases."""
    artifact = experiment.build_artifact()

    assert artifact["schema"] == "carnot.phase3.dsp_feasibility_channel.v1"
    assert artifact["experiment"] == "1292_dsp_feasibility_channel_diagnostic"
    assert artifact["run_date"] == "20260504"
    assert artifact["status"] == "complete"
    assert artifact["spec_refs"] == ["REQ-KONA-030", "SCENARIO-KONA-030"]
    assert artifact["source_context"]["experiment_1275_loaded"] is True
    assert artifact["source_context"]["experiment_1276_loaded"] is True
    assert "experiment_1291_loaded" in artifact["source_context"]
    assert artifact["n_cases"] == len(artifact["per_case"])
    assert artifact["feasibility_channel_auc"] >= 0.0
    assert artifact["repair_help_prediction_accuracy"] >= 0.0
    assert artifact["false_continue_rate"] >= 0.0
    assert artifact["false_stop_rate"] >= 0.0
    assert artifact["distortion_when_wrong"] >= 0.0
    assert isinstance(artifact["feasibility_channel_predictive"], bool)
    assert artifact["recommended_repair_stop_policy"]
    assert artifact["honest_verdict"] in {
        "feasibility_channel_predictive",
        "feasibility_channel_predictive_marginal",
        "feasibility_channel_not_predictive",
        "blocked_missing_required_repair_artifacts",
    }
    first_case = artifact["per_case"][0]
    assert "phi_local" in first_case
    assert "Phi_global" in first_case
    json.dumps(artifact)


def test_script_main_writes_terminal_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-KONA-030: experiment script writes the terminal JSON artifact."""
    output_path = tmp_path / "experiment_1292_dsp_feasibility_channel_diagnostic.json"
    monkeypatch.setattr(experiment, "RESULT_PATH", output_path)

    artifact = experiment.main()

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == artifact
    assert written["status"] == "complete"
    assert "honest_verdict" in written
