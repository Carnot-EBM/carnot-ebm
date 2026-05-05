"""Tests for Exp 1305 HardNet++/DSP feasibility stop policy.

Spec refs: REQ-KONA-031, SCENARIO-KONA-031.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase3.feasibility_stop_policy import evaluate_stop_policy
from scripts import experiment_1305_hardnetpp_dsp_feasibility_stop_policy as experiment


def test_stop_policy_continues_only_useful_high_signal_repairs() -> None:
    """REQ-KONA-031: continue only when violations and non-residual signal remain."""
    report = evaluate_stop_policy(
        [
            {
                "case_id": "hard_feasible",
                "cohort": "already_done",
                "before_violation_energy": 0.0,
                "before_violation_count": 0,
                "after_violation_energy": 0.0,
                "channel_score": 0.95,
                "repair_helped": False,
                "predicted_continue": True,
            },
            {
                "case_id": "helpful_hardnet",
                "cohort": "exp1291_raw_to_hardnetpp",
                "before_violation_energy": 0.25,
                "before_violation_count": 1,
                "after_violation_energy": 0.0,
                "channel_score": 0.85,
                "repair_helped": True,
                "predicted_continue": True,
            },
            {
                "case_id": "below_signal",
                "cohort": "weak_signal",
                "before_violation_energy": 0.20,
                "before_violation_count": 1,
                "after_violation_energy": 0.10,
                "channel_score": 0.20,
                "repair_helped": True,
                "predicted_continue": False,
            },
            {
                "case_id": "local_linear_plateau",
                "cohort": "exp1291_raw_to_fsnet_local_linear",
                "before_violation_energy": 0.20,
                "before_violation_count": 1,
                "after_violation_energy": 0.20,
                "channel_score": 0.85,
                "repair_helped": False,
                "predicted_continue": True,
            },
        ]
    )
    by_id = {row["case_id"]: row for row in report["per_case"]}

    assert by_id["hard_feasible"]["stop_reason"] == "hard_feasible"
    assert by_id["helpful_hardnet"]["conservative_continue"] is True
    assert by_id["below_signal"]["stop_reason"] == "below_feasibility_threshold"
    assert by_id["local_linear_plateau"]["stop_reason"] == (
        "residual_nonlinear_local_linear"
    )
    assert report["conservative_continue_recommendations"] == 1
    assert report["true_continue_recommendations"] == 1
    assert report["false_continue_recommendations"] == 0
    assert report["stop_policy_precision"] == pytest.approx(1.0)
    assert report["policy_false_stop_recommendations"] == 1


def test_stop_policy_rejects_malformed_replay_rows() -> None:
    """REQ-KONA-031: malformed deterministic replay rows fail before reporting."""
    valid_row = {
        "case_id": "ok",
        "cohort": "linear",
        "before_violation_energy": 0.1,
        "before_violation_count": 1,
        "after_violation_energy": 0.0,
        "channel_score": 0.8,
        "repair_helped": True,
        "predicted_continue": True,
    }

    with pytest.raises(ValueError, match="at least one"):
        evaluate_stop_policy([])
    with pytest.raises(ValueError, match="threshold"):
        evaluate_stop_policy([valid_row], threshold=1.5)
    with pytest.raises(ValueError, match="help_energy_tolerance"):
        evaluate_stop_policy([valid_row], help_energy_tolerance=-0.1)
    with pytest.raises(ValueError, match="missing"):
        evaluate_stop_policy([{"case_id": "missing"}])
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_stop_policy([{**valid_row, "before_violation_count": -1}])
    with pytest.raises(ValueError, match="finite"):
        evaluate_stop_policy([{**valid_row, "channel_score": float("nan")}])


def test_build_artifact_replays_exp1291_and_exp1292_metrics() -> None:
    """SCENARIO-KONA-031: artifact reports the conservative deterministic replay."""
    artifact = experiment.build_artifact()

    assert artifact["schema"] == (
        "carnot.phase3.hardnetpp_dsp_feasibility_stop_policy.v1"
    )
    assert artifact["experiment"] == "1305_hardnetpp_dsp_feasibility_stop_policy"
    assert artifact["run_date"] == "20260505"
    assert artifact["status"] == "complete"
    assert artifact["spec_refs"] == ["REQ-KONA-031", "SCENARIO-KONA-031"]
    assert artifact["feasibility_stop_policy_written"] is True
    assert artifact["hardnetpp_delta_over_snarenet"] == pytest.approx(
        1.2207222442957435
    )
    assert artifact["feasibility_channel_auc"] == pytest.approx(0.6604651162790698)
    assert artifact["stop_policy_precision"] == pytest.approx(1.0)
    assert artifact["benchmark_replay"]["candidate_transitions"] == 156
    assert artifact["benchmark_replay"]["conservative_continue_recommendations"] == 86
    assert artifact["benchmark_replay"]["false_continue_recommendations"] == 0
    assert sum(row["count"] for row in artifact["residual_nonlinear_cases"]) == 54
    assert "arXiv 2602.06737" in artifact["kan_pwa_abstraction_note"]
    assert artifact["honest_verdict"].startswith("complete: conservative replay")
    assert experiment._complete_verdict(0.50, 0.55).startswith(
        "complete: stop policy artifact written"
    )
    json.dumps(artifact)


def test_build_artifact_blocks_without_required_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-KONA-031: missing source artifacts produce a blocked artifact."""
    monkeypatch.setattr(experiment, "EXP1291_PATH", tmp_path / "missing_1291.json")
    monkeypatch.setattr(experiment, "EXP1292_PATH", tmp_path / "missing_1292.json")

    artifact = experiment.build_artifact()

    assert artifact["status"] == "blocked"
    assert artifact["feasibility_stop_policy_written"] is False
    assert artifact["stop_policy_precision"] == pytest.approx(0.0)
    assert artifact["honest_verdict"] == "blocked_missing_required_replay_artifacts"


def test_script_main_writes_terminal_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-KONA-031: experiment script writes the terminal JSON artifact."""
    output_path = tmp_path / "experiment_1305_hardnetpp_dsp_stop_policy.json"
    monkeypatch.setattr(experiment, "RESULT_PATH", output_path)

    artifact = experiment.main()

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == artifact
    assert written["status"] == "complete"
    assert written["feasibility_stop_policy_written"] is True
