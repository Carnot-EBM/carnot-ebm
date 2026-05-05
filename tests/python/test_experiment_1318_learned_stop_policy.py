"""Tests for Exp 1318 learned HardNet++/DSP stop policy.

Spec refs: REQ-KONA-032, SCENARIO-KONA-032.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase3.learned_stop_policy import (
    build_stop_policy_examples,
    conservative_replay_continue_predictions,
    evaluate_learned_stop_policy,
    fit_transparent_stop_policy,
    split_stop_policy_examples,
)
from scripts import experiment_1318_hardnetpp_dsp_learned_stop_policy as experiment


def _load_replay_rows() -> list[dict]:
    payload = json.loads(
        Path("results/experiment_1292_dsp_feasibility_channel_diagnostic.json").read_text()
    )
    return list(payload["per_case"])


def test_split_uses_validator_labels_and_is_deterministic() -> None:
    """REQ-KONA-032: held-out labels come from replay validator outcomes."""
    examples = build_stop_policy_examples(_load_replay_rows())

    split_a = split_stop_policy_examples(examples, holdout_modulus=5)
    split_b = split_stop_policy_examples(examples, holdout_modulus=5)

    assert [example.case_id for example in split_a.held_out] == [
        example.case_id for example in split_b.held_out
    ]
    assert len(split_a.train) == 120
    assert len(split_a.held_out) == 36
    assert all(example.label_source == "repair_helped" for example in examples)
    assert {example.should_continue for example in split_a.train} == {False, True}
    assert {example.should_continue for example in split_a.held_out} == {False, True}
    assert all(example.split_bucket == 0 for example in split_a.held_out)
    assert all(example.split_bucket != 0 for example in split_a.train)


def test_learned_policy_reports_held_out_metrics_against_replay_policy() -> None:
    """SCENARIO-KONA-032: learned policy is evaluated on held-out replay rows."""
    examples = build_stop_policy_examples(_load_replay_rows())
    split = split_stop_policy_examples(examples, holdout_modulus=5)
    policy = fit_transparent_stop_policy(split.train)
    baseline_predictions = conservative_replay_continue_predictions(split.held_out)

    report = evaluate_learned_stop_policy(
        policy,
        split.held_out,
        baseline_continue_predictions=baseline_predictions,
    )

    assert policy.label_source == "repair_helped"
    assert policy.family_continue_rates["hardnetpp"] == pytest.approx(1.0)
    assert policy.family_continue_rates["local_linear"] == pytest.approx(0.0)
    assert policy.predict(
        build_stop_policy_examples(
            [
                {
                    "case_id": "synthetic_seed99_unknown",
                    "cohort": "new_repair_family",
                    "before_violation_energy": 0.2,
                    "before_violation_count": 1,
                    "channel_score": 1.0,
                    "repair_helped": True,
                }
            ]
        )[0]
    )

    assert report["n_held_out"] == 36
    assert report["stop_policy_precision"] == pytest.approx(1.0)
    assert report["stop_policy_recall"] == pytest.approx(1.0)
    assert report["dsp_feasibility_auc"] == pytest.approx(0.640625)
    assert report["hardnetpp_delta_over_replay_policy"] == pytest.approx(0.0)
    assert report["replay_policy"]["stop_precision"] == pytest.approx(1.0)
    assert report["replay_policy"]["stop_recall"] == pytest.approx(1.0)


def test_learned_stop_policy_rejects_unbacked_or_malformed_rows() -> None:
    """REQ-KONA-032: rows without verifier-backed labels are rejected."""
    valid_row = {
        "case_id": "synthetic_seed1_unknown",
        "cohort": "unknown",
        "before_violation_energy": 0.1,
        "before_violation_count": 1,
        "channel_score": 0.8,
        "repair_helped": True,
    }

    with pytest.raises(ValueError, match="at least one"):
        build_stop_policy_examples([])
    with pytest.raises(ValueError, match="missing"):
        build_stop_policy_examples([{"case_id": "missing"}])
    with pytest.raises(ValueError, match="repair_helped"):
        build_stop_policy_examples([{**valid_row, "repair_helped": "yes"}])
    with pytest.raises(ValueError, match="finite"):
        build_stop_policy_examples([{**valid_row, "channel_score": float("nan")}])
    with pytest.raises(ValueError, match="non-negative"):
        build_stop_policy_examples([{**valid_row, "before_violation_energy": -1.0}])

    one_class_examples = build_stop_policy_examples([valid_row])
    assert (
        evaluate_learned_stop_policy(
            fit_transparent_stop_policy(one_class_examples),
            one_class_examples,
        )["dsp_feasibility_auc"]
        == pytest.approx(0.5)
    )

    with pytest.raises(ValueError, match="holdout_modulus"):
        split_stop_policy_examples(one_class_examples, holdout_modulus=1)
    with pytest.raises(ValueError, match="both training and held-out"):
        split_stop_policy_examples(one_class_examples, holdout_modulus=5)
    with pytest.raises(ValueError, match="training"):
        fit_transparent_stop_policy([])
    with pytest.raises(ValueError, match="held-out"):
        evaluate_learned_stop_policy(
            fit_transparent_stop_policy(one_class_examples),
            [],
        )
    with pytest.raises(ValueError, match="baseline predictions"):
        evaluate_learned_stop_policy(
            fit_transparent_stop_policy(one_class_examples),
            one_class_examples,
            baseline_continue_predictions=[],
        )


def test_build_artifact_contains_required_exp1318_fields() -> None:
    """SCENARIO-KONA-032: artifact reports held-out learned-policy metrics."""
    artifact = experiment.build_artifact()

    assert artifact["schema"] == "carnot.phase3.hardnetpp_dsp_learned_stop_policy.v1"
    assert artifact["experiment"] == "1318_hardnetpp_dsp_learned_stop_policy"
    assert artifact["run_date"] == "20260505"
    assert artifact["status"] == "complete"
    assert artifact["spec_refs"] == ["REQ-KONA-032", "SCENARIO-KONA-032"]
    assert artifact["learned_stop_policy_written"] is True
    assert artifact["generalization_split"]["train_count"] == 120
    assert artifact["generalization_split"]["held_out_count"] == 36
    assert artifact["stop_policy_precision"] == pytest.approx(1.0)
    assert artifact["stop_policy_recall"] == pytest.approx(1.0)
    assert artifact["dsp_feasibility_auc"] == pytest.approx(0.640625)
    assert artifact["hardnetpp_delta_over_replay_policy"] == pytest.approx(0.0)
    assert artifact["source_context"]["experiment_1305_status"] == "complete"
    assert artifact["source_context"]["prompt_named_paths_missing"] == [
        "results/experiment_1291_hardnetpp_nonlinear_repair.json",
        "results/experiment_1292_dsp_feasibility_channel.json",
    ]
    assert "not a broad general stop rule" in artifact["honest_verdict"]
    assert "too weak" in experiment._honest_verdict(
        {"stop_policy_precision": 0.0, "stop_policy_recall": 0.0}
    )
    json.dumps(artifact)


def test_build_artifact_blocks_without_required_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-KONA-032: missing replay sources produce a terminal blocker."""
    monkeypatch.setattr(experiment, "EXP1305_PATH", tmp_path / "missing_1305.json")
    monkeypatch.setattr(experiment, "EXP1291_PATH", tmp_path / "missing_1291.json")
    monkeypatch.setattr(experiment, "EXP1292_PATH", tmp_path / "missing_1292.json")

    artifact = experiment.build_artifact()

    assert artifact["status"] == "blocked"
    assert artifact["learned_stop_policy_written"] is False
    assert artifact["honest_verdict"] == "blocked_missing_required_replay_artifacts"
    assert artifact["missing_paths"] == [
        str(tmp_path / "missing_1305.json"),
        str(tmp_path / "missing_1291.json"),
        str(tmp_path / "missing_1292.json"),
    ]


def test_script_main_writes_terminal_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-KONA-032: script writes the terminal Exp 1318 JSON artifact."""
    output_path = tmp_path / "experiment_1318_hardnetpp_dsp_learned_stop_policy.json"
    monkeypatch.setattr(experiment, "RESULT_PATH", output_path)

    artifact = experiment.main()

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == artifact
    assert written["status"] == "complete"
    assert written["learned_stop_policy_written"] is True
