"""Tests for Exp 3090 FR-11 ReSyn KAN-CL completeness repair.

Spec refs: REQ-LEARN-3090, SCENARIO-LEARN-3090,
SCENARIO-LEARN-3090-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_resyn_kancl_completeness_repair_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
REQUIRED_FIELDS = {
    "fr11_resyn_kancl_ready",
    "continuous_self_learning_task",
    "promotion_decision",
    "soundness_mistakes",
    "completeness_mistakes",
    "family_holdout_delta",
    "prior_retention_delta",
    "no_feedback_control_delta",
    "shuffled_feedback_control_delta",
    "kancl_anchor_count",
    "rollback_count",
    "delayed_regression_delta",
    "preconditions_checked",
    "inference_substrate",
    "honest_verdict",
}


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / exp.OUTPUT_REL_PATH,
        exp3084_artifact_path=tmp_path / exp.EXP3084_ARTIFACT_REL_PATH,
        manifest_path=tmp_path / exp.EXP3084_MANIFEST_REL_PATH,
        started_at=100.0,
        clock=lambda: 104.25,
        tests_run=("focused-req-3090",),
    )


def _copy_fixture_bank(tmp_path: Path) -> None:
    for rel_path in (exp.EXP3084_ARTIFACT_REL_PATH, exp.EXP3084_MANIFEST_REL_PATH):
        source = REPO_ROOT / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_learn_3090_spec_anchor_exists() -> None:
    """REQ-LEARN-3090: OpenSpec declares the ReSyn KAN-CL pilot contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3090" in spec
    assert "SCENARIO-LEARN-3090" in spec
    assert "SCENARIO-LEARN-3090-BLOCKED" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "fr11_resyn_kancl_ready" in spec
    assert "blocked_fixture_precondition_failed" in spec
    assert "kancl_anchor_count" in spec


def test_scenario_learn_3090_writes_controller_only_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3090: KAN-CL anchors repair completeness with no soundness miss."""

    _copy_fixture_bank(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["fr11_resyn_kancl_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["honest_verdict"] == "complete_fr11_resyn_kancl_controller_only_ready"
    assert artifact["promotion_decision"] == "controller_only_resyn_kancl_budget_passed"
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["tests_run"] == ["focused-req-3090"]

    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 0
    assert artifact["baseline_completeness_mistakes"] > artifact["completeness_mistakes"]
    assert artifact["family_holdout_delta"] == pytest.approx(1.0)
    assert artifact["prior_retention_delta"] == pytest.approx(0.0)
    assert artifact["no_feedback_control_delta"] == pytest.approx(0.0)
    assert artifact["shuffled_feedback_control_delta"] == pytest.approx(0.0)
    assert artifact["delayed_regression_delta"] == pytest.approx(0.5)
    assert artifact["contradiction_rate_delta"] == pytest.approx(0.0)
    assert artifact["kancl_anchor_count"] == len(artifact["kancl_anchors"])
    assert artifact["kancl_anchor_count"] > 0
    assert artifact["rollback_count"] == 1

    preconditions = artifact["preconditions_checked"]
    assert preconditions["exp3084_artifact_ready"]["ok"] is True
    assert preconditions["fixture_manifest_exists"]["ok"] is True
    assert preconditions["exact_labels_available"]["ok"] is True
    assert preconditions["delayed_regression_labels_available"]["ok"] is True
    assert preconditions["fixture_count"]["observed"] == 72

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "deterministic_resyn_fixture_controller_replay"
    assert substrate["live_llm_inference"] is False
    assert substrate["live_model_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["controller_weight_update"] is True
    assert substrate["kan_model_weight_training"] is False

    assert artifact["control_report"]["non_vacuous_controls"] is True
    assert artifact["control_report"]["shuffled_candidate_rolled_back"] is True
    assert artifact["budget_gates"]["all_gates_passed"] is True
    assert artifact["split_report"]["family_holdout_perturbation_family"] == (
        "python_assertion_repair"
    )
    assert artifact["source_trace_counts"]["fixture_count"] == 72
    assert artifact["source_trace_counts"]["train_update_count"] > 0
    assert artifact["source_trace_counts"]["family_holdout_count"] > 0

    for anchor in artifact["kancl_anchors"]:
        assert anchor["family_knot"] >= 0.0
        assert anchor["constraint_local_basis_weights"]
        assert all(
            abs(weight) <= exp.MAX_ABS_WEIGHT
            for weight in anchor["constraint_local_basis_weights"].values()
        )

    exp.validate_artifact(artifact)


def test_req_learn_3090_policy_metrics_are_source_derived(tmp_path: Path) -> None:
    """REQ-LEARN-3090-2/3/4/5: splits, anchors, controls, and mistakes are auditable."""

    _copy_fixture_bank(tmp_path)
    config = _config(tmp_path)
    preconditions = exp.check_preconditions(config)
    split = exp.build_fixture_split(preconditions.rows)
    result = exp.run_online_policy(split)

    assert preconditions.ok is True
    assert exp.exact_accept_label(split.train_update[0]) is True
    assert exp.exact_accept_label(split.prior_cases[0]) is False
    assert {exp.exact_accept_label(row) for row in preconditions.rows} == {False, True}
    assert {row["family"] for row in split.train_update} == {
        "arithmetic_code_assertions",
        "repairable_invalid_candidates",
        "smt_constraints",
    }
    assert {row["perturbation_family"] for row in split.family_holdout} == {
        "python_assertion_repair"
    }

    baseline = exp.initial_controller_state()
    updated = exp.apply_online_feedback(baseline, split.train_update)
    all_positive_rows = tuple(row for row in preconditions.rows if exp.exact_accept_label(row))
    assert exp.accuracy(baseline, ()) == 0.0
    assert exp.contradiction_rate(updated, all_positive_rows[:2]) == 0.0
    assert exp.accuracy(baseline, split.family_holdout) == 0.0
    assert exp.accuracy(updated, split.family_holdout) == 1.0
    assert exp.count_decision_labels(exp.evaluate_cases(updated, split.family_holdout)) == {
        "correct": len(split.family_holdout)
    }

    assert result.metrics["family_holdout_delta"] > 0.0
    assert result.metrics["prior_retention_delta"] == 0.0
    assert result.metrics["no_feedback_control_delta"] == 0.0
    assert result.metrics["shuffled_feedback_control_delta"] == 0.0
    assert result.metrics["delayed_regression_delta"] > 0.0
    assert result.soundness_mistakes == 0
    assert result.completeness_mistakes == 0
    assert result.baseline_completeness_mistakes > 0
    assert result.rollback_count == 1
    assert result.kancl_anchor_count == len(result.anchors)
    assert exp.controller_has_model_weight_mutation(result.updated_state) is False

    json_repair = next(
        row for row in preconditions.rows if row["perturbation_family"] == "json_syntax_repair"
    )
    smt_sat = next(
        row for row in preconditions.rows if row["perturbation_family"] == "smt_sat_solving"
    )
    arithmetic_true = next(
        row
        for row in preconditions.rows
        if row["perturbation_family"] == "arithmetic_true_verification"
    )
    assert exp.exact_accept_label(json_repair) is True
    assert exp.exact_accept_label(smt_sat) is True
    assert exp.exact_accept_label(arithmetic_true) is True

    with pytest.raises(ValueError, match="unsupported exact label kind"):
        exp.exact_accept_label({"exact_label": {"kind": "unknown"}})
    with pytest.raises(ValueError, match="fixture split missing required partition"):
        exp.build_fixture_split(())
    assert exp._first_failed_precondition({"all_good": {"ok": True}}) == (
        "unknown_precondition_failure"
    )
    assert exp._relative_path(tmp_path, Path("/outside/root.json")) == "/outside/root.json"


def test_scenario_learn_3090_blocked_without_fixture_bank(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3090-BLOCKED: missing Exp 3084 evidence fails closed."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["fr11_resyn_kancl_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_decision"] == "blocked_fixture_precondition_failed"
    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 0
    assert artifact["family_holdout_delta"] == 0.0
    assert artifact["prior_retention_delta"] == 0.0
    assert artifact["no_feedback_control_delta"] == 0.0
    assert artifact["shuffled_feedback_control_delta"] == 0.0
    assert artifact["kancl_anchor_count"] == 0
    assert artifact["rollback_count"] == 0
    assert artifact["delayed_regression_delta"] == 0.0
    assert artifact["honest_verdict"] == "blocked_fixture_precondition_failed"
    assert artifact["preconditions_checked"]["exp3084_artifact_ready"]["ok"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / exp.OUTPUT_REL_PATH).is_file()
    exp.validate_artifact(artifact)


def test_req_learn_3090_validation_rejects_invalid_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3090-6: readiness and promotion boundaries are enforced."""

    _copy_fixture_bank(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")

    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"continuous_self_learning_task": False}, "continuous_self_learning_task"),
        (artifact | {"honest_verdict": "waiting"}, "honest_verdict"),
        (artifact | {"fr11_resyn_kancl_ready": False}, "blocked artifacts"),
        (artifact | {"soundness_mistakes": 1}, "soundness_mistakes"),
        (artifact | {"completeness_mistakes": 1}, "completeness_mistakes"),
        (artifact | {"family_holdout_delta": 0.0}, "family_holdout_delta"),
        (artifact | {"prior_retention_delta": -0.1}, "prior_retention_delta"),
        (artifact | {"no_feedback_control_delta": 0.1}, "no_feedback_control_delta"),
        (
            artifact | {"shuffled_feedback_control_delta": 0.1},
            "shuffled_feedback_control_delta",
        ),
        (artifact | {"kancl_anchor_count": 0}, "kancl_anchor_count"),
        (artifact | {"rollback_count": 0}, "rollback_count"),
        (artifact | {"delayed_regression_delta": -0.1}, "delayed_regression_delta"),
        (artifact | {"preconditions_checked": {}}, "preconditions_checked"),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_mutation": True}
            },
            "model weights",
        ),
        (
            artifact
            | {
                "budget_gates": artifact["budget_gates"] | {"all_gates_passed": False},
                "promotion_decision": "controller_only_resyn_kancl_budget_passed",
            },
            "promotion_decision",
        ),
    ]
    for invalid, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(invalid)
