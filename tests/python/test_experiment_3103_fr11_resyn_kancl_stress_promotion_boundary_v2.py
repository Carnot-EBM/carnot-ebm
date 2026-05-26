"""Tests for Exp 3103 FR-11 ReSyn/KAN-CL stress promotion boundary.

Spec refs: REQ-LEARN-3103, SCENARIO-LEARN-3103,
SCENARIO-LEARN-3103-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_resyn_kancl_stress_promotion_boundary_v2 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
REQUIRED_FIELDS = {
    "fr11_stress_ready",
    "continuous_self_learning_task",
    "promotion_decision",
    "soundness_mistakes",
    "completeness_mistakes",
    "family_holdout_delta",
    "prior_retention_delta",
    "delayed_regression_delta",
    "rollback_count",
    "negative_control_results",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _config(tmp_path: Path) -> exp.StressConfig:
    return exp.StressConfig(
        repo_root=tmp_path,
        output_path=tmp_path / exp.OUTPUT_REL_PATH,
        exp3090_artifact_path=tmp_path / exp.EXP3090_REL_PATH,
        exp3097_artifact_path=tmp_path / exp.EXP3097_REL_PATH,
        protocol_manifest_path=tmp_path / exp.STRATIFIED_MANIFEST_REL_PATH,
        started_s=10.0,
        clock=lambda: 14.25,
        tests_run=("focused-req-3103",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for rel_path in (
        exp.EXP3090_REL_PATH,
        exp.EXP3097_REL_PATH,
        exp.STRATIFIED_MANIFEST_REL_PATH,
    ):
        source = REPO_ROOT / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_learn_3103_spec_anchor_exists() -> None:
    """REQ-LEARN-3103: OpenSpec declares the stress-boundary contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3103" in spec
    assert "SCENARIO-LEARN-3103" in spec
    assert "SCENARIO-LEARN-3103-BLOCKED" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "fr11_stress_ready" in spec
    assert "negative_control_results" in spec
    assert "promotion_decision" in spec
    assert "controller_only" in spec
    assert "broader_promotion_candidate" in spec


def test_scenario_learn_3103_writes_blocked_promotion_boundary_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3103: stress evidence blocks broader promotion."""

    _copy_sources(tmp_path)
    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["fr11_stress_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_decision"] == "blocked"
    assert artifact["honest_verdict"] == "complete_fr11_stress_boundary_blocks_promotion"
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["tests_run"] == ["focused-req-3103"]

    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 12
    assert artifact["family_holdout_delta"] == pytest.approx(0.117647)
    assert artifact["prior_retention_delta"] == pytest.approx(-0.444444)
    assert artifact["delayed_regression_delta"] == pytest.approx(0.285714)
    assert artifact["rollback_count"] == 2

    split = artifact["stress_split_report"]
    assert split["train_update_count"] == 14
    assert split["family_holdout_count"] == 51
    assert split["prior_retention_count"] == 36
    assert split["delayed_regression_count"] == 7
    assert split["harder_holdout_perturbation_count"] == 7

    controls = artifact["negative_control_results"]
    assert controls["no_feedback"]["case_count"] == 51
    assert controls["no_feedback"]["family_holdout_delta"] == pytest.approx(0.0)
    assert controls["shuffled_label"]["case_count"] == 14
    assert controls["shuffled_label"]["family_holdout_delta"] < 0.0
    assert controls["shuffled_label"]["soundness_mistakes"] > 0
    assert controls["shuffled_label"]["rolled_back"] is True

    source_ids = {row["id"] for row in artifact["source_artifacts"]}
    assert {"exp3090_prior_fr11", "exp3097_protocol", "exp3097_manifest"} <= source_ids
    assert all(row["exists"] is True for row in artifact["source_artifacts"])

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "deterministic_fr11_stress_controller_replay"
    assert substrate["controller_weight_update"] is True
    assert substrate["base_model_weights_updated"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["kan_model_weight_training"] is False
    assert substrate["live_llm_inference"] is False
    assert substrate["live_model_inference"] is False
    exp.validate_artifact(artifact)


def test_req_learn_3103_metrics_are_source_derived(tmp_path: Path) -> None:
    """REQ-LEARN-3103-2/3/4/5: splits, mistakes, controls, and retention are auditable."""

    _copy_sources(tmp_path)
    preconditions = exp.load_preconditions(_config(tmp_path))
    split = exp.build_stress_split(preconditions)
    result = exp.run_stress_replay(preconditions, split)

    assert preconditions.ok is True
    assert {row.target_action for row in split.train_update} == {"accept", "reject"}
    assert {row.perturbation_type for row in split.family_holdout} == {
        "arithmetic_false_verification",
        "arithmetic_true_verification",
        "json_syntax_repair",
        "numeric_bound_repair",
        "python_assertion_repair",
        "smt_sat_solving",
        "smt_unsat_abstention",
    }

    prior_state = exp.reconstruct_prior_controller(preconditions.exp3090_artifact)
    candidate_state = exp.apply_protocol_feedback(prior_state, split.train_update)
    shuffled_state = exp.apply_protocol_feedback(
        prior_state,
        exp.shuffled_label_rows(split.train_update),
    )

    assert exp.accuracy(prior_state, split.family_holdout) == pytest.approx(0.705882)
    assert exp.accuracy(candidate_state, split.family_holdout) == pytest.approx(0.823529)
    assert exp.accuracy(shuffled_state, split.family_holdout) == pytest.approx(0.352941)
    assert exp.accuracy(prior_state, split.prior_retention) == pytest.approx(1.0)
    assert exp.accuracy(candidate_state, split.prior_retention) == pytest.approx(0.555556)

    stress_counts = exp.count_labels(exp.evaluate_rows(candidate_state, preconditions.protocol_rows))
    shuffled_counts = exp.count_labels(exp.evaluate_rows(shuffled_state, split.family_holdout))
    assert stress_counts.get("soundness_mistake", 0) == 0
    assert stress_counts["completeness_mistake"] == 12
    assert shuffled_counts["soundness_mistake"] == 33
    assert result.promotion_decision == "blocked"
    assert result.rollback_count == 2
    assert result.negative_control_results["shuffled_label"]["rolled_back"] is True
    assert result.metrics["prior_retention_delta"] < 0.0
    assert exp.accuracy(prior_state, ()) == 0.0

    assert exp.decision_label("accept", "accept") == "correct"
    assert exp.decision_label("reject", "reject") == "correct"
    assert exp.decision_label("accept", "reject") == "soundness_mistake"
    assert exp.decision_label("abstain", "accept") == "completeness_mistake"
    assert exp.decision_label("abstain", "reject") == "abstention"
    with pytest.raises(ValueError, match="unsupported target action"):
        exp.decision_label("accept", "defer")
    with pytest.raises(ValueError, match="stress split missing required partition"):
        exp.build_stress_split(
            exp.PreconditionResult(
                ok=True,
                checks={},
                exp3090_artifact={},
                exp3097_artifact={},
                protocol_rows=(),
                blocked_reason="",
            )
        )
    assert exp.relative_path(tmp_path, tmp_path / "results" / "x.json") == "results/x.json"
    assert exp.relative_path(tmp_path, Path("/outside/root.json")) == "/outside/root.json"


def test_scenario_learn_3103_blocked_without_sources(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3103-BLOCKED: missing prior/protocol evidence fails closed."""

    artifact = exp.write_artifact(_config(tmp_path))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["fr11_stress_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["promotion_decision"] == "blocked"
    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 0
    assert artifact["family_holdout_delta"] == 0.0
    assert artifact["prior_retention_delta"] == 0.0
    assert artifact["delayed_regression_delta"] == 0.0
    assert artifact["rollback_count"] == 0
    assert artifact["negative_control_results"] == {}
    assert artifact["honest_verdict"] == "blocked_precondition_failed"
    assert artifact["blocked_reason"] == "exp3090_artifact_missing_or_empty"
    assert artifact["preconditions_checked"]["exp3090_artifact_ready"]["ok"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / exp.OUTPUT_REL_PATH).is_file()
    exp.validate_artifact(artifact)
    with pytest.raises(ValueError, match="blocked stress artifacts"):
        exp.validate_artifact(artifact | {"honest_verdict": "waiting"})


def test_req_learn_3103_validation_rejects_invalid_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3103-6: promotion decisions and substrate boundaries are enforced."""

    _copy_sources(tmp_path)
    artifact = exp.write_artifact(_config(tmp_path))
    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")

    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"continuous_self_learning_task": False}, "continuous_self_learning_task"),
        (artifact | {"promotion_decision": "maybe"}, "promotion_decision"),
        (artifact | {"honest_verdict": "waiting"}, "honest_verdict"),
        (artifact | {"source_artifacts": []}, "source_artifacts"),
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
                "promotion_decision": "controller_only",
                "completeness_mistakes": 1,
            },
            "controller_only",
        ),
    ]
    for invalid, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(invalid)
