"""Tests for Exp 3116 FR-11 hard-family curriculum retention guard.

Spec refs: REQ-LEARN-3116, SCENARIO-LEARN-3116,
SCENARIO-LEARN-3116-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_unsolvable_curriculum_retention_guard_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"
REQUIRED_FIELDS = {
    "fr11_unsolvable_curriculum_ready",
    "continuous_self_learning_task",
    "controller_only",
    "no_weight_update_claim",
    "model_specs",
    "hard_family_count",
    "unsolvable_detection_summary",
    "hint_policy_summary",
    "soundness_mistakes",
    "completeness_mistakes",
    "prior_retention_delta",
    "delayed_regression_delta",
    "rollback_count",
    "promotion_decision",
    "negative_controls",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _config(tmp_path: Path) -> exp.GuardConfig:
    return exp.GuardConfig(
        repo_root=tmp_path,
        output_path=tmp_path / exp.OUTPUT_REL_PATH,
        exp3090_artifact_path=tmp_path / exp.EXP3090_REL_PATH,
        exp3097_artifact_path=tmp_path / exp.EXP3097_REL_PATH,
        exp3103_artifact_path=tmp_path / exp.EXP3103_REL_PATH,
        protocol_manifest_path=tmp_path / exp.STRATIFIED_MANIFEST_REL_PATH,
        started_s=100.0,
        clock=lambda: 103.5,
        tests_run=("focused-req-3116",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for rel_path in (
        exp.EXP3090_REL_PATH,
        exp.EXP3097_REL_PATH,
        exp.EXP3103_REL_PATH,
        exp.STRATIFIED_MANIFEST_REL_PATH,
    ):
        source = REPO_ROOT / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_learn_3116_spec_anchor_exists() -> None:
    """REQ-LEARN-3116: OpenSpec declares the curriculum guard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3116" in spec
    assert "SCENARIO-LEARN-3116" in spec
    assert "SCENARIO-LEARN-3116-BLOCKED" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "fr11_unsolvable_curriculum_ready" in spec
    assert "negative_controls" in spec
    assert "controller_only" in spec
    assert "no_weight_update_claim" in spec


def test_scenario_learn_3116_writes_controller_only_guard_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3116: abstract hints repair completeness and retention."""

    _copy_sources(tmp_path)
    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["fr11_unsolvable_curriculum_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["controller_only"] is True
    assert artifact["no_weight_update_claim"] is True
    assert artifact["model_specs"] == list(exp.MANDATED_MODEL_SPECS)
    assert artifact["hard_family_count"] == 4
    assert artifact["promotion_decision"] == "controller_only"
    assert artifact["honest_verdict"] == (
        "complete_fr11_unsolvable_curriculum_controller_only_guard_passed"
    )
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["tests_run"] == ["focused-req-3116"]

    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 0
    assert artifact["family_holdout_delta"] == pytest.approx(0.176471)
    assert artifact["prior_retention_delta"] == pytest.approx(0.444444)
    assert artifact["delayed_regression_delta"] == pytest.approx(0.142857)
    assert artifact["rollback_count"] == 3

    detection = artifact["unsolvable_detection_summary"]
    assert detection["source_exp3103_completeness_mistakes"] == 12
    assert detection["prior_retention_regression_case_count"] == 16
    assert detection["zero_pass_family_count"] == 4
    assert detection["hard_families"] == [
        "arithmetic_true_verification",
        "json_syntax_repair",
        "numeric_bound_repair",
        "python_assertion_repair",
    ]

    hints = artifact["hint_policy_summary"]
    assert hints["solver_derived"] is True
    assert hints["abstract_only"] is True
    assert hints["final_answers_revealed"] is False
    assert hints["hint_count"] == 4
    assert hints["live_llm_inference_used"] is False

    usefulness = artifact["hint_usefulness"]
    assert usefulness["exp3103_completeness_mistakes"] == 12
    assert usefulness["completeness_mistakes_reduced_by"] == 12
    assert usefulness["prior_retention_cases_recovered"] == 16

    controls = artifact["negative_controls"]
    assert set(controls) == {"no_feedback", "shuffled_hint", "stale_hint", "contradictory_hint"}
    assert controls["no_feedback"]["promotion_gate_passed"] is False
    assert controls["no_feedback"]["rolled_back"] is False
    assert controls["no_feedback"]["completeness_mistakes"] == 12
    assert controls["shuffled_hint"]["rolled_back"] is True
    assert controls["stale_hint"]["soundness_mistakes"] > 0
    assert controls["stale_hint"]["rolled_back"] is True
    assert controls["contradictory_hint"]["rolled_back"] is True
    assert all(control["failed_safely"] is True for control in controls.values())

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "deterministic_solver_derived_hint_controller_replay"
    assert substrate["controller_memory_update"] is True
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["base_model_weights_updated"] is False
    assert substrate["kan_model_weight_training"] is False
    assert substrate["live_llm_inference"] is False
    assert substrate["live_model_inference"] is False
    exp.validate_artifact(artifact)


def test_req_learn_3116_hints_are_source_derived_and_abstract(tmp_path: Path) -> None:
    """REQ-LEARN-3116-2/3/4/5/6: detection, hinting, controls, and gates are auditable."""

    _copy_sources(tmp_path)
    preconditions = exp.load_preconditions(_config(tmp_path))
    curriculum = exp.build_curriculum(preconditions)
    result = exp.run_curriculum_guard(preconditions, curriculum)

    assert preconditions.ok is True
    assert curriculum.hard_family_count == 4
    assert len(curriculum.protocol_hard_rows) == 12
    assert len(curriculum.prior_retention_regression_rows) == 16
    assert [hint.family_key for hint in curriculum.abstract_hints] == [
        "arithmetic_true_verification",
        "json_syntax_repair",
        "numeric_bound_repair",
        "python_assertion_repair",
    ]
    assert all(hint.solver_derived for hint in curriculum.abstract_hints)
    assert not any(hint.final_answer_revealed for hint in curriculum.abstract_hints)
    assert not any(
        exp.hint_leaks_final_answer(hint, curriculum.rows_by_id)
        for hint in curriculum.abstract_hints
    )
    first_hint = curriculum.abstract_hints[0]
    first_source = curriculum.rows_by_id[first_hint.source_fixture_ids[0]]
    leaky_hint = exp.AbstractHint(
        hint_id="leaky",
        family_key=first_hint.family_key,
        target_action=first_hint.target_action,
        target_context=first_hint.target_context,
        source_fixture_ids=first_hint.source_fixture_ids,
        feature_weights=first_hint.feature_weights,
        abstract_hint=str(first_source["expected_answer"]),
        evidence=first_hint.evidence,
    )
    assert exp.hint_leaks_final_answer(leaky_hint, curriculum.rows_by_id) is True

    base_counts = exp.count_labels(exp.evaluate_rows(result.base_state, preconditions.protocol_rows))
    guard_counts = exp.count_labels(exp.evaluate_rows(result.guarded_state, preconditions.protocol_rows))
    assert base_counts["completeness_mistake"] == 12
    assert guard_counts == {"correct": 72}
    assert result.promotion_decision == "controller_only"
    assert exp.promotion_gates_passed(result.metrics, result.negative_controls) is True
    assert exp.accuracy(result.base_state, ()) == 0.0

    bad_row = preconditions.protocol_rows[0]
    assert exp.with_target(bad_row, "reject", "unit-test").target_action == "reject"
    assert exp.decision_label("accept", "reject") == "soundness_mistake"
    assert exp.decision_label("reject", "accept") == "completeness_mistake"
    assert exp.decision_label("abstain", "reject") == "abstention"
    with pytest.raises(ValueError, match="unsupported target action"):
        exp.decision_label("accept", "defer")
    with pytest.raises(ValueError, match="curriculum requires hard-family rows"):
        exp.build_curriculum(
            exp.PreconditionResult(
                ok=True,
                checks={},
                exp3090_artifact=preconditions.exp3090_artifact,
                exp3097_artifact=preconditions.exp3097_artifact,
                exp3103_artifact={"online_decisions": []},
                protocol_rows=preconditions.protocol_rows,
                blocked_reason="",
                rows_by_id=preconditions.rows_by_id,
            )
        )
    with pytest.raises(ValueError, match="stress split missing required partition"):
        exp.build_stress_split(
            exp.PreconditionResult(
                ok=True,
                checks={},
                exp3090_artifact={},
                exp3097_artifact={},
                exp3103_artifact={},
                protocol_rows=(),
                blocked_reason="",
            )
        )
    assert exp.relative_path(tmp_path, tmp_path / "results" / "x.json") == "results/x.json"
    assert exp.relative_path(tmp_path, Path("/outside/root.json")) == "/outside/root.json"


def test_scenario_learn_3116_blocked_without_sources(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3116-BLOCKED: missing source evidence fails closed."""

    artifact = exp.write_artifact(_config(tmp_path))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["fr11_unsolvable_curriculum_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["controller_only"] is True
    assert artifact["no_weight_update_claim"] is True
    assert artifact["hard_family_count"] == 0
    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 0
    assert artifact["prior_retention_delta"] == 0.0
    assert artifact["delayed_regression_delta"] == 0.0
    assert artifact["rollback_count"] == 0
    assert artifact["promotion_decision"] == "blocked"
    assert artifact["negative_controls"] == {}
    assert artifact["honest_verdict"] == "blocked_precondition_failed"
    assert artifact["blocked_reason"] == "exp3090_artifact_missing_or_empty"
    assert artifact["preconditions_checked"]["exp3090_artifact_ready"]["ok"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / exp.OUTPUT_REL_PATH).is_file()
    exp.validate_artifact(artifact)
    with pytest.raises(ValueError, match="blocked curriculum artifacts"):
        exp.validate_artifact(artifact | {"honest_verdict": "waiting"})


def test_req_learn_3116_validation_rejects_invalid_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3116-6: artifact validation enforces controller-only gates."""

    _copy_sources(tmp_path)
    artifact = exp.write_artifact(_config(tmp_path))
    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")

    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"continuous_self_learning_task": False}, "continuous_self_learning_task"),
        (artifact | {"controller_only": False}, "controller_only"),
        (artifact | {"no_weight_update_claim": False}, "no_weight_update_claim"),
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
                "soundness_mistakes": 1,
            },
            "controller_only",
        ),
    ]
    for invalid, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(invalid)
