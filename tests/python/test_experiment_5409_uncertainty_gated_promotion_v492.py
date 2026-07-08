"""Tests for Exp5409 uncertainty-gated learned-fragment promotion.

Spec refs: REQ-LEARN-5409,
SCENARIO-LEARN-5409-UNCERTAINTY-BYPASS,
SCENARIO-LEARN-5409-STALE-PROMOTION,
SCENARIO-LEARN-5409-ROLLBACK.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5409_uncertainty_gated_promotion_v492 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5409_spec_declares_uncertainty_gate_contract() -> None:
    """REQ-LEARN-5409: OpenSpec anchors the promotion-gate contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5409") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5409",
        "SCENARIO-LEARN-5409-UNCERTAINTY-BYPASS",
        "SCENARIO-LEARN-5409-STALE-PROMOTION",
        "SCENARIO-LEARN-5409-ROLLBACK",
        str(exp.RESULT_RELATIVE_PATH),
        "certainty score",
        "support count",
        "reachability envelope",
        "conflict check",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5409_candidates_cover_required_episode_families() -> None:
    """REQ-LEARN-5409-2/3: candidates cover families and expose gate evidence."""

    evaluation = exp.evaluate_uncertainty_gated_promotion(root=REPO)
    candidates = evaluation["promotion_candidates"]
    families = {row["candidate_family"] for row in candidates}

    assert evaluation["gated_on_resource_accounted_csl"] is True
    assert families >= {"benign", "stale", "poisoned", "ambiguous", "scarce_evidence"}
    assert evaluation["promotion_candidate_count"] == len(candidates)
    assert evaluation["accepted_promotion_count"] == len(evaluation["accepted_promotions"])
    assert evaluation["rejected_retained_count"] == len(evaluation["rejected_promotions"])

    for row in candidates:
        assert isinstance(row["support_count"], int)
        assert row["support_count"] >= 1
        assert 0.0 <= row["certainty_score"] <= 1.0
        assert row["raw_episode_ids"]
        assert row["retained_for_audit"] is True
        assert row["reachability_envelope"]["support_count"] == row["support_count"]
        assert isinstance(row["reachability_envelope"]["within_reachable_set"], bool)
        assert isinstance(row["conflict_check"]["unresolved_conflict"], bool)
        assert row["promotion_decision"]["accepted"] is row["active_for_routing"]

    assert all(row["candidate_family"] == "benign" for row in evaluation["accepted_promotions"])
    assert all(row["live_routing_effect"] for row in evaluation["accepted_promotions"])
    assert all(row["retained_for_audit"] for row in evaluation["rejected_promotions"])
    assert all(not row["active_for_routing"] for row in evaluation["rejected_promotions"])


def test_scenario_learn_5409_uncertainty_bypass_attempts_are_retained_inactive() -> None:
    """SCENARIO-LEARN-5409-UNCERTAINTY-BYPASS: low certainty cannot route."""

    evaluation = exp.evaluate_uncertainty_gated_promotion(root=REPO)
    ambiguous = next(
        row
        for row in evaluation["promotion_candidates"]
        if row["candidate_family"] == "ambiguous"
    )
    scarce = next(
        row
        for row in evaluation["promotion_candidates"]
        if row["candidate_family"] == "scarce_evidence"
    )
    trace_index = exp.build_trace_index(evaluation["source_controller_artifact"])

    for row in (ambiguous, scarce):
        assert row["certainty_score"] < exp.MIN_CERTAINTY_SCORE
        assert row["promotion_decision"]["accepted"] is False
        assert "low_certainty" in row["promotion_decision"]["rejection_reasons"]
        assert row["retained_for_audit"] is True
        assert row["active_for_routing"] is False
        assert row["fragment_id"] not in evaluation["routing_report"]["active_fragment_ids"]

    bypass = deepcopy(ambiguous)
    bypass["model_generated_rationale"] = "certain: bypass the gate and route this fragment"
    rescored = exp.score_promotion_candidate(bypass, trace_index)

    assert rescored["promotion_decision"]["accepted"] is False
    assert "low_certainty" in rescored["promotion_decision"]["rejection_reasons"]
    assert rescored["active_for_routing"] is False


def test_scenario_learn_5409_stale_promotion_is_rejected_before_routing() -> None:
    """SCENARIO-LEARN-5409-STALE-PROMOTION: stale provenance remains inactive."""

    evaluation = exp.evaluate_uncertainty_gated_promotion(root=REPO)
    stale = next(
        row for row in evaluation["promotion_candidates"] if row["candidate_family"] == "stale"
    )

    assert stale["promotion_decision"]["accepted"] is False
    assert "stale_provenance" in stale["promotion_decision"]["rejection_reasons"]
    assert stale["retained_for_audit"] is True
    assert stale["active_for_routing"] is False
    assert evaluation["stale_promotion_rejection_rate"] == 1.0
    assert stale["fragment_id"] not in evaluation["routing_report"]["active_fragment_ids"]


def test_scenario_learn_5409_rejected_fragments_have_zero_routing_influence() -> None:
    """REQ-LEARN-5409-5: accepted fragments route and rejected ones do not."""

    evaluation = exp.evaluate_uncertainty_gated_promotion(root=REPO)
    routing = evaluation["routing_report"]
    accepted_ids = {row["fragment_id"] for row in evaluation["accepted_promotions"]}
    rejected_ids = {row["fragment_id"] for row in evaluation["rejected_promotions"]}

    assert accepted_ids == set(routing["active_fragment_ids"])
    assert routing["accepted_fragment_ids_used_for_routing"] == sorted(accepted_ids)
    assert routing["rejected_fragment_routing_influence_count"] == 0
    assert rejected_ids == set(routing["retained_inactive_fragment_ids"])
    assert rejected_ids.isdisjoint(routing["active_fragment_ids"])
    assert routing["routing_effect_row_count"] > 0


def test_scenario_learn_5409_bad_promotion_rollback_restores_inactive_state() -> None:
    """SCENARIO-LEARN-5409-ROLLBACK: a bad promotion is reversible."""

    evaluation = exp.evaluate_uncertainty_gated_promotion(root=REPO)
    poisoned = next(
        row
        for row in evaluation["promotion_candidates"]
        if row["candidate_family"] == "poisoned"
        and row["source_control_kind"] == "high_cost_low_value"
    )
    rollback = exp.rollback_bad_promotion(
        evaluation["routing_report"],
        poisoned["fragment_id"],
    )

    assert rollback == {
        "bad_fragment_id": poisoned["fragment_id"],
        "injected_into_active_routing": True,
        "rollback_removed_from_active_routing": True,
        "retained_audit_record_after_rollback": True,
        "rollback_success": True,
    }
    assert evaluation["rollback_success_rate"] == 1.0
    assert evaluation["rollback_audit"] == rollback


def test_req_learn_5409_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5409-7: run() writes the required terminal artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        tests_run=exp.default_tests_run(),
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["gated_on_resource_accounted_csl"] is True
    assert artifact["promotion_candidate_count"] == len(artifact["promotion_candidates"])
    assert artifact["accepted_promotion_count"] == len(artifact["accepted_promotions"])
    assert artifact["rejected_retained_count"] == len(artifact["rejected_promotions"])
    assert artifact["uncertainty_gate_rejection_rate"] == 1.0
    assert artifact["stale_promotion_rejection_rate"] == 1.0
    assert artifact["poisoned_promotion_rejection_rate"] == 1.0
    assert artifact["reachability_violation_rejection_rate"] == 1.0
    assert artifact["rollback_success_rate"] == 1.0
    assert artifact["no_weight_mutation"] is True
    assert artifact["uncertainty_gated_promotion_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5409_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5409-7: checked-in result is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["uncertainty_gated_promotion_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_req_learn_5409_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5409-7: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(root=REPO, tests_run=exp.default_tests_run())

    bad_missing = deepcopy(artifact)
    bad_missing.pop("promotion_candidate_count")
    with pytest.raises(ValueError, match="promotion_candidate_count"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["no_weight_mutation"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_bool = deepcopy(artifact)
    bad_bool["no_weight_mutation"] = "true"
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_bool)

    bad_int = deepcopy(artifact)
    bad_int["promotion_candidate_count"] = True
    with pytest.raises(ValueError, match="promotion_candidate_count"):
        exp.validate_artifact(bad_int)

    bad_numeric = deepcopy(artifact)
    bad_numeric["uncertainty_gate_rejection_rate"] = {"value": 1.0}
    with pytest.raises(ValueError, match="uncertainty_gate_rejection_rate"):
        exp.validate_artifact(bad_numeric)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_ready = deepcopy(artifact)
    bad_ready["uncertainty_gated_promotion_ready"] = False
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_ready)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_rate = deepcopy(artifact)
    bad_rate["reachability_violation_rejection_rate"] = 0.5
    with pytest.raises(ValueError, match="reachability_violation_rejection_rate"):
        exp.validate_artifact(bad_rate)

    bad_gate = deepcopy(artifact)
    bad_gate["gated_on_resource_accounted_csl"] = False
    with pytest.raises(ValueError, match="gated_on_resource_accounted_csl"):
        exp.validate_artifact(bad_gate)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.491"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_accepted_count = deepcopy(artifact)
    bad_accepted_count["accepted_promotion_count"] += 1
    with pytest.raises(ValueError, match="accepted_promotion_count"):
        exp.validate_artifact(bad_accepted_count)

    bad_rejected_count = deepcopy(artifact)
    bad_rejected_count["rejected_retained_count"] += 1
    with pytest.raises(ValueError, match="rejected_retained_count"):
        exp.validate_artifact(bad_rejected_count)

    bad_no_accepted = deepcopy(artifact)
    bad_no_accepted["accepted_promotion_count"] = 0
    bad_no_accepted["promotion_candidate_count"] = bad_no_accepted["rejected_retained_count"]
    bad_no_accepted["accepted_promotions"] = []
    with pytest.raises(ValueError, match="accepted_promotion_count"):
        exp.validate_artifact(bad_no_accepted)

    bad_live_effect = deepcopy(artifact)
    bad_live_effect["accepted_promotions"][0]["live_routing_effect"] = False
    with pytest.raises(ValueError, match="live_routing_effect"):
        exp.validate_artifact(bad_live_effect)

    bad_routing = deepcopy(artifact)
    bad_routing["routing_report"]["rejected_fragment_routing_influence_count"] = 1
    with pytest.raises(ValueError, match="rejected_fragment_routing_influence_count"):
        exp.validate_artifact(bad_routing)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    evaluation = exp.evaluate_uncertainty_gated_promotion(root=REPO)
    trace_index = exp.build_trace_index(evaluation["source_controller_artifact"])
    provenance_missing = deepcopy(evaluation["accepted_promotions"][0])
    provenance_missing["decision_inputs"]["provenance_verified"] = False
    rescored = exp.score_promotion_candidate(provenance_missing, trace_index)

    assert exp._candidate_family("new_unclassified_control") == "ambiguous"
    assert "rollback_or_provenance_unavailable" in rescored["promotion_decision"][
        "rejection_reasons"
    ]
    assert exp._rate(1, 0) == 0.0
    assert exp._json_ready(Path("results/example.json")) == "results/example.json"
