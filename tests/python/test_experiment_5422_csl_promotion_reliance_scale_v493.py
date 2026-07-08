"""Tests for Exp5422 gated CSL promotion reliance scale-up.

Spec refs: REQ-LEARN-5422,
SCENARIO-LEARN-5422-THRESHOLDS, SCENARIO-LEARN-5422-QUARANTINE,
SCENARIO-LEARN-5422-ROLLBACK, SCENARIO-LEARN-5422-NO-WEIGHT-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5422_csl_promotion_reliance_scale_v493 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5422_spec_declares_promotion_reliance_contract() -> None:
    """REQ-LEARN-5422: OpenSpec anchors the scale-up contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5422") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5422",
        "SCENARIO-LEARN-5422-THRESHOLDS",
        "SCENARIO-LEARN-5422-QUARANTINE",
        "SCENARIO-LEARN-5422-ROLLBACK",
        "SCENARIO-LEARN-5422-NO-WEIGHT-MUTATION",
        str(exp.RESULT_RELATIVE_PATH),
        "reachable fragments, unsupported fragments, stale fragments, poisoned fragments",
        "ambiguous evidence reliance",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5422_candidates_cover_pool_and_precondition() -> None:
    """REQ-LEARN-5422-1/2/3: source readiness, pool scale, and thresholds exist."""

    evaluation = exp.evaluate_csl_promotion_reliance_scale(root=REPO)
    candidates = evaluation["promotion_candidates"]
    families = {row["candidate_family"] for row in candidates}

    assert evaluation["source_readiness"] == {
        "exp5421_evidence_reliance_csl_ready": True,
        "exp5409_uncertainty_gated_promotion_ready": True,
    }
    assert families >= exp.REQUIRED_CANDIDATE_FAMILIES
    assert evaluation["candidate_fragment_count"] == len(candidates)
    assert evaluation["candidate_fragment_count"] > 9
    assert evaluation["promoted_fragment_count"] == len(evaluation["promoted_fragments"])
    assert evaluation["rejected_fragment_count"] == len(evaluation["rejected_fragments"])
    assert evaluation["abstained_fragment_count"] == len(evaluation["abstained_fragments"])
    assert evaluation["promotion_thresholds"] == exp.THRESHOLDS
    assert evaluation["reliance_drift_threshold"] == exp.RELIANCE_DRIFT_THRESHOLD
    assert evaluation["accepted_risk_threshold"] == exp.ACCEPTED_RISK_THRESHOLD

    for row in candidates:
        assert row["record_type"] == "csl_promotion_scale_candidate"
        assert row["raw_episode_ids"]
        assert row["audit_retained"] is True
        assert row["fragment_kind"] in {"learned_memory", "world_fragment"}
        assert 0.0 <= row["uncertainty_score"] <= 1.0
        assert 0.0 <= row["accepted_risk"] <= 1.0
        assert row["promotion_status"] in {"promoted", "rejected", "abstained"}
        assert row["promotion_status"] == row["promotion_decision"]["status"]
        assert row["active_for_routing"] is (row["promotion_status"] == "promoted")
        assert isinstance(row["threshold_results"]["uncertainty"], bool)
        assert isinstance(row["threshold_results"]["grounding"], bool)
        assert isinstance(row["threshold_results"]["accepted_risk"], bool)
        assert isinstance(row["threshold_results"]["resource_savings"], bool)
        assert isinstance(row["threshold_results"]["rollback"], bool)
        assert isinstance(row["threshold_results"]["reliance_drift"], bool)


def test_scenario_learn_5422_thresholds_gate_promotion_rejection_and_abstention() -> None:
    """SCENARIO-LEARN-5422-THRESHOLDS: only all-pass candidates route."""

    evaluation = exp.evaluate_csl_promotion_reliance_scale(root=REPO)

    assert {row["candidate_family"] for row in evaluation["promoted_fragments"]} == {
        "reachable"
    }
    assert {row["candidate_family"] for row in evaluation["rejected_fragments"]} >= {
        "stale",
        "poisoned",
    }
    assert {row["candidate_family"] for row in evaluation["abstained_fragments"]} >= {
        "unsupported",
        "ambiguous_evidence_reliance",
    }

    for row in evaluation["promoted_fragments"]:
        assert all(row["threshold_results"].values())
        assert row["routing_influence"] > 0
        assert row["promotion_decision"]["reasons"] == ["all_thresholds_passed"]

    for row in evaluation["rejected_fragments"]:
        assert not all(row["threshold_results"].values())
        assert row["routing_influence"] == 0
        assert row["promotion_decision"]["reasons"]
        assert any(
            reason
            in {
                "grounding_not_allowed",
                "accepted_risk_exceeds_threshold",
                "rollback_unavailable",
                "resource_savings_below_threshold",
            }
            for reason in row["promotion_decision"]["reasons"]
        )

    for row in evaluation["abstained_fragments"]:
        assert row["routing_influence"] == 0
        assert any(
            reason
            in {
                "uncertainty_exceeds_threshold",
                "reliance_drift_exceeds_threshold",
                "insufficient_support",
                "ambiguous_evidence_reliance",
            }
            for reason in row["promotion_decision"]["reasons"]
        )

    promoted = deepcopy(evaluation["promoted_fragments"][0])
    promoted["reliance_drift"] = exp.RELIANCE_DRIFT_THRESHOLD + 0.1
    rescored = exp.score_candidate(promoted)

    assert rescored["promotion_status"] == "abstained"
    assert "reliance_drift_exceeds_threshold" in rescored["promotion_decision"]["reasons"]
    assert rescored["active_for_routing"] is False

    low_resource = deepcopy(evaluation["promoted_fragments"][0])
    low_resource["resource_savings"] = exp.MIN_RESOURCE_SAVINGS - 1.0
    rescored_low_resource = exp.score_candidate(low_resource)

    assert rescored_low_resource["promotion_status"] == "rejected"
    assert "resource_savings_below_threshold" in rescored_low_resource[
        "promotion_decision"
    ]["reasons"]

    no_rollback = deepcopy(evaluation["promoted_fragments"][0])
    no_rollback["rollback_available"] = False
    rescored_no_rollback = exp.score_candidate(no_rollback)

    assert rescored_no_rollback["promotion_status"] == "rejected"
    assert "rollback_unavailable" in rescored_no_rollback["promotion_decision"]["reasons"]


def test_scenario_learn_5422_rejected_and_abstained_fragments_are_quarantined() -> None:
    """SCENARIO-LEARN-5422-QUARANTINE: inactive fragments cannot silently route."""

    evaluation = exp.evaluate_csl_promotion_reliance_scale(root=REPO)
    routing = evaluation["routing_report"]
    promoted_ids = {row["fragment_id"] for row in evaluation["promoted_fragments"]}
    rejected_ids = {row["fragment_id"] for row in evaluation["rejected_fragments"]}
    abstained_ids = {row["fragment_id"] for row in evaluation["abstained_fragments"]}

    assert promoted_ids == set(routing["active_fragment_ids"])
    assert rejected_ids == set(routing["quarantined_rejected_fragment_ids"])
    assert abstained_ids == set(routing["retained_abstained_fragment_ids"])
    assert promoted_ids.isdisjoint(rejected_ids | abstained_ids)
    assert routing["rejected_fragment_routing_influence_count"] == 0
    assert routing["abstained_fragment_routing_influence_count"] == 0
    assert routing["routing_effect_row_count"] > 0
    assert evaluation["rejected_fragments_quarantined"] is True


def test_scenario_learn_5422_rollback_restores_active_sidecar() -> None:
    """SCENARIO-LEARN-5422-ROLLBACK: injected bad routing state is reversible."""

    evaluation = exp.evaluate_csl_promotion_reliance_scale(root=REPO)
    rollback = evaluation["rollback_audit"]

    assert rollback == {
        "bad_fragment_id": "frag5422-poisoned-rollback-probe",
        "injected_into_active_routing": True,
        "rollback_removed_from_active_routing": True,
        "prior_active_sidecar_restored": True,
        "retained_audit_record_after_rollback": True,
        "rollback_success": True,
    }
    assert evaluation["rollback_verified"] is True


def test_scenario_learn_5422_no_weight_mutation_boundary() -> None:
    """SCENARIO-LEARN-5422-NO-WEIGHT-MUTATION: promotion is sidecar-only."""

    evaluation = exp.evaluate_csl_promotion_reliance_scale(root=REPO)

    assert evaluation["no_weight_mutation"] is True
    assert evaluation["weight_mutation_receipt"] == {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "controller_promotion_reliance_sidecar_only",
    }


def test_req_learn_5422_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5422-6: run() writes the required terminal artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        tests_run=exp.default_tests_run(),
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["candidate_fragment_count"] == len(artifact["promotion_candidates"])
    assert artifact["promoted_fragment_count"] == len(artifact["promoted_fragments"])
    assert artifact["rejected_fragment_count"] == len(artifact["rejected_fragments"])
    assert artifact["abstained_fragment_count"] == len(artifact["abstained_fragments"])
    assert artifact["grounding_preserved"] is True
    assert artifact["reliance_drift_threshold"] == exp.RELIANCE_DRIFT_THRESHOLD
    assert artifact["accepted_risk_threshold"] == exp.ACCEPTED_RISK_THRESHOLD
    assert artifact["rollback_verified"] is True
    assert artifact["rejected_fragments_quarantined"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["csl_promotion_reliance_scale_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5422_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5422-6: checked-in result is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["csl_promotion_reliance_scale_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_req_learn_5422_blocked_artifact_reports_missing_tests() -> None:
    """REQ-LEARN-5422-6: missing test evidence keeps readiness blocked."""

    artifact = exp.build_artifact(root=REPO, tests_run=[])

    assert artifact["status"] == "blocked"
    assert artifact["csl_promotion_reliance_scale_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "tests_recorded" in artifact["readiness_checks"]["failed_checks"]
    exp.validate_artifact(artifact)


def test_req_learn_5422_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5422-6: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(root=REPO, tests_run=exp.default_tests_run())

    bad_missing = deepcopy(artifact)
    bad_missing.pop("candidate_fragment_count")
    with pytest.raises(ValueError, match="candidate_fragment_count"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["candidate_fragment_count"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_bool = deepcopy(artifact)
    bad_bool["grounding_preserved"] = "true"
    with pytest.raises(ValueError, match="grounding_preserved"):
        exp.validate_artifact(bad_bool)

    bad_int = deepcopy(artifact)
    bad_int["candidate_fragment_count"] = True
    with pytest.raises(ValueError, match="candidate_fragment_count"):
        exp.validate_artifact(bad_int)

    bad_numeric = deepcopy(artifact)
    bad_numeric["accepted_risk_threshold"] = {"value": exp.ACCEPTED_RISK_THRESHOLD}
    with pytest.raises(ValueError, match="accepted_risk_threshold"):
        exp.validate_artifact(bad_numeric)

    bad_drift_threshold = deepcopy(artifact)
    bad_drift_threshold["reliance_drift_threshold"] = exp.RELIANCE_DRIFT_THRESHOLD + 0.01
    with pytest.raises(ValueError, match="reliance_drift_threshold"):
        exp.validate_artifact(bad_drift_threshold)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_ready = deepcopy(artifact)
    bad_ready["csl_promotion_reliance_scale_ready"] = False
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_ready)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.492"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    for field in (
        "grounding_preserved",
        "rollback_verified",
        "rejected_fragments_quarantined",
        "no_weight_mutation",
    ):
        bad = deepcopy(artifact)
        bad[field] = False
        with pytest.raises(ValueError, match=field):
            exp.validate_artifact(bad)

    bad_count = deepcopy(artifact)
    bad_count["promoted_fragment_count"] += 1
    with pytest.raises(ValueError, match="promoted_fragment_count"):
        exp.validate_artifact(bad_count)

    bad_no_promoted = deepcopy(artifact)
    bad_no_promoted["promoted_fragment_count"] = 0
    with pytest.raises(ValueError, match="promoted_fragment_count"):
        exp.validate_artifact(bad_no_promoted)

    bad_rejected_count = deepcopy(artifact)
    bad_rejected_count["rejected_fragment_count"] = 0
    with pytest.raises(ValueError, match="rejected_fragment_count"):
        exp.validate_artifact(bad_rejected_count)

    bad_abstained_count = deepcopy(artifact)
    bad_abstained_count["abstained_fragment_count"] = 0
    with pytest.raises(ValueError, match="abstained_fragment_count"):
        exp.validate_artifact(bad_abstained_count)

    bad_rejected_route = deepcopy(artifact)
    bad_rejected_route["routing_report"]["rejected_fragment_routing_influence_count"] = 1
    with pytest.raises(ValueError, match="rejected_fragment_routing_influence_count"):
        exp.validate_artifact(bad_rejected_route)

    bad_abstained_route = deepcopy(artifact)
    bad_abstained_route["routing_report"]["abstained_fragment_routing_influence_count"] = 1
    with pytest.raises(ValueError, match="abstained_fragment_routing_influence_count"):
        exp.validate_artifact(bad_abstained_route)

    bad_promoted_route = deepcopy(artifact)
    bad_promoted_route["promoted_fragments"][0]["routing_influence"] = 0
    with pytest.raises(ValueError, match="routing_influence"):
        exp.validate_artifact(bad_promoted_route)

    bad_no_tests = deepcopy(artifact)
    bad_no_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_no_tests)

    assert exp._json_ready(Path("results/example.json")) == "results/example.json"
