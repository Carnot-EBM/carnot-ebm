"""Tests for Exp5421 evidence-reliance CSL hidden-forgetting diagnostic.

Spec refs: REQ-LEARN-5421,
SCENARIO-LEARN-5421-DRIFT, SCENARIO-LEARN-5421-RAW-RETENTION,
SCENARIO-LEARN-5421-SAFETY, SCENARIO-LEARN-5421-ROLLBACK.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5421_evidence_reliance_csl_v493 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5421_spec_declares_evidence_reliance_contract() -> None:
    """REQ-LEARN-5421: OpenSpec anchors the evidence-reliance diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5421") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5421",
        "SCENARIO-LEARN-5421-DRIFT",
        "SCENARIO-LEARN-5421-RAW-RETENTION",
        "SCENARIO-LEARN-5421-SAFETY",
        "SCENARIO-LEARN-5421-ROLLBACK",
        str(exp.RESULT_RELATIVE_PATH),
        "stale, poisoned, underspecified, and distribution-shift",
        "deterministic verifiers",
        "SHALL NOT load, fine-tune, write, or mutate model weights or adapter weights",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5421_paired_episodes_track_reliance_fields() -> None:
    """REQ-LEARN-5421-2: paired rows expose routing, evidence, and costs."""

    evaluation = exp.evaluate_evidence_reliance_csl(root=REPO)
    rows = evaluation["paired_episodes"]
    families = {row["episode_family"] for row in rows}

    assert evaluation["episode_count"] == len(rows)
    assert families >= exp.REQUIRED_EPISODE_FAMILIES
    assert evaluation["raw_episodes_retained"] is True
    assert evaluation["quality_preserved"] is True
    assert evaluation["accuracy_before_rate"] == 1.0
    assert evaluation["accuracy_after_rate"] == 1.0
    assert evaluation["resource_delta"] > 0.0
    assert evaluation["verifier_cost_delta"] > 0.0

    for row in rows:
        assert row["answer_correct_before"] is True
        assert row["answer_correct_after"] is True
        assert row["surface_answer_before"] == row["surface_answer_after"]
        assert row["raw_episode_ids"]
        assert all(item.startswith("raw") for item in row["raw_episode_ids"])
        assert row["raw_episode_receipts"]
        assert all(
            receipt["raw_payload_checksum"].startswith("sha256:")
            for receipt in row["raw_episode_receipts"]
        )
        assert sum(row["influence_shares_before"].values()) == 100
        assert sum(row["influence_shares_after"].values()) == 100
        assert row["routing_decision_before"] in exp.ROUTING_DECISIONS
        assert row["routing_decision_after"] in exp.ROUTING_DECISIONS
        assert row["grounding_source_before"]
        assert row["grounding_source_after"]
        assert row["verifier_dependency_before"]
        assert row["verifier_dependency_after"]
        assert row["constraint_evidence_before"]
        assert row["constraint_evidence_after"]
        assert row["evidence_reliance_label_before"]
        assert row["evidence_reliance_label_after"]
        assert row["resource_cost_before"] >= 0
        assert row["resource_cost_after"] >= 0
        assert row["verifier_calls_before"] >= 0
        assert row["verifier_calls_after"] >= 0


def test_scenario_learn_5421_detects_hidden_forgetting_under_stable_accuracy() -> None:
    """SCENARIO-LEARN-5421-DRIFT: stable accuracy still reports drift."""

    evaluation = exp.evaluate_evidence_reliance_csl(root=REPO)
    drifted = [
        row
        for row in evaluation["paired_episodes"]
        if row["surface_success_stable"]
        and row["evidence_reliance_drift"] >= exp.HIDDEN_FORGETTING_THRESHOLD
    ]

    assert drifted
    assert evaluation["reliance_drift_metric"] >= exp.HIDDEN_FORGETTING_THRESHOLD
    assert evaluation["hidden_forgetting_detected"] is True
    assert all(row["answer_correct_before"] and row["answer_correct_after"] for row in drifted)
    assert any(
        row["grounding_source_before"] != row["grounding_source_after"]
        or row["evidence_reliance_label_before"] != row["evidence_reliance_label_after"]
        for row in drifted
    )


def test_scenario_learn_5421_unsafe_and_uncertain_reliance_is_deflected() -> None:
    """SCENARIO-LEARN-5421-SAFETY: unsafe memory cannot route unverified."""

    evaluation = exp.evaluate_evidence_reliance_csl(root=REPO)
    rows = evaluation["paired_episodes"]
    unsafe = [
        row
        for row in rows
        if row["episode_family"] in {"stale", "poisoned", "underspecified", "distribution_shift"}
    ]
    stale_poison = [
        row for row in rows if row["episode_family"] in {"stale", "poisoned"}
    ]

    assert unsafe
    assert stale_poison
    assert evaluation["stale_poison_deflection_rate"] == 1.0
    assert evaluation["uncertain_reliance_deflection_rate"] == 1.0
    assert all(row["promoted_after_learning"] is False for row in unsafe)
    assert all(
        row["controller_action_after"]
        in {"route_to_deterministic_verifier", "abstain", "retain_non_promoted_evidence"}
        for row in unsafe
    )
    assert all(row["active_learned_memory_routing_after"] is False for row in stale_poison)
    assert all(row["raw_episode_retained"] is True for row in unsafe)


def test_scenario_learn_5421_raw_retention_and_rollback_are_auditable() -> None:
    """SCENARIO-LEARN-5421-RAW-RETENTION/ROLLBACK: audit and recovery hold."""

    evaluation = exp.evaluate_evidence_reliance_csl(root=REPO)
    rollback = evaluation["rollback_audit"]

    assert rollback == {
        "bad_fragment_id": "frag5421-poisoned-reliance-drift",
        "injected_into_active_routing": True,
        "rollback_removed_from_active_routing": True,
        "prior_routing_restored": True,
        "retained_audit_record_after_rollback": True,
        "rollback_success": True,
    }
    assert evaluation["rollback_verified"] is True
    assert evaluation["raw_episodes_retained"] is True
    assert evaluation["weight_mutation_receipt"] == {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "controller_evidence_reliance_sidecar_only",
    }


def test_req_learn_5421_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5421-6: run() writes the required terminal artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        tests_run=exp.default_tests_run(),
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["episode_count"] == len(artifact["paired_episodes"])
    assert artifact["raw_episodes_retained"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["rollback_verified"] is True
    assert artifact["quality_preserved"] is True
    assert artifact["resource_delta"] > 0.0
    assert artifact["verifier_cost_delta"] > 0.0
    assert artifact["reliance_drift_metric"] >= exp.HIDDEN_FORGETTING_THRESHOLD
    assert artifact["hidden_forgetting_detected"] is True
    assert artifact["stale_poison_deflection_rate"] == 1.0
    assert artifact["evidence_reliance_csl_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5421_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5421-6: checked-in result is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["evidence_reliance_csl_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_req_learn_5421_blocked_artifact_reports_failed_readiness() -> None:
    """REQ-LEARN-5421-6: missing test evidence keeps readiness blocked."""

    artifact = exp.build_artifact(root=REPO, tests_run=[])

    assert artifact["status"] == "blocked"
    assert artifact["evidence_reliance_csl_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "tests_recorded" in artifact["readiness_checks"]["failed_checks"]
    exp.validate_artifact(artifact)


def test_req_learn_5421_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5421-6: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(root=REPO, tests_run=exp.default_tests_run())

    bad_missing = deepcopy(artifact)
    bad_missing.pop("episode_count")
    with pytest.raises(ValueError, match="episode_count"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["episode_count"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_bool = deepcopy(artifact)
    bad_bool["raw_episodes_retained"] = "true"
    with pytest.raises(ValueError, match="raw_episodes_retained"):
        exp.validate_artifact(bad_bool)

    bad_int = deepcopy(artifact)
    bad_int["episode_count"] = True
    with pytest.raises(ValueError, match="episode_count"):
        exp.validate_artifact(bad_int)

    bad_numeric = deepcopy(artifact)
    bad_numeric["resource_delta"] = {"value": 1.0}
    with pytest.raises(ValueError, match="resource_delta"):
        exp.validate_artifact(bad_numeric)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_ready = deepcopy(artifact)
    bad_ready["evidence_reliance_csl_ready"] = False
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

    bad_resource = deepcopy(artifact)
    bad_resource["resource_delta"] = 0.0
    with pytest.raises(ValueError, match="resource_delta"):
        exp.validate_artifact(bad_resource)

    for field in (
        "raw_episodes_retained",
        "no_weight_mutation",
        "rollback_verified",
        "quality_preserved",
        "hidden_forgetting_detected",
    ):
        bad = deepcopy(artifact)
        bad[field] = False
        with pytest.raises(ValueError, match=field):
            exp.validate_artifact(bad)

    bad_rate = deepcopy(artifact)
    bad_rate["stale_poison_deflection_rate"] = 0.5
    with pytest.raises(ValueError, match="stale_poison_deflection_rate"):
        exp.validate_artifact(bad_rate)

    bad_drift = deepcopy(artifact)
    bad_drift["reliance_drift_metric"] = exp.HIDDEN_FORGETTING_THRESHOLD - 0.01
    with pytest.raises(ValueError, match="reliance_drift_metric"):
        exp.validate_artifact(bad_drift)

    bad_count = deepcopy(artifact)
    bad_count["episode_count"] += 1
    with pytest.raises(ValueError, match="episode_count"):
        exp.validate_artifact(bad_count)

    bad_no_tests = deepcopy(artifact)
    bad_no_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_no_tests)
