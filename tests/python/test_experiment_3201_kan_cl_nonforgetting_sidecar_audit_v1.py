"""Tests for Exp 3201 FR-11 KAN-CL nonforgetting sidecar audit.

Spec refs: REQ-LEARN-3201, SCENARIO-LEARN-3201,
SCENARIO-LEARN-3201-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_kan_cl_nonforgetting_sidecar_audit_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trace(
    row_id: str,
    role: str,
    key: str,
    *,
    exact_label: str = "EXACT_ACCEPT",
    expected_action: str = "accept",
    decision: str = "answer",
    routing: str = "verify_then_answer",
    consistency: str = "consistent",
    changed: bool = False,
    family: str = "modular_balance",
) -> dict[str, Any]:
    return {
        "trace_id": f"trace-{role}-{row_id}",
        "row_id": row_id,
        "replay_role": role,
        "fixture_family": family,
        "source_artifact": "results/unit_source.json",
        "initial_answer": f"Row {row_id}",
        "verification_query": f"Verify {row_id}",
        "consistency_judgment": consistency,
        "answer_abstain_decision": decision,
        "exact_label": exact_label,
        "routing_outcome": routing,
        "expected_action": expected_action,
        "ledger_action": expected_action,
        "observed_action": expected_action,
        "observed_action_changed": changed,
        "historical_exact_evidence_key": key,
        "redundant_check_suppressed": routing == "skip_redundant_recheck",
    }


def _exp3200_payload(
    *,
    traces: list[dict[str, Any]] | None = None,
    fresh_live_calls: int = 0,
) -> dict[str, Any]:
    trace_records = traces or [
        _trace("safe-accept", "heldout", "key-safe"),
        _trace(
            "known-reject",
            "heldout",
            "key-reject",
            exact_label="INVALID",
            expected_action="reject",
            decision="abstain",
            routing="abstain_or_escalate",
        ),
        _trace("safe-accept", "drift", "key-safe", routing="skip_redundant_recheck"),
        _trace("safe-accept", "negative_control", "key-safe", routing="skip_redundant_recheck"),
    ]
    return {
        "artifact": "experiment_3200_fr11_verify_trace_memory_controller_v1",
        "schema_version": "1.0",
        "experiment_id": "experiment_3200_fr11_verify_trace_memory_controller_v1",
        "trace_count": len(trace_records),
        "heldout_row_count": sum(row["replay_role"] == "heldout" for row in trace_records),
        "drift_row_count": sum(row["replay_role"] == "drift" for row in trace_records),
        "negative_control_row_count": sum(
            row["replay_role"] == "negative_control" for row in trace_records
        ),
        "negative_control_regression_count": 0,
        "promotion_allowed": True,
        "model_weight_update_performed": False,
        "trace_records": trace_records,
        "inference_substrate": {
            "controller_memory_replay_only": True,
            "trace_memory_policy_only": True,
            "uses_checked_in_artifacts_only": True,
            "executes_live_model_inference": False,
            "fresh_live_inference_calls": fresh_live_calls,
            "model_weight_learning": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
            "base_model_weights_updated": False,
            "kan_model_weight_training": False,
            "hidden_state_mutation_claimed": False,
        },
        "honest_verdict": "complete: unit trace memory",
    }


def _exp3187_payload(
    *,
    negative_control_regression_count: int = 0,
    drift_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_3187_fr11_cross_environment_drift_replay_v1",
        "fr11_cross_environment_drift_replay_v1_ready": True,
        "promotion_allowed": True,
        "rollback_triggered": bool(negative_control_regression_count or drift_cases),
        "no_model_weight_update_claimed": True,
        "heldout_row_count": 2,
        "cross_environment_row_count": 1,
        "negative_control_regression_count": negative_control_regression_count,
        "negative_control_regressions": [
            {"row_id": "source-negative", "reason": "source negative regression"}
        ]
        if negative_control_regression_count
        else [],
        "drift_cases": drift_cases or [],
        "inference_substrate": {
            "controller_memory_replay_only": True,
            "uses_checked_in_artifacts_only": True,
            "executes_live_model_inference": False,
            "fresh_live_inference_calls": 0,
            "model_weight_learning": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
            "base_model_weights_updated": False,
            "kan_model_weight_training": False,
            "hidden_state_mutation_claimed": False,
        },
        "honest_verdict": "complete: unit drift replay",
    }


def _write_sources(
    root: Path,
    *,
    exp3200: Mapping[str, Any] | None = None,
    exp3187: Mapping[str, Any] | None = None,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no sidecar promotion\n", encoding="utf-8")
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3201\nSCENARIO-LEARN-3201\nSCENARIO-LEARN-3201-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3200_REL_PATH, exp3200 or _exp3200_payload())
    _write_json(root, mod.EXP3187_REL_PATH, exp3187 or _exp3187_payload())


def test_req_learn_3201_spec_anchor_exists() -> None:
    """REQ-LEARN-3201: OpenSpec declares the sidecar audit artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3201" in spec
    assert "SCENARIO-LEARN-3201" in spec
    assert "SCENARIO-LEARN-3201-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "audit_metric_schema" in spec
    assert "sidecar_promotion_allowed" in spec
    assert "model_weight_update_performed=false" in spec


def test_req_learn_3201_metric_schema_and_replay_metrics() -> None:
    """REQ-LEARN-3201-2/3/4: exact labels define replay and locality metrics."""

    exp3200 = _exp3200_payload()
    audit = mod.audit_traces(exp3200["trace_records"], _exp3187_payload())
    schema = mod.audit_metric_schema()

    assert schema["schema_id"] == "carnot.fr11.kan_cl_nonforgetting_audit_metrics.v1"
    assert {"exact_label_consistency", "routing_bin_retention", "locality_boundary"} <= set(
        schema["metrics"]
    )
    assert audit["heldout_replay_count"] == 2
    assert audit["drift_replay_count"] == 1
    assert audit["negative_control_replay_count"] == 1
    assert audit["heldout_regression_count"] == 0
    assert audit["drift_regression_count"] == 0
    assert audit["negative_control_regression_count"] == 0
    assert audit["locality_violation_count"] == 0
    assert audit["routing_bin_summary"]["answer_path"]["count"] == 3
    assert audit["routing_bin_summary"]["abstain_path"]["count"] == 1


def test_scenario_learn_3201_writes_complete_sidecar_audit(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3201: sidecar audit preserves boundaries without promotion."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.25,
        tests_run=["REQ-LEARN-3201 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["heldout_replay_count"] == 2
    assert artifact["drift_replay_count"] == 1
    assert artifact["negative_control_regression_count"] == 0
    assert artifact["locality_violation_count"] == 0
    assert artifact["rollback_triggered"] is False
    assert artifact["model_weight_update_performed"] is False
    assert artifact["sidecar_promotion_allowed"] is False
    assert artifact["tests_run"] == ["REQ-LEARN-3201 focused"]
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "sidecar_promotion_allowed=false" in artifact["honest_verdict"]
    assert artifact["inference_substrate"]["fresh_live_inference_calls"] == 0
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3201_rollback_for_negative_and_locality_violations(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3201-BLOCKED: regressions and locality breaches trigger rollback."""

    negative_traces = [
        _trace("safe-accept", "heldout", "key-safe"),
        _trace("safe-accept", "drift", "key-safe", routing="skip_redundant_recheck"),
        _trace(
            "bad-negative",
            "negative_control",
            "key-bad-negative",
            exact_label="INVALID",
            expected_action="reject",
            decision="answer",
            routing="verify_then_answer",
        ),
    ]
    _write_sources(tmp_path, exp3200=_exp3200_payload(traces=negative_traces))
    regressed = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert regressed["negative_control_regression_count"] == 1
    assert regressed["rollback_triggered"] is True
    assert regressed["sidecar_promotion_allowed"] is False
    assert regressed["rollback_reasons"] == ["negative_control_regression"]
    mod.validate_artifact(regressed)

    locality_traces = [
        _trace("safe-accept", "heldout", "key-safe"),
        _trace(
            "safe-accept",
            "drift",
            "key-safe",
            exact_label="INVALID",
            expected_action="reject",
            decision="abstain",
            routing="abstain_or_escalate",
        ),
        _trace("safe-accept", "negative_control", "key-safe", routing="skip_redundant_recheck"),
    ]
    _write_sources(tmp_path, exp3200=_exp3200_payload(traces=locality_traces))
    violated = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)

    assert violated["locality_violation_count"] == 1
    assert violated["rollback_triggered"] is True
    assert violated["rollback_reasons"] == ["drift_regression", "locality_violation"]
    assert violated["locality_violations"][0]["historical_exact_evidence_key"] == "key-safe"
    mod.validate_artifact(violated)


def test_req_learn_3201_blocked_and_helper_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3201-1/6: unsafe sources and overclaims fail closed."""

    missing = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= missing.keys()
    assert missing["heldout_replay_count"] == 0
    assert missing["rollback_triggered"] is True
    assert missing["blocked_reason"] == "exp3200_missing_or_not_terminal"
    assert missing["model_weight_update_performed"] is False
    assert missing["sidecar_promotion_allowed"] is False
    mod.validate_artifact(missing)

    unsafe = _exp3200_payload(fresh_live_calls=1)
    _write_sources(tmp_path, exp3200=unsafe)
    blocked = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    assert blocked["blocked_reason"] == "exp3200_live_inference_or_weight_update_claimed"
    assert blocked["rollback_triggered"] is True
    mod.validate_artifact(blocked)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.rows_from_trace_payload({"trace_records": "bad"}) == []
    assert mod.rows_from_trace_payload({"trace_records": [None, {"row_id": "ok"}]}) == [
        {"row_id": "ok"}
    ]
    assert mod.exact_action_for_label("VALID", "reject") == "accept"
    assert mod.exact_action_for_label("INVALID", "accept") == "reject"
    assert mod.exact_action_for_label("", "reject") == "reject"
    assert (
        mod.route_bin_for(
            {"answer_abstain_decision": "answer", "routing_outcome": "skip_redundant_recheck"}
        )
        == "answer_path"
    )
    assert mod.route_bin_for({"answer_abstain_decision": "abstain"}) == "abstain_path"
    assert mod.route_bin_for({"answer_abstain_decision": "x", "routing_outcome": "y"}) == "y"
    assert mod.source_blocker({"exp3200": _exp3200_payload(), "exp3187": {}}) == (
        "exp3187_missing_or_not_ready"
    )
    bad_weight = _exp3200_payload()
    bad_weight["model_weight_update_performed"] = True
    assert mod.source_blocker({"exp3200": bad_weight, "exp3187": _exp3187_payload()}) == (
        "exp3200_model_weight_update_claimed"
    )
    bad_exp3187_claim = _exp3187_payload()
    bad_exp3187_claim["no_model_weight_update_claimed"] = False
    assert mod.source_blocker({"exp3200": _exp3200_payload(), "exp3187": bad_exp3187_claim}) == (
        "exp3187_model_weight_update_claimed"
    )
    bad_exp3187_live = _exp3187_payload()
    bad_exp3187_live["inference_substrate"]["fresh_live_inference_calls"] = 1
    assert mod.source_blocker({"exp3200": _exp3200_payload(), "exp3187": bad_exp3187_live}) == (
        "exp3187_live_inference_or_weight_update_claimed"
    )
    assert mod.source_claims_live_or_mutation({"inference_substrate": []}) is True
    assert mod.source_claims_live_or_mutation(
        {"inference_substrate": {"model_weight_training": True}}
    )
    assert mod.source_claims_live_or_mutation(
        {"inference_substrate": {"fresh_live_inference_calls": 1}}
    )
    assert (
        mod.trace_regression(_trace("inconsistent", "heldout", "key-i", consistency="bad"))[
            "reason"
        ]
        == "inconsistent_exact_replay"
    )
    assert mod.trace_regression(_trace("changed", "heldout", "key-c", changed=True))["reason"] == (
        "observed_action_changed"
    )
    assert (
        mod.trace_regression(_trace("accept-abstain", "heldout", "key-aa", decision="abstain"))[
            "reason"
        ]
        == "exact_accept_not_answered"
    )
    assert mod.rollback_reasons_for(1, 0, 0, 0, []) == ["heldout_regression"]
    assert mod.sha256_file(tmp_path / "absent.txt") is None


def test_req_learn_3201_validation_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-LEARN-3201-5/6: validation rejects rollback and authority overclaims."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=5.0)

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="model_weight_update_performed"):
        mod.validate_artifact(artifact | {"model_weight_update_performed": True})
    with pytest.raises(ValueError, match="sidecar_promotion_allowed"):
        mod.validate_artifact(artifact | {"sidecar_promotion_allowed": True})
    with pytest.raises(ValueError, match="audit_metric_schema"):
        mod.validate_artifact(artifact | {"audit_metric_schema": []})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": []})
    live_substrate = dict(artifact["inference_substrate"])
    live_substrate["fresh_live_inference_calls"] = 1
    with pytest.raises(ValueError, match="fresh_live_inference_calls"):
        mod.validate_artifact(artifact | {"inference_substrate": live_substrate})
    mutation_substrate = dict(artifact["inference_substrate"])
    mutation_substrate["model_weight_training"] = True
    with pytest.raises(ValueError, match="model mutation flags"):
        mod.validate_artifact(artifact | {"inference_substrate": mutation_substrate})
    with pytest.raises(ValueError, match="rollback_triggered"):
        mod.validate_artifact(
            artifact | {"negative_control_regression_count": 1, "rollback_triggered": False}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
