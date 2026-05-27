"""Tests for Exp 3215 FR-11 evidence-gated trace replay labels.

Spec refs: REQ-LEARN-3215, SCENARIO-LEARN-3215,
SCENARIO-LEARN-3215-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_evidence_gated_trace_replay_controller_v2 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trace(
    row_id: str,
    role: str,
    *,
    exact_label: str = "EXACT_ACCEPT",
    expected_action: str = "accept",
    decision: str = "answer",
    routing: str = "verify_then_answer",
    consistency: str = "consistent",
    suppressed: bool = False,
    changed: bool = False,
    verification_query: str | None = "Verify exact row evidence",
) -> dict[str, Any]:
    row = {
        "trace_id": f"trace-{role}-{row_id}",
        "row_id": row_id,
        "replay_role": role,
        "fixture_family": "unit_family",
        "source_artifact": "results/unit_source.json",
        "initial_answer": f"Row {row_id}",
        "verification_query": verification_query,
        "consistency_judgment": consistency,
        "answer_abstain_decision": decision,
        "exact_label": exact_label,
        "routing_outcome": routing,
        "expected_action": expected_action,
        "ledger_action": expected_action,
        "observed_action": expected_action,
        "observed_action_changed": changed,
        "historical_exact_evidence_key": f"key-{row_id}",
        "redundant_check_suppressed": suppressed,
    }
    return {key: value for key, value in row.items() if value is not None}


def _default_traces() -> list[dict[str, Any]]:
    return [
        _trace("accept-heldout", "heldout"),
        _trace(
            "reject-heldout",
            "heldout",
            exact_label="INVALID",
            expected_action="reject",
            decision="abstain",
            routing="abstain_or_escalate",
        ),
        _trace("accept-drift", "drift", routing="skip_redundant_recheck", suppressed=True),
        _trace(
            "accept-negative",
            "negative_control",
            routing="skip_redundant_recheck",
            suppressed=True,
        ),
    ]


def _exp3200_payload(
    *,
    traces: list[dict[str, Any]] | None = None,
    negative_control_regression_count: int = 0,
    fresh_live_calls: int = 0,
    model_weight_update: bool = False,
) -> dict[str, Any]:
    trace_records = traces or _default_traces()
    return {
        "artifact": "experiment_3200_fr11_verify_trace_memory_controller_v1",
        "schema_version": "1.0",
        "experiment_id": "experiment_3200_fr11_verify_trace_memory_controller_v1",
        "trace_schema": {
            "schema_id": "carnot.fr11.verify_trace_memory_record.v1",
            "schema_version": "1.0",
            "fields": list(mod.REUSED_TRACE_FIELDS),
        },
        "trace_count": len(trace_records),
        "heldout_row_count": sum(row["replay_role"] == "heldout" for row in trace_records),
        "drift_row_count": sum(row["replay_role"] == "drift" for row in trace_records),
        "negative_control_row_count": sum(
            row["replay_role"] == "negative_control" for row in trace_records
        ),
        "negative_control_regression_count": negative_control_regression_count,
        "redundant_check_suppression_count": sum(
            bool(row.get("redundant_check_suppressed")) for row in trace_records
        ),
        "routing_accuracy_delta": 0.142857,
        "promotion_allowed": negative_control_regression_count == 0,
        "model_weight_update_performed": model_weight_update,
        "trace_records": trace_records,
        "inference_substrate": {
            "controller_memory_replay_only": True,
            "trace_memory_policy_only": True,
            "uses_checked_in_artifacts_only": True,
            "executes_live_model_inference": False,
            "fresh_live_inference_calls": fresh_live_calls,
            "model_weight_learning": False,
            "model_weight_training": model_weight_update,
            "model_weight_mutation": False,
            "base_model_weights_updated": False,
            "kan_model_weight_training": False,
            "hidden_state_mutation_claimed": False,
        },
        "honest_verdict": "complete: unit trace memory",
    }


def _exp3201_payload(
    *,
    rollback_triggered: bool = False,
    negative_control_regression_count: int = 0,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1",
        "schema_version": "1.0",
        "experiment_id": "experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1",
        "negative_control_regression_count": negative_control_regression_count,
        "rollback_triggered": rollback_triggered,
        "rollback_reasons": ["negative_control_regression"] if rollback_triggered else [],
        "model_weight_update_performed": False,
        "sidecar_promotion_allowed": False,
        "inference_substrate": {
            "controller_memory_replay_only": True,
            "sidecar_audit_only": True,
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
        "honest_verdict": "complete: unit sidecar audit",
    }


def _write_sources(
    root: Path,
    *,
    exp3200: Mapping[str, Any] | None = None,
    exp3201: Mapping[str, Any] | None = None,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no model-weight update claims\n", encoding="utf-8")
    (root / "research-program.md").write_text("Evidence Over Plans\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "Verifier-rewarded distillation\n", encoding="utf-8"
    )
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3215\nSCENARIO-LEARN-3215\nSCENARIO-LEARN-3215-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3200_REL_PATH, exp3200 or _exp3200_payload())
    _write_json(root, mod.EXP3201_REL_PATH, exp3201 or _exp3201_payload())


def test_req_learn_3215_spec_anchor_exists() -> None:
    """REQ-LEARN-3215: OpenSpec declares evidence-gated replay labels."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3215" in spec
    assert "SCENARIO-LEARN-3215" in spec
    assert "SCENARIO-LEARN-3215-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "evidence_backed_trace_count" in spec
    assert "replay_utility_label_count" in spec
    assert "model_weight_update_claimed=false" in spec


def test_req_learn_3215_labels_only_evidence_backed_traces() -> None:
    """REQ-LEARN-3215-1/2/3: labels require exact evidence, not planned rows."""

    planned_only = _trace("planned-only", "heldout", verification_query=None)
    traces = _default_traces() + [planned_only]
    labels = mod.label_replay_candidates(traces)
    by_row = {label["row_id"]: label for label in labels}

    assert set(mod.REUSED_TRACE_FIELDS) <= set(mod.label_schema()["reused_trace_fields"])
    assert len(labels) == 4
    assert "planned-only" not in by_row
    assert by_row["accept-heldout"]["exact_verifier_outcome"] == "exact_accept_answered"
    assert by_row["reject-heldout"]["exact_verifier_outcome"] == "exact_reject_abstained"
    assert by_row["reject-heldout"]["prior_route_utility"] == "safe_abstain_for_exact_reject"
    assert by_row["accept-drift"]["prior_route_utility"] == "suppress_redundant_recheck"
    assert by_row["accept-drift"]["redundant_check_suppressed"] is True
    assert by_row["accept-drift"]["reward_weight"] == pytest.approx(1.0)
    assert all(label["controller_utility_label_only"] is True for label in labels)
    assert all(label["model_weight_update_claimed"] is False for label in labels)


def test_scenario_learn_3215_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3215: evidence-gated labels promote controller replay only."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.25,
        tests_run=["REQ-LEARN-3215 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["prior_trace_memory_artifact"] == mod.EXP3200_REL_PATH.as_posix()
    assert artifact["trace_count"] == 4
    assert artifact["evidence_backed_trace_count"] == 4
    assert artifact["replay_utility_label_count"] == 4
    assert artifact["redundant_check_suppression_count"] == 2
    assert artifact["heldout_row_count"] == 2
    assert artifact["drift_row_count"] == 1
    assert artifact["routing_improvement_count"] == 3
    assert artifact["negative_control_regression_count"] == 0
    assert artifact["rollback_event_count"] == 0
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["promotion_allowed"] is True
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["tests_run"] == ["REQ-LEARN-3215 focused"]
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "model_weight_update_claimed=false" in artifact["honest_verdict"]
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3215_blocks_planned_only_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3215-BLOCKED: missing evidence or rollback denies promotion."""

    planned_only = _trace("planned-only", "heldout", verification_query=None)
    exp3200 = _exp3200_payload(traces=_default_traces() + [planned_only])
    _write_sources(tmp_path, exp3200=exp3200)
    planned = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert planned["trace_count"] == 5
    assert planned["evidence_backed_trace_count"] == 4
    assert planned["replay_utility_label_count"] == 4
    assert planned["promotion_allowed"] is False
    assert "missing_evidence_backed_labels" in planned["promotion_blockers"]
    mod.validate_artifact(planned)

    _write_sources(tmp_path, exp3201=_exp3201_payload(rollback_triggered=True))
    rollback = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)

    assert rollback["rollback_event_count"] == 1
    assert rollback["promotion_allowed"] is False
    assert rollback["promotion_blockers"] == ["rollback_event"]
    mod.validate_artifact(rollback)


def test_req_learn_3215_blocked_sources_and_validation_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3215-2/5: unsafe sources and overclaims fail closed."""

    missing = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= missing.keys()
    assert missing["trace_count"] == 0
    assert missing["promotion_allowed"] is False
    assert missing["blocked_reason"] == "exp3200_missing_or_not_terminal"
    assert missing["model_weight_update_claimed"] is False
    mod.validate_artifact(missing)

    _write_sources(tmp_path, exp3200=_exp3200_payload(fresh_live_calls=1))
    live_blocked = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    assert live_blocked["blocked_reason"] == "exp3200_live_inference_or_weight_update_claimed"
    mod.validate_artifact(live_blocked)

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
    assert mod.trace_is_evidence_backed(_trace("ok", "heldout")) is True
    assert mod.trace_is_evidence_backed({"trace_id": "planned"}) is False
    assert mod.source_claims_live_or_mutation({"inference_substrate": []}) is True
    assert mod.source_claims_live_or_mutation(
        {"inference_substrate": {"fresh_live_inference_calls": 1}}
    )
    assert mod.detected_model_weight_update({"bad": []}) is False
    assert mod.detected_model_weight_update({"bad": {"model_weight_update_performed": True}})
    assert mod.detected_model_weight_update(
        {"bad": {"inference_substrate": {"model_weight_training": True}}}
    )
    assert mod.rollback_status_for({"rollback_or_retraction_status": "retracted"}, "x") == (
        "retracted"
    )
    assert mod.route_utility_for({}, "x", "rollback_required") == (
        "block_rollback_or_retraction",
        -1.0,
    )
    assert mod.route_utility_for({}, "exact_replay_failed", "none") == (
        "block_failed_exact_replay",
        -1.0,
    )
    assert mod.sha256_file(tmp_path / "absent.txt") is None
    assert mod.source_blocker({"exp3200": _exp3200_payload(), "exp3201": {}}) == (
        "exp3201_missing_or_not_terminal"
    )
    bad_weight = _exp3200_payload(model_weight_update=True)
    assert mod.source_blocker({"exp3200": bad_weight, "exp3201": _exp3201_payload()}) == (
        "exp3200_model_weight_update_claimed"
    )

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=5.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="model_weight_update_claimed"):
        mod.validate_artifact(artifact | {"model_weight_update_claimed": True})
    with pytest.raises(ValueError, match="conductor_file_modified"):
        mod.validate_artifact(artifact | {"conductor_file_modified": True})
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(artifact | {"active_roadmap_modified": True})
    with pytest.raises(ValueError, match="evidence-backed"):
        mod.validate_artifact(
            artifact
            | {
                "promotion_allowed": True,
                "evidence_backed_trace_count": 3,
            }
        )
    with pytest.raises(ValueError, match="negative-control"):
        mod.validate_artifact(
            artifact
            | {
                "promotion_allowed": True,
                "negative_control_regression_count": 1,
            }
        )
    with pytest.raises(ValueError, match="rollback"):
        mod.validate_artifact(
            artifact
            | {
                "promotion_allowed": True,
                "rollback_event_count": 1,
            }
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
