"""Tests for Exp 3216 FR-11 grounded continuation nonforgetting queue.

Spec refs: REQ-LEARN-3216, SCENARIO-LEARN-3216,
SCENARIO-LEARN-3216-FALLBACK.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_grounded_continuation_nonforgetting_queue_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _label(
    row_id: str,
    role: str,
    *,
    exact_label: str = "EXACT_ACCEPT",
    decision: str = "answer",
    consistency: str = "consistent",
    routing: str = "verify_then_answer",
    outcome: str = "exact_accept_answered",
    utility: str = "verified_answer_for_exact_accept",
    reward: float = 0.75,
    rollback: str = "none",
) -> dict[str, Any]:
    return {
        "trace_id": f"trace-{role}-{row_id}",
        "row_id": row_id,
        "replay_role": role,
        "verification_query": f"Does {row_id} satisfy exact replay?",
        "consistency_judgment": consistency,
        "answer_abstain_decision": decision,
        "exact_label": exact_label,
        "exact_action": "reject" if exact_label == "INVALID" else "accept",
        "routing_outcome": routing,
        "exact_verifier_outcome": outcome,
        "prior_route_utility": utility,
        "reward_weight": reward,
        "redundant_check_suppressed": False,
        "rollback_or_retraction_status": rollback,
        "controller_utility_label_only": True,
        "model_weight_update_claimed": False,
    }


def _exp3215_payload(
    *,
    negative_control_regression_count: int = 0,
    terminal: bool = True,
) -> dict[str, Any]:
    labels = [
        _label("heldout-ok", "heldout"),
        _label("drift-ok", "drift", routing="skip_redundant_recheck", utility="suppress_redundant_recheck", reward=1.0),
        _label(
            "drift-stale",
            "drift",
            consistency="inconsistent",
            outcome="exact_replay_failed",
            utility="block_failed_exact_replay",
            reward=-1.0,
            rollback="rollback_required",
        ),
    ]
    return {
        "artifact": "experiment_3215_fr11_evidence_gated_trace_replay_controller_v2",
        "schema_version": "1.0",
        "experiment_id": "experiment_3215_fr11_evidence_gated_trace_replay_controller_v2",
        "milestone": "2026.05.297",
        "continuous_self_learning_task": True,
        "trace_count": len(labels),
        "heldout_row_count": 1,
        "drift_row_count": 2,
        "negative_control_regression_count": negative_control_regression_count,
        "rollback_event_count": 0,
        "model_weight_update_claimed": False,
        "promotion_allowed": negative_control_regression_count == 0,
        "replay_utility_labels": labels,
        "inference_substrate": {
            "controller_memory_replay_only": True,
            "trace_memory_policy_only": True,
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
        "honest_verdict": "complete: unit source" if terminal else "draft: unit source",
    }


def _trace(row_id: str, role: str) -> dict[str, Any]:
    return {
        "trace_id": f"trace-{role}-{row_id}",
        "row_id": row_id,
        "replay_role": role,
        "verification_query": f"Does {row_id} satisfy exact replay?",
        "consistency_judgment": "consistent",
        "answer_abstain_decision": "answer",
        "exact_label": "EXACT_ACCEPT",
        "routing_outcome": "verify_then_answer",
        "expected_action": "accept",
        "observed_action_changed": False,
        "redundant_check_suppressed": False,
    }


def _exp3200_payload() -> dict[str, Any]:
    records = [_trace("fallback-heldout", "heldout"), _trace("fallback-drift", "drift")]
    return {
        "artifact": "experiment_3200_fr11_verify_trace_memory_controller_v1",
        "schema_version": "1.0",
        "experiment_id": "experiment_3200_fr11_verify_trace_memory_controller_v1",
        "continuous_self_learning_task": True,
        "trace_count": len(records),
        "heldout_row_count": 1,
        "drift_row_count": 1,
        "negative_control_regression_count": 0,
        "model_weight_update_performed": False,
        "promotion_allowed": True,
        "trace_records": records,
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
        "honest_verdict": "complete: fallback unit source",
    }


def _write_sources(root: Path, *, exp3215: Mapping[str, Any] | None = None) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("controller-memory only\n", encoding="utf-8")
    (root / "research-program.md").write_text("Continuous Self-Learning\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "Grounded Continuation\nDrift-Plus-Penalty\n", encoding="utf-8"
    )
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text("REQ-LEARN-3216\nSCENARIO-LEARN-3216\n", encoding="utf-8")
    if exp3215 is not None:
        _write_json(root, mod.EXP3215_REL_PATH, exp3215)
    _write_json(root, mod.EXP3200_REL_PATH, _exp3200_payload())


def test_req_learn_3216_spec_anchor_exists() -> None:
    """REQ-LEARN-3216: OpenSpec declares graph and queue artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3216" in spec
    assert "SCENARIO-LEARN-3216" in spec
    assert "SCENARIO-LEARN-3216-FALLBACK" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "trace_graph_node_count" in spec
    assert "nonforgetting_queue_value" in spec
    assert "model_weight_update_claimed=false" in spec


def test_req_learn_3216_graph_propagates_stale_premises() -> None:
    """REQ-LEARN-3216-2/3: retractions invalidate dependent routes."""

    rows = mod.trace_rows_from_payload(_exp3215_payload(), source_kind="exp3215")
    graph = mod.build_trace_graph(rows, trace_limit=3)
    propagation = mod.propagate_stale_premises(graph)

    assert graph["source_trace_count"] == 3
    assert graph["node_count"] == 11
    assert graph["edge_count"] == 9
    assert {node["node_kind"] for node in graph["nodes"]} >= {
        "claim",
        "evidence",
        "retraction",
        "repair",
        "route",
    }
    assert propagation["stale_premise_invalidations"] == 2
    assert propagation["affected_route_count"] == 1
    assert propagation["affected_routes"] == [
        {
            "route_node_id": "route:trace-drift-drift-stale",
            "row_id": "drift-stale",
            "replay_role": "drift",
            "routing_outcome": "verify_then_answer",
        }
    ]


def test_scenario_learn_3216_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3216: graph and queue stay within budget."""

    _write_sources(tmp_path, exp3215=_exp3215_payload())
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.0,
        tests_run=["SCENARIO-LEARN-3216 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["source_trace_artifact"] == mod.EXP3215_REL_PATH.as_posix()
    assert artifact["source_selection"]["fallback_used"] is False
    assert artifact["trace_graph_node_count"] == 11
    assert artifact["trace_graph_edge_count"] == 9
    assert artifact["stale_premise_invalidations"] == 2
    assert artifact["affected_route_count"] == 1
    assert artifact["nonforgetting_queue_defined"] is True
    assert artifact["nonforgetting_queue_value"] == pytest.approx(1.0)
    assert artifact["nonforgetting_budget_exceeded"] is False
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["controller_memory_promotion_allowed"] is False
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["tests_run"] == ["SCENARIO-LEARN-3216 focused"]
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "model_weight_update_claimed=false" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_scenario_learn_3216_falls_back_to_exp3200(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3216-FALLBACK: missing Exp 3215 uses Exp 3200."""

    _write_sources(tmp_path, exp3215=None)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["source_trace_artifact"] == mod.EXP3200_REL_PATH.as_posix()
    assert artifact["source_selection"]["fallback_used"] is True
    assert artifact["source_selection"]["fallback_reason"] == "exp3215_missing_or_not_terminal"
    assert artifact["trace_graph_node_count"] == 6
    assert artifact["trace_graph_edge_count"] == 4
    assert artifact["stale_premise_invalidations"] == 0
    assert artifact["affected_route_count"] == 0
    assert artifact["nonforgetting_queue_defined"] is True
    assert artifact["nonforgetting_queue_value"] == pytest.approx(0.0)
    assert artifact["nonforgetting_budget_exceeded"] is False
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["controller_memory_promotion_allowed"] is False
    mod.validate_artifact(artifact)


def test_req_learn_3216_budget_and_validation_guards(tmp_path: Path) -> None:
    """REQ-LEARN-3216-4/5: budget pressure is explicit and overclaims fail."""

    _write_sources(
        tmp_path,
        exp3215=_exp3215_payload(negative_control_regression_count=1),
    )
    exceeded = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.25)

    assert exceeded["nonforgetting_queue_value"] == pytest.approx(3.0)
    assert exceeded["nonforgetting_budget_exceeded"] is True
    assert exceeded["controller_memory_promotion_allowed"] is False
    assert "budget_exceeded" in exceeded["honest_verdict"]
    mod.validate_artifact(exceeded)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.is_terminal({"honest_verdict": "complete_ok"}) is True
    assert mod.is_terminal({"honest_verdict": "draft"}) is False
    assert mod.source_claims_live_or_mutation({"inference_substrate": []}) is True
    assert mod.source_claims_live_or_mutation(
        {"inference_substrate": {"fresh_live_inference_calls": 1}}
    )
    assert mod.source_claims_live_or_mutation(
        {"inference_substrate": {"model_weight_training": True}}
    )
    assert mod.select_source({"exp3215": {}, "exp3200": {}})["blocked_reason"] == (
        "no_terminal_safe_trace_source"
    )
    assert mod.detected_model_weight_update({"model_weight_update_claimed": True})
    assert mod.detected_model_weight_update({"model_weight_update_performed": True})
    assert mod.detected_model_weight_update(
        {"inference_substrate": {"base_model_weights_updated": True}}
    )
    assert mod.trace_rows_from_payload({"replay_utility_labels": "bad"}, source_kind="exp3215") == []
    assert mod.trace_rows_from_payload({"trace_records": [None, {"row_id": "ok"}]}, source_kind="exp3200") == [
        {"row_id": "ok", "source_kind": "exp3200"}
    ]
    assert mod.select_trace_rows([_trace("a", "heldout")], trace_limit=0) == []
    assert mod.select_trace_rows([{"row_id": "unmatched", "replay_role": "other"}]) == [
        {"row_id": "unmatched", "replay_role": "other"}
    ]
    assert mod.trace_needs_retraction({"replay_role": "drift", "redundant_check_suppressed": True})
    assert mod.trace_needs_retraction({"exact_verifier_outcome": "exact_replay_failed"})
    assert mod.trace_needs_retraction({"consistency_judgment": "inconsistent"})
    assert mod.trace_needs_retraction({"rollback_triggered": True})
    assert mod.retraction_text({"redundant_check_suppressed": True}) == (
        "stale_premise_probe=redundant_check_suppressed"
    )
    assert mod.stable_trace_id({"row_id": "no-trace"}).startswith("trace-")
    assert mod.normalize_token("Needs Repair") == "needs_repair"
    assert mod.sha256_file(tmp_path / "absent.txt") is None
    duplicate_graph = {
        "nodes": [{"node_id": "claim:x", "node_kind": "claim"}],
        "edges": [
            {"source": "retraction:a", "target": "claim:x", "relation": "invalidates"},
            {"source": "retraction:b", "target": "claim:x", "relation": "invalidates"},
        ],
    }
    assert mod.propagate_stale_premises(duplicate_graph)["stale_premise_invalidations"] == 1

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="model_weight_update_claimed"):
        mod.validate_artifact(exceeded | {"model_weight_update_claimed": True})
    with pytest.raises(ValueError, match="controller_memory_promotion_allowed"):
        mod.validate_artifact(exceeded | {"controller_memory_promotion_allowed": True})
    with pytest.raises(ValueError, match="conductor_file_modified"):
        mod.validate_artifact(exceeded | {"conductor_file_modified": True})
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(exceeded | {"active_roadmap_modified": True})
    with pytest.raises(ValueError, match="queue"):
        mod.validate_artifact(exceeded | {"nonforgetting_queue_defined": False})
    with pytest.raises(ValueError, match="queue value"):
        mod.validate_artifact(exceeded | {"nonforgetting_queue_value": "bad"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(exceeded | {"honest_verdict": "blocked"})
