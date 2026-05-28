"""Tests for Exp 3229 FR-11 nonforgetting-aware promotion governance.

Spec refs: REQ-LEARN-3229, SCENARIO-LEARN-3229,
SCENARIO-LEARN-3229-DEFERRED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_nonforgetting_promotion_controller_v3 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_substrate() -> dict[str, Any]:
    return {
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
    }


def _label(
    row_id: str,
    role: str,
    *,
    trace_id: str | None = None,
    outcome: str = "exact_accept_answered",
    utility: str = "verified_answer_for_exact_accept",
    reward: float = 0.75,
    rollback: str = "none",
    routing: str = "verify_then_answer",
) -> dict[str, Any]:
    return {
        "trace_id": trace_id or f"trace-{role}-{row_id}",
        "row_id": row_id,
        "replay_role": role,
        "verification_query": f"Verify {row_id}",
        "consistency_judgment": "consistent",
        "answer_abstain_decision": "answer",
        "exact_label": "EXACT_ACCEPT",
        "exact_action": "accept",
        "routing_outcome": routing,
        "exact_verifier_outcome": outcome,
        "prior_route_utility": utility,
        "reward_weight": reward,
        "redundant_check_suppressed": routing == "skip_redundant_recheck",
        "rollback_or_retraction_status": rollback,
        "controller_utility_label_only": True,
        "model_weight_update_claimed": False,
    }


def _labels() -> list[dict[str, Any]]:
    return [
        _label("heldout-ok", "heldout"),
        _label("drift-stale", "drift", routing="skip_redundant_recheck"),
        _label("drift-ok", "drift"),
        _label("negative-ok", "negative_control", routing="skip_redundant_recheck"),
    ]


def _exp3215_payload(
    *,
    labels: list[dict[str, Any]] | None = None,
    terminal: bool = True,
    negative_control_regression_count: int = 0,
    model_weight_update_claimed: bool = False,
    fresh_live_calls: int = 0,
) -> dict[str, Any]:
    replay_labels = labels or _labels()
    substrate = _safe_substrate()
    substrate["fresh_live_inference_calls"] = fresh_live_calls
    return {
        "artifact": "experiment_3215_fr11_evidence_gated_trace_replay_controller_v2",
        "schema_version": "1.0",
        "experiment_id": "experiment_3215_fr11_evidence_gated_trace_replay_controller_v2",
        "milestone": "2026.05.297",
        "continuous_self_learning_task": True,
        "trace_count": len(replay_labels),
        "evidence_backed_trace_count": len(replay_labels),
        "replay_utility_label_count": len(replay_labels),
        "heldout_row_count": sum(row["replay_role"] == "heldout" for row in replay_labels),
        "drift_row_count": sum(row["replay_role"] == "drift" for row in replay_labels),
        "negative_control_regression_count": negative_control_regression_count,
        "rollback_event_count": 0,
        "model_weight_update_claimed": model_weight_update_claimed,
        "promotion_allowed": negative_control_regression_count == 0,
        "replay_utility_labels": replay_labels,
        "inference_substrate": substrate,
        "honest_verdict": "complete: unit exp3215" if terminal else "draft: unit exp3215",
    }


def _affected_route(row_id: str, role: str = "drift") -> dict[str, str]:
    trace_id = f"trace-{role}-{row_id}"
    return {
        "route_node_id": f"route:{trace_id}",
        "row_id": row_id,
        "replay_role": role,
        "routing_outcome": "skip_redundant_recheck",
    }


def _exp3216_payload(
    *,
    affected_routes: list[dict[str, str]] | None = None,
    budget_exceeded: bool = False,
    negative_control_regressions: int = 0,
    terminal: bool = True,
) -> dict[str, Any]:
    routes = affected_routes if affected_routes is not None else [_affected_route("drift-stale")]
    queue_value = 3.0 if budget_exceeded else float(len(routes) + negative_control_regressions)
    return {
        "artifact": "experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1",
        "schema_version": "1.0",
        "experiment_id": "experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1",
        "milestone": "2026.05.297",
        "continuous_self_learning_task": True,
        "source_trace_artifact": mod.EXP3215_REL_PATH.as_posix(),
        "trace_graph_node_count": 5,
        "trace_graph_edge_count": 4,
        "stale_premise_invalidations": len(routes),
        "affected_route_count": len(routes),
        "affected_routes": routes,
        "nonforgetting_queue_defined": True,
        "nonforgetting_queue_value": queue_value,
        "nonforgetting_budget_exceeded": budget_exceeded,
        "nonforgetting_queue": {
            "nonforgetting_budget": 2.0,
            "nonforgetting_budget_exceeded": budget_exceeded,
            "nonforgetting_queue_defined": True,
            "nonforgetting_queue_value": queue_value,
            "pressure_terms": {
                "affected_heldout_or_drift_routes": len(routes),
                "negative_control_regressions": negative_control_regressions,
                "unrouted_retraction_pressure": 0,
            },
        },
        "model_weight_update_claimed": False,
        "controller_memory_promotion_allowed": False,
        "inference_substrate": _safe_substrate(),
        "honest_verdict": "complete: unit exp3216" if terminal else "draft: unit exp3216",
    }


def _write_sources(
    root: Path,
    *,
    exp3215: Mapping[str, Any] | None = None,
    exp3216: Mapping[str, Any] | None = None,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("controller-memory only\n", encoding="utf-8")
    (root / "research-program.md").write_text("Continuous Self-Learning\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "Evidence Over Plans\nGrounded Continuation\n", encoding="utf-8"
    )
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3229\nSCENARIO-LEARN-3229\nSCENARIO-LEARN-3229-DEFERRED\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3215_REL_PATH, exp3215 or _exp3215_payload())
    _write_json(root, mod.EXP3216_REL_PATH, exp3216 or _exp3216_payload())


def test_req_learn_3229_spec_anchor_exists() -> None:
    """REQ-LEARN-3229: OpenSpec declares promotion governance fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3229" in spec
    assert "SCENARIO-LEARN-3229" in spec
    assert "SCENARIO-LEARN-3229-DEFERRED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "accepted_trace_count" in spec
    assert "stale_premise_rejection_count" in spec
    assert "model_weight_update_claimed=false" in spec


def test_req_learn_3229_admission_rules_reject_stale_premise() -> None:
    """REQ-LEARN-3229-2/3/6: stale routes are rejected before promotion."""

    candidates = mod.candidate_traces(_exp3215_payload())
    stale = mod.stale_premise_index(_exp3216_payload())
    rules = mod.admission_context(_exp3215_payload(), _exp3216_payload())
    simulation = mod.simulate_replay(candidates, stale, rules)

    assert len(candidates) == 4
    assert mod.candidate_is_evidence_backed(candidates[0]) is True
    assert mod.candidate_is_evidence_backed({"trace_id": "planned"}) is False
    assert simulation["accepted_trace_count"] == 3
    assert simulation["rejected_trace_count"] == 1
    assert simulation["deferred_trace_count"] == 0
    assert simulation["stale_premise_rejection_count"] == 1
    assert simulation["rejected_traces"][0]["row_id"] == "drift-stale"
    assert simulation["rejected_traces"][0]["decision_reasons"] == ["stale_premise_failure"]
    assert mod.rollback_policy()["rollback_trigger_count"] == 3


def test_scenario_learn_3229_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3229: accepted promotion excludes stale premises."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.5,
        tests_run=["SCENARIO-LEARN-3229 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["source_trace_artifacts"] == [
        mod.EXP3215_REL_PATH.as_posix(),
        mod.EXP3216_REL_PATH.as_posix(),
    ]
    assert artifact["candidate_trace_count"] == 4
    assert artifact["accepted_trace_count"] == 3
    assert artifact["rejected_trace_count"] == 1
    assert artifact["deferred_trace_count"] == 0
    assert artifact["promotion_allowed"] is True
    assert artifact["controller_memory_promotion_allowed"] is True
    assert artifact["nonforgetting_budget_exceeded"] is False
    assert artifact["rollback_policy_defined"] is True
    assert artifact["rollback_trigger_count"] == 3
    assert artifact["negative_control_regression_count"] == 0
    assert artifact["stale_premise_rejection_count"] == 1
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["controller_memory_training_boundary"]["model_weight_training"] is False
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["tests_run"] == ["SCENARIO-LEARN-3229 focused"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "model_weight_update_claimed=false" in artifact["honest_verdict"]
    assert "controller_memory_updates_are_not_training" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_scenario_learn_3229_deferred_when_queue_budget_exceeded(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3229-DEFERRED: queue pressure defers otherwise good traces."""

    exp3216 = _exp3216_payload(affected_routes=[], budget_exceeded=True)
    _write_sources(tmp_path, exp3216=exp3216)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert artifact["candidate_trace_count"] == 4
    assert artifact["accepted_trace_count"] == 0
    assert artifact["rejected_trace_count"] == 0
    assert artifact["deferred_trace_count"] == 4
    assert artifact["nonforgetting_budget_exceeded"] is True
    assert artifact["promotion_allowed"] is False
    assert artifact["controller_memory_promotion_allowed"] is False
    assert all(
        row["decision_reasons"] == ["nonforgetting_budget_exceeded"]
        for row in artifact["deferred_traces"]
    )
    assert "promotion_allowed=false" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_learn_3229_guards_and_negative_controls(tmp_path: Path) -> None:
    """REQ-LEARN-3229-1/3/5: unsafe sources and regressions fail closed."""

    missing = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)
    assert missing["candidate_trace_count"] == 0
    assert missing["promotion_allowed"] is False
    assert missing["blocked_reason"] == "exp3215_missing_or_not_terminal"
    mod.validate_artifact(missing)

    _write_sources(tmp_path, exp3215=_exp3215_payload(fresh_live_calls=1))
    live_blocked = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    assert live_blocked["blocked_reason"] == "exp3215_live_inference_or_weight_update_claimed"
    mod.validate_artifact(live_blocked)

    exp3215 = _exp3215_payload(negative_control_regression_count=1)
    exp3216 = _exp3216_payload(affected_routes=[], negative_control_regressions=1)
    _write_sources(tmp_path, exp3215=exp3215, exp3216=exp3216)
    regressed = mod.build_artifact(tmp_path, started_s=3.0, now_s=4.0)

    assert regressed["negative_control_regression_count"] == 1
    assert regressed["accepted_trace_count"] == 0
    assert regressed["rejected_trace_count"] == 4
    assert regressed["promotion_allowed"] is False
    assert all(
        "negative_control_regression" in row["decision_reasons"]
        for row in regressed["rejected_traces"]
    )
    mod.validate_artifact(regressed)

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
    assert mod.detected_model_weight_update({"model_weight_update_claimed": True})
    assert mod.detected_model_weight_update({"model_weight_update_performed": True})
    assert mod.detected_model_weight_update(
        {"inference_substrate": {"base_model_weights_updated": True}}
    )
    assert mod.source_blocker({"exp3215": _exp3215_payload(), "exp3216": {}}) == (
        "exp3216_missing_or_not_terminal"
    )
    bad_weight = _exp3215_payload(model_weight_update_claimed=True)
    assert mod.source_blocker({"exp3215": bad_weight, "exp3216": _exp3216_payload()}) == (
        "exp3215_model_weight_update_claimed"
    )
    assert mod.candidate_traces({"replay_utility_labels": "bad"}) == []
    assert mod.stale_premise_index({"affected_routes": "bad"}) == {
        "route_node_ids": set(),
        "trace_ids": set(),
        "route_keys": set(),
    }
    assert mod.stale_premise_index({"affected_routes": [None, _affected_route("x")]})[
        "trace_ids"
    ] == {"trace-drift-x"}
    no_count_context = mod.admission_context(
        {"replay_utility_labels": [_label("heldout", "heldout"), _label("drift", "drift")]},
        _exp3216_payload(affected_routes=[]),
    )
    assert no_count_context["heldout_row_count"] == 1
    assert no_count_context["drift_row_count"] == 1
    rejection_reasons = mod.trace_rejection_reasons(
        {"trace_id": "planned", "reward_weight": "-1", "rollback_or_retraction_status": "bad"},
        {"route_node_ids": set(), "trace_ids": set(), "route_keys": set()},
        {"heldout_check_passed": False, "drift_check_passed": False},
    )
    assert rejection_reasons == [
        "missing_evidence_label",
        "missing_heldout_or_drift_check",
        "rollback_or_retraction",
        "negative_utility_label",
    ]
    assert mod.nonforgetting_queue_entries({"nonforgetting_queue": []}) == []
    assert mod.nonforgetting_queue_entries({"nonforgetting_queue": {"pressure_terms": []}}) == []
    assert mod.normalize_token("Needs Repair") == "needs_repair"
    assert mod.safe_int("not-int") == 0
    assert mod.safe_float("not-float") == 0.0
    assert mod.sha256_file(tmp_path / "absent.txt") is None

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="model_weight_update_claimed"):
        mod.validate_artifact(regressed | {"model_weight_update_claimed": True})
    with pytest.raises(ValueError, match="controller_memory_promotion_allowed"):
        mod.validate_artifact(regressed | {"controller_memory_promotion_allowed": True})
    with pytest.raises(ValueError, match="conductor_file_modified"):
        mod.validate_artifact(regressed | {"conductor_file_modified": True})
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(regressed | {"active_roadmap_modified": True})
    with pytest.raises(ValueError, match="training"):
        mod.validate_artifact(
            regressed
            | {
                "controller_memory_training_boundary": regressed[
                    "controller_memory_training_boundary"
                ]
                | {"model_weight_training": True}
            }
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(regressed | {"inference_substrate": "bad"})
    with pytest.raises(ValueError, match="controller_memory_training_boundary"):
        mod.validate_artifact(regressed | {"controller_memory_training_boundary": []})
    with pytest.raises(ValueError, match="separate from training"):
        mod.validate_artifact(
            regressed
            | {
                "controller_memory_training_boundary": regressed[
                    "controller_memory_training_boundary"
                ]
                | {"controller_memory_updates_are_not_training": False}
            }
        )
    with pytest.raises(ValueError, match="rollback policy"):
        mod.validate_artifact(regressed | {"rollback_policy_defined": False})
    with pytest.raises(ValueError, match="rollback trigger"):
        mod.validate_artifact(regressed | {"rollback_trigger_count": 2})
    with pytest.raises(ValueError, match="counts"):
        mod.validate_artifact(regressed | {"candidate_trace_count": 99})
    with pytest.raises(ValueError, match="accepted"):
        mod.validate_artifact(
            regressed
            | {
                "promotion_allowed": True,
                "controller_memory_promotion_allowed": True,
                "accepted_trace_count": 0,
            }
        )
    _write_sources(tmp_path)
    allowed = mod.build_artifact(tmp_path, started_s=4.0, now_s=5.0)
    with pytest.raises(ValueError, match="budget"):
        mod.validate_artifact(
            allowed
            | {
                "nonforgetting_budget_exceeded": True,
            }
        )
    with pytest.raises(ValueError, match="negative-control"):
        mod.validate_artifact(
            allowed
            | {
                "negative_control_regression_count": 1,
            }
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(regressed | {"honest_verdict": "blocked"})
