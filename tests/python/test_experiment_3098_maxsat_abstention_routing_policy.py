"""Tests for Exp 3098 MaxSAT abstention routing policy.

Spec refs: REQ-VERIFY-3098, SCENARIO-VERIFY-3098.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import maxsat_abstention_routing_policy_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
REQUIRED_ARTIFACT_FIELDS = {
    "maxsat_policy_ready",
    "routing_policy_path",
    "hard_constraints",
    "soft_constraints",
    "objective_terms",
    "fallback_evaluator",
    "downstream_usage",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
REQUIRED_HARD_TOPICS = {
    "exact_label_disagreement",
    "model_cache_availability",
    "formal_feedback_lift",
    "syntax_schema_validity",
    "repair_intent_preservation",
    "no_tiny_panel_disqualification",
}
REQUIRED_SOFT_TOPICS = {
    "accept_exact_consistent",
    "reject_exact_inconsistent",
    "abstain_on_uncertainty",
    "prefer_formal_feedback_lift",
    "preserve_repair_intent",
    "minimize_false_accept",
    "minimize_false_reject",
    "minimize_unnecessary_abstention",
}


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 1.5
        return self.value


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_required_sources(root: Path) -> None:
    (root / "CODEX.md").write_text("Spec First\nWrite Tests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No tiny panels\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "OpenReview Qmr9VbwRaB: weighted MaxSAT routing.\n", encoding="utf-8"
    )
    _write_json(
        root,
        exp.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "minimum_live_eval_count": 48,
            "usable_fixture_count": 72,
            "stratified_eval_manifest_path": "results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl",
            "honest_verdict": "complete: eval_protocol_ready=true",
        },
    )
    _write_json(
        root,
        exp.EXP3085_REL_PATH,
        {
            "artifact": "experiment_3085_icalm_task_abstention_sota_panel_v2",
            "abstention_precision": 0.0,
            "rejection_recall": 0.333333,
            "exact_ground_truth_count": 9,
            "honest_verdict": "complete_below_gate: abstention_precision=0.0",
        },
    )
    _write_json(
        root,
        exp.EXP3087_REL_PATH,
        {
            "artifact": "experiment_3087_gated_local_sota_verifier_calibration_v3",
            "status": "blocked",
            "gate_check_summary": "abstention_precision gate failed",
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        exp.EXP3094_REL_PATH,
        {
            "artifact": "experiment_3094_capstone_v288",
            "capstone_ready": True,
            "verifier_gain_status": "flagged_or_gated_verifier_gain_recovery_incomplete",
            "repair_claim_status": "bounded_flagged_gated_missing_verifier_gated",
            "honest_verdict": "complete: capstone_ready=true",
        },
    )


def _config(root: Path) -> exp.PolicyConfig:
    return exp.PolicyConfig(repo_root=root, started_s=10.0, clock=lambda: 14.25)


def _safe_case(**updates: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "expected_action": "accept",
        "exact_label_match": True,
        "model_cache_available": True,
        "headline_claim": True,
        "exact_ground_truth_count": 72,
        "minimum_live_eval_count": 48,
        "syntax_valid": True,
        "schema_valid": True,
        "repair_candidate": False,
        "repair_intent_preserved": True,
        "repair_promotion": False,
        "formal_feedback_delta": 0.1,
        "confidence": 0.92,
    }
    case.update(updates)
    return case


def test_req_verify_3098_spec_anchor_exists() -> None:
    """REQ-VERIFY-3098: OpenSpec declares the MaxSAT routing contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3098" in spec
    assert "SCENARIO-VERIFY-3098" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "exact-label disagreement" in spec
    assert "tie" in spec and "abstain" in spec
    assert "Exp 3099" in spec and "Exp 3101" in spec and "Exp 3102" in spec


def test_scenario_verify_3098_writes_policy_artifact_and_reference_spec(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3098: policy artifact is ready and downstream-consumable."""
    _write_required_sources(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    policy_path = tmp_path / artifact["routing_policy_path"]
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    assert REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["maxsat_policy_ready"] is True
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert policy["schema"] == exp.POLICY_SCHEMA
    assert policy["actions"] == ["accept", "reject", "abstain"]
    assert policy["fallback_evaluator"]["tie_break_order"] == ["abstain", "reject", "accept"]
    assert policy["objective_terms"] == artifact["objective_terms"]

    hard_topics = {constraint["topic"] for constraint in artifact["hard_constraints"]}
    soft_topics = {constraint["topic"] for constraint in artifact["soft_constraints"]}
    assert REQUIRED_HARD_TOPICS <= hard_topics
    assert REQUIRED_SOFT_TOPICS <= soft_topics
    assert all(constraint["weight"] > 0 for constraint in artifact["soft_constraints"])

    downstream = artifact["downstream_usage"]
    assert set(downstream) == {"exp3099", "exp3101", "exp3102"}
    assert downstream["exp3099"]["must_load_policy_from"] == artifact["routing_policy_path"]
    assert downstream["exp3099"]["required_field"] == "maxsat_policy_used"
    assert downstream["exp3101"]["required_metric"] == "verifier_gain_delta_with_maxsat"
    assert downstream["exp3102"]["gate"] == "exp3101.verifier_gain_delta_with_maxsat > 0.0"

    substrate = artifact["inference_substrate"]
    assert substrate["no_live_llm_inference"] is True
    assert substrate["executes_models"] is False
    assert substrate["live_llm_calls"] == 0

    source_paths = {source.get("path") for source in artifact["source_artifacts"]}
    assert exp.EXP3097_REL_PATH.as_posix() in source_paths
    assert exp.EXP3085_REL_PATH.as_posix() in source_paths
    assert exp.EXP3087_REL_PATH.as_posix() in source_paths
    assert "https://openreview.net/forum?id=Qmr9VbwRaB" in {
        source.get("url") for source in artifact["source_artifacts"]
    }
    exp.validate_artifact(artifact)


def test_req_verify_3098_fallback_routes_fail_closed() -> None:
    """REQ-VERIFY-3098: deterministic fallback never accepts unsafe cases."""
    policy = exp.build_policy_document()

    safe_accept = exp.evaluate_route(_safe_case(), policy=policy)
    assert safe_accept["decision"] == "accept"
    assert safe_accept["hard_feasible_actions"] == ["accept", "abstain"]
    assert safe_accept["used_solver"] == "deterministic_reference_evaluator"

    exact_reject = exp.evaluate_route(
        _safe_case(expected_action="reject", exact_label_match=False, confidence=0.88),
        policy=policy,
    )
    assert exact_reject["decision"] == "reject"
    assert "accept" in exact_reject["blocked_actions"]

    tiny_panel = exp.evaluate_route(
        _safe_case(exact_ground_truth_count=9, confidence=0.99),
        policy=policy,
    )
    assert tiny_panel["decision"] == "abstain"
    assert tiny_panel["hard_feasible_actions"] == ["abstain"]

    missing_cache = exp.evaluate_route(
        _safe_case(model_cache_available=False, confidence=0.99),
        policy=policy,
    )
    assert missing_cache["decision"] == "abstain"

    syntax_failure = exp.evaluate_route(
        _safe_case(exact_label_match=False, syntax_valid=False, confidence=0.99),
        policy=policy,
    )
    assert syntax_failure["decision"] == "reject"

    repair_drift = exp.evaluate_route(
        _safe_case(
            exact_label_match=False,
            repair_candidate=True,
            repair_intent_preserved=False,
            confidence=0.99,
        ),
        policy=policy,
    )
    assert repair_drift["decision"] == "reject"

    negative_formal_lift = exp.evaluate_route(
        _safe_case(
            repair_promotion=True,
            formal_feedback_delta=-0.2,
            confidence=0.99,
        ),
        policy=policy,
    )
    assert negative_formal_lift["decision"] == "abstain"

    uncertain = exp.evaluate_route(_safe_case(confidence=0.2), policy=policy)
    abstain_topics = {item["topic"] for item in uncertain["score_breakdown"]["abstain"]}
    assert "abstain_on_uncertainty" in abstain_topics

    no_feasible_action_policy = dict(policy) | {"actions": ["accept"]}
    no_feasible = exp.evaluate_route(
        _safe_case(exact_label_match=False, model_cache_available=False),
        policy=no_feasible_action_policy,
    )
    assert no_feasible["decision"] == "abstain"
    assert no_feasible["hard_feasible_actions"] == ["abstain"]


def test_req_verify_3098_validation_and_blocked_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3098: missing policy prerequisites cannot report readiness."""
    artifact = exp.write_artifact(_config(tmp_path))

    assert artifact["maxsat_policy_ready"] is False
    assert artifact["honest_verdict"].startswith(
        "blocked_maxsat_policy_precondition_failed"
    )
    assert artifact["inference_substrate"]["no_live_llm_inference"] is True
    exp.validate_artifact(artifact)

    _write_required_sources(tmp_path)
    ready = exp.write_artifact(_config(tmp_path))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: missing fields"})
    with pytest.raises(ValueError, match="no live model inference"):
        bad_substrate = dict(ready["inference_substrate"]) | {"no_live_llm_inference": False}
        exp.validate_artifact(ready | {"inference_substrate": bad_substrate})
    with pytest.raises(ValueError, match="must not execute models"):
        bad_substrate = dict(ready["inference_substrate"]) | {"executes_models": True}
        exp.validate_artifact(ready | {"inference_substrate": bad_substrate})
    with pytest.raises(ValueError, match="success prefix"):
        exp.validate_artifact(ready | {"honest_verdict": "ready without prefix"})
    with pytest.raises(ValueError, match="routing policy path"):
        exp.validate_artifact(ready | {"routing_policy_path": ""})
    with pytest.raises(ValueError, match="blocked verdict"):
        exp.validate_artifact(
            ready
            | {
                "maxsat_policy_ready": False,
                "honest_verdict": "blocked without expected prefix",
            }
        )
    with pytest.raises(ValueError, match="hard constraints"):
        exp.validate_artifact(ready | {"hard_constraints": ready["hard_constraints"][1:]})
    with pytest.raises(ValueError, match="soft constraints"):
        exp.validate_artifact(ready | {"soft_constraints": ready["soft_constraints"][1:]})
    with pytest.raises(ValueError, match="fail closed"):
        exp.validate_artifact(
            ready
            | {
                "fallback_evaluator": dict(ready["fallback_evaluator"])
                | {"fail_closed_default": "accept"}
            }
        )
    with pytest.raises(ValueError, match="downstream usage"):
        exp.validate_artifact(ready | {"downstream_usage": {"exp3099": {}}})

    assert exp.safe_load_json(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert exp.safe_load_json(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[1]", encoding="utf-8")
    assert exp.safe_load_json(list_payload) == {}
