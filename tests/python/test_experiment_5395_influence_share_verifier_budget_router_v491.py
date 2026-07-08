"""Tests for Exp5395 influence-share verifier-budget routing.

Spec refs: REQ-LEARN-5395, SCENARIO-LEARN-5395-SHARES,
SCENARIO-LEARN-5395-ROUTING, SCENARIO-LEARN-5395-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5395_influence_share_verifier_budget_router_v491 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5395_spec_declares_router_contract() -> None:
    """REQ-LEARN-5395: OpenSpec anchors the influence-share router."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5395") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5395",
        "SCENARIO-LEARN-5395-SHARES",
        "SCENARIO-LEARN-5395-ROUTING",
        "SCENARIO-LEARN-5395-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        "baseline routing variant",
        "fixed self-learning routing variant",
        "influence-share routing variant",
        "SHALL NOT load, fine-tune, write, or mutate model weights",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5395_influence_shares_sum_to_100() -> None:
    """SCENARIO-LEARN-5395-SHARES: every routing row sums to 100."""

    evaluation = exp.evaluate_routing_variants(root=REPO)
    decisions = evaluation["routing_decisions"]

    assert evaluation["workflow"]["session_count"] >= exp.MIN_SESSIONS
    assert evaluation["checked_event_count"] >= exp.MIN_CHECKED_EVENTS
    assert evaluation["influence_factor_names"] == exp.INFLUENCE_FACTOR_NAMES
    assert evaluation["routed_decision_count"] == len(decisions)
    assert evaluation["influence_share_sum_valid_rate"] == 1.0

    for row in decisions:
        assert row["variant_name"] == exp.INFLUENCE_VARIANT
        assert row["raw_evidence"]["event_id"] == row["event_id"]
        assert row["selected_verifier_tier"] in exp.VERIFIER_TIER_COSTS
        assert row["rejected_tier"] in exp.VERIFIER_TIER_COSTS
        assert row["selected_verifier_tier"] != row["rejected_tier"]
        assert row["reason"]
        assert row["rollback_status"] in {"not_required", "recovered"}
        assert list(row["influence_shares"]) == exp.INFLUENCE_FACTOR_NAMES
        assert sum(row["influence_shares"].values()) == 100


def test_scenario_learn_5395_tier_escalation_is_budgeted() -> None:
    """SCENARIO-LEARN-5395-ROUTING: local SOTA is selected only for justified rows."""

    evaluation = exp.evaluate_routing_variants(root=REPO)
    variants = evaluation["variant_metrics"]
    baseline = variants[exp.BASELINE_VARIANT]
    fixed = variants[exp.FIXED_VARIANT]
    routed = variants[exp.INFLUENCE_VARIANT]

    assert baseline["event_ids"] == fixed["event_ids"] == routed["event_ids"]
    assert evaluation["trace_count"] >= exp.MIN_TRACES
    assert evaluation["verifier_cost_delta_vs_baseline"] == round(
        baseline["verifier_cost"] - routed["verifier_cost"],
        6,
    )
    assert evaluation["context_efficiency_delta_vs_baseline"] == round(
        routed["context_efficiency"] - baseline["context_efficiency"],
        6,
    )
    assert evaluation["quality_delta_vs_baseline"] == round(
        routed["quality"] - baseline["quality"],
        6,
    )
    assert routed["verifier_cost"] < baseline["verifier_cost"]
    assert routed["verifier_cost"] <= fixed["verifier_cost"]
    assert routed["quality"] >= baseline["quality"]

    decisions_by_tier = {
        tier: [
            row for row in evaluation["routing_decisions"] if row["selected_verifier_tier"] == tier
        ]
        for tier in exp.VERIFIER_TIER_COSTS
    }
    assert decisions_by_tier["cheap_deterministic"]
    assert decisions_by_tier["rich_deterministic"]
    assert decisions_by_tier["local_sota"]

    for row in decisions_by_tier["local_sota"]:
        evidence = row["raw_evidence"]
        assert evidence["uncertainty"] >= exp.LOCAL_SOTA_UNCERTAINTY_MIN
        assert evidence["user_impact"] >= exp.LOCAL_SOTA_USER_IMPACT_MIN
        assert evidence["budget_remaining_before"] >= exp.VERIFIER_TIER_COSTS["local_sota"]
        assert row["reason"] == "high_uncertainty_high_impact_budget_headroom_use_local_sota"

    clean_cheap = [
        row
        for row in decisions_by_tier["cheap_deterministic"]
        if row["raw_evidence"]["memory_variant"] == "clean"
    ]
    assert clean_cheap
    assert all(row["raw_evidence"]["evidence_confidence"] >= 0.8 for row in clean_cheap)


def test_req_learn_5395_safety_and_no_weight_mutation() -> None:
    """REQ-LEARN-5395-4: routed checks deflect stale and poison controls."""

    evaluation = exp.evaluate_routing_variants(root=REPO)
    controls = evaluation["safety_controls"]

    assert evaluation["stale_memory_deflection_rate"] == 1.0
    assert evaluation["poison_deflection_rate"] == 1.0
    assert evaluation["rollback_success_rate"] == 1.0
    assert evaluation["unsafe_false_accepts"] == 0
    assert controls["stale_probe_count"] > 0
    assert controls["poison_probe_count"] > 0
    assert controls["rollback_required_count"] > 0
    assert evaluation["weight_mutation_receipt"] == {
        "no_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "learned_state_scope": "controller_routing_policy_only",
    }


def test_req_learn_5395_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5395-ARTIFACT: run() writes the gated artifact."""

    tests_run = [
        {
            "command": (
                ".venv/bin/pytest "
                "tests/python/test_experiment_5395_influence_share_verifier_budget_router_v491.py "
                "-q --no-cov"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5395_influence_share_verifier_budget_router_v491.py "
                "-m pytest "
                "tests/python/test_experiment_5395_influence_share_verifier_budget_router_v491.py "
                "-q --no-cov -n 0 && .venv/bin/coverage report --fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["session_count"] >= exp.MIN_SESSIONS
    assert artifact["trace_count"] >= exp.MIN_TRACES
    assert artifact["checked_event_count"] >= exp.MIN_CHECKED_EVENTS
    assert artifact["influence_factor_names"] == exp.INFLUENCE_FACTOR_NAMES
    assert artifact["influence_share_sum_valid_rate"] == 1.0
    assert artifact["routed_decision_count"] == len(artifact["routing_decisions"])
    assert artifact["verifier_cost_delta_vs_baseline"] > 0.0
    assert artifact["context_efficiency_delta_vs_baseline"] > 0.0
    assert artifact["quality_delta_vs_baseline"] >= 0.0
    assert artifact["stale_memory_deflection_rate"] == 1.0
    assert artifact["poison_deflection_rate"] == 1.0
    assert artifact["rollback_success_rate"] == 1.0
    assert artifact["no_weight_mutation"] is True
    assert artifact["continuous_self_learning_router_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5395_repository_artifact_matches_replay() -> None:
    """REQ-LEARN-5395-5: checked-in result is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["continuous_self_learning_router_ready"] is True
    assert result["no_weight_mutation"] is True


def test_req_learn_5395_validation_rejects_claim_drift() -> None:
    """REQ-LEARN-5395-5: validation rejects malformed ready claims."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5395", "outcome": "passed"}],
    )

    bad_missing = deepcopy(artifact)
    bad_missing.pop("status")
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["status"] = "changed"
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
    bad_int["routed_decision_count"] = True
    with pytest.raises(ValueError, match="routed_decision_count"):
        exp.validate_artifact(bad_int)

    bad_numeric = deepcopy(artifact)
    bad_numeric["verifier_cost_delta_vs_baseline"] = {"value": 1.0}
    with pytest.raises(ValueError, match="verifier_cost_delta_vs_baseline"):
        exp.validate_artifact(bad_numeric)

    bad_ready_status = deepcopy(artifact)
    bad_ready_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_ready_status)

    bad_router = deepcopy(artifact)
    bad_router["continuous_self_learning_router_ready"] = False
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_router)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.490"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_sessions = deepcopy(artifact)
    bad_sessions["session_count"] = exp.MIN_SESSIONS - 1
    with pytest.raises(ValueError, match="session_count"):
        exp.validate_artifact(bad_sessions)

    bad_shares = deepcopy(artifact)
    bad_shares["influence_share_sum_valid_rate"] = 0.99
    with pytest.raises(ValueError, match="influence_share_sum_valid_rate"):
        exp.validate_artifact(bad_shares)

    bad_cost = deepcopy(artifact)
    bad_cost["verifier_cost_delta_vs_baseline"] = -0.1
    bad_cost["context_efficiency_delta_vs_baseline"] = 0.0
    with pytest.raises(ValueError, match="cost or context"):
        exp.validate_artifact(bad_cost)

    bad_quality = deepcopy(artifact)
    bad_quality["quality_delta_vs_baseline"] = -0.1
    with pytest.raises(ValueError, match="quality_delta_vs_baseline"):
        exp.validate_artifact(bad_quality)

    bad_stale = deepcopy(artifact)
    bad_stale["stale_memory_deflection_rate"] = 0.5
    with pytest.raises(ValueError, match="stale_memory_deflection_rate"):
        exp.validate_artifact(bad_stale)

    bad_poison = deepcopy(artifact)
    bad_poison["poison_deflection_rate"] = 0.5
    with pytest.raises(ValueError, match="poison_deflection_rate"):
        exp.validate_artifact(bad_poison)

    bad_rollback = deepcopy(artifact)
    bad_rollback["rollback_success_rate"] = 0.5
    with pytest.raises(ValueError, match="rollback_success_rate"):
        exp.validate_artifact(bad_rollback)

    bad_weight = deepcopy(artifact)
    bad_weight["no_weight_mutation"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_weight)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    assert exp._rate(1, 0) == 0.0
    assert exp._json_ready(Path("results/example.json")) == "results/example.json"
