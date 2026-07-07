"""Tests for Exp5368 budget-curated memory governance.

Spec refs: REQ-LEARN-5368, SCENARIO-LEARN-5368-BUDGET,
SCENARIO-LEARN-5368-SAFETY, SCENARIO-LEARN-5368-SHARE-TRUST.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5368_budget_curated_memory_governance_v489 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5368_spec_declares_budget_governance_contract() -> None:
    """REQ-LEARN-5368: OpenSpec anchors fields, rules, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5368") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5368",
        "SCENARIO-LEARN-5368-BUDGET",
        "SCENARIO-LEARN-5368-SAFETY",
        "SCENARIO-LEARN-5368-SHARE-TRUST",
        str(exp.RESULT_RELATIVE_PATH),
        "value-minus-harm per byte",
        "provenance, byte cost, estimated verifier value, stale risk, poison",
        "KEEP, DROP, SHARE, QUARANTINE, TRUST, and UNTRUST",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5368_source_gate_reuses_clean_prior_fixtures() -> None:
    """REQ-LEARN-5368-1: prior readiness gates are required before curation."""

    gate = exp.confirm_source_gate(root=REPO)

    assert gate["all_passed"] is True
    assert gate["dependency_provenance_ready"] is True
    assert gate["memory_tool_drift_ready"] is True
    assert gate["self_learning_scaleup_ready"] is True
    assert gate["source_unsafe_false_accepts_zero"] is True
    assert gate["rollback_recovery_ready"] is True
    assert gate["no_weight_mutation"] is True
    assert str(exp.EXP5355_RELATIVE_PATH) in gate["source_artifacts"]
    assert str(exp.EXP5356_RELATIVE_PATH) in gate["source_artifacts"]
    assert str(exp.EXP5357_RELATIVE_PATH) in gate["source_artifacts"]


def test_req_learn_5368_memory_items_have_required_governance_fields() -> None:
    """REQ-LEARN-5368-2: every item carries value, cost, harm, and trust data."""

    items = exp.build_memory_items()
    rows = [item.as_dict() for item in items]

    assert len(items) == exp.MEMORY_ITEM_COUNT
    assert {row["memory_variant"] for row in rows} >= {
        "clean",
        "stale",
        "poisoned",
        "unverified",
        "low_value",
    }
    for row in rows:
        assert {
            "provenance",
            "byte_cost",
            "estimated_verifier_value",
            "stale_risk",
            "poison_risk",
            "sharing_risk",
            "trust_label",
            "useful",
            "harmful",
            "rollback_available",
        }.issubset(row)
        assert row["byte_cost"] > 0
        assert row["provenance"]["source_artifact"].startswith("results/")


def test_scenario_learn_5368_budget_keeps_highest_score_items_first() -> None:
    """SCENARIO-LEARN-5368-BUDGET: KEEP/DROP follows score under budget."""

    curation = exp.curate_memory_items(exp.build_memory_items())
    kept_rows = [row for row in curation["decision_rows"] if row["keep_decision"] == "KEEP"]
    dropped_rows = [row for row in curation["decision_rows"] if row["keep_decision"] == "DROP"]

    assert curation["retained_bytes"] <= exp.DEFAULT_BUDGET_BYTES
    assert [row["memory_id"] for row in kept_rows] == [
        "mem5368-clean-rollback-route",
        "mem5368-clean-dependency-edge",
        "mem5368-clean-scaleup-summary",
    ]
    assert [row["score_rank"] for row in kept_rows] == [1, 2, 3]
    assert any(row["memory_id"] == "mem5368-clean-low-value-overbudget" for row in dropped_rows)
    assert all(row["value_minus_harm_per_byte"] > 0.0 for row in kept_rows)


def test_scenario_learn_5368_safety_and_share_trust_metrics_gate_ready() -> None:
    """SCENARIO-LEARN-5368-SAFETY/SHARE-TRUST: unsafe memory is deflected."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit budget memory governance", "outcome": "passed"}],
    )
    shared_rows = [
        row for row in artifact["memory_decision_rows"] if row["share_decision"] == "SHARE"
    ]
    trusted_rows = [
        row for row in artifact["memory_decision_rows"] if row["trust_decision"] == "TRUST"
    ]
    stale_rows = [
        row for row in artifact["memory_decision_rows"] if row["memory_variant"] == "stale"
    ]
    poisoned_rows = [
        row for row in artifact["memory_decision_rows"] if row["memory_variant"] == "poisoned"
    ]
    expected_mean = round(
        sum(row["value_minus_harm_per_byte"] for row in artifact["memory_decision_rows"])
        / artifact["memory_item_count"],
        6,
    )

    assert artifact["budget_curated_memory_ready"] is True
    assert artifact["memory_item_count"] == exp.MEMORY_ITEM_COUNT
    assert artifact["budget_bytes"] == exp.DEFAULT_BUDGET_BYTES
    assert artifact["retained_bytes"] == sum(row["byte_cost"] for row in artifact["memory_decision_rows"] if row["keep_decision"] == "KEEP")
    assert artifact["retained_bytes"] <= artifact["budget_bytes"]
    assert artifact["value_minus_harm_per_byte_mean"] == expected_mean
    assert artifact["keep_precision"] == 1.0
    assert artifact["stale_memory_deflection_rate"] == 1.0
    assert artifact["poison_memory_deflection_rate"] == 1.0
    assert artifact["share_decision_precision"] == 1.0
    assert artifact["trust_decision_precision"] == 1.0
    assert artifact["rollback_recovery_rate"] == 1.0
    assert artifact["unsafe_false_accepts"] == 0
    assert all(row["trust_decision"] == "UNTRUST" or row["keep_decision"] != "KEEP" for row in stale_rows)
    assert all(row["trust_decision"] == "UNTRUST" or row["keep_decision"] != "KEEP" for row in poisoned_rows)
    assert all(row["provenance_verified"] and row["trust_decision"] == "TRUST" for row in shared_rows)
    assert all(row["useful"] and not row["harmful"] for row in trusted_rows)


def test_req_learn_5368_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5368-7: run() writes the required terminal artifact."""

    tests_run = [
        {"command": "pytest tests/python/test_experiment_5368_budget_curated_memory_governance_v489.py -q", "outcome": "passed"},
        {"command": "coverage run --source=python/carnot/experiment_5368_budget_curated_memory_governance_v489.py -m pytest tests/python/test_experiment_5368_budget_curated_memory_governance_v489.py -q", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "budget_curated_memory_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["budget_curated_memory_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5368_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5368: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["budget_curated_memory_ready"] is True
    assert result["no_weight_mutation"] is True
    assert result["unsafe_false_accepts"] == 0
    exp.validate_artifact(result)


def test_req_learn_5368_blocked_when_tests_not_recorded() -> None:
    """REQ-LEARN-5368-7: ready gate stays false without test records."""

    artifact = exp.build_result_artifact(root=REPO, tests_run=[])

    assert artifact["status"]["value"] == "blocked_budget_curated_memory_gate"
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_budget_curated_memory_not_ready:"
    )
    assert "tests_recorded" in artifact["readiness_gate"]["failed_gates"]
    assert "tests_not_recorded" in artifact["honest_verdict"]["value"]
    assert artifact["budget_curated_memory_ready"] is False
    assert artifact["unsafe_false_accepts"] == 0
    exp.validate_artifact(artifact)


def test_req_learn_5368_blocked_when_source_gate_fails(monkeypatch) -> None:
    """REQ-LEARN-5368-1: failed upstream gates block the budget fixture."""

    failed_gate = {
        "all_passed": False,
        "dependency_provenance_ready": False,
        "memory_tool_drift_ready": True,
        "self_learning_scaleup_ready": True,
        "source_unsafe_false_accepts_zero": True,
        "rollback_recovery_ready": True,
        "no_weight_mutation": True,
        "failed_gates": ["dependency_provenance_ready"],
        "source_artifacts": [],
    }
    monkeypatch.setattr(exp, "confirm_source_gate", lambda root=REPO: failed_gate)

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit budget memory governance", "outcome": "passed"}],
    )

    assert artifact["budget_curated_memory_ready"] is False
    assert artifact["memory_item_count"] == 0
    assert artifact["retained_bytes"] == 0
    assert artifact["readiness_gate"]["source_gate_passed"] is False
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_budget_curated_memory_not_ready:"
    )
    exp.validate_artifact(artifact)


def test_req_learn_5368_helper_branches_are_deterministic() -> None:
    """REQ-LEARN-5368-3: helper branches fail closed deterministically."""

    untrusted_low_value = exp.MemoryItem(
        memory_id="mem5368-helper-untrusted-low-value",
        memory_variant="helper",
        provenance={
            "source_artifact": "results/helper.json",
            "source_ref": "helper",
            "evidence_summary": "helper branch",
            "verified": False,
        },
        byte_cost=16,
        estimated_verifier_value=0.01,
        stale_risk=0.0,
        poison_risk=0.0,
        sharing_risk=0.0,
        trust_label="unverified",
        useful=False,
        harmful=False,
        rollback_available=False,
    )

    assert exp._keep_decision(untrusted_low_value, "UNTRUST", 0, 64) == "DROP"
    assert exp._wrapped_value("plain") == "plain"
    assert exp._json_ready(Path("helper")) == "helper"


def test_req_learn_5368_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5368-7: artifact validation rejects scalar and gate drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit budget governance", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_context_memory"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_learning = deepcopy(artifact)
    bad_learning["continuous_self_learning_target"] = {"value": True}
    with pytest.raises(ValueError, match="continuous_self_learning_target"):
        exp.validate_artifact(bad_learning)

    bad_weight = deepcopy(artifact)
    bad_weight["no_weight_mutation"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_weight)

    bad_count = deepcopy(artifact)
    bad_count["memory_item_count"] = True
    with pytest.raises(ValueError, match="memory_item_count"):
        exp.validate_artifact(bad_count)

    bad_numeric = deepcopy(artifact)
    bad_numeric["keep_precision"] = {"value": 1.0}
    with pytest.raises(ValueError, match="keep_precision"):
        exp.validate_artifact(bad_numeric)

    bad_ready = deepcopy(artifact)
    bad_ready["budget_curated_memory_ready"] = "yes"
    with pytest.raises(ValueError, match="budget_curated_memory_ready"):
        exp.validate_artifact(bad_ready)

    bad_budget = deepcopy(artifact)
    bad_budget["retained_bytes"] = bad_budget["budget_bytes"] + 1
    with pytest.raises(ValueError, match="retained_bytes"):
        exp.validate_artifact(bad_budget)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_unsafe)

    bad_missing_tests = deepcopy(artifact)
    bad_missing_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_missing_tests)

    bad_gate = deepcopy(artifact)
    bad_gate["readiness_gate"]["value_cost_harm_trust_measured"] = False
    with pytest.raises(ValueError, match="value_cost_harm_trust_measured"):
        exp.validate_artifact(bad_gate)

    bad_field = deepcopy(artifact)
    bad_field["status"] = {"value": "budget_curated_memory_ready"}
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_field)
