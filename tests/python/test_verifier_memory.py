"""Tests for the V477 verifier-memory promotion ledger.

Spec refs: REQ-LEARN-5214, SCENARIO-LEARN-5214.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline.verifier_memory import (
    assert_no_test_gold_leak,
    decide_promotion,
    dedupe_memory_entries,
    make_memory_entry,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / "results" / "experiment_5214_continuous_self_learning_verifier_memory_v477.json"
MEMORY_PATH = REPO / "results" / "verifier_memory_v477.json"


def _passing_guard() -> dict[str, object]:
    return {
        "passed": True,
        "checks": {
            "deterministic_scoring": True,
            "leakage_audit_passed": True,
        },
        "no_test_gold_leak": True,
    }


def test_req_learn_5214_spec_declares_verifier_memory_contract() -> None:
    """REQ-LEARN-5214: OpenSpec names the durable memory schema and policy."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5214") :]

    for marker in (
        "REQ-LEARN-5214",
        "SCENARIO-LEARN-5214",
        "failure_signature",
        "candidate_predicate_or_set",
        "deterministic_guard_result",
        "heldout_delta",
        "promotion_state",
        "rollback_reason",
        "source_artifacts",
    ):
        assert marker in section


def test_req_learn_5214_promotes_when_guard_and_heldout_delta_clear() -> None:
    """REQ-LEARN-5214-1: passing guards plus useful held-out delta promote."""

    decision = decide_promotion(
        deterministic_guard_result=_passing_guard(),
        heldout_delta=0.041797,
        promotion_threshold=0.02,
    )

    assert decision.promotion_state == "promoted"
    assert decision.reason == "heldout_delta_clears_promotion_threshold"
    assert decision.rollback_reason is None


def test_req_learn_5214_holds_small_positive_delta() -> None:
    """REQ-LEARN-5214-2: weak positive held-out evidence is held."""

    decision = decide_promotion(
        deterministic_guard_result=_passing_guard(),
        heldout_delta=0.005,
        promotion_threshold=0.02,
    )

    assert decision.promotion_state == "held"
    assert decision.reason == "heldout_delta_below_promotion_threshold"
    assert decision.rollback_reason is None


@pytest.mark.parametrize(
    ("guard", "delta", "rollback_reason"),
    [
        ({"passed": False, "checks": {"leakage_audit_passed": True}}, 0.04, "deterministic_guard_failed"),
        (_passing_guard(), None, "heldout_delta_missing"),
        (_passing_guard(), 0.0, "heldout_delta_null"),
        (_passing_guard(), -0.01, "heldout_delta_negative"),
    ],
)
def test_req_learn_5214_rolls_back_failed_or_null_candidates(
    guard: dict[str, object],
    delta: float | None,
    rollback_reason: str,
) -> None:
    """REQ-LEARN-5214-3: failed guards and null deltas roll back."""

    decision = decide_promotion(
        deterministic_guard_result=guard,
        heldout_delta=delta,
        promotion_threshold=0.02,
    )

    assert decision.promotion_state == "rolled_back"
    assert decision.rollback_reason == rollback_reason


def test_req_learn_5214_policy_rejects_invalid_guard_and_delta_shapes() -> None:
    """REQ-LEARN-5214-1: malformed guards or deltas cannot promote."""

    assert (
        decide_promotion(
            deterministic_guard_result="not-a-guard",  # type: ignore[arg-type]
            heldout_delta=0.1,
        ).rollback_reason
        == "deterministic_guard_failed"
    )
    assert (
        decide_promotion(
            deterministic_guard_result={"passed": True, "no_test_gold_leak": False},
            heldout_delta=0.1,
        ).rollback_reason
        == "deterministic_guard_failed"
    )
    assert (
        decide_promotion(
            deterministic_guard_result={"passed": True, "leakage_audit_passed": False},
            heldout_delta=0.1,
        ).rollback_reason
        == "deterministic_guard_failed"
    )
    assert (
        decide_promotion(
            deterministic_guard_result={"passed": True, "checks": "unchecked"},
            heldout_delta=0.1,
        ).promotion_state
        == "promoted"
    )
    assert (
        decide_promotion(
            deterministic_guard_result=True,
            heldout_delta={"metric": "delta", "value": "not-a-number"},
        ).rollback_reason
        == "heldout_delta_invalid"
    )


def test_scenario_learn_5214_duplicate_memory_entries_are_idempotent() -> None:
    """SCENARIO-LEARN-5214: repeated upstream sightings collapse to one entry."""

    first = make_memory_entry(
        failure_signature="GAP-1: square transpose orientation blind spot",
        candidate_predicate_or_set={
            "kind": "discriminator_set",
            "members": ["color_centroid_orientation", "row_column_run_profile"],
        },
        provenance={"experiment": "experiment_5209"},
        deterministic_guard_result=_passing_guard(),
        heldout_delta={"metric": "min_delta_vs_baselines", "value": 0.041797},
        source_artifacts=["results/experiment_5209_gap1_set_search_holdout_hardening_v477.json"],
        promotion_threshold=0.02,
    )
    duplicate = make_memory_entry(
        failure_signature="GAP-1: square transpose orientation blind spot",
        candidate_predicate_or_set={
            "kind": "discriminator_set",
            "members": ["color_centroid_orientation", "row_column_run_profile"],
        },
        provenance={"experiment": "experiment_5209_replay"},
        deterministic_guard_result=_passing_guard(),
        heldout_delta={"metric": "min_delta_vs_baselines", "value": 0.041797},
        source_artifacts=["ops/verifier_gaps.md"],
        promotion_threshold=0.02,
    )

    entries = dedupe_memory_entries([first, duplicate])

    assert len(entries) == 1
    assert entries[0]["memory_id"] == first["memory_id"]
    assert entries[0]["promotion_state"] == "promoted"
    assert entries[0]["source_artifacts"] == [
        "ops/verifier_gaps.md",
        "results/experiment_5209_gap1_set_search_holdout_hardening_v477.json",
    ]

    scalar_entry = make_memory_entry(
        failure_signature="GAP-1: scalar-delta smoke",
        candidate_predicate_or_set={"kind": "discriminator_set", "members": ["row_run"]},
        provenance={},
        deterministic_guard_result=True,
        heldout_delta=0.03,
        source_artifacts=[],
    )
    assert scalar_entry["deterministic_guard_result"] == {"passed": True, "checks": {}}
    assert scalar_entry["heldout_delta"]["metric"] == "heldout_delta"

    reconstructed = dedupe_memory_entries(
        [
            {
                "failure_signature": "GAP-X",
                "candidate_predicate_or_set": "not-a-mapping",
                "source_artifacts": [],
            }
        ]
    )
    assert reconstructed[0]["memory_id"].startswith("verifier-memory:")


def test_req_learn_5214_no_test_gold_leak_guard_rejects_forbidden_payloads() -> None:
    """REQ-LEARN-5214-4: memory entries do not carry test-gold labels."""

    clean_entry = make_memory_entry(
        failure_signature="GAP-4: guarded local candidate expansion lacks held-out delta",
        candidate_predicate_or_set={
            "kind": "candidate_pool_guard",
            "name": "same_shape_demo_perfect_transform_code_pool",
        },
        provenance={"experiment": "experiment_5211", "evaluation_boundary": "demo_only"},
        deterministic_guard_result={
            "passed": True,
            "checks": {"restricted_execution": True, "leakage_audit_passed": True},
            "no_test_gold_leak": True,
        },
        heldout_delta={"metric": "heldout_pass_at_2_delta", "value": None},
        source_artifacts=["results/experiment_5211_gap4_sota_local_candidate_expansion_v477.json"],
        promotion_threshold=0.02,
    )

    payload = {"entries": [clean_entry]}
    assert assert_no_test_gold_leak(payload) is True
    serialized = json.dumps(payload, sort_keys=True)
    assert "correct_for_eval_only" not in serialized
    assert "test_output" not in serialized
    assert clean_entry["promotion_state"] == "rolled_back"
    assert clean_entry["rollback_reason"] == "heldout_delta_missing"

    with pytest.raises(ValueError, match="test_output"):
        assert_no_test_gold_leak({"entry": {"test_output": [[1, 2], [3, 4]]}})
    with pytest.raises(ValueError, match="z_gold"):
        assert_no_test_gold_leak({"entry": {"candidate_id": "z_gold"}})


def test_req_learn_5214_result_artifact_reports_memory_policy() -> None:
    """REQ-LEARN-5214-5: Exp 5214 writes the result and memory artifact path."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))

    assert result["continuous_self_learning_task"]["value"] is True
    assert result["memory_artifact_path"]["value"] == "results/verifier_memory_v477.json"
    assert result["memory_entries_written"]["value"] == len(memory["entries"])
    assert result["promotions"]["value"] == 1
    assert result["rollbacks"]["value"] == 1
    assert result["deterministic_guardrails_enforced"]["value"] is True
    assert result["heldout_gate_required_for_promotion"]["value"] is True
    assert result["inference_substrate"]["value"] == "verifier_memory_from_upstream_artifacts"
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert "promotions_1" in result["honest_verdict"]["value"]
    assert "rollbacks_1" in result["honest_verdict"]["value"]
    assert result["tests_run"]["value"]

    assert_no_test_gold_leak(memory)
    assert {entry["promotion_state"] for entry in memory["entries"]} == {
        "promoted",
        "rolled_back",
    }
