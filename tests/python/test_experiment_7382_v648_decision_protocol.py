"""Tests for the experiment-local calibrated decision protocol.

Spec refs: REQ-AUTO-018 and SCENARIO-AUTO-7382-01 through
SCENARIO-AUTO-7382-05.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7382_v648_decision_protocol as protocol


ROOT = Path(__file__).resolve().parents[2]


def _row(question_id: object, text: object, label: object) -> dict[str, object]:
    """Build one compact archive row for grouping tests."""

    return {"question_id": question_id, "step_text": text, "label": label}


def test_scenario_auto_7382_01_connected_groups_and_quarantine() -> None:
    """SCENARIO-AUTO-7382-01 joins transitive duplicates and quarantines bad rows."""

    rows = [
        _row("q1", "alpha", "correct"),
        _row("q1", "beta", "incorrect"),
        _row("q2", " beta\n", "correct"),
        _row("q3", "gamma", "mystery"),
        _row(None, "delta", "correct"),
        _row("q4", "", "correct"),
    ]
    groups, quarantine = protocol.build_connected_groups(rows)

    assert len(groups) == 1
    assert groups[0]["row_indices"] == [0, 1, 2]
    assert groups[0]["incorrect_count"] == 1
    assert [row["reason"] for row in quarantine] == [
        "unknown_label",
        "missing_question_id",
        "missing_step_text",
    ]
    assert protocol.normalize_step_text("  beta\n") == "beta"


def test_scenario_auto_7382_01_partition_is_fixed_and_group_disjoint() -> None:
    """SCENARIO-AUTO-7382-01 assigns every complete group to exactly one role."""

    rows = [_row(f"q{i}", f"step {i}", "incorrect" if i % 7 == 0 else "correct") for i in range(80)]
    groups, quarantine = protocol.build_connected_groups(rows)
    first = protocol.assign_partitions(groups)
    second = protocol.assign_partitions(deepcopy(groups))

    assert quarantine == []
    assert first == second
    assert set(first) == {str(group["group_id"]) for group in groups}
    assert set(first.values()) <= set(protocol.PARTITION_NAMES)


def test_req_auto_018_actual_archive_has_locked_class_support() -> None:
    """REQ-AUTO-018 checks the frozen salt against the exact 6,548-row archive."""

    rows = json.loads((ROOT / "data/fover_corpus_v4.json").read_text(encoding="utf-8"))
    groups, quarantine = protocol.build_connected_groups(rows)
    memberships = protocol.assign_partitions(groups)
    summary = protocol.partition_summary(groups, memberships)

    assert len(rows) == 6548
    assert sum(group["incorrect_count"] for group in groups) == 114
    assert quarantine == []
    assert all(summary[name]["incorrect_rows"] >= 10 for name in protocol.PARTITION_NAMES)
    assert all(summary[name]["effective_groups"] > 0 for name in protocol.PARTITION_NAMES)


@pytest.mark.parametrize(
    ("energy", "expected"),
    [(-1000.0, 0.0), (0.0, 0.5), (1000.0, 1.0)],
)
def test_scenario_auto_7382_02_stable_probability_extremes(energy: float, expected: float) -> None:
    """SCENARIO-AUTO-7382-02 keeps extreme energy conversion finite."""

    probability = protocol.energy_to_probability(energy)
    assert probability == expected
    assert math.isfinite(probability)


def test_scenario_auto_7382_02_prior_correction_and_affine_control() -> None:
    """SCENARIO-AUTO-7382-02 uses training odds and a separate affine map."""

    prevalence = 0.02
    corrected = protocol.energy_to_probability(0.0, prevalence=prevalence)
    affine = protocol.energy_to_probability(0.0, affine=(2.0, -1.0))

    assert corrected == pytest.approx(prevalence)
    assert affine == pytest.approx(1.0 / (1.0 + math.e))
    with pytest.raises(ValueError, match="prevalence"):
        protocol.energy_to_probability(0.0, prevalence=0.0)
    with pytest.raises(ValueError, match="finite"):
        protocol.energy_to_probability(float("nan"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        protocol.energy_to_probability(0.0, prevalence=0.2, affine=(1.0, 0.0))
    with pytest.raises(ValueError, match="affine parameters"):
        protocol.energy_to_probability(0.0, affine=(float("inf"), 0.0))


@pytest.mark.parametrize(
    ("probability", "decision", "confidence"),
    [(0.005, "accept", 0.995), (0.4, "escalate", 0.6), (0.9, "reject", 0.1)],
)
def test_scenario_auto_7382_03_typed_decision_confidence_semantics(
    probability: float, decision: str, confidence: float
) -> None:
    """SCENARIO-AUTO-7382-03 always defines confidence as correctness."""

    result = protocol.typed_decision(
        probability,
        accept_threshold=0.01,
        reject_threshold=0.75,
        model_version="fixture-v1",
    )

    assert result == {
        "decision": decision,
        "p_incorrect": probability,
        "confidence_correct": pytest.approx(confidence),
        "model_version": "fixture-v1",
        "reason": f"p_incorrect_{decision}_region",
    }


@pytest.mark.parametrize("probability", [None, -0.1, 1.1, float("nan")])
def test_scenario_auto_7382_03_invalid_input_escalates(probability: float | None) -> None:
    """SCENARIO-AUTO-7382-03 escalates invalid input without invented confidence."""

    result = protocol.typed_decision(
        probability,
        accept_threshold=0.01,
        reject_threshold=0.75,
        model_version="fixture-v1",
    )

    assert result["decision"] == "escalate"
    assert result["p_incorrect"] is None
    assert result["confidence_correct"] is None
    assert result["reason"] == "invalid_probability"


def test_scenario_auto_7382_04_exact_certificate_and_empty_action() -> None:
    """SCENARIO-AUTO-7382-04 disables empty actions and retains exact counts."""

    empty = protocol.exact_risk_certificate([], risk_budget=0.05)
    nonempty = protocol.exact_risk_certificate([0] * 200, risk_budget=0.05)

    assert empty == {
        "selected_groups": 0,
        "harmful_outcomes": 0,
        "upper_risk_bound": None,
        "risk_budget": 0.05,
        "alpha_familywise": protocol.FAMILYWISE_ALPHA,
        "alpha_per_test": protocol.ALPHA_PER_TEST,
        "certified": False,
        "action_enabled": False,
        "reason": "no_selected_groups_no_certificate",
    }
    assert nonempty["selected_groups"] == 200
    assert nonempty["harmful_outcomes"] == 0
    assert 0.0 < nonempty["upper_risk_bound"] < 0.05
    assert nonempty["certified"] is True
    one_harm = protocol.exact_risk_certificate([1, *([0] * 199)], risk_budget=0.05)
    all_harm = protocol.exact_risk_certificate([1, 1], risk_budget=0.05)
    assert one_harm["upper_risk_bound"] > nonempty["upper_risk_bound"]
    assert all_harm["upper_risk_bound"] == 1.0
    assert protocol._binomial_cdf(0, 1, 0.0) == 1.0
    assert protocol._binomial_cdf(0, 1, 1.0) == 0.0
    assert protocol._binomial_cdf(1, 1, 1.0) == 1.0
    with pytest.raises(ValueError, match="binary"):
        protocol.exact_risk_certificate([0, 2], risk_budget=0.05)


def test_req_auto_018_policy_metrics_handles_empty_accept_and_reject() -> None:
    """REQ-AUTO-018 does not describe an all-escalate policy as useful."""

    metrics = protocol.typed_policy_metrics(
        labels=[0, 1],
        decisions=["escalate", "escalate"],
    )

    assert metrics["coverage"] == 0.0
    assert metrics["incorrect_accept_risk"] is None
    assert metrics["correct_reject_risk"] is None
    assert metrics["useful"] is False
    with pytest.raises(ValueError, match="length"):
        protocol.typed_policy_metrics([0], [])
    with pytest.raises(ValueError, match="labels"):
        protocol.typed_policy_metrics([2], ["accept"])
    with pytest.raises(ValueError, match="decision"):
        protocol.typed_policy_metrics([0], ["defer"])

    decided = protocol.typed_policy_metrics(
        labels=[0, 1, 1, 0],
        decisions=["accept", "accept", "reject", "reject"],
    )
    assert decided["coverage"] == 1.0
    assert decided["incorrect_accept_risk"] == 0.5
    assert decided["correct_reject_risk"] == 0.5
    assert decided["utility"] == 0.0


def test_req_auto_018_group_proper_scores_and_one_class_rejection() -> None:
    """REQ-AUTO-018 scores complete groups and rejects one-class fitting data."""

    metrics = protocol.per_group_proper_scores(
        labels=[0, 1, 0],
        probabilities=[0.1, 0.8, 0.3],
        group_ids=["a", "a", "b"],
    )

    assert metrics["effective_groups"] == 2
    assert metrics["brier"] == pytest.approx(((0.01 + 0.04) / 2 + 0.09) / 2)
    assert metrics["log_loss"] > 0.0
    protocol.require_two_classes([0, 1])
    with pytest.raises(ValueError, match="both labels"):
        protocol.require_two_classes([0, 0])
    with pytest.raises(ValueError, match="common length"):
        protocol.per_group_proper_scores([], [], [])
    with pytest.raises(ValueError, match="valid"):
        protocol.per_group_proper_scores([2], [0.5], ["a"])


def test_scenario_auto_7382_05_reader_boundaries_keep_test_labels_sealed() -> None:
    """SCENARIO-AUTO-7382-05 reserves final-test labels for the evaluator."""

    rows = [
        {"row_index": i, "group_id": f"g{i}", "partition": name, "label": i % 2}
        for i, name in enumerate(protocol.PARTITION_NAMES)
    ]
    readers = protocol.PartitionReaders(rows)

    assert readers.read_training()[0]["label"] == 0
    assert readers.read_probability_calibration()[0]["label"] == 1
    assert readers.read_policy_calibration()[0]["label"] == 0
    assert "label" not in readers.read_final_test()[0]
    with pytest.raises(PermissionError, match="trusted evaluator"):
        readers.read_final_test_labels("wrong")
    assert readers.read_final_test_labels(protocol.TRUSTED_EVALUATOR_TOKEN)[0]["label"] == 1


def test_req_auto_018_online_replay_is_ordered_and_group_disjoint() -> None:
    """REQ-AUTO-018 freezes an independent ordered replay within training only."""

    groups = [{"group_id": f"g{i}", "row_indices": [i], "incorrect_count": i % 2} for i in range(8)]
    memberships = {f"g{i}": "training" if i < 6 else "final_test" for i in range(8)}
    replay = protocol.build_online_replay(groups, memberships)

    initial = replay["initialization_group_ids"]
    later = replay["later_group_ids"]
    assert len(initial) == 3
    assert len(later) == 3
    assert set(initial).isdisjoint(later)
    assert set(initial) | set(later) == {f"g{i}" for i in range(6)}
    assert replay["moving_block_draws"] == 10_000
    assert replay["block_length"] == 32
    assert replay["sensitivity_block_length"] == 64
    assert replay["random_seed"] == 7_386_307


def test_req_auto_018_manifest_and_fixture_are_frozen() -> None:
    """REQ-AUTO-018 publishes the complete frozen protocol without a value claim."""

    manifest = protocol.protocol_manifest()
    fixture = protocol.run_analytic_fixture()

    assert manifest["architecture"] == {
        "input_dim": 2,
        "hidden_dims": [4],
        "output_dim": 1,
        "parameter_count": 17,
    }
    assert manifest["seeds"] == [7_382_001, 7_382_002, 7_382_003, 7_382_004, 7_382_005]
    assert manifest["optimizer"]["max_steps"] == 500
    assert len(manifest["arms"]) == 5
    assert manifest["threshold_pairs"] == [list(pair) for pair in protocol.THRESHOLD_PAIRS]
    assert manifest["bootstrap"] == {"draws": 10_000, "seed": 7_382_307, "unit": "group"}
    assert all(row["passed"] for row in fixture["analytic_checks"])
    assert all(row["passed"] for row in fixture["mutation_checks"])
    assert {row["decision"] for row in fixture["typed_decision_fixture_rows"]} == {
        "accept",
        "reject",
        "escalate",
    }


def test_req_auto_018_readiness_reducer_fails_closed() -> None:
    """REQ-AUTO-018 requires partitions, class support, analytic checks, and validation."""

    checks = {
        "source_ready": True,
        "partitions_sealed": True,
        "class_support": True,
        "analytic_checks": True,
        "mutation_checks": True,
        "required_validation": True,
        "safety": True,
    }
    assert protocol.reduce_readiness(checks) == 1
    for name in checks:
        mutated = {**checks, name: False}
        assert protocol.reduce_readiness(mutated) == 0


def test_req_auto_018_artifact_validator_rejects_mutation() -> None:
    """REQ-AUTO-018 independently checks identity, zero LLM calls, and checksum."""

    artifact = protocol.build_fixture_artifact()
    assert protocol.validate_artifact(artifact) == []

    mutated = deepcopy(artifact)
    mutated["model_invoked"] = True
    assert "current_model_declaration_mismatch" in protocol.validate_artifact(mutated)
    mutated = deepcopy(artifact)
    mutated["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in protocol.validate_artifact(mutated)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "wrong", "identity_mismatch"),
        ("invocation_counts", {}, "current_invocation_counts_nonzero"),
        ("inference_substrate_class", "aggregation", "substrate_class_mismatch"),
        ("execution_venue", "kv260", "execution_venue_mismatch"),
        ("verdict_class", "success", "verdict_class_invalid"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("decision_protocol_ready_score", 0, "readiness_reduction_mismatch"),
        ("calibration_value_score", 1, "forbidden_value_or_promotion"),
        ("flagged_adversarial", True, "adversarial_readiness_nonzero"),
    ],
)
def test_req_auto_018_artifact_validator_covers_fail_closed_fields(
    field: str, value: object, error: str
) -> None:
    """REQ-AUTO-018 rejects every closed declaration when independently mutated."""

    artifact = protocol.build_fixture_artifact()
    artifact[field] = value
    artifact["reproducibility_checksum"] = protocol.reproducibility_checksum(artifact)
    assert error in protocol.validate_artifact(artifact)


def test_req_auto_018_artifact_validator_and_gate_reducer_boundaries() -> None:
    """REQ-AUTO-018 keeps malformed objects and failed gate details explicit."""

    assert protocol.validate_artifact([]) == ["artifact_not_object"]
    passed = protocol._gate("support", "safety", 10, 10, ">=")
    failed_required = protocol._gate("reader", "safety", True, False)
    failed_science = protocol._gate("value", "scientific_efficacy", True, False)
    summary = protocol._gate_summary([passed, failed_required, failed_science])
    assert passed["passed"] is True
    assert summary["failed_required_count"] == 1
    assert summary["first_required_failure"]["check"] == "reader"
    assert summary["failed_scientific_gate_count"] == 1
    assert summary["first_scientific_failure"]["check"] == "value"
