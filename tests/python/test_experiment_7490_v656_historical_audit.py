"""Tests for the V656 historical probability and feedback audit."""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from carnot import experiment_7490_v656_historical_audit as audit


# REQ-REPORT-7490; SCENARIO-REPORT-7490-POLARITY
def test_pinned_annotation_semantics_determine_probability_polarity() -> None:
    evaluator_rows = [
        {
            "group_id": "supported",
            "label": 1,
            "annotation_disposition": "supported",
            "label_policy": "one_if_no_human_unsupported_span",
        },
        {
            "group_id": "unsupported",
            "label": 0,
            "annotation_disposition": "contains_unsupported",
            "label_policy": "one_if_no_human_unsupported_span",
        },
    ]

    semantics = audit.resolve_label_polarity(evaluator_rows)

    assert semantics["evaluator_label_one"] == "supported"
    assert semantics["audit_target_one"] == "contains_unsupported"
    assert semantics["transform"] == "audit_target=1-evaluator_label"
    assert semantics["valid"] is True


# REQ-REPORT-7490; SCENARIO-REPORT-7490-PROBABILITY
def test_probability_accounting_reduces_groups_before_scoring() -> None:
    rows = [
        {
            "role": "external",
            "group_id": "a",
            "arm": "gibbs",
            "seed": 1,
            "label": 0,
            "probability": 0.2,
            "failed": False,
        },
        {
            "role": "external",
            "group_id": "a",
            "arm": "gibbs",
            "seed": 2,
            "label": 0,
            "probability": 0.4,
            "failed": False,
        },
        {
            "role": "external",
            "group_id": "b",
            "arm": "gibbs",
            "seed": 1,
            "label": 1,
            "probability": 0.6,
            "failed": False,
        },
        {
            "role": "external",
            "group_id": "b",
            "arm": "gibbs",
            "seed": 2,
            "label": 1,
            "probability": 0.8,
            "failed": False,
        },
        {
            "role": "external",
            "group_id": "a",
            "arm": "temperature",
            "seed": None,
            "label": 0,
            "probability": 0.1,
            "failed": False,
        },
        {
            "role": "external",
            "group_id": "b",
            "arm": "temperature",
            "seed": None,
            "label": 1,
            "probability": 0.9,
            "failed": False,
        },
    ]

    reduced = audit.recompute_probability_accounting(rows)

    gibbs = reduced["arms"]["gibbs"]
    assert gibbs["group_count"] == 2
    assert gibbs["fit_seed_count"] == 2
    assert gibbs["coverage"] == 1.0
    assert gibbs["class_support"] == {"supported": 1, "contains_unsupported": 1}
    assert math.isclose(gibbs["brier"], 0.09)
    assert math.isclose(gibbs["log_loss"], -math.log(0.7))
    assert reduced["arms"]["temperature"]["fit_seed_count"] == 0


# REQ-REPORT-7490; SCENARIO-REPORT-7490-PROBABILITY
def test_probability_accounting_rejects_bad_rows_and_retains_missing_coverage() -> None:
    base = [
        {
            "role": "external",
            "group_id": "a",
            "arm": "gibbs",
            "seed": 1,
            "label": 0,
            "probability": 0.2,
            "failed": False,
        },
        {
            "role": "external",
            "group_id": "b",
            "arm": "temperature",
            "seed": None,
            "label": 1,
            "probability": 0.8,
            "failed": False,
        },
    ]
    reduced = audit.recompute_probability_accounting(base)
    assert reduced["arms"]["gibbs"]["coverage"] == 0.5

    disagreement = [dict(base[0]), {**base[0], "label": 1}]
    with pytest.raises(ValueError, match="probability_label_disagreement"):
        audit.recompute_probability_accounting(disagreement)

    invalid = [dict(base[0], probability=1.5)]
    with pytest.raises(ValueError, match="probability_invalid"):
        audit.recompute_probability_accounting(invalid)


def _ledger_rows() -> list[dict[str, object]]:
    return [
        {
            "order_seed": 1,
            "audit_seed": 7,
            "delay": 0,
            "arm": "shuffled_feedback",
            "group_id": f"g{index}",
            "event_time": index,
            "prediction_time": index,
            "feedback_time": index,
            "feedback_due_time": index,
            "label": index % 2,
            "label_revealed": True,
            "update_accepted": True,
        }
        for index in range(6)
    ]


# REQ-REPORT-7490; SCENARIO-REPORT-7490-CHRONOLOGY
def test_global_shuffle_records_future_label_origins() -> None:
    rows = audit.reconstruct_shuffled_chronology(
        _ledger_rows(), order_seeds=[1], audit_seeds=[7], delays=[0]
    )
    summary = audit.validate_feedback_chronology(rows)

    assert len(rows) == 6
    assert summary["future_origin_assignment_count"] > 0
    assert summary["valid"] is False
    assert all(row["origin_group_id"] for row in rows)
    assert all("origin_available_time" in row for row in rows)

    with pytest.raises(ValueError, match="seed_count_mismatch"):
        audit.reconstruct_shuffled_chronology(
            _ledger_rows(), order_seeds=[1], audit_seeds=[], delays=[0]
        )


# REQ-REPORT-7490; SCENARIO-REPORT-7490-MUTATIONS
def test_future_mutation_fails_and_causal_block_permutation_passes() -> None:
    causal = audit.chronology_preserving_block_permutation(_ledger_rows(), block_size=2)
    assert audit.validate_feedback_chronology(causal)["valid"] is True

    mutated = deepcopy(causal)
    mutated[0]["origin_available_time"] = int(mutated[0]["assignment_time"]) + 1
    failed = audit.validate_feedback_chronology(mutated)

    assert failed["valid"] is False
    assert failed["future_origin_assignment_count"] == 1

    with pytest.raises(ValueError, match="block_size_must_be_positive"):
        audit.chronology_preserving_block_permutation(_ledger_rows(), block_size=0)


# REQ-REPORT-7490; SCENARIO-REPORT-7490-CHRONOLOGY
def test_claim_limits_do_not_transfer_control_leakage_to_main_learner() -> None:
    limits = audit.historical_claim_limits(
        probability_benefit=False,
        typed_utility=True,
        passed_cost_cells=7,
        total_cost_cells=9,
        online_benefit=False,
        future_control_assignments=3,
        main_feedback_violations=0,
    )

    assert limits["typed_utility"]["scope"] == "previously_evaluated_diagnostic_history"
    assert limits["probability_quality"]["finding"] == "null"
    assert limits["shuffled_control_validity"]["finding"] == "invalid_negative_control"
    assert limits["main_learner_future_label_use"]["finding"] == "not_observed"
    assert limits["counterfactual_control_used_for_efficacy"] is False

    main_rows = [
        {
            "arm": "importance_anchor",
            "update_accepted": True,
            "label_revealed": True,
            "prediction_time": 1,
            "feedback_time": 1,
        },
        {
            "arm": "importance_anchor",
            "update_accepted": True,
            "label_revealed": False,
            "prediction_time": 2,
            "feedback_time": 1,
        },
    ]
    assert audit._main_feedback_summary(main_rows) == {
        "accepted_update_count": 2,
        "future_label_violation_count": 1,
        "source_branch": "importance_anchor_uses_own_source_label_after_release",
    }


# REQ-REPORT-7490; SCENARIO-REPORT-7490-ARTIFACT
def test_artifact_contract_requires_canonical_current_provenance() -> None:
    value = audit.fixture_artifact()

    assert audit.validate_artifact(value, verify_files=False) == []
    assert value["historical_audit_complete_score"] == 1
    assert value["verdict_class"] == "null"
    assert set(value["field_principles"]) == set(value)

    changed = deepcopy(value)
    changed["inference_substrate"] = "aggregation_from_hash_bound_rows"
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    assert "current_provenance_invalid" in audit.validate_artifact(changed, verify_files=False)


# REQ-REPORT-7490; SCENARIO-REPORT-7490-ARTIFACT
def test_missing_external_input_is_blocked_not_partial() -> None:
    terminal = audit.classify_terminal(
        inputs_available=False,
        current_validation_passed=False,
        reduction_complete=False,
        scientific_benefit=False,
    )

    assert terminal == {
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_missing_historical_input",
        "historical_audit_complete_score": 0,
    }
    assert (
        audit.classify_terminal(
            inputs_available=True,
            current_validation_passed=False,
            reduction_complete=True,
            scientific_benefit=False,
        )["verdict_class"]
        == "disqualified"
    )
    assert (
        audit.classify_terminal(
            inputs_available=True,
            current_validation_passed=True,
            reduction_complete=True,
            scientific_benefit=True,
        )["verdict_class"]
        == "positive"
    )
