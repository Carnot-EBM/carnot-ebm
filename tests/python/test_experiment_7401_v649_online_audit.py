"""Spec-linked tests for the independent V649 online audit.

Spec refs: REQ-REPORT-7401 and SCENARIO-REPORT-7401-DIAGNOSIS through
SCENARIO-REPORT-7401-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from carnot import experiment_7401_v649_online_audit as audit


def _bundle() -> dict:
    checkpoint = {
        "schema": "carnot.exp7399.numeric_checkpoint.v1",
        "seed": 7397001,
        "training_authority": "frozen_initialization_groups_only",
        "initial_group_count": 2,
        "initial_group_ids_sha256": audit.canonical_hash(["initial-0", "initial-1"]),
        "future_stream_groups_used": 0,
        "gibbs_steps": 2,
        "weights": {
            "w1": [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-0.5, 0.5]],
            "b1": [0.0, 0.0, 0.0, 0.0],
            "w_out": [1.0, -1.0, 0.5, -0.5],
            "b_out": 0.1,
        },
        "affine": {"a": 1.0, "b": -1.0},
        "logistic_weights": {"coef": [0.2, -0.1], "bias": -1.5},
    }
    stream = [
        {
            "group_id": f"later-{index}",
            "source_row_index": 10 + index,
            "partition": "training",
            "entity_uptake": uptake,
            "falsifiability_score": falsifiability,
            "label": label,
        }
        for index, (uptake, falsifiability, label) in enumerate(
            ((0.2, 0.1, 0), (0.8, 0.4, 1), (0.4, 0.9, 0), (0.7, 0.2, 1))
        )
    ]
    policy = {
        "accept_enabled": True,
        "accept_threshold": 0.25,
        "reject_enabled": True,
        "reject_threshold": 0.75,
    }
    condition = {"name": "primary_delay_1", "delay": 1, "missing_fraction": 0.0}
    return {
        "checkpoint": checkpoint,
        "initial_labels": [0, 1],
        "stream": stream,
        "policy": policy,
        "condition": condition,
    }


def test_independent_numeric_replay_covers_updates_and_actions() -> None:
    """REQ-REPORT-7401; SCENARIO-REPORT-7401-REPLAY."""

    bundle = _bundle()
    first_energy = audit.gibbs_energy(
        bundle["checkpoint"]["weights"],
        [bundle["stream"][0]["entity_uptake"], bundle["stream"][0]["falsifiability_score"]],
    )
    assert math.isfinite(first_energy)
    replay = audit.replay_registered_unit(
        bundle["checkpoint"],
        bundle["initial_labels"],
        bundle["stream"],
        ordering="fixed_hash_order",
        condition=bundle["condition"],
        seed=7397001,
        policy=bundle["policy"],
    )

    assert len(replay["rows"]) == len(audit.ARMS) * len(bundle["stream"])
    adaptive = [row for row in replay["rows"] if row["arm"] == "adaptive_affine_gibbs"]
    assert adaptive[0]["update_count"] == 0
    assert adaptive[-1]["update_count"] == 2
    assert all(row["brier_loss"] == pytest.approx((row["probability"] - row["label"]) ** 2) for row in adaptive)
    assert replay["restart_receipt"]["prediction_parity"] is True
    assert replay["erasure_receipt"]["passed"] is True
    assert all("original_answer_correct" in row and "typed_action_changed" in row for row in adaptive)


def test_producer_comparison_rejects_changed_probability_and_cost() -> None:
    """REQ-REPORT-7401; SCENARIO-REPORT-7401-REPLAY."""

    bundle = _bundle()
    replay = audit.replay_registered_unit(
        bundle["checkpoint"],
        bundle["initial_labels"],
        bundle["stream"],
        ordering="fixed_hash_order",
        condition=bundle["condition"],
        seed=7397001,
        policy=bundle["policy"],
    )
    producer = []
    for row in replay["rows"]:
        producer.append(
            {
                **{key: row[key] for key in audit.ROW_ID_FIELDS},
                "probability": row["probability"],
                "brier_loss": row["brier_loss"],
                "log_loss": row["log_loss"],
                "typed_action": row["typed_action"],
                "full_cost_s": 0.001,
                "prediction_before_feedback": True,
            }
        )
    compared = audit.compare_producer_rows(replay["rows"], producer)
    assert audit.audit_row_errors(compared, len(compared)) == []

    changed = deepcopy(compared)
    changed[0]["producer_probability"] += 0.1
    changed[0]["probability_matches"] = False
    assert "probability_mismatch" in audit.audit_row_errors(changed, len(changed))
    changed = deepcopy(compared)
    changed[0]["full_cost_s"] = None
    assert "missing_cost" in audit.audit_row_errors(changed, len(changed))


def test_all_six_private_mutations_are_rejected() -> None:
    """REQ-REPORT-7401; SCENARIO-REPORT-7401-MUTATIONS."""

    fixture = audit.build_fixture_artifact()
    mutations = audit.run_mutation_suite(fixture)
    assert [row["mutation"] for row in mutations] == list(audit.MUTATION_NAMES)
    assert all(row["rejected"] is True and row["rejecting_check"] for row in mutations)


def test_complete_null_keeps_audit_and_value_separate() -> None:
    """REQ-REPORT-7401; SCENARIO-REPORT-7401-VALUE."""

    fixture = audit.build_fixture_artifact()
    fixture["producer_eligibility"]["eligible"] = True
    fixture["producer_registered_efficacy"]["passed"] = False
    fixture["mutation_rows"] = audit.run_mutation_suite(fixture)
    audit.finalize_fixture(fixture)

    reduced = audit.independent_reduce(fixture)
    assert reduced["online_audit_complete_score"] == 1
    assert reduced["online_value_confirmed_score"] == 0
    assert audit.classify_terminal(True, True, False) == (
        "complete_online_audit_null",
        "null",
        "complete_null_online_registered_benefit_not_confirmed",
    )
    assert audit.validate_artifact(fixture) == []


def test_ineligible_present_evidence_is_disqualified_but_diagnosed() -> None:
    """REQ-REPORT-7401; SCENARIO-REPORT-7401-DIAGNOSIS."""

    assert audit.classify_terminal(False, True, False) == (
        "disqualified_online_audit_input",
        "disqualified",
        "complete_disqualified_online_input_diagnosed",
    )
    assert audit.classify_terminal(None, False, False)[1] == "blocked"
    bad = audit.build_fixture_artifact()
    bad["producer_eligibility"]["eligible"] = False
    bad["online_value_confirmed_score"] = 1
    assert "value_requires_eligible_producer" in audit.validate_artifact(bad)


def test_artifact_validation_rejects_identity_scores_and_checksum() -> None:
    """REQ-REPORT-7401; SCENARIO-REPORT-7401-ARTIFACT."""

    fixture = audit.build_fixture_artifact()
    fixture["mutation_rows"] = audit.run_mutation_suite(fixture)
    fixture["producer_eligibility"]["eligible"] = True
    fixture["producer_registered_efficacy"]["passed"] = False
    audit.finalize_fixture(fixture)
    assert audit.validate_artifact(fixture) == []

    changed = deepcopy(fixture)
    changed["MODEL_SPECS"] = ["historical-model"]
    changed["promotion_score"] = 1
    changed["reproducibility_checksum"] = "sha256:changed"
    errors = audit.validate_artifact(changed)
    assert "substrate_declaration_mismatch" in errors
    assert "promotion_nonzero" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_precondition_rows_keep_exact_failure_and_source_class() -> None:
    """REQ-REPORT-7401; SCENARIO-REPORT-7401-DIAGNOSIS."""

    passed = audit.precondition_row("producer_status", "producer.json", "status", "complete", "complete")
    failed = audit.precondition_row("producer_flag", "producer.json", "flag", False, True)
    summary = audit.gate_summary([passed, failed])
    assert passed["passed"] is True
    assert summary["first_required_failure"]["artifact_field"] == "flag"
    assert audit.source_outcome(False, []) == "blocked"
    assert audit.source_outcome(True, [failed]) == "disqualified"
    assert audit.source_outcome(True, [passed]) == "eligible"
