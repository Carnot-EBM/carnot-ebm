"""Tests for the primary-receipt reliability opportunity stream.

Spec refs: REQ-LEARN-6871 and SCENARIO-LEARN-6871-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6871_observable_reliability_opportunity_stream as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(label: str) -> str:
    """Make a stable test identity with the production hash helper."""

    return exp.sha256_json(label)


def _transaction(
    row_id: str = "order_1::family-a::event-1::remove",
    *,
    position: int = 1,
    order_id: str = "order_1",
    family: str = "family-a",
    event_id: str = "family-a|ordinary-01|seed-1|typed",
) -> dict[str, object]:
    """Build one complete row-level transaction receipt."""

    return {
        "row_id": row_id,
        "event_id": event_id,
        "chronological_key": f"{position:08d}:{event_id}",
        "chronological_position": position,
        "order_id": order_id,
        "source_family": family,
        "split": "development",
        "operation_kind": "retrieve",
        "operation_admitted": True,
        "operation_id": f"{order_id}::{family}::{position:03d}",
        "write_operation_id": f"{order_id}::{family}::{position - 1:03d}",
        "read_operation_id": f"{order_id}::{family}::{position:03d}",
        "action_identity": _sha(f"action:{row_id}"),
        "causal_edge_id": _sha(f"edge:{row_id}"),
        "outcome_identity": _sha(f"outcome:{row_id}"),
        "parent_state_sha256": _sha(f"parent:{row_id}"),
        "new_state_sha256": _sha(f"next:{row_id}"),
        "decision_snapshot_sha256": _sha(f"snapshot:{row_id}"),
        "source_proposal_sha256": _sha(f"proposal:{row_id}"),
        "receipt_sha256": _sha(f"receipt:{row_id}"),
        "counterfactual_applicable": True,
        "counterfactual_kind": "remove",
        "counterfactual_transformation": {
            "operation": "remove_writer",
            "expected_read": "absent",
        },
        "legal_alternative_count": 1,
    }


def _outcome(
    source_event_row_id: str,
    *,
    direction: int = 1,
    correction_family: str | None = None,
) -> dict[str, object]:
    """Build one later row-level exact-outcome receipt."""

    exact = {
        "outcome_identity": _sha(f"outcome:{source_event_row_id}"),
        "exact_outcome_hash": _sha(f"exact:{source_event_row_id}:{direction}"),
        "signed_direction": direction,
    }
    row: dict[str, object] = {
        "row_id": f"outcome::{source_event_row_id}",
        "row_sha256": _sha(f"outcome-row:{source_event_row_id}"),
        "source_event_row_id": source_event_row_id,
        "arm": "no_memory",
        "seed": 1,
        "exact_outcome": exact,
        "decision_frozen_before_outcome_reveal": True,
        "outcome_revealed_after_decision": True,
    }
    if correction_family is not None:
        row["correction_family"] = correction_family
        row["reveal_delay_events"] = 2
        row["correction_latency_events"] = 2
    return row


def _one_reconstructed_row() -> dict[str, object]:
    """Reconstruct one valid event through the public receipt path."""

    transaction = _transaction()
    outcomes, conflicts = exp.build_outcome_index(
        {"rows": [_outcome(str(transaction["row_id"]))]}, {"rows": []}
    )
    rows, rejected = exp.reconstruct_opportunities([transaction], outcomes)
    assert conflicts == []
    assert rejected == []
    assert len(rows) == 1
    return rows[0]


def test_missing_primary_receipt_writes_complete_blocked_artifact(tmp_path: Path) -> None:
    # SCENARIO-LEARN-6871-MISSING-PRIMARY
    artifact = exp.build_artifact(tmp_path, "20260902", live_reports={})

    assert artifact["observable_reliability_stream_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == (
        "complete_blocked_observable_reliability_opportunity_stream"
    )
    assert artifact["gate_check_summary"]["failed_check"] == "required_source_readability"
    assert "transactions" in artifact["gate_check_summary"]["observed"]
    assert exp.validate_artifact(artifact) == []


def test_duplicate_event_identity_fails_closed() -> None:
    # SCENARIO-LEARN-6871-DUPLICATE
    row = _one_reconstructed_row()
    duplicate = deepcopy(row)
    duplicate["decision_sequence_index"] = 2

    errors = exp.validate_stream([row, duplicate])

    assert any(error.startswith("duplicate_event_identity:") for error in errors)


def test_outcome_known_at_decision_is_reported_as_leakage() -> None:
    # SCENARIO-LEARN-6871-LEAKAGE
    row = _one_reconstructed_row()
    row["decision_features"]["later_exact_outcome"] = 1

    witnesses = exp.find_leakage_witnesses([row])

    assert witnesses == [
        {
            "event_identity": row["event_identity"],
            "feature": "later_exact_outcome",
            "reason": "offline_or_unknown_feature_in_decision_context",
        }
    ]
    assert "decision_feature_leakage" in exp.validate_stream([row])


def test_reordered_events_fail_the_frozen_base_order() -> None:
    # SCENARIO-LEARN-6871-ORDER
    first = _transaction(position=1, row_id="order_1::family-a::event-1::remove")
    second = _transaction(position=2, row_id="order_1::family-a::event-2::remove")
    outcome_rows = [_outcome(str(first["row_id"])), _outcome(str(second["row_id"]))]
    outcome_index, _ = exp.build_outcome_index({"rows": outcome_rows}, {"rows": []})
    rows, rejected = exp.reconstruct_opportunities([second, first], outcome_index, sort_rows=False)

    assert rejected == []
    assert "chronological_order_mismatch" in exp.validate_stream(rows)


def test_absent_counterfactual_is_rejected_without_fabrication() -> None:
    # SCENARIO-LEARN-6871-COUNTERFACTUAL
    transaction = _transaction()
    transaction["counterfactual_applicable"] = False
    transaction["legal_alternative_count"] = 0
    outcome_index, _ = exp.build_outcome_index(
        {"rows": [_outcome(str(transaction["row_id"]))]}, {"rows": []}
    )

    rows, rejected = exp.reconstruct_opportunities([transaction], outcome_index)

    assert rows == []
    assert rejected[0]["event_identity"] == transaction["row_id"]
    assert "absent_pre_action_counterfactual" in rejected[0]["reasons"]
    assert "fabricated_outcome" not in rejected[0]


def test_stale_evidence_keeps_quarantine_and_disables_update() -> None:
    # SCENARIO-LEARN-6871-STALE
    transaction = _transaction(event_id="family-a|stale_prerequisites-01|seed-1|typed")
    outcome_index, _ = exp.build_outcome_index(
        {"rows": [_outcome(str(transaction["row_id"]))]}, {"rows": []}
    )
    rows, _ = exp.reconstruct_opportunities([transaction], outcome_index)

    support = exp.action_support(rows[0])

    assert support["bounded_update"]["supported"] is False
    assert support["bounded_update"]["reason"] == "stale_evidence_at_decision"
    assert support["quarantine"]["supported"] is True


def test_delayed_correction_is_offline_and_routes_attack_to_quarantine() -> None:
    # SCENARIO-LEARN-6871-DELAYED-CORRECTION
    transaction = _transaction()
    outcome_index, _ = exp.build_outcome_index(
        {
            "rows": [
                _outcome(
                    str(transaction["row_id"]),
                    correction_family="stale_evidence",
                )
            ]
        },
        {"rows": []},
    )
    rows, _ = exp.reconstruct_opportunities([transaction], outcome_index)
    delayed = rows[0]["later_exact_outcome"]["delayed_correction"]
    attacks = {row["attack_id"]: row for row in exp.transition_attack_manifest()}

    assert delayed["correction_family"] == "stale_evidence"
    assert "delayed_correction" not in rows[0]["decision_features"]
    assert attacks["delayed_invalidation"]["failure_action"] == "quarantine"
    assert attacks["delayed_invalidation"]["detected"] is True


def test_poison_restart_and_rollback_attacks_fail_closed() -> None:
    # SCENARIO-LEARN-6871-POISON
    # SCENARIO-LEARN-6871-RESTART
    # SCENARIO-LEARN-6871-ROLLBACK
    attacks = {row["attack_id"]: row for row in exp.transition_attack_manifest()}

    assert attacks["poison_nonfinite"]["detected"] is True
    assert attacks["poison_over_bound"]["detected"] is True
    assert attacks["restart_loss"]["detected"] is True
    assert attacks["rollback_mismatch"]["detected"] is True
    assert all(
        attacks[name]["failure_action"] == "quarantine"
        for name in (
            "poison_nonfinite",
            "poison_over_bound",
            "restart_loss",
            "rollback_mismatch",
        )
    )
    assert exp.initial_reliability_state_schema()["initial_state_unchanged_by_fixture"] is True


def test_outcome_conflict_and_missing_outcome_are_rejected() -> None:
    # REQ-LEARN-6871
    transaction = _transaction()
    first = _outcome(str(transaction["row_id"]), direction=1)
    second = _outcome(str(transaction["row_id"]), direction=-1)
    outcome_index, conflicts = exp.build_outcome_index({"rows": [first, second]}, {"rows": []})
    rows, rejected = exp.reconstruct_opportunities([transaction], {})

    assert outcome_index == {}
    assert conflicts[0]["source_event_identity"] == transaction["row_id"]
    assert rows == []
    assert rejected[0]["reasons"] == ["missing_later_exact_outcome"]


def test_malformed_primary_documents_and_validator_loader_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SCENARIO-LEARN-6871-MISSING-PRIMARY
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")

    assert exp._read_json(invalid)[1] == "unreadable:JSONDecodeError"
    assert exp._read_json(array) == ({}, "json_object_required")

    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *_args: None)
    with pytest.raises(RuntimeError, match="adversarial verifier import failed"):
        exp._load_validator(REPO_ROOT)


def test_malformed_transaction_and_outcome_receipts_are_ignored() -> None:
    # REQ-LEARN-6871
    transaction = _transaction()
    transaction["action_identity"] = "not-a-hash"
    reasons = exp._transaction_rejection_reasons(transaction)
    assert "invalid_content_identity:action_identity" in reasons

    invalid_outcomes = [
        None,
        {},
        {**_outcome("event"), "exact_outcome": {"outcome_identity": "bad"}},
        {
            **_outcome("event"),
            "exact_outcome": {
                "outcome_identity": _sha("outcome"),
                "exact_outcome_hash": _sha("exact"),
                "signed_direction": True,
            },
        },
        {**_outcome("event"), "decision_frozen_before_outcome_reveal": False},
        {**_outcome("event"), "outcome_revealed_after_decision": False},
    ]
    index, conflicts = exp.build_outcome_index({"rows": "not-a-list"}, {"rows": invalid_outcomes})
    assert index == {}
    assert conflicts == []


def test_non_mapping_context_and_all_stream_integrity_failures_are_visible() -> None:
    # SCENARIO-LEARN-6871-LEAKAGE
    row = _one_reconstructed_row()
    non_mapping = deepcopy(row)
    non_mapping["decision_features"] = []
    assert exp.find_leakage_witnesses([non_mapping]) == [
        {
            "event_identity": row["event_identity"],
            "feature": "decision_features",
            "reason": "decision_context_must_be_mapping",
        }
    ]

    broken = deepcopy(row)
    broken["decision_sequence_index"] = 7
    broken["primary_source"]["transaction_receipt_sha256"] = "bad"
    broken["counterfactual"]["valid_pre_action"] = False
    broken["later_exact_outcome"]["revealed_after_decision"] = False
    broken["primary_content_sha256"] = _sha("wrong-content")
    errors = exp.validate_stream([broken])
    assert "decision_sequence_mismatch" in errors
    assert any(error.startswith("invalid_primary_provenance:") for error in errors)
    assert any(error.startswith("invalid_counterfactual:") for error in errors)
    assert any(error.startswith("invalid_later_exact_outcome:") for error in errors)
    assert any(error.startswith("primary_content_identity_mismatch:") for error in errors)


def test_sealed_outcome_identity_joins_to_primary_transaction() -> None:
    # REQ-LEARN-6871
    transaction = _transaction()
    sealed_identity = exp.sha256_json(str(transaction["row_id"]))
    outcome_index, conflicts = exp.build_outcome_index(
        {"rows": []}, {"rows": [_outcome(sealed_identity, direction=-1)]}
    )

    rows, rejected = exp.reconstruct_opportunities([transaction], outcome_index)

    assert conflicts == []
    assert rejected == []
    assert rows[0]["later_exact_outcome"]["signed_direction"] == -1
    assert rows[0]["later_exact_outcome"]["source_artifact"] == "outcomes_b"


def test_order_replicates_are_deterministic_and_preserve_anchors() -> None:
    # REQ-LEARN-6871
    rows = []
    for position, family in enumerate(("family-a", "family-b", "family-c"), start=1):
        transaction = _transaction(
            row_id=f"order_1::{family}::event-{position}::remove",
            position=position,
            family=family,
            event_id=f"{family}|ordinary-01|seed-{position}|typed",
        )
        outcome_index, _ = exp.build_outcome_index(
            {"rows": [_outcome(str(transaction["row_id"]))]}, {"rows": []}
        )
        rebuilt, _ = exp.reconstruct_opportunities([transaction], outcome_index)
        rows.extend(rebuilt)
    anchors = exp.select_old_family_anchors(rows)

    first = exp.deterministic_order_replicates(rows, anchors)
    second = exp.deterministic_order_replicates(rows, anchors)

    assert first == second
    assert len(first) >= 5
    anchor_ids = {row["event_identity"] for row in anchors}
    for replicate in first:
        assert set(replicate["old_family_anchor_event_identities"]) == anchor_ids
        assert replicate["held_out_family"] in {"family-a", "family-b", "family-c"}
        assert set(replicate["event_identities"]) == {row["event_identity"] for row in rows}


def test_real_artifact_reconstructs_only_primary_receipts() -> None:
    # REQ-LEARN-6871
    reports = exp.collect_live_adversarial_reports(REPO_ROOT)
    artifact = exp.build_artifact(REPO_ROOT, "20260902", live_reports=reports)

    assert artifact["observable_reliability_stream_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"] == (
        "complete_positive_observable_reliability_opportunity_stream_ready"
    )
    assert len(artifact["rows"]) == 765
    assert all(
        row["primary_source"]["transaction_artifact"] == "exp6827" for row in artifact["rows"]
    )
    assert all(
        row["later_exact_outcome"]["source_artifact"] in {"outcomes_a", "outcomes_b"}
        for row in artifact["rows"]
    )
    assert len(artifact["order_replicate_manifest"]) >= 5
    assert len(artifact["old_family_anchor_rows"]) == 3
    assert set(artifact["action_manifest"]) == {
        "no_memory",
        "read_only_retrieval",
        "bounded_update",
        "quarantine",
        "v599_unsafe_reference",
    }
    assert all(
        row["authoritative"] is False
        for row in artifact["primary_receipt_manifest"]["excluded_v599_aggregates"]
    )
    assert artifact["gate_check_summary"]["passed"] is True
    assert exp.validate_artifact(artifact) == []


def test_artifact_validation_catches_readiness_and_checksum_tampering() -> None:
    # REQ-LEARN-6871
    artifact = exp.build_artifact(
        REPO_ROOT, "20260902", live_reports=exp.collect_live_adversarial_reports(REPO_ROOT)
    )
    changed = deepcopy(artifact)
    changed["rows"][0]["decision_features"]["audit_label"] = "helpful"
    changed["observable_reliability_stream_ready_score"] = 1
    changed["gate_check_summary"]["passed"] = False
    changed["verdict_class"] = "null"

    errors = exp.validate_artifact(changed)

    assert "ready_artifact_has_failed_gate" in errors
    assert "ready_artifact_has_leakage" in errors
    assert "ready_artifact_must_be_positive" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_artifact_validation_rejects_malformed_and_inconsistent_blocked_shapes() -> None:
    # REQ-LEARN-6871
    malformed_errors = exp.validate_artifact({})
    assert any(error.startswith("missing_required_fields:") for error in malformed_errors)
    assert "field_principles_must_cover_every_field" in malformed_errors
    assert "invalid_inference_substrate" in malformed_errors
    assert "verifier_is_oracle_must_be_false" in malformed_errors
    assert "invalid_verdict_class" in malformed_errors
    assert "honest_verdict_not_terminal" in malformed_errors
    assert "invalid_ready_score" in malformed_errors

    blocked = exp.build_artifact(Path("/nonexistent/exp6871-test"), "20260902", live_reports={})
    blocked["gate_check_summary"] = {}
    blocked["verdict_class"] = "null"
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked_artifact_requires_failed_check" in blocked_errors
    assert "blocked_artifact_must_use_blocked_class" in blocked_errors


def test_main_refuses_to_write_an_invalid_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # REQ-LEARN-6871
    monkeypatch.setattr(exp, "build_artifact", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="missing_required_fields"):
        exp.main(["--date", "20260902", "--output", str(tmp_path / "bad.json")])


def test_main_writes_requested_output_without_touching_the_real_result(
    tmp_path: Path,
) -> None:
    # REQ-LEARN-6871
    output = tmp_path / "experiment_6871.json"

    assert exp.main(["--date", "20260902", "--output", str(output)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["observable_reliability_stream_ready_score"] == 1
    assert exp.validate_artifact(artifact) == []
