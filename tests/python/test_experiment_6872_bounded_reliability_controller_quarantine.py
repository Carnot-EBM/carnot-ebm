"""Tests for bounded reliability updates and exact write quarantine.

Spec refs: REQ-LEARN-6872 and SCENARIO-LEARN-6872-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6872_bounded_reliability_controller_quarantine as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(label: str) -> str:
    """Return one stable content identity for a compact exact fixture."""

    return exp.sha256_json(label)


def _event(
    identity: str = "event-1",
    *,
    direction: int = 1,
    bucket: int = 0,
    stale: bool = False,
    delayed: bool = False,
    family: str = "family-a",
) -> dict[str, object]:
    """Build one event with separate decision and later-outcome fields."""

    suffix = f"{bucket:02x}"
    return {
        "event_identity": identity,
        "primary_content_sha256": _sha(f"primary:{identity}"),
        "primary_source": {
            "transaction_artifact": "exp6827",
            "transaction_receipt_sha256": _sha(f"receipt:{identity}"),
            "transaction_row_id": identity,
        },
        "decision_features": {
            "source_identity": "exp6827",
            "source_content_sha256": f"sha256:{'0' * 62}{suffix}",
            "family": family,
            "chronological_position": bucket + 1,
            "operation_kind": "retrieve",
            "candidate_action_identity": _sha(f"action:{identity}"),
            "pre_action_state_sha256": _sha(f"parent:{identity}"),
            "decision_snapshot_sha256": _sha(f"snapshot:{identity}"),
            "counterfactual_kind": "remove",
            "counterfactual_available": True,
            "evidence_status_at_decision": "stale" if stale else "fresh",
        },
        "counterfactual": {
            "kind": "remove",
            "legal_alternative_count": 1,
            "valid_pre_action": True,
            "action_specific_outcome_fabricated": False,
        },
        "action_support": {
            "bounded_update": {"supported": not stale},
            "no_memory": {"supported": True},
            "read_only_retrieval": {"supported": True},
            "quarantine": {"supported": True},
        },
        "later_exact_outcome": {
            "signed_direction": direction,
            "exact_outcome_hash": _sha(f"outcome:{identity}:{direction}"),
            "outcome_identity": _sha(f"outcome-id:{identity}"),
            "revealed_after_decision": True,
            "source_artifact": "outcomes_a",
            "source_row_sha256": _sha(f"outcome-row:{identity}"),
            "delayed_correction": {
                "correction_family": "stale_evidence" if delayed else None,
                "correction_latency_events": 2 if delayed else 0,
                "reveal_delay_events": 2 if delayed else 0,
            },
        },
    }


def _source_document(events: list[dict[str, object]] | None = None) -> dict[str, object]:
    """Build the complete Exp6871 contract consumed by the controller."""

    rows = events or [
        _event("event-write", bucket=0),
        _event("event-read", bucket=3),
        _event("event-none", bucket=5, direction=-1),
        _event("event-abstain", bucket=6),
        _event("event-stale", bucket=1, stale=True),
    ]
    identities = [str(row["event_identity"]) for row in rows]
    return {
        "observable_reliability_stream_ready_score": 1,
        "leakage_witnesses": {"passed": True, "clean_stream_witnesses": []},
        "counterfactual_support_rows": [
            {"event_identity": identity, "valid_pre_action": True} for identity in identities
        ],
        "action_manifest": {
            name: {}
            for name in (
                "no_memory",
                "read_only_retrieval",
                "bounded_update",
                "quarantine",
                "v599_unsafe_reference",
            )
        },
        "bounded_update_contract": {
            "updates_applied": False,
            "update_timing": "after_action_freeze_and_later_exact_outcome_reveal",
            "same_event_outcome_may_select_action": False,
            "max_abs_entry_delta_per_event": 0.05,
            "max_spectral_norm_delta_per_event": 0.1,
            "nonfinite_feedback_policy": "reject",
            "over_bound_feedback_policy": "reject",
            "state_before_and_after_hash_required": True,
        },
        "chronological_order_manifest": {
            "event_identities": identities,
            "base_order_sha256": exp.sha256_json(identities),
            "base_order_is_chronological": True,
        },
        "order_replicate_manifest": [
            {
                "replicate_id": "order_replicate_1",
                "seed": 6_871_011,
                "event_identities": identities,
                "all_events_preserved": True,
            }
        ],
        "old_family_anchor_rows": [
            {
                "family": "family-a",
                "event_identity": identities[0],
                "primary_content_sha256": rows[0]["primary_content_sha256"],
            }
        ],
        "rows": rows,
    }


def _write_source(root: Path, document: dict[str, object] | None = None) -> Path:
    """Write one source fixture at the production relative path."""

    path = root / exp.SOURCE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document or _source_document()), encoding="utf-8")
    return path


def test_state_asymmetry_and_excessive_spectral_change_are_rejected() -> None:
    # SCENARIO-LEARN-6872-STATE
    state = exp.initialize_reliability_state()
    asymmetric = deepcopy(state)
    asymmetric[0][1] = 0.01

    after_asymmetry, asymmetry = exp.bounded_symmetric_update(
        asymmetric, "exp6827_transactions", "write", 1.0
    )
    after_excess, excess = exp.bounded_symmetric_update(
        state,
        "exp6827_transactions",
        "write",
        1.0,
        learning_rate=0.11,
    )

    assert asymmetry["applied"] is False
    assert "state_asymmetry" in asymmetry["failure_reasons"]
    assert after_asymmetry == asymmetric
    assert excess["applied"] is False
    assert "spectral_change_exceeds_bound" in excess["failure_reasons"]
    assert after_excess == state
    assert excess["state_before_sha256"] == excess["state_after_sha256"]


def test_update_is_symmetric_bounded_and_applied_only_after_outcome() -> None:
    # REQ-LEARN-6872
    state = exp.initialize_reliability_state()
    after, receipt = exp.bounded_symmetric_update(state, "exp6840_outcomes", "read_only", 1.0)

    assert after != state
    assert exp.matrix_is_symmetric(after)
    assert receipt["applied"] is True
    assert receipt["max_abs_entry_delta"] == pytest.approx(0.05)
    assert receipt["spectral_norm_delta"] <= 0.1
    assert receipt["update_timing"] == ("after_action_freeze_and_later_exact_outcome_reveal")
    assert receipt["state_before_sha256"] != receipt["state_after_sha256"]

    unchanged, early = exp.bounded_symmetric_update(
        state,
        "exp6840_outcomes",
        "read_only",
        1.0,
        outcome_revealed_after_action=False,
    )
    assert unchanged == state
    assert early["failure_reasons"] == ["outcome_not_revealed_after_action"]


def test_nonfinite_unknown_nodes_and_same_event_access_fail_closed() -> None:
    # SCENARIO-LEARN-6872-STATE
    # SCENARIO-LEARN-6872-TIMING
    state = exp.initialize_reliability_state()
    for kwargs, reason in (
        ({"reward": float("nan")}, "nonfinite_feedback"),
        ({"source_node": "unknown"}, "unknown_state_node"),
        ({"action_frozen": False}, "action_not_frozen"),
        ({"same_event_outcome_access": True}, "same_event_outcome_access"),
    ):
        call = {
            "source_node": "exp6827_transactions",
            "action_node": "write",
            "reward": 1.0,
            **kwargs,
        }
        after, receipt = exp.bounded_symmetric_update(state, **call)
        assert after == state
        assert reason in receipt["failure_reasons"]

    assert exp.matrix_is_symmetric([[0.0]]) is False
    invalid = exp.initialize_reliability_state()
    invalid[0][0] = "not-a-number"  # type: ignore[assignment]
    assert exp.matrix_is_symmetric(invalid) is False


def test_same_event_outcome_is_rejected_from_decision_features() -> None:
    # SCENARIO-LEARN-6872-TIMING
    features = deepcopy(_event()["decision_features"])
    features["later_exact_outcome"] = {"signed_direction": 1}

    assert exp.decision_feature_failures(features) == [
        "offline_feature_in_decision_context:later_exact_outcome"
    ]
    with pytest.raises(ValueError, match="offline_feature_in_decision_context"):
        exp.choose_action("bounded_update", features, exp.initialize_reliability_state())
    with pytest.raises(ValueError, match="unknown arm"):
        exp.choose_action(
            "unknown", _event()["decision_features"], exp.initialize_reliability_state()
        )


def test_harmful_and_unsupported_writes_are_quarantined() -> None:
    # SCENARIO-LEARN-6872-QUARANTINE
    memory = [{"event_identity": "anchor", "family": "family-a"}]
    harmful, unchanged_harm = exp.evaluate_write_transition(_event(direction=-1), memory)
    unsupported, unchanged_insert = exp.evaluate_write_transition(
        _event(), memory, attack={"proposal_kind": "unsupported_insert"}
    )

    assert harmful["admission_decision"] == "quarantined"
    assert harmful["checks"]["source_faithfulness"] is False
    assert unsupported["checks"]["coverage"] is False
    assert unchanged_harm == memory
    assert unchanged_insert == memory
    assert harmful["parent_memory_sha256"] == harmful["restored_memory_sha256"]


@pytest.mark.parametrize(
    ("event", "attack", "failed_check"),
    [
        (_event(delayed=True), {}, "delayed_invalidation"),
        (_event(), {"drop_existing": True}, "preservation"),
        (_event(), {"drop_anchor_family": "family-a"}, "old_family_retention"),
        (_event(), {"replay_passed": False}, "replay"),
        (_event(), {"restart_drift": True}, "restart"),
        (_event(), {"rollback_failure": True}, "rollback"),
        (_event(), {"remove_provenance": True}, "provenance"),
    ],
)
def test_exact_transition_attacks_keep_parent_bytes(
    event: dict[str, object], attack: dict[str, object], failed_check: str
) -> None:
    # SCENARIO-LEARN-6872-QUARANTINE
    memory = [{"event_identity": "anchor", "family": "family-a"}]
    receipt, after = exp.evaluate_write_transition(event, memory, attack=attack)

    assert receipt["admission_decision"] == "quarantined"
    assert receipt["checks"][failed_check] is False
    assert after == memory
    assert receipt["rollback_byte_exact"] is (failed_check != "rollback")


def test_exact_supported_write_is_admitted_and_restartable() -> None:
    # REQ-LEARN-6872
    memory = [{"event_identity": "anchor", "family": "family-a"}]
    receipt, after = exp.evaluate_write_transition(_event(), memory)

    assert receipt["admission_decision"] == "admitted"
    assert receipt["failed_checks"] == []
    assert len(after) == 2
    assert all(receipt["checks"].values())
    assert receipt["restart_byte_exact"] is True
    assert receipt["rollback_byte_exact"] is True


def test_always_abstain_and_always_write_collapse_readiness() -> None:
    # SCENARIO-LEARN-6872-COLLAPSE
    base = {
        "arm": "bounded_update",
        "admission_decision": "admitted",
        "harmful_write": False,
    }
    spectral = [{"arm": "bounded_update", "within_bound": True}]

    for action in ("abstain", "write"):
        rows = [{**base, "proposed_action": action} for _ in range(2)]
        ready, checks = exp.compute_readiness(rows, spectral, rows, [], expected_event_count=2)
        assert ready == 0
        assert checks["multiple_actions"] is False


def test_complete_fixture_compares_all_arms_and_is_ready(tmp_path: Path) -> None:
    # SCENARIO-LEARN-6872-ROWS
    source_path = _write_source(tmp_path)
    artifact = exp.build_artifact(tmp_path, "20260902", source_relative_path=source_path)

    assert artifact["bounded_reliability_controller_ready_score"] == 1
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert len(artifact["rows"]) == 5 * 5
    assert {row["arm"] for row in artifact["rows"]} == set(exp.ARMS)
    assert all(row["action_frozen_before_outcome"] for row in artifact["rows"])
    assert all(row["outcome_revealed_after_action"] for row in artifact["rows"])
    assert artifact["action_entropy_by_arm"]["bounded_update"] > 0.0
    assert artifact["admitted_update_rows"]
    assert not [
        row
        for row in artifact["harmful_write_rows"]
        if row["arm"] in {"bounded_update", "exact_quarantine"}
    ]
    assert (
        artifact["model_immutability_receipt"]["before_sha256"]
        == (artifact["model_immutability_receipt"]["after_sha256"])
    )
    assert exp.validate_artifact(artifact) == []

    with pytest.raises(ValueError, match="event decision and outcome objects are required"):
        exp.run_comparison([{"event_identity": "bad", "decision_features": None}], order_seed=1)


def test_missing_precondition_writes_complete_blocked_shape(tmp_path: Path) -> None:
    # REQ-LEARN-6872
    artifact = exp.build_artifact(tmp_path, "20260902")

    assert artifact["bounded_reliability_controller_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == (
        "complete_blocked_bounded_reliability_controller_quarantine"
    )
    assert artifact["gate_check_summary"]["failed_check"] == "source_artifact_readable"
    assert artifact["gate_check_summary"]["expected"] == "readable JSON object"
    assert exp.validate_artifact(artifact) == []


def test_bad_stream_contract_is_blocked_with_exact_gate(tmp_path: Path) -> None:
    # REQ-LEARN-6872
    document = _source_document()
    document["bounded_update_contract"]["same_event_outcome_may_select_action"] = True
    source_path = _write_source(tmp_path, document)

    artifact = exp.build_artifact(tmp_path, "20260902", source_relative_path=source_path)

    assert artifact["gate_check_summary"]["failed_check"] == "frozen_update_contract"
    assert (
        artifact["gate_check_summary"]["observed"]["same_event_outcome_may_select_action"] is True
    )


def test_malformed_sources_and_collapsed_fixture_fail_closed(tmp_path: Path) -> None:
    # REQ-LEARN-6872
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp._read_json(malformed)[1] == "unreadable:JSONDecodeError"
    assert exp._read_json(array) == ({}, "json_object_required")

    one_event = _source_document([_event("only-write", bucket=0)])
    source_path = _write_source(tmp_path, one_event)
    artifact = exp.build_artifact(tmp_path, "20260902", source_relative_path=source_path)
    assert artifact["bounded_reliability_controller_ready_score"] == 0
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"] == (
        "complete_partial_bounded_reliability_controller_readiness_not_met"
    )


def test_weight_mutation_disqualifies_and_validator_detects_tampering(tmp_path: Path) -> None:
    # SCENARIO-LEARN-6872-WEIGHTS
    source_path = _write_source(tmp_path)
    artifact = exp.build_artifact(
        tmp_path,
        "20260902",
        source_relative_path=source_path,
        no_model_after_sha256=_sha("mutated-model-baseline"),
    )

    assert artifact["bounded_reliability_controller_ready_score"] == 0
    assert artifact["no_model_weight_mutation"] is False
    assert artifact["verdict_class"] == "disqualified"
    assert exp.validate_artifact(artifact) == []

    bad = deepcopy(artifact)
    bad["no_model_weight_mutation"] = True
    bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
    assert "model_weight_immutability_mismatch" in exp.validate_artifact(bad)


def test_artifact_validator_reports_schema_and_terminal_inconsistency(tmp_path: Path) -> None:
    # REQ-LEARN-6872
    source_path = _write_source(tmp_path)
    artifact = exp.build_artifact(tmp_path, "20260902", source_relative_path=source_path)
    bad = deepcopy(artifact)
    del bad["rows"]
    bad["field_principles"] = {}
    bad["inference_substrate"] = "wrong"
    bad["verifier_is_oracle"] = True
    bad["verdict_class"] = "invented"
    bad["honest_verdict"] = "not-terminal"
    bad["bounded_reliability_controller_ready_score"] = 2

    errors = exp.validate_artifact(bad)

    assert any(error.startswith("missing_required_fields:rows") for error in errors)
    assert "field_principles_must_cover_every_field" in errors
    assert "invalid_inference_substrate" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "invalid_verdict_class" in errors
    assert "honest_verdict_not_terminal" in errors
    assert "invalid_ready_score" in errors
    assert "reproducibility_checksum_mismatch" in errors

    invalid_ready = deepcopy(artifact)
    invalid_ready["gate_check_summary"]["passed"] = False
    invalid_ready["verdict_class"] = "partial"
    invalid_ready["no_model_weight_mutation"] = False
    invalid_ready["model_immutability_receipt"]["after_sha256"] = _sha("changed")
    invalid_ready["reproducibility_checksum"] = exp.reproducibility_checksum(invalid_ready)
    invalid_ready_errors = exp.validate_artifact(invalid_ready)
    assert "ready_artifact_has_failed_gate" in invalid_ready_errors
    assert "ready_artifact_must_be_positive" in invalid_ready_errors
    assert "ready_artifact_mutated_model" in invalid_ready_errors

    invalid_blocked = deepcopy(artifact)
    invalid_blocked["bounded_reliability_controller_ready_score"] = 0
    invalid_blocked["verdict_class"] = "blocked"
    invalid_blocked["gate_check_summary"]["failed_check"] = None
    invalid_blocked["reproducibility_checksum"] = exp.reproducibility_checksum(invalid_blocked)
    assert "blocked_artifact_requires_failed_check" in exp.validate_artifact(invalid_blocked)


def test_atomic_write_and_cli_emit_valid_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # REQ-LEARN-6872
    source_path = _write_source(tmp_path)
    output = tmp_path / "result.json"

    exit_code = exp.main(
        [
            "--date",
            "20260902",
            "--root",
            str(tmp_path),
            "--source",
            str(source_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert exp.validate_artifact(written) == []
    assert json.loads(capsys.readouterr().out)["ready_score"] == 1
    assert not output.with_suffix(".json.tmp").exists()


def test_cli_refuses_an_invalid_built_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-LEARN-6872
    _write_source(tmp_path)
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["invalid"])
    with pytest.raises(ValueError, match="invalid"):
        exp.main(
            [
                "--date",
                "20260902",
                "--root",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_checked_in_source_build_is_complete_and_prospective() -> None:
    # SCENARIO-LEARN-6872-ROWS
    artifact = exp.build_artifact(REPO_ROOT, "20260902")
    source = json.loads((REPO_ROOT / exp.SOURCE_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert len(artifact["rows"]) == len(source["rows"]) * len(exp.ARMS)
    assert artifact["bounded_reliability_controller_ready_score"] == 1
    assert artifact["gate_check_summary"]["passed"] is True
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert exp.validate_artifact(artifact) == []
