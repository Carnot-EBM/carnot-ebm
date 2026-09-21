"""Tests for REQ-REPORT-7484 and SCENARIO-REPORT-7484-*.

Small fixtures exercise the independent math and fail-closed controls. The
capability test also reduces the checked-in producer rows without changing them.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7484_v655_decision_audit as audit


def _native_pair(group: str = "g1", role: str = "external") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order in (
        ["supported", "contains_unsupported"],
        ["contains_unsupported", "supported"],
    ):
        mapping = {" A": order[0], " B": order[1]}
        rows.append(
            {
                "cell_id": f"{group}-{order[0]}",
                "source_group_id": group,
                "group_hash": f"hash-{group}",
                "role": role,
                "arm": "full_source_response",
                "eligible": True,
                "disposition": "complete",
                "option_order": order,
                "label_to_option_id": mapping,
                "display_labels": [" A", " B"],
                "label_token_ids": [357, 417],
                "raw_logits_by_option_id": {
                    "supported": 2.0,
                    "contains_unsupported": 1.0,
                },
                "probabilities_by_option_id": {
                    "supported": 1.0 / (1.0 + math.exp(-1.0)),
                    "contains_unsupported": 1.0 / (1.0 + math.exp(1.0)),
                },
                "gold_label": 1,
                "generated_tokens": 0,
            }
        )
    return rows


def _ledger_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "group_id": "g1",
        "source_family": "qa",
        "order_seed": 10,
        "audit_seed": 20,
        "delay": 0,
        "event_time": 0,
        "arm": "importance_anchor",
        "probability": 0.4,
        "prediction_state_hash": "state-0",
        "prediction_time": 0,
        "feedback_due_time": 0,
        "feedback_time": 0,
        "label": 1,
        "label_revealed": True,
        "learner_event_id": "10-0-0",
        "state_hash_before": "state-0",
        "state_hash_after": "state-1",
        "update_accepted": True,
        "update_status": "committed",
        "role": "online",
        "propensity": 0.5,
    }
    row.update(changes)
    row["prediction_event_hash"] = audit.prediction_event_hash(row)
    return row


def test_native_option_reconstruction_rejects_swapped_ids() -> None:
    """SCENARIO-REPORT-7484-STATIC: both token orders map to one risk score."""

    reduced = audit.reconstruct_native_groups(_native_pair())
    assert reduced["errors"] == []
    assert reduced["groups"][0]["native_log_odds"] == pytest.approx(-1.0)
    assert reduced["groups"][0]["label"] == 0

    swapped = deepcopy(_native_pair())
    swapped[0]["label_to_option_id"] = {" A": "contains_unsupported", " B": "supported"}
    assert (
        "option_id_mapping_mismatch:g1-supported"
        in audit.reconstruct_native_groups(swapped)["errors"]
    )


def test_frozen_checkpoint_scoring_and_metrics_are_independent() -> None:
    """REQ-REPORT-7484: frozen checkpoint math reproduces proper scores."""

    views = {"full": [1.0, 0.0], "verifier_only": [0.0, 0.0], "source_removal": [0.0, 0.0]}
    logistic = {
        "seed": 7,
        "checkpoint": {"coef": [2.0, 0.0], "bias": -1.0},
        "calibration_temperature": 1.0,
    }
    assert audit.score_frozen_state("logistic", logistic, views) == pytest.approx(
        1.0 / (1.0 + math.exp(-1.0))
    )
    rows = [
        {
            "role": "external",
            "group_id": "g1",
            "arm": "gibbs",
            "seed": 1,
            "label": 1,
            "probability": 0.8,
        },
        {
            "role": "external",
            "group_id": "g1",
            "arm": "gibbs",
            "seed": 2,
            "label": 1,
            "probability": 0.6,
        },
        {
            "role": "external",
            "group_id": "g1",
            "arm": "temperature",
            "seed": None,
            "label": 1,
            "probability": 0.5,
        },
        {
            "role": "external",
            "group_id": "g2",
            "arm": "gibbs",
            "seed": 1,
            "label": 0,
            "probability": 0.2,
        },
        {
            "role": "external",
            "group_id": "g2",
            "arm": "gibbs",
            "seed": 2,
            "label": 0,
            "probability": 0.4,
        },
        {
            "role": "external",
            "group_id": "g2",
            "arm": "temperature",
            "seed": None,
            "label": 0,
            "probability": 0.5,
        },
    ]
    metrics = audit.reduce_probability_rows(rows)
    assert metrics["external"]["gibbs"]["brier"] == pytest.approx(0.09)
    assert metrics["external"]["gibbs"]["log_loss"] > 0.0
    assert metrics["external"]["gibbs"]["class_support"] == {"0": 1, "1": 1}


def test_roster_and_delayed_replay_fail_closed() -> None:
    """SCENARIO-REPORT-7484-UPDATES: missing units and future labels fail."""

    pair = _native_pair()
    plan = [
        {key: row[key] for key in ("cell_id", "source_group_id", "role", "eligible")}
        for row in pair
    ]
    assert audit.audit_roster(plan, pair)["errors"] == []
    assert (
        "missing_observed_cell:g1-contains_unsupported"
        in audit.audit_roster(plan, pair[:1])["errors"]
    )

    first = _ledger_row()
    second = _ledger_row(
        group_id="g2",
        event_time=1,
        prediction_time=1,
        feedback_due_time=1,
        feedback_time=1,
        learner_event_id="10-0-1",
        prediction_state_hash="state-1",
        state_hash_before="state-1",
        state_hash_after="state-2",
    )
    replay = audit.replay_delayed_updates([first, second])
    assert replay["errors"] == []
    assert replay["accepted_updates"] == 2
    future = deepcopy([first, second])
    future[1]["feedback_time"] = 0
    future[1]["prediction_event_hash"] = audit.prediction_event_hash(future[1])
    assert "feedback_before_prediction:10-0-1" in audit.replay_delayed_updates(future)["errors"]


def test_all_private_attacks_reject_their_named_claims() -> None:
    """SCENARIO-REPORT-7484-ATTACKS: all seven corruptions are detected."""

    attacks = audit.run_attack_controls()
    assert [row["attack"] for row in attacks] == list(audit.REQUIRED_ATTACKS)
    assert all(row["rejected"] is True for row in attacks)
    assert len({row["check"] for row in attacks}) == 7


def test_real_reduction_reproduces_branches_and_scientific_null() -> None:
    """SCENARIO-REPORT-7484-BRANCHES: checked-in source rows reduce independently."""

    reduction = audit.audit_sources(audit.REPO_ROOT)
    assert [row["availability"] for row in reduction["branch_dispositions"]] == [
        "available",
        "available",
        "available",
        "available",
    ]
    assert reduction["errors"] == []
    assert reduction["static_summary"]["external_group_count"] == 74
    assert reduction["online_summary"]["independent_group_count"] == 159
    assert reduction["online_summary"]["scientific_benefit_passed"] is False


def test_artifact_contract_and_fresh_readers(tmp_path: Path, capsys) -> None:
    """SCENARIO-REPORT-7484-ARTIFACT: checksum and cold replay bind the result."""

    artifact = audit.build_artifact_for_test()
    assert audit.validate_artifact(artifact) == []
    assert artifact["audit_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert all("principle" in gate for gate in artifact["acceptance_gate_results"])
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.cold_replay(path) == []
    assert audit.independent_replay(path) == []
    assert audit.cold_replay(tmp_path / "missing.json") == ["candidate_json_invalid"]

    changed = deepcopy(artifact)
    changed["verifier_is_oracle"] = True
    assert "oracle_flag_mismatch" in audit.validate_artifact(changed)
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    assert "oracle_flag_mismatch" in audit.validate_artifact(changed)
    audit.progress(0.0, "test", "complete", completed_units=1)
    assert "completed_units=1" in capsys.readouterr().out


def test_fail_closed_shapes_and_exact_producer_identity(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7484-BRANCHES: malformed evidence cannot borrow a name."""

    results = tmp_path / "results"
    results.mkdir()
    spec = audit.PRODUCERS[0]
    wrong = results / spec.path.name
    wrong.write_text(json.dumps({"experiment_id": "wrong", "milestone": audit.MILESTONE}))
    assert audit.locate_producer(tmp_path, spec)["availability"] == "invalid"
    wrong.unlink()
    pre_gate = results / "conductor-pre-gate.json"
    pre_gate.write_text(
        json.dumps(
            {
                "experiment_id": spec.experiment_id,
                "milestone": audit.MILESTONE,
                "blocked_at_layer": "conductor_pre_gate",
                "verdict_class": "blocked",
                "flagged_adversarial": False,
            }
        )
    )
    assert audit.locate_producer(tmp_path, spec)["availability"] == "pre_gate"

    with pytest.raises(ValueError, match="feature_view_invalid"):
        audit.score_frozen_state("logistic", {"checkpoint": {}}, {})
    assert audit.validate_service_costs(
        [
            {
                "operation": "prediction",
                "observed_operations": 2,
                "total_s": 1.0,
                "mean_s_per_event": 0.8,
            }
        ]
    ) == ["fabricated_speedup:prediction"]


def test_defensive_reducers_cover_each_corruption_boundary(tmp_path: Path) -> None:
    """REQ-REPORT-7484: every independent guard returns its named contradiction."""

    pair = _native_pair()
    assert (
        "option_order_pair_invalid:g1:full_source_response"
        in audit.reconstruct_native_groups(pair[:1])["errors"]
    )
    malformed = deepcopy(pair)
    malformed[0].pop("raw_logits_by_option_id")
    assert any(
        "native_readout_missing" in error
        for error in audit.reconstruct_native_groups(malformed)["errors"]
    )
    malformed = deepcopy(pair)
    malformed[0]["probabilities_by_option_id"] = {
        "supported": 0.5,
        "contains_unsupported": 0.5,
    }
    malformed[0]["generated_tokens"] = 1
    errors = audit.reconstruct_native_groups(malformed)["errors"]
    assert any("native_probability_mismatch" in error for error in errors)
    assert any("generation_not_zero" in error for error in errors)

    plan = [{"cell_id": "a", "source_group_id": "g", "role": "external", "eligible": True}]
    roster_errors = audit.audit_roster(plan, [{"cell_id": "b"}, {"cell_id": "b"}])["errors"]
    assert "unexpected_observed_cell:b" in roster_errors
    assert "duplicate_observed_cell:b" in roster_errors

    with pytest.raises(ValueError, match="feature_view_invalid"):
        audit.score_frozen_state(
            "logistic",
            {"checkpoint": {"coef": [1.0], "bias": "bad"}},
            {"full": [1.0]},
        )
    events = [_ledger_row(), _ledger_row(event_time=1, learner_event_id="x")]
    events[0]["prediction_event_hash"] = "changed"
    events[0]["label_revealed"] = False
    events[1]["state_hash_before"] = "wrong"
    events[1]["prediction_event_hash"] = audit.prediction_event_hash(events[1])
    replay_errors = audit.replay_delayed_updates(events)["errors"]
    assert any("prediction_event_hash_mismatch" in error for error in replay_errors)
    assert any("unrevealed_label_update" in error for error in replay_errors)
    assert any("state_chain_mismatch" in error for error in replay_errors)

    assert audit._invocation_errors(
        {
            "invocation_counts": {
                "model_loads": {"attempted": 1, "completed": 0, "in_flight": 1},
                "generation_calls": {"attempted": 1, "completed": 1},
            }
        },
        "fixture",
    ) == [
        "invocation_imbalance:fixture:model_loads",
        "generation_calls_nonzero:fixture",
    ]
    bad_bundle = {
        "heads": {"logistic": [{"seed": 1, "checkpoint": {}, "checkpoint_sha256": "bad"}]},
        "bundle_sha256": "bad",
    }
    assert set(audit._checkpoint_hash_errors(bad_bundle)) == {
        "checkpoint_hash_mismatch:logistic:1",
        "bundle_hash_mismatch",
    }
    reconstructed = [
        {
            "role": "external",
            "group_id": "g",
            "arm": "gibbs",
            "seed": 1,
            "label": 1,
            "probability": 0.7,
        }
    ]
    assert audit._prediction_errors(reconstructed, []) == ["prediction_roster_mismatch"]
    changed = deepcopy(reconstructed)
    changed[0]["probability"] = 0.2
    assert any(
        "prediction_value_mismatch" in error
        for error in audit._prediction_errors(reconstructed, changed)
    )
    assert audit._retention_summary([{"used_for_update": True}])["errors"] == [
        "retention_label_used_for_update"
    ]
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}", encoding="utf-8")
    assert audit._checkpoint_file_errors(
        tmp_path,
        [{"path": "checkpoint.json", "sha256": "bad", "order_seed": 1, "delay": 0}],
    ) == ["changed_checkpoint:1:0"]
    ledger = audit._ledger_fixture()
    corrupt_checkpoint = {
        "checkpoint_sha256": "bad",
        "predictions": [
            {
                "event_id": "not-in-ledger",
                "label": 1,
                "pre_update_probability": 0.9,
                "prediction_time": 0,
                "reveal_time": 0,
                "predictor_state_hash": "wrong",
            }
        ],
    }
    checkpoint.write_text(json.dumps(corrupt_checkpoint), encoding="utf-8")
    prediction_errors = audit._checkpoint_prediction_errors(
        tmp_path,
        [{"path": "checkpoint.json", "order_seed": 1, "delay": 0}],
        ledger,
    )
    assert {
        "checkpoint_content_hash_mismatch:1:0",
        "future_label_in_prediction_record:not-in-ledger",
        "checkpoint_prediction_mismatch:not-in-ledger",
        "checkpoint_prediction_roster_mismatch",
    } <= set(prediction_errors)


def test_terminal_classification_and_contract_mutations_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7484-ARTIFACT: terminal classes do not borrow readiness."""

    artifact = audit.build_artifact_for_test()
    reduction = audit.audit_sources(audit.REPO_ROOT)
    disqualified = audit._build_artifact(reduction, [])
    assert disqualified["verdict_class"] == "disqualified"
    blocked_reduction = deepcopy(reduction)
    blocked_reduction["branch_dispositions"][0]["availability"] = "pre_gate"
    blocked = audit._build_artifact(blocked_reduction, audit._passing_receipts())
    assert blocked["verdict_class"] == "blocked"
    positive_reduction = deepcopy(reduction)
    positive_reduction["static_summary"]["scientific_benefit_passed"] = True
    positive_reduction["online_summary"]["scientific_benefit_passed"] = True
    positive = audit._build_artifact(positive_reduction, audit._passing_receipts())
    assert positive["verdict_class"] == "positive"

    mutations = (
        ("field_principles", {}),
        ("schema", "wrong"),
        ("run_date", "wrong"),
        ("MODEL_SPECS", ["wrong"]),
        ("invocation_counts", {}),
        ("execution_venue", "wrong"),
        ("attack_rows", []),
        ("branch_dispositions", []),
        ("rows", []),
        ("acceptance_gate_results", []),
        ("audit_complete_score", 0),
    )
    observed = set()
    missing = deepcopy(artifact)
    missing.pop("schema")
    observed.update(audit.validate_artifact(missing))
    for field, replacement in mutations:
        changed = deepcopy(artifact)
        changed[field] = replacement
        observed.update(audit.validate_artifact(changed))
    assert {
        "required_fields_missing",
        "artifact_identity_mismatch",
        "run_date_mismatch",
        "current_model_provenance_mismatch",
        "current_invocation_counts_nonzero",
        "substrate_or_venue_mismatch",
        "attack_controls_incomplete",
        "branch_dispositions_incomplete",
        "independent_rows_missing",
        "gate_principles_missing",
        "audit_incomplete",
        "reproducibility_checksum_mismatch",
    } <= observed
    assert audit.independent_replay(tmp_path / "missing.json") == ["candidate_json_invalid"]
    candidate = tmp_path / "candidate.json"
    changed = deepcopy(artifact)
    changed["static_summary"] = {}
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    candidate.write_text(json.dumps(changed), encoding="utf-8")
    assert "independent_replay_mismatch:static_summary" in audit.independent_replay(candidate)

    absent = audit.audit_sources(tmp_path)
    assert absent["independent_metric_rows"] == []
    assert all(row["availability"] == "absent" for row in absent["branch_dispositions"])
