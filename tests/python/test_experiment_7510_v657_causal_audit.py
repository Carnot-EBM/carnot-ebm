"""Tests for REQ-REPORT-7510 and its independent causal-audit scenarios."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7510_v657_causal_audit as exp


def _write_json(path: Path, value: object) -> None:
    """Write compact fixture bytes where exact file identity is under test."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _valid_receipts(names: tuple[str, ...]) -> list[dict[str, object]]:
    """Build one successful receipt for each required upstream command."""

    return [{"name": name, "passed": True, "exit_code": 0, "timed_out": False} for name in names]


def _prototype() -> dict[str, object]:
    """Build the smallest terminal prototype accepted by the inventory gate."""

    return {
        "milestone": exp.MILESTONE,
        "terminal_status": "complete",
        "honest_verdict": "complete_circular_positive_fixture",
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "causal_update_ready_score": 1,
        "checkpoint_schema": {
            "required_parts": ["models", "pending_queue", "audit_rng_state", "order_cursor"]
        },
        "validation_receipts": _valid_receipts(exp.REQUIRED_UPSTREAM_RECEIPTS[7506]),
    }


def _online() -> dict[str, object]:
    """Use the module's compact complete event ledger in focused tests."""

    return exp.fixture_online_artifact()


# REQ-REPORT-7510; SCENARIO-REPORT-7510-INVENTORY.
def test_inventory_distinguishes_absent_invalid_and_valid(tmp_path: Path) -> None:
    inventory = exp.inventory_upstreams(tmp_path)
    assert [row["state"] for row in inventory] == ["absent", "absent"]

    _write_json(tmp_path / exp.UPSTREAM_PATHS[7506], _prototype())
    _write_json(tmp_path / exp.UPSTREAM_PATHS[7509], {"terminal_status": "complete"})
    inventory = exp.inventory_upstreams(tmp_path)
    assert [row["state"] for row in inventory] == ["valid", "invalid"]

    _write_json(tmp_path / exp.UPSTREAM_PATHS[7509], _online())
    inventory = exp.inventory_upstreams(tmp_path)
    assert [row["state"] for row in inventory] == ["valid", "valid"]
    assert all(str(row["sha256"]).startswith("sha256:") for row in inventory)


# REQ-REPORT-7510; SCENARIO-REPORT-7510-DAG.
def test_event_graph_reconciles_predictions_reveals_updates_and_retention() -> None:
    audited = exp.audit_event_history(_online())
    assert audited["errors"] == []
    assert audited["chronology_violation_count"] == 0
    assert audited["prediction_hash_mismatch_count"] == 0
    assert audited["retention_leakage_count"] == 0
    assert audited["sample_size_budget"] == {
        "planned": 4,
        "attempted": 4,
        "completed": 4,
        "excluded": 0,
        "failed": 0,
        "censored": 0,
        "unstarted": 0,
        "independent_unit": "source_group_by_role",
    }
    stream = audited["audit_rows"][0]
    assert stream["prediction_group_count"] == 2
    assert stream["released_label_count"] == 2
    assert stream["update_row_count"] == 2 * len(exp.ARMS)


# REQ-REPORT-7510; SCENARIO-REPORT-7510-DAG/MUTATIONS.
@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("future_label", "label_origin_outside_release_batch"),
        ("cross_batch_permutation", "shuffle_origin_not_a_permutation"),
        ("retroactive_prediction", "prediction_hash_mismatch"),
        ("lost_pending_update", "update_roster_mismatch"),
        ("favorable_seed_filtering", "stream_roster_mismatch"),
        ("retention_driven_rollback", "retention_control_leakage"),
    ],
)
def test_named_private_mutations_fail_closed(mutation: str, expected_error: str) -> None:
    value = _online()
    exp.apply_private_mutation(value, mutation)
    assert expected_error in exp.audit_event_history(value)["errors"]


# REQ-REPORT-7510; SCENARIO-REPORT-7510-MUTATIONS.
def test_all_private_mutation_controls_reject_without_publishing_payloads() -> None:
    results = exp.run_private_mutations()
    assert results == {name: True for name in exp.MUTATION_NAMES}


# REQ-REPORT-7510; SCENARIO-REPORT-7510-REDUCTION.
def test_source_averaging_precedes_bootstrap_and_holm() -> None:
    source = _online()
    audited = exp.audit_event_history(source)
    reduced = exp.reduce_measurement(
        source["per_source_results"],
        source["retention_rows"],
        audited["audit_rows"],
        settings=exp.fixture_reduction_settings(),
    )
    assert reduced["independent_unit"] == "schedule_seed_mean_within_source"
    assert reduced["primary_holm_family_size"] == 5
    assert len(reduced["primary_contrasts"]) == 5
    assert reduced["primary_contrasts"][0]["source_count"] == 2
    assert reduced["retention_contrast"]["source_count"] == 2
    assert {(row["delay"], row["block_length"]) for row in reduced["sensitivity_contrasts"]} == {
        (8, 8),
        (8, 32),
        (0, 16),
    }
    assert reduced["support_passed"] is False
    assert reduced["qualified_online_benefit"] is False


# REQ-REPORT-7510; SCENARIO-REPORT-7510-RESTART.
def test_checkpoint_audit_resolves_hashes_and_complete_restart_state(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    uninterrupted = root / "seed-1-delay-8" / "schedule-1-d8-b0.json"
    restarted = root / "restart-seed-1" / "schedule-1-d8-b0.json"
    payload = {
        "schema": exp.ONLINE_CHECKPOINT_SCHEMA,
        "cursor": 2,
        "heads": {"local_brier": {"coefficients": [0.1]}},
        "predictions": [{"group_id": "g0", "prediction_hash": "sha256:p"}],
        "updates": [{"group_id": "g0", "parameter_hash": "sha256:u"}],
        "reveal_rows": [{"block_id": 0, "event_ids": ["g0"]}],
        "pending_blocks": {"1": ["g1"]},
    }
    _write_json(uninterrupted, payload)
    _write_json(restarted, payload)
    producer = {
        "test_fixture": True,
        "protocol": {"schedule_seeds": [1], "primary_delay": 8},
        "checkpoint_hashes": [
            {
                "block_id": 0,
                "path": str(uninterrupted),
                "sha256": exp.sha256_file(uninterrupted),
            }
        ],
        "restart_parity_rows": [
            {
                "schedule_seed": 1,
                "passed": True,
                "comparisons": {
                    "per_source_results": True,
                    "per_update_rows": True,
                    "retention_rows": True,
                    "reveal_counts_by_batch": True,
                    "final_state_hashes": True,
                    "pending_queue": True,
                },
            }
        ],
    }
    audited = exp.audit_checkpoints(producer)
    assert audited["errors"] == []
    assert audited["declared_checkpoint_count"] == 1
    assert audited["restart_pair_count"] == 1
    assert audited["restart_bytes_equal"] is True
    assert audited["prediction_rows_equal"] is True
    assert audited["update_hashes_equal"] is True
    assert audited["pending_queues_equal"] is True
    assert audited["model_states_equal"] is True
    assert audited["rng_states_equal"] is True

    restarted.write_text("{}\n", encoding="utf-8")
    assert "restart_checkpoint_bytes_mismatch" in exp.audit_checkpoints(producer)["errors"]


# REQ-REPORT-7510; SCENARIO-REPORT-7510-INVENTORY.
@pytest.mark.parametrize(
    ("states", "errors", "expected"),
    [
        (("valid", "valid"), (), ("null", 1)),
        (("valid", "absent"), (), ("blocked", 0)),
        (("valid", "invalid"), (), ("disqualified", 0)),
        (("valid", "valid"), ("row_invalid",), ("disqualified", 0)),
    ],
)
def test_terminal_class_keeps_completion_separate_from_claims(
    states: tuple[str, str], errors: tuple[str, ...], expected: tuple[str, int]
) -> None:
    inventory = [
        {"producer": producer, "state": state, "path": f"exp{producer}.json"}
        for producer, state in zip((7506, 7509), states, strict=True)
    ]
    classified = exp.classify_terminal(inventory, errors)
    assert classified["verdict_class"] == expected[0]
    assert classified["causal_audit_complete_score"] == 1
    assert classified["causal_claims_qualified_score"] == expected[1]
    assert classified["honest_verdict"].startswith("complete_")


# REQ-REPORT-7510; SCENARIO-REPORT-7510-E2E.
def test_fixture_artifact_has_required_aggregation_fields_and_validates() -> None:
    value = exp.fixture_artifact()
    assert value["schema"] == exp.SCHEMA
    assert value["run_date"] == "20260922"
    assert value["MODEL_SPECS"] == value["model_specs"] == []
    assert value["model_invoked"] is False
    assert value["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert value["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert value["inference_substrate_class"] == "aggregation"
    assert value["causal_audit_complete_score"] == 1
    assert value["causal_claims_qualified_score"] == 1
    assert value["qualified_online_benefit_score"] == 0
    assert value["chronology_violation_count"] == 0
    assert value["verdict_class"] == "null"
    assert set(value["field_principles"]) == set(value)
    assert exp.validate_artifact(value, verify_sources=False) == []


# REQ-REPORT-7510; SCENARIO-REPORT-7510-E2E.
@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("inference_substrate", "no_model_load", "current_provenance_invalid"),
        ("causal_claims_qualified_score", 0, "terminal_reduction_mismatch"),
        ("chronology_violation_count", 1, "terminal_reduction_mismatch"),
        ("reproducibility_checksum", "sha256:wrong", "reproducibility_checksum_mismatch"),
    ],
)
def test_terminal_mutations_fail_cold_validation(
    field: str, replacement: object, error: str
) -> None:
    value = exp.fixture_artifact()
    value[field] = replacement
    assert error in exp.validate_artifact(value, verify_sources=False)


# REQ-REPORT-7510; SCENARIO-REPORT-7510-E2E.
def test_source_and_sidecar_hashes_fail_after_byte_change(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    sidecar = tmp_path / "sidecar.json"
    _write_json(source, {"value": 1})
    _write_json(sidecar, {"rows": []})
    value = exp.fixture_artifact()
    value["source_artifact_hashes"] = [exp.source_row(source, tmp_path, "fixture")]
    value["raw_sidecars"] = {"fixture": exp.source_row(sidecar, tmp_path, "fixture_sidecar")}
    value["field_principles"] = exp.field_principles(value)
    value["reproducibility_checksum"] = exp.reproducibility_checksum(value)
    assert exp.validate_artifact(value, root=tmp_path) == []
    _write_json(sidecar, {"rows": [1]})
    assert "source_hash_mismatch:sidecar.json" in exp.validate_artifact(value, root=tmp_path)


# REQ-REPORT-7510; SCENARIO-REPORT-7510-E2E.
def test_module_does_not_import_the_exp7509_producer_reducer() -> None:
    tree = ast.parse((exp.REPO_ROOT / exp.MODULE_PATH).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any("experiment_7509_v657_causal_online" in name for name in imported)


# REQ-REPORT-7510; SCENARIO-REPORT-7510-DAG/REDUCTION/RESTART.
def test_real_upstream_bytes_reproduce_without_the_producer_reducer() -> None:
    preconditions = exp.collect_preconditions(exp.REPO_ROOT)
    assert preconditions["missing_external"] == []
    assert preconditions["invalid_present"] == []
    assert all(row["passed"] for row in preconditions["rows"])

    prototype = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATHS[7506])
    online = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATHS[7509])
    assert exp.audit_prototype(prototype) == []
    history = exp.audit_event_history(online)
    assert history["errors"] == []
    assert len(history["rows"]) == 275
    source_audit = exp.audit_original_inputs(exp.REPO_ROOT, online)
    assert source_audit["errors"] == []
    checkpoint = exp.audit_checkpoints(online)
    assert checkpoint["errors"] == []
    assert checkpoint["restart_pair_count"] == 90
    reduced = exp.reduce_measurement(
        online["per_source_results"], online["retention_rows"], history["audit_rows"]
    )
    assert exp.compare_producer_reduction(online, reduced) == []
    assert reduced["support_passed"] is False
    assert reduced["retention_passed"] is True

    complete = exp._reduce_real(exp.REPO_ROOT)
    assert complete["errors"] == []
    assert exp._checkpoint_summary(complete["checkpoint_audit"])["rng_states_equal"] is True
    duplicated = [*complete["source_artifact_hashes"], *complete["source_artifact_hashes"]]
    assert exp._unique_sources(duplicated) == complete["source_artifact_hashes"]
    assert exp._empty_reduction()["primary_holm_family_size"] == 0
    assert exp._empty_checkpoint_audit()["restart_pair_count"] == 0
    assert exp._empty_history("missing")["errors"] == ["missing"]
    assert exp._source_pairs([], comparator="frozen_base", delay=8) == []
    assert (
        exp._source_pairs(
            [
                {
                    "group_id": "only-local",
                    "source_family": "fixture",
                    "delay": 8,
                    "arm": "local_brier",
                    "brier_loss": 0.1,
                }
            ],
            comparator="frozen_base",
            delay=8,
        )
        == []
    )
    assert (
        exp._contrast([], comparator="frozen_base", delay=8, block_length=16, replicates=2, seed=1)[
            "source_count"
        ]
        == 0
    )
    assert exp._temperature_probability(0.8) > 0.5
    with pytest.raises(ValueError, match="unknown_private_mutation"):
        exp.apply_private_mutation({}, "unknown")


# REQ-REPORT-7510; SCENARIO-REPORT-7510-E2E.
def test_defensive_terminal_reader_names_each_contract_failure(tmp_path: Path) -> None:
    value = exp.fixture_artifact()
    del value["schema"]
    assert any(
        error.startswith("required_fields_missing") for error in exp.validate_artifact(value)
    )

    cases: list[tuple[dict[str, object], str]] = [
        ({"schema": "wrong"}, "artifact_identity_invalid"),
        ({"verdict_class": "unknown"}, "verdict_class_invalid"),
        ({"honest_verdict": "unfinished"}, "honest_verdict_not_terminal"),
        ({"causal_audit_complete_score": 2}, "score_not_bare_binary"),
        ({"field_principles": {}}, "field_principles_incomplete"),
        ({"acceptance_gate_results": []}, "gate_contract_invalid"),
        ({"validation_receipts": []}, "required_validation_failed"),
        ({"flagged_adversarial": True}, "failed_guard_not_disqualified"),
        (
            {"qualified_online_benefit_score": 1, "producer_online_benefit_score": 0},
            "qualified_benefit_exceeds_producer",
        ),
        ({"raw_sidecars": []}, "raw_sidecars_invalid"),
        ({"source_artifact_hashes": ["bad"]}, "source_reference_invalid"),
        (
            {"source_artifact_hashes": [{"path": "missing.json", "sha256": "sha256:none"}]},
            "source_missing:missing.json",
        ),
    ]
    for changes, expected in cases:
        candidate = exp.fixture_artifact()
        candidate.update(changes)
        errors = exp.validate_artifact(candidate, root=tmp_path)
        assert any(error.startswith(expected) for error in errors)

    positive = exp.fixture_artifact()
    positive["independent_reduction"]["qualified_online_benefit"] = True
    positive["producer_online_benefit_score"] = 1
    expected = exp._expected_terminal(positive)
    assert expected["verdict_class"] == "positive"


# REQ-REPORT-7510; SCENARIO-REPORT-7510-INVENTORY.
def test_prototype_reader_fails_each_missing_or_invalid_boundary() -> None:
    missing = exp.audit_prototype({})
    assert set(missing) == {
        "prototype_causality_rows_invalid",
        "prototype_restart_invalid",
        "prototype_checkpoint_schema_invalid",
        "prototype_retention_selection_invalid",
    }
    changed = _prototype()
    changed["causality_rows"] = [
        {"future_access_violations": 1, "predict_before_update": True},
        {"future_access_violations": 0, "predict_before_update": True},
    ]
    changed["fixture_rows"] = {
        "restart_check": {
            "passed": True,
            "stable_trace_equal": True,
            "terminal_bytes_equal": True,
        }
    }
    changed["retention_labels_used_for_selection_or_rollback"] = False
    assert exp.audit_prototype(changed) == ["prototype_chronology_invalid"]
