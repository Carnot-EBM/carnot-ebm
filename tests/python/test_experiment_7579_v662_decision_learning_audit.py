"""Contract tests for the V662 independent decision-learning audit.

Spec refs: REQ-REPORT-7579 and SCENARIO-REPORT-7579-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7579_v662_decision_learning_audit as audit


def _heads() -> dict:
    return {
        "heads": {
            "proper_loss_monotone": {
                "theta": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
                "head_sha256": "sha256:proper",
            },
            "raw_original": {"head_sha256": "sha256:raw"},
            "temperature_original": {
                "temperature": 0.5,
                "head_sha256": "sha256:temperature",
            },
            "unconstrained_nine_knot": {
                "theta": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                "head_sha256": "sha256:unconstrained",
            },
        }
    }


def _static_fixture() -> tuple[list[dict], list[dict]]:
    features = [
        {"source_id": "a", "group_id": "g-a", "probability": 0.25, "label": 0},
        {"source_id": "b", "group_id": "g-b", "probability": 0.75, "label": 1},
    ]
    rows = audit.reconstruct_static_rows(features, _heads(), seed=7575101)
    return features, rows


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-STATIC
def test_independent_energy_static_reconstruction_and_reduction() -> None:
    features, rows = _static_fixture()
    assert audit.probability_from_binary_energies(audit.binary_energies(0.25)) == pytest.approx(
        0.25
    )
    assert len(rows) == 10
    assert {row["arm"] for row in rows} == {*audit.STATIC_ARMS, "escalate_all"}
    assert all(row["raw_brier_denominator"] == 1 for row in rows)
    reduced = audit.reduce_static_rows(rows, draws=16, seed=7)
    assert reduced["unit_count"] == len(features)
    assert reduced["metrics"]["raw_original"]["raw_brier_denominator"] == 2
    assert reduced["contrasts"]["raw_original"]["direction"] == (
        "control_minus_candidate_positive_is_better"
    )
    assert reduced["coverage"]["denominator"] == 2


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-STATIC
def test_static_published_probability_and_roster_mutations_fail() -> None:
    features, rows = _static_fixture()
    assert audit.compare_static_rows(rows, rows) == []
    changed = deepcopy(rows)
    changed[0]["q"] += 0.1
    assert "static_probability_mismatch" in audit.compare_static_rows(rows, changed)
    assert "static_row_roster_mismatch" in audit.compare_static_rows(rows, rows[:-1])
    leaked = deepcopy(features)
    leaked[0]["future_label"] = 1
    with pytest.raises(ValueError, match="future_label"):
        audit.reconstruct_static_rows(leaked, _heads(), seed=1)


def _causal_fixture() -> dict:
    predictions = [
        {
            "operation": "sealed_prediction",
            "event_id": event_id,
            "event_count": index + 1,
            "release_index": 0,
            "label_available_at_prediction": False,
            "state_hash_before_prediction": "sha256:s0",
            "original_source_q": probability,
            "arms": {
                arm: {"q": probability, "typed_action": "escalate"} for arm in audit.LEARNING_ARMS
            },
            "seed": 11,
        }
        for index, (event_id, probability) in enumerate((("e0", 0.2), ("e1", 0.8)))
    ]
    release = {
        "operation": "release_update_persist",
        "release_index": 0,
        "event_count": 2,
        "event_ids": ["e0", "e1"],
        "feedback": [["e0", 0], ["e1", 1]],
        "update_count": 2,
        "state_hash_before": "sha256:s0",
        "state_hash_after": "sha256:s1",
        "persisted_state_hash_before_ack": "sha256:persisted",
        "original_labels": [0, 1],
        "shuffled_labels": [1, 0],
    }
    acknowledgment = {
        "operation": "durable_acknowledgment",
        "release_index": 0,
        "event_count": 2,
        "ack_count": 1,
        "state_hash": "sha256:final",
    }
    state = {
        "acknowledgments": [0],
        "next_release_index": 1,
        "predictions": {row["event_id"]: row for row in predictions},
        "release_receipts": {
            "0": {"release_index": 0, "feedback": release["feedback"], "update_count": 2}
        },
    }
    return {
        "predictions": predictions,
        "release_ack": [release, acknowledgment],
        "states": {"11": state},
        "orders": {"11": ["e0", "e1"]},
    }


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-CAUSAL
def test_causal_prediction_release_update_persist_reload() -> None:
    fixture = _causal_fixture()
    result = audit.audit_causal_rows(**fixture, block_size=2)
    assert result["qualified"] is True
    assert result["prediction_count"] == 2
    assert result["release_count"] == 1
    assert result["update_count"] == 2
    assert result["acknowledgment_count"] == 1
    assert result["persist_reload_exercised"] is True


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-MUTATIONS
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("future_label", "future_label_visible"),
        ("order", "prediction_order_mismatch"),
        ("missing_row", "prediction_order_mismatch"),
        ("hash", "state_hash_chain_mismatch"),
        ("duplicate_update", "duplicate_update"),
    ],
)
def test_causal_mutations_close_qualification(mutation: str, error: str) -> None:
    fixture = _causal_fixture()
    audit.mutate_causal_fixture(fixture, mutation)
    result = audit.audit_causal_rows(**fixture, block_size=2)
    assert result["qualified"] is False
    assert error in result["errors"]


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-MUTATIONS
def test_private_mutation_panel_includes_sign_and_all_fail_closed() -> None:
    rows = audit.run_private_mutations()
    assert {row["mutation"] for row in rows} == {
        "future_label",
        "order",
        "missing_row",
        "sign",
        "hash",
        "duplicate_update",
    }
    assert all(row["passed"] is True for row in rows)


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-ABSENCE
def test_missing_external_input_builds_complete_blocked_artifact(tmp_path: Path) -> None:
    artifact = audit.build_blocked_artifact(
        tmp_path,
        [
            audit.check_row(
                "producer_exists",
                "exp7578",
                "results/missing.json",
                "exists",
                True,
                False,
                "eq",
            )
        ],
        duration_s=0.1,
    )
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["first_failure"] == {
        "check": "producer_exists",
        "upstream": "exp7578",
        "path": "results/missing.json",
        "field": "exists",
        "op": "eq",
        "expected": True,
        "observed": False,
    }
    assert artifact["static_claims_qualified_score"] == 0
    assert artifact["learning_claims_qualified_score"] == 0


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-VALIDATION
def test_terminal_schema_checksum_and_cold_replay(tmp_path: Path) -> None:
    artifact = audit.build_test_artifact(tmp_path)
    assert audit.validate_artifact(artifact, root=tmp_path)["valid"] is True
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["fresh_confirmatory_claim_allowed"] is False
    assert all(
        count == 0
        for operation in artifact["invocation_counts"].values()
        for count in operation.values()
    )
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.cold_replay(path, root=tmp_path)["valid"] is True
    changed = deepcopy(artifact)
    changed["fresh_confirmatory_claim_allowed"] = True
    with pytest.raises(ValueError, match="freshness"):
        audit.validate_artifact(changed, root=tmp_path)


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-VALIDATION
def test_validation_plan_is_scoped_and_cli_parser_is_fixed(tmp_path: Path) -> None:
    commands = audit.build_validation_commands(tmp_path, tmp_path / "private")
    assert [command.name for command in commands] == list(audit.AFFECTED_CHECK_NAMES)
    assert all("tests/python" not in command.argv for command in commands)
    assert any(
        any("experiment_7579_v662_decision_learning_audit.py" in part for part in command.argv)
        for command in commands
        if command.name in {"changed_module_coverage", "changed_module_coverage_report"}
    )
    args = audit.parse_args(["--root", str(tmp_path), "--date", "20260924"])
    assert args.root == tmp_path.resolve()
    assert args.date == "20260924"


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-VALIDATION
def test_source_hash_and_principle_guards(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    receipt = audit.source_receipt(source, tmp_path, "producer")
    assert audit.authenticate_source_receipt(receipt, tmp_path) == source
    source.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="source_hash"):
        audit.authenticate_source_receipt(receipt, tmp_path)
    assert set(audit.REQUIRED_PRINCIPLE_FIELDS) <= set(audit.field_principles())


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-STATIC / SCENARIO-REPORT-7579-CAUSAL
def test_exact_v662_producer_rows_reconstruct_independently() -> None:
    root = audit.REPO_ROOT
    checks, producers = audit.collect_preconditions(root)
    assert all(row["passed"] is True for row in checks)
    inputs, source_hashes = audit.load_authenticated_inputs(root, producers)

    features = [row for row in inputs["roles"] if row["role"] == "test"]
    static_rows = audit.reconstruct_static_rows(features, inputs["heads"], seed=7575101)
    static_errors = audit.compare_static_rows(static_rows, inputs["published_static"])
    static_reduction = audit.reduce_static_rows(static_rows, draws=1000, seed=7575101)
    assert static_errors == []
    assert static_reduction["unit_count"] == 80

    causal = audit.audit_causal_rows(
        inputs["causal"],
        inputs["release_ack"],
        inputs["learned"],
        inputs["protocol"]["orders"],
        block_size=inputs["protocol"]["release_block_size"],
    )
    online = audit.reduce_learning_metric_rows(inputs["comparison"])
    retention = audit.reduce_learning_metric_rows(inputs["retention"])
    bootstrap = audit.reduce_bootstrap_rows(inputs["bootstrap"])
    assert causal["qualified"] is True, causal["errors"]
    assert online["bounded"]["raw_squared_error_denominator"] == 800
    assert retention["bounded"]["raw_squared_error_denominator"] == 1600
    assert bootstrap["completed_replays"] == 5000
    assert bootstrap["benefit_passed"] is False
    assert bootstrap["retention_passed"] is False

    receipts = audit.build_test_artifact(root)["validation_receipts"]
    artifact = audit.build_artifact(
        root=root,
        run_date="20260924",
        preconditions=checks,
        source_hashes=source_hashes,
        static_rows=static_rows,
        static_reduction=static_reduction,
        static_errors=static_errors,
        causal_reduction=causal,
        online_reduction=online,
        retention_reduction=retention,
        bootstrap_rows=inputs["bootstrap"],
        bootstrap_reduction=bootstrap,
        mutations=audit.run_private_mutations(),
        validation_receipts=receipts,
        duration_s=1.0,
        phase_spans=[],
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["static_claims_qualified_score"] == 1
    assert artifact["learning_claims_qualified_score"] == 1
    assert audit.validate_artifact(artifact, root=root)["valid"] is True


def _bootstrap_fixture() -> list[dict]:
    rows = []
    for index, arm in enumerate(audit.LEARNING_ARMS):
        brier = 0.2 + index * 0.01
        retention = 0.15 + index * 0.01
        rows.append(
            {
                "unit_id": "u0",
                "seed": 11,
                "arm": arm,
                "raw_squared_error_numerator": brier * 2,
                "raw_squared_error_denominator": 2,
                "mean_brier": brier,
                "retention_brier_numerator": retention * 2,
                "retention_denominator": 2,
                "retention_mean_brier": retention,
                "action_cost_numerator": 0.4,
                "action_cost_denominator": 2,
                "mean_action_cost": 0.2,
                "coverage_numerator": 1,
                "coverage_denominator": 2,
                "coverage": 0.5,
                "metric_direction": "lower_brier_and_cost_are_better",
                "censoring": {"prediction_loss": False},
                "provenance": "fixture",
            }
        )
    return rows


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-MUTATIONS
def test_numerical_and_roster_guards_fail_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    assert audit.load_json(invalid) == {}
    invalid.write_text("[]", encoding="utf-8")
    assert audit.load_json(invalid) == {}
    lines = tmp_path / "rows.jsonl"
    lines.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object"):
        audit.load_jsonl(lines)
    assert audit._path_label(Path("/outside"), tmp_path) == "/outside"

    with pytest.raises(ValueError, match="probability_invalid"):
        audit.binary_energies(float("nan"))
    with pytest.raises(ValueError, match="binary_energies_invalid"):
        audit.probability_from_binary_energies([0.0])
    with pytest.raises(ValueError, match="head_theta_invalid"):
        audit._piecewise_probability(0.5, [0.0])
    assert audit._piecewise_probability(1.0, list(range(9))) == 8.0
    with pytest.raises(ValueError, match="static_head_missing"):
        audit._head_probability("missing", 0.5, {})
    with pytest.raises(ValueError, match="temperature_invalid"):
        audit._head_probability(
            "temperature_original", 0.5, {"temperature_original": {"temperature": 0}}
        )
    with pytest.raises(ValueError, match="static_label_invalid"):
        audit._static_metric_row({"source_id": "x", "label": 2}, "raw", 0.5, seed=1)
    with pytest.raises(ValueError, match="static_head_roster_invalid"):
        audit.reconstruct_static_rows([], {"heads": {}}, seed=1)

    features, rows = _static_fixture()
    duplicate = deepcopy(features)
    duplicate.append(deepcopy(features[0]))
    with pytest.raises(ValueError, match="static_source_roster_invalid"):
        audit.reconstruct_static_rows(duplicate, _heads(), seed=1)
    bad_probability = deepcopy(features)
    bad_probability[0]["probability"] = 2.0
    with pytest.raises(ValueError, match="static_probability_invalid"):
        audit.reconstruct_static_rows(bad_probability, _heads(), seed=1)
    decision = deepcopy(rows)
    decision[0]["action"] = "accept"
    assert audit.compare_static_rows(rows, decision) == ["static_decision_mismatch"]
    with pytest.raises(ValueError, match="static_duplicate_row"):
        audit.reduce_static_rows([*rows, rows[0]], draws=2, seed=1)
    with pytest.raises(ValueError, match="static_arm_roster_invalid"):
        audit.reduce_static_rows(rows[:-1], draws=2, seed=1)
    with pytest.raises(ValueError, match="release_ack_partition_invalid"):
        audit._release_blocks([{}], ["a", "b"])
    assert audit._design(1.0)[-1] == 1.0


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-MUTATIONS
def test_learning_row_and_state_guards_fail_closed() -> None:
    fixture = _causal_fixture()
    fixture["predictions"][0]["arms"].pop("raw")
    assert (
        "prediction_arm_roster_mismatch"
        in audit.audit_causal_rows(**fixture, block_size=2)["errors"]
    )
    missing_state = _causal_fixture()
    missing_state["states"] = {}
    assert (
        "persisted_state_missing"
        in audit.audit_causal_rows(**missing_state, block_size=2)["errors"]
    )
    malformed_state = _causal_fixture()
    malformed_state["states"]["11"] = {"acknowledgments": [], "next_release_index": 9}
    assert (
        "persisted_state_shape_invalid"
        in audit.audit_causal_rows(**malformed_state, block_size=2)["errors"]
    )
    unknown = _causal_fixture()
    with pytest.raises(ValueError, match="unknown_mutation"):
        audit.mutate_causal_fixture(unknown, "unknown")

    metric_rows = []
    for arm in audit.LEARNING_ARMS:
        probability = 0.5
        metric_rows.append(
            {
                "unit_id": "u",
                "seed": 1,
                "phase": "online",
                "arm": arm,
                "event_count": 1,
                "q": probability,
                "label": 1,
                "raw_squared_error_numerator": 0.25,
                "raw_squared_error_denominator": 1,
                "typed_action": "escalate",
                "realized_action_cost": 0.2,
                "non_escalated": False,
            }
        )
    assert audit.reduce_learning_metric_rows(metric_rows)["raw"]["mean_brier"] == 0.25
    for field, value, error in (
        ("raw_squared_error_numerator", 0.5, "sign_or_value"),
        ("raw_squared_error_denominator", 2, "denominator"),
        ("typed_action", "accept", "decision"),
    ):
        changed = deepcopy(metric_rows)
        changed[0][field] = value
        with pytest.raises(ValueError, match=error):
            audit.reduce_learning_metric_rows(changed)
    with pytest.raises(ValueError, match="learning_metric_duplicate"):
        audit.reduce_learning_metric_rows([*metric_rows, metric_rows[0]])
    with pytest.raises(ValueError, match="learning_metric_arm_roster_invalid"):
        audit.reduce_learning_metric_rows(metric_rows[:-1])
    assert audit._interval([])["count"] == 0

    bootstrap = _bootstrap_fixture()
    assert audit.reduce_bootstrap_rows(bootstrap)["completed_replays"] == 1
    for field, value, error in (
        ("raw_squared_error_denominator", 0, "denominator"),
        ("mean_brier", 9.0, "operand"),
    ):
        changed = deepcopy(bootstrap)
        changed[0][field] = value
        with pytest.raises(ValueError, match=error):
            audit.reduce_bootstrap_rows(changed)
    with pytest.raises(ValueError, match="bootstrap_duplicate_arm"):
        audit.reduce_bootstrap_rows([*bootstrap, bootstrap[0]])
    with pytest.raises(ValueError, match="bootstrap_arm_roster_invalid"):
        audit.reduce_bootstrap_rows(bootstrap[:-1])

    partition = _causal_fixture()
    partition["orders"]["12"] = []
    partition["release_ack"].append({})
    assert audit.audit_causal_rows(**partition, block_size=2)["errors"] == [
        "release_ack_partition_invalid"
    ]

    compound = _causal_fixture()
    compound["release_ack"][0]["release_index"] = 1
    compound["release_ack"][0]["feedback"][0][1] = 2
    compound["release_ack"][0]["update_count"] = 1
    compound["release_ack"][1]["release_index"] = 1
    compound["release_ack"][1]["ack_count"] = 2
    state = compound["states"]["11"]
    state["predictions"].pop("e1")
    state["acknowledgments"] = []
    state["next_release_index"] = 9
    state["release_receipts"] = {"1": {"feedback": []}}
    errors = audit.audit_causal_rows(**compound, block_size=2)["errors"]
    assert {
        "release_order_mismatch",
        "acknowledgment_order_mismatch",
        "release_label_invalid",
        "update_count_mismatch",
        "acknowledgment_count_mismatch",
        "persisted_prediction_roster_mismatch",
        "persisted_acknowledgment_mismatch",
        "persisted_release_index_mismatch",
        "persisted_update_mismatch",
    } <= set(errors)


def test_full_state_parameter_corruptions_are_detected() -> None:
    checks, producers = audit.collect_preconditions(audit.REPO_ROOT)
    assert all(row["passed"] for row in checks)
    inputs, _hashes = audit.load_authenticated_inputs(audit.REPO_ROOT, producers)
    seed = "7578001"
    events = {row["event_id"]: row for row in inputs["causal"] if str(row["seed"]) == seed}
    block = audit._release_blocks(inputs["release_ack"], sorted(inputs["protocol"]["orders"]))[seed]
    releases = [row for row in block if row["operation"] == "release_update_persist"]
    state = deepcopy(inputs["learned"][seed])
    releases[0]["shuffled_labels"].pop()
    state["arms"]["constrained"]["gram"][0][0] += 1
    state["arms"]["constrained"]["target"][0] += 1
    state["arms"]["shuffled_constrained"]["gram"][0][0] += 1
    state["arms"]["shuffled_constrained"]["target"][0] += 1
    state["arms"]["constrained"]["sample_count"] = -1
    state["arms"]["global_count"]["counts"][0][0] += 1
    state["arms"]["local_count"]["counts"][0][0] += 1
    state["arms"]["constrained"]["theta"][1] = -1
    state["journal"][0]["entry_hash"] = "sha256:bad"
    first_prediction = next(iter(state["predictions"].values()))
    first_prediction["arms"]["raw"]["energies"] = [0.0, 0.0]
    errors = audit._audit_final_parameters(state, events, releases)
    assert {
        "shuffled_label_roster_mismatch",
        "constrained_gram_mismatch",
        "constrained_target_mismatch",
        "shuffled_gram_mismatch",
        "shuffled_target_mismatch",
        "parameter_sample_count_mismatch",
        "global_count_update_mismatch",
        "local_count_update_mismatch",
        "constrained_constraint_mismatch",
        "persisted_journal_hash_mismatch",
        "persisted_energy_probability_mismatch",
    } <= set(errors)


# REQ-REPORT-7579 / SCENARIO-REPORT-7579-VALIDATION
def test_defensive_artifact_and_terminal_readers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert audit.check_row("x", "u", "p", "f", (1, 2), 1, "in")["passed"] is True
    with pytest.raises(ValueError, match="unknown_check_op"):
        audit.check_row("x", "u", "p", "f", 1, 1, "bad")
    with pytest.raises(ValueError, match="requires_failed_check"):
        audit._blocked_summary([{"passed": True}])

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    receipt = audit.source_receipt(source, tmp_path, "fixture")
    wrong_size = dict(receipt)
    wrong_size["bytes"] += 1
    with pytest.raises(ValueError, match="source_size"):
        audit.authenticate_source_receipt(wrong_size, tmp_path)
    hashes: list[dict] = []
    absolute = {**receipt, "path": str(source)}
    assert audit.read_sidecar(tmp_path, absolute, "fixture", hashes) == source
    bad_hash = dict(absolute)
    bad_hash["sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="sidecar_hash"):
        audit.read_sidecar(tmp_path, bad_hash, "fixture", [])
    bad_size = dict(absolute)
    bad_size["bytes"] += 1
    with pytest.raises(ValueError, match="sidecar_size"):
        audit.read_sidecar(tmp_path, bad_size, "fixture", [])

    metric_errors = audit._selected_metric_errors(
        {"a": {"x": 1.0}, "b": {"x": "left"}, "c": {"x": 1}},
        {"a": {"x": 2.0}, "b": {"x": "right"}},
        ("x",),
    )
    assert metric_errors == [
        "producer_metric_mismatch:a:x",
        "producer_metric_mismatch:b:x",
        "producer_metric_arm_missing:c",
    ]

    artifact = audit.build_test_artifact(tmp_path)
    invalid = deepcopy(artifact)
    invalid.update(
        {
            "schema": "bad",
            "milestone": "bad",
            "honest_verdict": "bad",
            "verdict_class": "bad",
            "MODEL_SPECS": ["bad"],
            "model_invoked": True,
            "inference_substrate_class": "bad",
            "field_principles": {},
            "branch_conclusions": {},
            "static_claims_qualified_score": 2,
            "learning_claims_qualified_score": 2,
            "validation_receipts": [],
            "mutation_rows": [{"mutation": "bad", "passed": False}],
            "rows": [{"arm": "bad"}],
            "source_artifact_hashes": [bad_hash],
        }
    )
    with pytest.raises(ValueError) as exc:
        audit.validate_artifact(invalid, root=tmp_path)
    message = str(exc.value)
    for expected in (
        "identity_mismatch",
        "milestone_mismatch",
        "terminal_prefix_missing",
        "verdict_class_invalid",
        "model_specs_not_empty",
        "current_model_calls_nonzero",
        "substrate_class_mismatch",
        "field_principles_incomplete",
        "branch_conclusions_incomplete",
        "static_qualification_invalid",
        "learning_qualification_invalid",
        "comparison_row_schema_invalid",
        "validation_receipts_failed",
        "mutation_panel_failed",
        "checksum_mismatch",
        "source_hash_mismatch",
    ):
        assert expected in message

    blocked = audit.build_blocked_artifact(
        tmp_path,
        [audit.check_row("missing", "up", "path", "exists", True, False, "eq")],
        duration_s=0.1,
    )
    blocked["gate_check_summary"]["first_failure"] = {}
    blocked["validation_receipts"] = artifact["validation_receipts"]
    blocked["reproducibility_checksum"] = audit.reproducibility_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_gate_summary_invalid"):
        audit.validate_artifact(blocked, root=tmp_path)

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_not_object"):
        audit.cold_replay(empty, root=tmp_path)
    terminal = audit.terminal_commands(tmp_path / "candidate.json", audit.REPO_ROOT)
    assert [row.name for row in terminal] == list(audit.TERMINAL_CHECK_NAMES)
    assert len(audit._provisional_terminal_receipts()) == 4
    manifest_path, manifest = audit._write_manifest(tmp_path)
    assert manifest_path.is_file() and manifest["experiment_id"] == audit.EXPERIMENT_ID
    audit.progress(0.0, "test", "boundary", units=1)
    assert "phase=test" in capsys.readouterr().out
    assert audit._span("test", 2.0, 1.0, 1)["completed_units"] == 1
    with pytest.raises(ValueError, match="run_date"):
        audit.parse_args(["--root", str(tmp_path), "--date", "20260923"])


def test_fresh_independent_reader_reduces_embedded_rows(tmp_path: Path) -> None:
    _features, static_rows = _static_fixture()
    static_reduction = audit.reduce_static_rows(static_rows, draws=16, seed=7)
    bootstrap_rows = _bootstrap_fixture()
    bootstrap_reduction = audit.reduce_bootstrap_rows(bootstrap_rows)
    artifact = audit.build_test_artifact(tmp_path)
    artifact["rows"] = audit._published_rows(static_rows, bootstrap_rows)
    artifact["static_reconstruction"]["reduction"] = static_reduction
    artifact["learning_interval_reduction"] = bootstrap_reduction
    artifact["reproducibility_checksum"] = audit.reproducibility_checksum(artifact)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    result = audit.independent_replay(path, root=tmp_path)
    assert result == {"valid": True, "static_rows": 10, "learning_rows": 5}

    changed = deepcopy(artifact)
    changed["static_reconstruction"]["reduction"]["unit_count"] = 99
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="independent_static_reduction_mismatch"):
        audit.independent_replay(path, root=tmp_path)

    changed = deepcopy(artifact)
    changed["learning_interval_reduction"]["completed_replays"] = 99
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="independent_learning_reduction_mismatch"):
        audit.independent_replay(path, root=tmp_path)
