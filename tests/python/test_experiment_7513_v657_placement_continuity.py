"""Tests for the V657 numeric placement and board continuity contract."""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from carnot import experiment_7513_v657_placement_continuity as exp


def _real_numeric_evidence() -> tuple[list[dict[str, object]], dict[str, object]]:
    fit_rows = exp.load_fit_rows(exp.REPO_ROOT)
    checkpoint = exp.load_checkpoint_bundle(exp.REPO_ROOT)
    rows, benchmark = exp.evaluate_numeric_placement(
        [*fit_rows["training"], *fit_rows["calibration"]],
        checkpoint,
        warmup_batches=1,
        timed_batches=2,
    )
    return rows, benchmark


def test_req_hw_7513_symmetric_integer_formats_are_finite() -> None:
    """REQ-HW-7513: both signed formats expose their exact scale and bounds."""

    values = np.asarray([-2.0, -0.25, 0.0, 0.5, 1.5])
    for bits, dtype in ((16, np.int16), (8, np.int8)):
        quantized, scale = exp.quantize_symmetric(values, bits)
        assert quantized.dtype == dtype
        assert scale > 0.0
        assert np.max(np.abs(quantized.astype(np.int64))) <= (1 << (bits - 1)) - 1
        assert np.all(np.isfinite(exp.dequantize(quantized, scale)))

    zeros, scale = exp.quantize_symmetric(np.zeros(3), 8)
    assert zeros.tolist() == [0, 0, 0]
    assert scale == 1.0
    with pytest.raises(ValueError, match="integer_bits_invalid"):
        exp.quantize_symmetric(values, 7)


def test_scenario_hw_7513_numeric_ready_on_frozen_rows() -> None:
    """SCENARIO-HW-7513-NUMERIC-READY: both formats pass the frozen gate."""

    rows, benchmark = _real_numeric_evidence()
    reduced = exp.reduce_quantization_rows(rows)

    assert len(rows) == 236 * 2
    assert reduced["numeric_placement_ready_score"] == 1
    assert reduced["calibration_group_count"] == 60
    assert reduced["policy_count"] == 9
    assert reduced["overflow_count"] == 0
    assert reduced["action_disagreement_count"] == 0
    assert reduced["max_abs_probability_error"] <= 0.01
    assert {row["arithmetic"] for row in rows} == {"int16", "int8"}
    assert all(len(row["policy_comparisons"]) == 9 for row in rows)
    assert all("parameter_footprint" in row for row in rows)
    assert benchmark["timed_batches"] == 2
    assert benchmark["warmup_batches"] == 1
    assert set(benchmark["rows"][0]["component_ns"]) == {
        "allocation",
        "quantization",
        "kernel",
        "dequantization",
        "policy_decision",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (("overflow_count", 1), ("action_disagreement_count", 1), ("abs_probability_error", 0.02)),
)
def test_req_hw_7513_numeric_reducer_fails_closed(field: str, value: object) -> None:
    """REQ-HW-7513: favorable rows cannot hide one failed numeric operand."""

    rows, _benchmark = _real_numeric_evidence()
    changed = deepcopy(rows)
    changed[-1][field] = value
    assert exp.reduce_quantization_rows(changed)["numeric_placement_ready_score"] == 0

    assert exp.reduce_quantization_rows(rows[:-1])["numeric_placement_ready_score"] == 0


def test_scenario_hw_7513_board_continuity_is_independent() -> None:
    """SCENARIO-HW-7513-BOARD-CONTINUITY: all historical scopes remain narrow."""

    source = json.loads(exp.BOARD_ARTIFACT.read_text(encoding="utf-8"))
    rows, summary = exp.reduce_board_rows(source)

    assert summary["board_continuity_complete_score"] == 1
    assert [row["board"] for row in rows] == ["KV260", "PolarFire", "GateMate"]
    assert rows[0]["exact_claim_scope"] == "historical_kv260_fpga_fabric_sampling_only"
    assert rows[1]["exact_claim_scope"] == (
        "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
    )
    assert rows[2]["current_disposition"] == "blocked_unchanged_physical_prerequisite"
    assert all(row["current_reachability"] == "unknown_not_probed" for row in rows)
    assert all(row["hardware_operations_issued"] == [] for row in rows)

    assert (
        exp.reduce_board_rows({"board_rows": rows[:2]})[1]["board_continuity_complete_score"] == 0
    )


def test_req_hw_7513_inventory_refuses_service_and_device_claims() -> None:
    """REQ-HW-7513: the inventory keeps unknown transfer and service costs open."""

    inventory = exp.build_operation_inventory(parameter_count=80, policy_count=9)
    assert [row["operation"] for row in inventory] == [
        "prediction",
        "sparse_brier_update",
        "guard",
        "serialization",
        "durable_acknowledgement",
    ]
    assert all("operations" in row and "bytes" in row and "state" in row for row in inventory)
    assert all(row["transfer_cost"] == "unknown" for row in inventory)
    assert inventory[0]["conditional_binary_energy_normalization"] == "exact_two_state"
    assert inventory[0]["tsu_sampling_required"] is False


def test_scenario_hw_7513_numeric_blocked_keeps_board_score(tmp_path) -> None:
    """SCENARIO-HW-7513-NUMERIC-BLOCKED: board accounting still completes."""

    artifact = exp.build_fixture_artifact(tmp_path, numeric_available=False)
    assert artifact["numeric_placement_ready_score"] == 0
    assert artifact["board_continuity_complete_score"] == 1
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    failure = artifact["gate_check_summary"]["failed_checks"][0]
    assert failure["path"] == "results/experiment_7505_v657_energy_fit.json"
    assert failure["observed"] is None
    assert exp.validate_artifact(artifact, verify_sources=False) == []


def test_req_hw_7513_artifact_reduction_and_mutations(tmp_path) -> None:
    """REQ-HW-7513: cold validation recomputes scores, gates, and checksum."""

    artifact = exp.build_fixture_artifact(tmp_path, numeric_available=True)
    assert artifact["numeric_placement_ready_score"] == 1
    assert artifact["board_continuity_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["whole_service_speedup"] is None
    assert artifact["fpga_performance_measured"] is False
    assert artifact["gatemate_blocker_changed"] is False
    assert set(artifact) == set(artifact["field_principles"])
    assert exp.validate_artifact(artifact, verify_sources=False) == []

    changed = deepcopy(artifact)
    changed["numeric_placement_ready_score"] = 0
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed, verify_sources=False)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, verify_sources=False
    )

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = [{"name": "forbidden"}]
    assert "current_inference_declaration_invalid" in exp.validate_artifact(
        changed, verify_sources=False
    )


def test_req_hw_7513_validation_scope_is_frozen(tmp_path) -> None:
    """REQ-HW-7513: validation names exact files and terminal readers."""

    commands = exp.build_validation_commands(tmp_path)
    argv_text = "\n".join(" ".join(command.argv) for command in commands)
    assert "tests/python/test_experiment_7513_v657_placement_continuity.py" in argv_text
    assert "tests/python -q" not in argv_text
    assert "--fail-under=100" in argv_text
    assert {command.name for command in commands} == set(exp.AFFECTED_CHECK_NAMES)

    terminal = exp.terminal_commands(tmp_path / "candidate.json")
    assert {row.spec.name for row in terminal} == set(exp.TERMINAL_CHECK_NAMES)
    assert all(row.required for row in terminal)


def test_req_hw_7513_expected_action_tie_order() -> None:
    """REQ-HW-7513: integer parity uses the upstream frozen action rule."""

    assert exp.frozen_action(0.0, 1.0, 0.5, 1.0) == "accept"
    assert exp.frozen_action(1.0, 1.0, 0.5, 1.0) == "reject"
    assert exp.frozen_action(0.5, 20.0, 0.1, 1.0) == "escalate"


def test_req_hw_7513_preconditions_authenticate_real_inputs(tmp_path) -> None:
    """REQ-HW-7513: current source bytes pass each branch before measurement."""

    context = exp.collect_preconditions(exp.REPO_ROOT)
    assert context["required_ready"] is True
    assert context["numeric_ready"] is True
    assert context["board_ready"] is True
    assert all(row["passed"] for row in context["rows"])
    assert exp.FIT_ARTIFACT.as_posix() in context["source_artifact_hashes"]

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    scalar = tmp_path / "scalar.json"
    scalar.write_text("1", encoding="utf-8")
    assert exp._load_object(scalar) == {}


def test_req_hw_7513_fail_closed_numeric_inputs(monkeypatch) -> None:
    """REQ-HW-7513: malformed transforms, rows, and batch counts are rejected."""

    with pytest.raises(ValueError, match="quantization_input_nonfinite"):
        exp.quantize_symmetric([np.nan], 8)
    with pytest.raises(ValueError, match="quantization_scale_invalid"):
        exp.dequantize([1], 0.0)

    fit_rows = exp.load_fit_rows(exp.REPO_ROOT)
    checkpoint = exp.load_checkpoint_bundle(exp.REPO_ROOT)
    all_rows = [*fit_rows["training"], *fit_rows["calibration"]]
    with pytest.raises(ValueError, match="benchmark_batch_count_invalid"):
        exp.evaluate_numeric_placement(all_rows, checkpoint, warmup_batches=-1)
    with pytest.raises(ValueError, match="placement_design_shape_invalid"):
        exp._numeric_inputs(all_rows[:-1], checkpoint)
    bad_policy = deepcopy(checkpoint)
    bad_policy["frozen_policies"]["rows"] = []
    with pytest.raises(ValueError, match="frozen_policy_count_invalid"):
        exp._numeric_inputs(all_rows, bad_policy)

    monkeypatch.setattr(exp, "canonical_hash", lambda _value: "sha256:bad")
    with pytest.raises(ValueError, match="frozen_transform_hash_mismatch"):
        exp.load_checkpoint_bundle(exp.REPO_ROOT)


def test_req_hw_7513_null_and_disqualified_classification(tmp_path) -> None:
    """REQ-HW-7513: a measured miss is null, while failed validation disqualifies."""

    artifact = exp.build_fixture_artifact(tmp_path, numeric_available=True)
    raw = deepcopy(artifact)
    raw["quantization_rows"][-1]["overflow_count"] = 1
    result = exp._finalize(raw)
    assert result["verdict_class"] == "null"
    assert result["honest_verdict"].startswith("complete_null_numeric_placement_not_ready")

    raw = deepcopy(artifact)
    raw["validation_receipts"][0]["passed"] = False
    raw["validation_receipts"][0]["exit_code"] = 1
    result = exp._finalize(raw)
    assert result["verdict_class"] == "disqualified"
    assert result["honest_verdict"] == "complete_disqualified_required_validation_failed"


def test_req_hw_7513_validator_rejects_malformed_terminal_rows(tmp_path) -> None:
    """REQ-HW-7513: terminal validation detects each protected schema surface."""

    artifact = exp.build_fixture_artifact(tmp_path, numeric_available=True)
    mutations = (
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("honest_verdict", "null_without_prefix", "terminal_verdict_prefix_invalid"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("acceptance_gate_results", [], "acceptance_gate_results_mismatch"),
        ("board_continuity_complete_score", 0, "independent_reduction_mismatch"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed, verify_sources=False)

    changed = deepcopy(artifact)
    changed["quantization_rows"] = [{"abs_probability_error": "not-a-number"}]
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed, verify_sources=False)

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = {
        "missing": {"path": "missing", "sha256": "sha256:" + "0" * 64}
    }
    assert "source_hash_invalid:missing" in exp.validate_artifact(changed, verify_sources=True)

    changed = deepcopy(artifact)
    changed["acceptance_gate_results"][0]["principle"] = ""
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "gate_principle_missing" in exp.validate_artifact(changed, verify_sources=False)
