"""Tests for the retained-state learning placement measurement.

Spec refs: REQ-KAN-7487 and SCENARIO-KAN-7487-01 through
SCENARIO-KAN-7487-09.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7487_v655_learning_placement as exp


def test_learner_authentication_gates_only_numeric_branch(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-01."""

    root = exp.build_fixture_root(tmp_path)
    checks, sources, context = exp.collect_preconditions(root)
    assert all(row["passed"] for row in checks)
    assert context["numeric_branch_available"] is True
    assert context["board_branch_available"] is True
    assert sources[exp.IMPORTANCE_ARTIFACT.as_posix()]["original_verdict_class"] == (
        "circular_positive"
    )

    (root / exp.IMPORTANCE_ARTIFACT).unlink()
    checks, _sources, context = exp.collect_preconditions(root)
    assert context["numeric_branch_available"] is False
    assert context["board_branch_available"] is True
    assert any(row["check"] == "importance_learner_present" and not row["passed"] for row in checks)


def test_frozen_fixture_and_numeric_arms_share_protocol(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-02."""

    fixture = exp.freeze_numeric_fixture(32, 748700, 8)
    rows = exp.benchmark_size_seed(fixture, tmp_path, updates=8)
    updates = [row for row in rows if row["mode"] == "update"]
    assert {row["arithmetic"] for row in updates} == {
        "float64",
        "float32",
        exp.FIXED_ARITHMETIC,
    }
    assert len({row["fixture_hash"] for row in rows}) == 1
    assert all(row["active_data_gradient_count"] == exp.ACTIVE_COEFFICIENTS for row in updates)
    assert all(row["full_anchor_coefficient_count"] == 32 for row in updates)
    assert all(set(exp.TIMED_COMPONENTS).issubset(row["component_total_ns"]) for row in rows)


def test_no_update_control_is_durable_and_unchanged(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-03."""

    fixture = exp.freeze_numeric_fixture(64, 748701, 6)
    rows = exp.benchmark_size_seed(fixture, tmp_path, updates=6)
    controls = [row for row in rows if row["mode"] == "no_update_control"]
    assert len(controls) == 3
    assert all(row["state_unchanged"] for row in controls)
    assert all(row["exact_recovery"] for row in controls)
    assert all(row["acknowledged_updates"] == 0 for row in controls)
    assert all(row["durable_records"] == 6 for row in controls)


def test_fixed_point_replay_gate_precedes_timing(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-04."""

    fixture = exp.freeze_numeric_fixture(32, 748702, 12)
    rows = exp.benchmark_size_seed(fixture, tmp_path, updates=12)
    reduction = exp.reduce_numeric_rows(rows, sizes=(32,), seeds=(748702,), updates=12)
    fixed = next(
        row
        for row in rows
        if row["arithmetic"] == exp.FIXED_ARITHMETIC and row["mode"] == "update"
    )
    assert fixed["decision_flips"] == 0
    assert fixed["max_probability_error"] <= exp.FIXED_PROBABILITY_ERROR_LIMIT
    assert reduction["fixed_point_deployable"] is True

    changed = deepcopy(rows)
    next(
        row
        for row in changed
        if row["arithmetic"] == exp.FIXED_ARITHMETIC and row["mode"] == "update"
    )["decision_flips"] = 1
    assert exp.reduce_numeric_rows(changed, sizes=(32,), seeds=(748702,), updates=12)[
        "fixed_point_deployable"
    ] is False


def test_numeric_reducer_requires_every_size_seed_arm_and_control(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-02/03/04."""

    fixture = exp.freeze_numeric_fixture(32, 748700, 3)
    rows = exp.benchmark_size_seed(fixture, tmp_path, updates=3)
    reduction = exp.reduce_numeric_rows(rows, sizes=(32,), seeds=(748700,), updates=3)
    assert reduction["numeric_complete"] is True
    assert reduction["planned_rows"] == 6

    assert exp.reduce_numeric_rows(rows[:-1], sizes=(32,), seeds=(748700,), updates=3)[
        "numeric_complete"
    ] is False


def test_service_envelope_refuses_incomplete_denominator() -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-05."""

    fit_capture = {
        "duration_components_s": {"forward": 12.0},
        "raw_logit_shards": [{"role": "training", "row_count": 10}],
    }
    eval_capture = {
        "duration_components_s": {"forward": 8.0},
        "raw_logit_shards": [{"role": "online", "row_count": 6}],
    }
    online = {
        "service_cost_rows": [
            {"operation": "feedback_processing", "total_s": 1.0},
            {"operation": "update", "total_s": 2.0},
            {"operation": "serialization", "total_s": 0.1},
            {"operation": "fsync", "total_s": 0.2},
        ]
    }
    envelope = exp.build_service_envelope(fit_capture, eval_capture, online)
    assert envelope["status"] == "conditional_missing_complete_components"
    assert envelope["measured_end_to_end_speedup"] is None
    assert {"tokenization", "verifier_work", "durable_write"}.issubset(
        envelope["missing_components"]
    )


@pytest.mark.parametrize(
    ("fraction", "kernel_speed", "expected"),
    [
        (0.0, 10.0, 1.0),
        (0.5, 2.0, 4.0 / 3.0),
        (0.99, math.inf, 100.0),
        (0.999, math.inf, 1000.0),
    ],
)
def test_amdahl_formula(fraction: float, kernel_speed: float, expected: float) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-06."""

    assert exp.amdahl_speedup(fraction, kernel_speed) == pytest.approx(expected)


def test_amdahl_rejects_invalid_inputs() -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-06."""

    with pytest.raises(ValueError, match="accelerated_fraction_invalid"):
        exp.amdahl_speedup(1.1, 2.0)
    with pytest.raises(ValueError, match="kernel_speed_invalid"):
        exp.amdahl_speedup(0.5, 0.5)


def test_board_reduction_preserves_scope_and_issues_no_operations(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-07."""

    root = exp.build_fixture_root(tmp_path)
    board = json.loads((root / exp.BOARD_ARTIFACT).read_text(encoding="utf-8"))
    rows, reduction = exp.reduce_board_evidence(board)
    by_board = {row["board"]: row for row in rows}
    assert by_board["KV260"]["last_authenticated_date"] == "20260915"
    assert by_board["KV260"]["future_access"] == "ssh kria only"
    assert by_board["KV260"]["architecture_limit"] == "k_max<=5"
    assert by_board["PolarFire"]["exact_claim_scope"] == (
        "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
    )
    assert by_board["GateMate"]["current_disposition"] == (
        "blocked_unchanged_physical_prerequisite"
    )
    assert reduction["hardware_operations_issued"] == []
    assert reduction["board_branch_complete"] is True


def test_fixture_artifact_keeps_completion_separate_from_speed(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-08."""

    artifact = exp.build_fixture_artifact(tmp_path)
    assert artifact["placement_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["service_envelope"]["measured_end_to_end_speedup"] is None
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    assert exp.validate_artifact(artifact) == []


def test_terminal_reader_detects_row_and_board_mutations(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-09."""

    artifact = exp.build_fixture_artifact(tmp_path)
    changed = deepcopy(artifact)
    changed["numeric_cost_rows"][0]["decision_flips"] = 1
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["board_rows"][0]["last_authenticated_date"] = "20990101"
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed)


def test_terminal_reader_detects_invocation_and_checksum_mutations(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-09."""

    artifact = exp.build_fixture_artifact(tmp_path)
    changed = deepcopy(artifact)
    changed["model_invoked"] = True
    assert "current_inference_declaration_invalid" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_validation_manifest_is_scoped_and_frozen(tmp_path: Path) -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-09."""

    private = tmp_path / "private"
    plan = exp.build_command_plan(exp.REPO_ROOT, exp.VALIDATION_MANIFEST, private)
    assert exp.validate_command_plan(exp.REPO_ROOT, exp.VALIDATION_MANIFEST, plan) == []
    assert {command.name for command in plan} == set(exp.AFFECTED_CHECK_NAMES)
    assert all("tests/python " not in " ".join(command.argv) for command in plan)


def test_parse_args_supports_terminal_reader_modes() -> None:
    """REQ-KAN-7487; SCENARIO-KAN-7487-09."""

    args = exp.parse_args(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"])
    assert args.date == exp.RUN_DATE
    assert args.cold_replay == Path("candidate.json")
    assert args.independent_reduce is None
