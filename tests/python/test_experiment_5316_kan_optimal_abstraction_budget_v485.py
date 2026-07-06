"""Tests for Exp 5316 bounded KAN optimal abstraction budget.

Spec refs: REQ-KAN-5316, SCENARIO-KAN-5316.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5316_kan_optimal_abstraction_budget_v485 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _method(comparison: dict[str, object], method_id: str) -> dict[str, object]:
    methods = comparison["methods"]
    assert isinstance(methods, list)
    return next(row for row in methods if row["method_id"] == method_id)


def test_req_kan_5316_spec_declares_optimal_budget_contract() -> None:
    """REQ-KAN-5316: OpenSpec anchors the bounded allocation experiment."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5316") : spec.index("## Implementation Status")]

    for marker in (
        "REQ-KAN-5316",
        "SCENARIO-KAN-5316",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "static allocation",
        "`.484` dynamic spot-check allocation",
        "DP/knapsack-style",
        "global error budget",
        "`bounded_fixture_only` MUST be true",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_kan_5316_dp_allocation_meets_budget_and_allocates_curvature() -> None:
    """REQ-KAN-5316: DP allocation satisfies the bounded global error budget."""

    plan = mod.allocate_optimal_piece_budget(
        piece_budget=mod.PIECE_BUDGET,
        global_error_budget=mod.GLOBAL_ERROR_BUDGET,
    )

    assert plan.strategy_id == "dp_knapsack_min_pieces_then_gap"
    assert plan.piece_counts == (4, 3, 3)
    assert plan.total_pieces == mod.PIECE_BUDGET
    assert plan.global_error_bound <= mod.GLOBAL_ERROR_BUDGET
    assert plan.global_error_bound < mod.static_piece_plan().global_error_bound
    assert plan.global_error_bound > mod.dynamic_piece_plan().global_error_bound


def test_scenario_kan_5316_compares_static_dynamic_and_optimal_methods() -> None:
    """SCENARIO-KAN-5316: method comparison reports the required metrics."""

    comparison = mod.run_budget_comparison()
    static = _method(comparison, "static_allocation")
    dynamic = _method(comparison, "dynamic_spotcheck_allocation_v484")
    optimal = _method(comparison, "optimal_budget_allocation")

    assert [row["method_id"] for row in comparison["methods"]] == [
        "static_allocation",
        "dynamic_spotcheck_allocation_v484",
        "optimal_budget_allocation",
    ]
    assert static["piece_count"] == 6
    assert dynamic["piece_count"] == 12
    assert optimal["piece_count"] == mod.PIECE_BUDGET
    assert optimal["allocation_piece_counts"] == [4, 3, 3]
    assert static["envelope_gap"] > optimal["envelope_gap"] > dynamic["envelope_gap"]
    assert optimal["global_error_budget_met"] is True
    assert comparison["false_property_rejection_rate"] == pytest.approx(1.0)
    assert comparison["certificate_success_delta"] == pytest.approx(0.0)
    assert comparison["envelope_gap_delta"] > 0.0
    assert isinstance(comparison["milp_solve_time_delta_s"], float)
    assert all(row["certificate_success"] for row in comparison["methods"])
    assert all(row["false_property_rejected"] for row in comparison["methods"])


def test_req_kan_5316_artifact_schema_and_validation(tmp_path: Path) -> None:
    """REQ-KAN-5316: artifact exposes principle fields and bare scalar gates."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5316", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.25,
        tests_run=tests_run,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "experiment_id") == mod.EXPERIMENT_ID
    assert _value(artifact, "milestone") == mod.MILESTONE
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert artifact["kan_optimal_abstraction_ready"] is True
    assert _value(artifact, "allocation_strategy")["selected_piece_counts"] == [4, 3, 3]
    assert artifact["piece_budget"] == mod.PIECE_BUDGET
    assert artifact["envelope_gap_delta"] > 0.0
    assert artifact["certificate_success_delta"] == pytest.approx(0.0)
    assert artifact["false_property_rejection_rate"] == pytest.approx(1.0)
    assert isinstance(artifact["milp_solve_time_delta_s"], int | float)
    assert artifact["bounded_fixture_only"] is True
    assert _value(artifact, "tests_run") == tests_run
    assert "REQ-KAN-5316" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_kan_5316_validation_fails_closed_on_schema_drift() -> None:
    """REQ-KAN-5316: invalid readiness, scope, or substrate claims fail."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5316", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["bounded_fixture_only"] = False
    with pytest.raises(AssertionError, match="bounded fixture"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["kan_optimal_abstraction_ready"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["piece_budget"] = {"value": mod.PIECE_BUDGET}
    with pytest.raises(AssertionError, match="bare integer"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "wrong")
    with pytest.raises(AssertionError, match="inference"):
        mod.validate_artifact(broken)


def test_req_kan_5316_honest_verdict_prefixes_cover_blocked_and_null() -> None:
    """REQ-KAN-5316: terminal verdict prefixes cover blocked and null outcomes."""

    comparison = mod.run_budget_comparison()

    assert mod.honest_verdict(comparison).startswith("complete:")

    null_case = copy.deepcopy(comparison)
    null_case["envelope_gap_delta"] = 0.0
    assert mod.honest_verdict(null_case).startswith("null:")

    blocked_case = copy.deepcopy(comparison)
    blocked_case["kan_optimal_abstraction_ready"] = False
    assert mod.honest_verdict(blocked_case).startswith("blocked_")


def test_deliverable_file_validates_for_scenario_kan_5316() -> None:
    """SCENARIO-KAN-5316: deliverable JSON satisfies the V485 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["kan_optimal_abstraction_ready"] is True
    assert artifact["bounded_fixture_only"] is True
    assert artifact["piece_budget"] == mod.PIECE_BUDGET
