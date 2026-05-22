"""REQ-KAN-2876 tests for the Exp 2871 KAN PWA/MILP corrigendum.

Scenario: SCENARIO-KAN-2876.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify.kan_pwa_milp_corrigendum import (
    REQUIRED_ARTIFACT_FIELDS,
    build_corrigendum_fixture,
    build_experiment_artifact,
    detect_milp_backend,
    solve_property,
    validate_artifact,
    write_experiment_artifact,
)


def test_req_kan_2876_is_spec_anchored() -> None:
    """REQ-KAN-2876: the corrigendum requirement exists before code."""
    spec = Path("openspec/capabilities/kan/spec.md").read_text(encoding="utf-8")

    assert "REQ-KAN-2876" in spec
    assert "SCENARIO-KAN-2876" in spec


def test_bounds_are_distinct_by_construction() -> None:
    """REQ-KAN-2876: local and global bounds come from distinct procedures."""
    fixture = build_corrigendum_fixture()

    assert fixture.local_error_bound == pytest.approx(0.0625)
    assert fixture.global_error_bound == pytest.approx(0.09375)
    assert fixture.bounds_distinct_by_construction is True
    assert fixture.bound_procedures() == {
        "local_error_bound": "max_per_segment_midpoint_residual",
        "global_error_bound": "weighted_output_error_propagation",
    }

    payload = fixture.as_serializable()
    assert payload["units"][0]["local_error_bound"] == pytest.approx(0.0625)
    assert payload["units"][1]["local_error_bound"] == pytest.approx(0.015625)
    assert fixture.evaluate_upper(0.5) == pytest.approx(0.53125)
    assert fixture.units[0].true_value(0.5) == pytest.approx(0.25)
    assert fixture.units[0].upper(0.25) == pytest.approx(0.125)

    with pytest.raises(ValueError, match="outside the PWA domain"):
        fixture.evaluate_upper(1.5)
    with pytest.raises(ValueError, match="outside the PWA domain"):
        fixture.units[0].upper(1.5)


def test_z3_milp_solver_reports_optimal_certificate_when_available() -> None:
    """SCENARIO-KAN-2876: a real local solver is used before enumeration."""
    if detect_milp_backend() != "z3":
        pytest.skip("Z3 is not available in this environment")

    result = solve_property(build_corrigendum_fixture(), backend_name="z3")

    assert result.milp_backend_available is True
    assert result.milp_backend_name == "z3"
    assert result.solver_status == "optimal"
    assert result.exact_enumeration_used_only_as_fallback is False
    assert result.property_verified is True
    assert result.certified_upper_bound == pytest.approx(0.53125)
    assert result.counterexample_or_certificate["kind"] == "certificate"
    assert result.counterexample_or_certificate["method"] == "z3_mixed_integer_linear_pwa"
    assert result.as_serializable()["solver_status"] == "optimal"


def test_blocked_solver_dependency_uses_enumeration_only_as_fallback() -> None:
    """REQ-KAN-2876: absent solver dependencies are reported honestly."""
    result = solve_property(build_corrigendum_fixture(), backend_name="")

    assert result.milp_backend_available is False
    assert result.milp_backend_name == ""
    assert result.solver_status == "blocked_solver_dependency"
    assert result.exact_enumeration_used_only_as_fallback is True
    assert result.property_verified is True
    assert result.certified_upper_bound == pytest.approx(0.53125)
    assert result.counterexample_or_certificate["kind"] == "fallback_certificate"
    assert result.counterexample_or_certificate["method"] == "exact_enumerated_pwa_vertices"


def test_experiment_artifact_schema_and_validation(tmp_path: Path) -> None:
    """SCENARIO-KAN-2876: the deliverable carries every required field."""
    path = tmp_path / "experiment_2876_kan_pwa_milp_corrigendum_v2.json"

    artifact = write_experiment_artifact(path, backend_name="z3")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["honest_verdict"].startswith("complete_corrigendum")
    assert payload["kan_corrigendum_ready"] is True
    assert payload["tautology_flag_cleared"] is True
    assert payload["bounds_distinct_by_construction"] is True
    assert payload["milp_backend_available"] is True
    assert payload["milp_backend_name"] == "z3"
    assert payload["exact_enumeration_used_only_as_fallback"] is False
    assert payload["solver_status"] == "optimal"
    assert payload["run_date"] == "20260522"
    assert isinstance(payload["duration_s"], float)
    assert "exp_2871_flag_fields" in payload["field_principles"]

    assert validate_artifact(payload) == payload
    incomplete = dict(payload)
    incomplete.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_artifact(incomplete)


def test_blocked_artifact_schema_when_solver_is_forced_absent() -> None:
    """REQ-KAN-2876: blocked artifacts still prove solver checks ran."""
    artifact = build_experiment_artifact(backend_name="")

    assert artifact["honest_verdict"].startswith("blocked_solver_dependency")
    assert artifact["kan_corrigendum_ready"] is False
    assert artifact["tautology_flag_cleared"] is True
    assert artifact["milp_backend_available"] is False
    assert artifact["solver_status"] == "blocked_solver_dependency"
    assert artifact["exact_enumeration_used_only_as_fallback"] is True
    assert artifact["solver_preconditions_checked"][0]["available"] is False
