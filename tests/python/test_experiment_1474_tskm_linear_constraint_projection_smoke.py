"""Tests for Exp 1474 T-SKM linear constraint projection smoke.

Spec: REQ-VERIFY-1474, SCENARIO-VERIFY-1474.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify.skm_projection import (
    REQUIRED_ARTIFACT_FIELDS,
    LinearConstraintSystem,
    build_toy_linear_cases,
    evaluate_toy_cases,
    project_skm,
)
from scripts.experiment_1474_tskm_linear_constraint_projection_smoke import (
    run_experiment,
    write_in_progress_artifact,
)


def test_req_verify_1474_equality_constraints_become_two_inequalities() -> None:
    """REQ-VERIFY-1474: equalities are represented as paired half-space rows."""

    system = LinearConstraintSystem.from_constraints(
        less_equal=[("upper_bound", [1.0, 0.0], 3.0)],
        equalities=[("fixed_y", [0.0, 1.0], 2.0)],
    )

    assert system.matrix.shape == (3, 2)
    assert system.names == ("upper_bound", "fixed_y<=2.0", "fixed_y>=2.0")
    assert system.max_violation([3.0, 2.0]) == pytest.approx(0.0)
    assert system.max_violation([3.0, 4.0]) == pytest.approx(2.0)


def test_req_verify_1474_projection_reaches_known_toy_feasible_points() -> None:
    """REQ-VERIFY-1474: bounded SKM projection reaches known toy feasible points."""

    for case in build_toy_linear_cases():
        result = project_skm(case.system, case.start, max_iterations=32)

        assert result.converged is True
        assert result.max_constraint_violation <= 1e-9
        assert np.allclose(result.vector, case.expected_solution, atol=1e-8)


def test_req_verify_1474_projection_reports_bounded_nonconvergence() -> None:
    """REQ-VERIFY-1474: max_iterations bounds the projection loop."""

    case = build_toy_linear_cases()[0]

    result = project_skm(case.system, case.start, max_iterations=0)

    assert result.converged is False
    assert result.iterations == 0
    assert result.max_constraint_violation > 0.0


def test_scenario_verify_1474_projected_cases_agree_with_carnot_z3_and_ising() -> None:
    """SCENARIO-VERIFY-1474: projected toy cases agree with existing baselines."""

    summary = evaluate_toy_cases()

    assert summary.toy_cases_evaluated == 3
    assert summary.zero_violation_projection is True
    assert summary.max_constraint_violation <= 1e-9
    assert summary.baseline_verifier_agreement is True
    assert summary.projection_iterations_p95 >= summary.projection_iterations_p50
    assert all(case["carnot_verdict"] for case in summary.case_results)
    assert all(case["z3_verdict"] for case in summary.case_results)
    assert all(case["ising_verdict"] for case in summary.case_results)


def test_req_verify_1474_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1474: the deliverable is seeded before projection evaluation."""

    output = tmp_path / "experiment_1474.json"

    artifact = write_in_progress_artifact(output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "in_progress"
    assert artifact["honest_verdict"] == "in_progress"


def test_scenario_verify_1474_run_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1474: the runner writes the required terminal schema."""

    output = tmp_path / "experiment_1474.json"

    artifact = run_experiment(
        output_path=output,
        tests_run=[".venv/bin/pytest tests/python/test_experiment_1474_tskm_linear_constraint_projection_smoke.py -q"],
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["toy_cases_evaluated"] == 3
    assert artifact["zero_violation_projection"] is True
    assert artifact["max_constraint_violation"] <= 1e-9
    assert artifact["baseline_verifier_agreement"] is True
    assert artifact["helper_path"] == "python/carnot/verify/skm_projection.py"
    assert "cpu_only" in artifact["honest_verdict"]
