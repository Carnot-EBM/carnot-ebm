"""Tests for Exp 1675 LagONN toy Max-3-SAT prototype.

Spec traces: REQ-ISING-042, SCENARIO-ISING-042.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import experiment_1675_lagonn as exp


REQUIRED_FIELDS = {
    "status",
    "experiment_id",
    "spec_refs",
    "algorithm",
    "baseline_algorithm",
    "toy_problem",
    "initial_assignment",
    "steps_to_convergence_lagonn",
    "steps_to_convergence_soft_penalty",
    "lagonn_converged",
    "soft_penalty_converged",
    "final_violations_lagonn",
    "final_violations_soft_penalty",
    "convergence_speedup_lagonn_over_soft_penalty",
    "lagrange_multiplier_trace",
    "soft_penalty_trace",
    "cpu_only",
    "simulator_only",
    "hardware_execution_performed",
    "hardware_claim_allowed",
    "honest_verdict",
}


def test_req_ising_042_spec_anchor_exists() -> None:
    """REQ-ISING-042, SCENARIO-ISING-042: Exp 1675 is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/ising-backend/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-ISING-042" in spec
    assert "SCENARIO-ISING-042" in spec
    assert "scripts/experiment_1675_lagonn.py" in spec
    assert "results/experiment_1675_lagonn.json" in spec


def test_req_ising_042_toy_problem_is_deterministic_max3sat() -> None:
    """REQ-ISING-042: the toy instance has deterministic three-literal clauses."""

    problem = exp.build_toy_max3sat_problem()
    initial = problem.initial_assignment()
    feasible = np.asarray([0, 1, 1, 0, 1, 1], dtype=np.int8)

    assert problem.n_variables == 6
    assert len(problem.clauses) == 6
    assert all(len(clause) == 3 for clause in problem.clauses)
    assert problem.violation_vector(initial).tolist() == [0, 0, 1, 1, 0, 0]
    assert problem.violation_count(initial) == 2
    assert problem.violation_count(feasible) == 0
    assert exp.ising_bias_energy(initial, one_bias=1.25) == pytest.approx(-7.5)
    assert exp.fixed_soft_energy(problem, initial, one_bias=1.25, penalty_weight=0.75) == (
        pytest.approx(-6.0)
    )

    with pytest.raises(ValueError, match="three literals"):
        exp.ToyMax3SatProblem(n_variables=3, clauses=(((0, True), (1, True)),))
    with pytest.raises(ValueError, match="out of range"):
        exp.ToyMax3SatProblem(
            n_variables=3,
            clauses=(((0, True), (1, True), (3, False)),),
        )


def test_req_ising_042_config_validation() -> None:
    """REQ-ISING-042: solver controls reject invalid step and penalty settings."""

    assert exp.LagONNConfig(max_steps=3, one_bias=1.0).max_steps == 3

    with pytest.raises(ValueError, match="max_steps"):
        exp.LagONNConfig(max_steps=0)
    with pytest.raises(ValueError, match="one_bias"):
        exp.LagONNConfig(one_bias=0.0)
    with pytest.raises(ValueError, match="soft_penalty_weight"):
        exp.LagONNConfig(soft_penalty_weight=-0.1)
    with pytest.raises(ValueError, match="initial_lambda"):
        exp.LagONNConfig(initial_lambda=-0.1)
    with pytest.raises(ValueError, match="dual_lr"):
        exp.LagONNConfig(dual_lr=0.0)
    with pytest.raises(ValueError, match="lambda_decay"):
        exp.LagONNConfig(lambda_decay=1.0)


def test_req_ising_042_fixed_soft_penalty_stalls_at_infeasible_state() -> None:
    """REQ-ISING-042: fixed soft penalties can remain trapped in violations."""

    problem = exp.build_toy_max3sat_problem()
    config = exp.LagONNConfig(max_steps=6)

    result = exp.run_soft_penalty_baseline(problem, config=config)

    assert result.method == "fixed_soft_penalty_ising"
    assert result.converged is False
    assert result.steps_to_convergence == config.max_steps
    assert result.final_assignment == [1, 1, 1, 1, 1, 1]
    assert result.final_violations == 2
    assert len(result.trace) == config.max_steps + 1
    assert all(record["violations"] == 2 for record in result.trace)
    assert result.as_dict()["method"] == "fixed_soft_penalty_ising"

    high_penalty = exp.run_soft_penalty_baseline(
        problem,
        config=exp.LagONNConfig(max_steps=4, soft_penalty_weight=2.0),
    )
    assert high_penalty.converged is True
    assert high_penalty.final_violations == 0


def test_req_ising_042_lagonn_multiplier_oscillation_converges() -> None:
    """REQ-ISING-042: violated clauses grow multipliers until flips are feasible."""

    problem = exp.build_toy_max3sat_problem()
    config = exp.LagONNConfig(max_steps=8)

    result = exp.run_lagonn_solver(problem, config=config)

    assert result.method == "lagonn_lagrange_multiplier"
    assert result.converged is True
    assert result.steps_to_convergence < config.max_steps
    assert result.final_violations == 0
    assert problem.violation_count(np.asarray(result.final_assignment, dtype=np.int8)) == 0
    assert max(record["lambda"][2] for record in result.trace) > config.initial_lambda
    assert max(record["lambda"][3] for record in result.trace) > config.initial_lambda
    assert any(record["flipped_variable"] is not None for record in result.trace)

    capped = exp.run_lagonn_solver(problem, config=exp.LagONNConfig(max_steps=1))
    assert capped.converged is False
    assert capped.final_violations == 2


def test_req_ising_042_verdict_branches() -> None:
    """REQ-ISING-042: verdicts reflect observed convergence outcomes."""

    assert (
        exp._verdict(lagonn_converged=True, soft_converged=False, violation_delta=2)
        == "complete_lagonn_converged_soft_penalty_stalled_cpu_only"
    )
    assert (
        exp._verdict(lagonn_converged=True, soft_converged=True, violation_delta=0)
        == "complete_both_methods_converged_cpu_only"
    )
    assert (
        exp._verdict(lagonn_converged=False, soft_converged=True, violation_delta=-1)
        == "complete_soft_penalty_better_on_toy_cpu_only"
    )
    assert (
        exp._verdict(lagonn_converged=False, soft_converged=False, violation_delta=0)
        == "complete_no_convergence_difference_cpu_only"
    )


def test_scenario_ising_042_run_experiment_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-ISING-042: Exp 1675 writes the complete CPU-only artifact."""

    output_path = tmp_path / "experiment_1675_lagonn.json"

    artifact = exp.run_experiment(output_path=output_path, config=exp.LagONNConfig(max_steps=8))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1675
    assert artifact["spec_refs"] == ["REQ-ISING-042", "SCENARIO-ISING-042"]
    assert artifact["toy_problem"]["n_variables"] == 6
    assert artifact["initial_assignment"] == [1, 1, 1, 1, 1, 1]
    assert artifact["lagonn_converged"] is True
    assert artifact["soft_penalty_converged"] is False
    assert artifact["final_violations_lagonn"] == 0
    assert artifact["final_violations_soft_penalty"] == 2
    assert artifact["convergence_speedup_lagonn_over_soft_penalty"] > 1.0
    assert artifact["cpu_only"] is True
    assert artifact["simulator_only"] is True
    assert artifact["hardware_execution_performed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert "cpu_only" in artifact["honest_verdict"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_scenario_ising_042_writes_in_progress_marker(tmp_path: Path) -> None:
    """SCENARIO-ISING-042: runner emits a bootstrap marker before solving."""

    marker_path = tmp_path / "experiment_1675_marker.json"

    marker = exp.write_in_progress_artifact(marker_path)

    assert marker["status"] == "in_progress"
    assert marker["experiment_id"] == 1675
    assert marker["spec_refs"] == ["REQ-ISING-042", "SCENARIO-ISING-042"]
    assert marker["cpu_only"] is True
    assert marker["simulator_only"] is True
    assert marker["hardware_execution_performed"] is False
    assert marker["hardware_claim_allowed"] is False
    assert json.loads(marker_path.read_text(encoding="utf-8")) == marker


def test_scenario_ising_042_main_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-ISING-042: CLI writes marker, runs experiment, and prints summary."""

    calls: list[str] = []

    def fake_marker(path: Path = exp.DEFAULT_RESULT_PATH) -> dict:
        calls.append(f"marker:{path.name}")
        return {"status": "in_progress"}

    def fake_run() -> dict:
        calls.append("run")
        return {
            "steps_to_convergence_lagonn": 3,
            "steps_to_convergence_soft_penalty": 8,
            "final_violations_lagonn": 0,
            "final_violations_soft_penalty": 2,
            "hardware_claim_allowed": False,
            "honest_verdict": "complete_lagonn_converged_soft_penalty_stalled_cpu_only",
        }

    monkeypatch.setattr(exp, "write_in_progress_artifact", fake_marker)
    monkeypatch.setattr(exp, "run_experiment", fake_run)

    exp.main()

    assert calls == ["marker:experiment_1675_lagonn.json", "run"]
    assert "3 8 0 2 False complete_lagonn_converged_soft_penalty_stalled_cpu_only" in (
        capsys.readouterr().out
    )
