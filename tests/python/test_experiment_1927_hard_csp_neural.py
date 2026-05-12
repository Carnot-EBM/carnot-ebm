"""Tests for Exp 1927 hard-CSP neural-solver reality check.

Spec traces: REQ-ISING-044, SCENARIO-ISING-044.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_1927_hard_csp_neural as exp


REQUIRED_FIELDS = {
    "status",
    "experiment_id",
    "spec_refs",
    "run_date",
    "solver_name",
    "csp_family",
    "problem",
    "config",
    "time_budget_s",
    "wall_time_s",
    "timeout_exceeded",
    "assignments_evaluated",
    "true_constraint_satisfaction_rate",
    "best_satisfied_constraints",
    "total_constraints",
    "best_assignment",
    "attempts",
    "cpu_only",
    "hardware_execution_performed",
    "hardware_claim_allowed",
    "honest_verdict",
}


def test_req_ising_044_spec_anchor_exists() -> None:
    """REQ-ISING-044, SCENARIO-ISING-044: Exp 1927 is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/ising-backend/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-ISING-044" in spec
    assert "SCENARIO-ISING-044" in spec
    assert "scripts/experiment_1927_hard_csp_neural.py" in spec
    assert "results/experiment_1927_hard_csp_neural.json" in spec


def test_req_ising_044_direct_clause_satisfaction_rate() -> None:
    """REQ-ISING-044: scoring checks the true 3-SAT clauses directly."""

    instance = exp.Hard3SatInstance(
        n_variables=3,
        clauses=((1, -2, 3), (1, 2, -3), (-1, -2, -3)),
        planted_assignment=(True, False, True),
        seed=11,
    )

    assert instance.satisfied_constraints((True, False, True)) == 3
    assert instance.constraint_satisfaction_rate((True, False, True)) == pytest.approx(1.0)
    assert instance.satisfied_constraints((False, True, False)) == 2
    assert instance.constraint_satisfaction_rate((False, True, False)) == pytest.approx(2 / 3)
    assert instance.as_dict()["clauses"][0] == [1, -2, 3]

    with pytest.raises(ValueError, match="three literals"):
        exp.Hard3SatInstance(
            n_variables=3,
            clauses=((1, -2),),
            planted_assignment=(True, False, True),
            seed=11,
        )
    with pytest.raises(ValueError, match="out of range"):
        exp.Hard3SatInstance(
            n_variables=3,
            clauses=((1, 0, 3),),
            planted_assignment=(True, False, True),
            seed=11,
        )
    with pytest.raises(ValueError, match="planted assignment"):
        exp.Hard3SatInstance(
            n_variables=3,
            clauses=((1, -2, 3),),
            planted_assignment=(True, False),
            seed=11,
        )


def test_req_ising_044_hard_3sat_instance_is_deterministic_and_planted_sat() -> None:
    """REQ-ISING-044: the generated hard 3-SAT instance is deterministic."""

    config = exp.HardCspNeuralConfig(n_variables=6, n_clauses=26, seed=1927)
    first = exp.build_hard_3sat_instance(config)
    second = exp.build_hard_3sat_instance(config)

    assert first == second
    assert first.n_variables == 6
    assert len(first.clauses) == 26
    assert first.clause_density == pytest.approx(26 / 6)
    assert first.constraint_satisfaction_rate(first.planted_assignment) == pytest.approx(1.0)
    assert all(len(set(abs(literal) for literal in clause)) == 3 for clause in first.clauses)


def test_req_ising_044_config_validation() -> None:
    """REQ-ISING-044: invalid solver and problem budgets are rejected."""

    assert exp.HardCspNeuralConfig(max_steps=3, attempts=1).max_steps == 3

    with pytest.raises(ValueError, match="n_variables"):
        exp.HardCspNeuralConfig(n_variables=2)
    with pytest.raises(ValueError, match="n_clauses"):
        exp.HardCspNeuralConfig(n_clauses=0)
    with pytest.raises(ValueError, match="unique planted"):
        exp.HardCspNeuralConfig(n_variables=3, n_clauses=9)
    with pytest.raises(ValueError, match="attempts"):
        exp.HardCspNeuralConfig(attempts=0)
    with pytest.raises(ValueError, match="max_steps"):
        exp.HardCspNeuralConfig(max_steps=0)
    with pytest.raises(ValueError, match="time_budget_s"):
        exp.HardCspNeuralConfig(time_budget_s=0.0)
    with pytest.raises(ValueError, match="multiplier_lr"):
        exp.HardCspNeuralConfig(multiplier_lr=0.0)
    with pytest.raises(ValueError, match="multiplier_decay"):
        exp.HardCspNeuralConfig(multiplier_decay=1.0)


def test_req_ising_044_neural_solver_reports_true_csr_within_budget() -> None:
    """REQ-ISING-044: evaluation reports direct true CSR and bounded work."""

    config = exp.HardCspNeuralConfig(
        n_variables=6,
        n_clauses=26,
        attempts=2,
        max_steps=10,
        time_budget_s=1.0,
        seed=1927,
    )

    report = exp.evaluate_neural_solver_on_hard_csp(config)

    assert report["status"] == "complete"
    assert report["solver_name"] == "lagrange_neural_clause_weight_local_search"
    assert report["csp_family"] == "planted_3sat"
    assert report["time_budget_s"] == pytest.approx(1.0)
    assert report["wall_time_s"] <= report["time_budget_s"]
    assert report["assignments_evaluated"] > 0
    assert report["total_constraints"] == 26
    assert report["best_satisfied_constraints"] <= report["total_constraints"]
    assert report["true_constraint_satisfaction_rate"] == pytest.approx(
        report["best_satisfied_constraints"] / report["total_constraints"]
    )
    assert len(report["best_assignment"]) == config.n_variables
    assert len(report["attempts"]) <= config.attempts
    assert all("true_constraint_satisfaction_rate" in row for row in report["attempts"])


def test_req_ising_044_timeout_stops_before_launching_solver_work() -> None:
    """REQ-ISING-044: exhausted wall-clock budget prevents solver attempts."""

    config = exp.HardCspNeuralConfig(
        n_variables=6,
        n_clauses=26,
        attempts=3,
        max_steps=10,
        time_budget_s=0.1,
        seed=1927,
    )
    ticks = iter([0.0, 0.2, 0.2])

    report = exp.evaluate_neural_solver_on_hard_csp(config, clock=lambda: next(ticks))

    assert report["timeout_exceeded"] is True
    assert report["assignments_evaluated"] == 0
    assert report["attempts"] == []
    assert report["best_assignment"] == []
    assert report["honest_verdict"] == "timeout_before_any_assignment_cpu_only"


def test_req_ising_044_timeout_during_attempt_is_reported() -> None:
    """REQ-ISING-044: timeout inside a launched attempt propagates to the artifact."""

    config = exp.HardCspNeuralConfig(
        n_variables=6,
        n_clauses=26,
        attempts=3,
        max_steps=10,
        time_budget_s=0.1,
        seed=1927,
    )
    ticks = iter([0.0, 0.0, 0.2, 0.2])

    report = exp.evaluate_neural_solver_on_hard_csp(config, clock=lambda: next(ticks))

    assert report["timeout_exceeded"] is True
    assert report["assignments_evaluated"] == 1
    assert report["attempts"][0]["timed_out"] is True
    assert report["honest_verdict"] == "timeout_partial_constraint_satisfaction_cpu_only"


def test_req_ising_044_attempt_timeout_and_exploration_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ISING-044: attempt-level timeout and exploratory flips are bounded."""

    instance = exp.Hard3SatInstance(
        n_variables=3,
        clauses=((1, -2, 3), (1, 2, -3), (-1, -2, -3)),
        planted_assignment=(True, False, True),
        seed=11,
    )
    config = exp.HardCspNeuralConfig(
        n_variables=3,
        n_clauses=3,
        attempts=1,
        max_steps=2,
        time_budget_s=1.0,
        seed=1,
    )
    candidate_ticks = iter([0.0, 2.0])

    timed = exp._run_neural_attempt(
        instance,
        config,
        attempt_index=0,
        deadline=1.0,
        clock=lambda: next(candidate_ticks),
    )
    assert timed["timed_out"] is True
    assert timed["steps_run"] == 1

    tautology = exp.Hard3SatInstance(
        n_variables=3,
        clauses=((1, -1, 2),),
        planted_assignment=(True, True, True),
        seed=12,
    )
    solved = exp._run_neural_attempt(
        tautology,
        config,
        attempt_index=0,
        deadline=1.0,
        clock=lambda: 0.0,
    )
    assert solved["best_true_constraint_satisfaction_rate"] == 1.0

    monkeypatch.setattr(exp, "_weighted_clause_score", lambda *_args: 0.0)
    explored = exp._run_neural_attempt(
        instance,
        config,
        attempt_index=0,
        deadline=1.0,
        clock=lambda: 0.0,
    )
    assert explored["steps_run"] >= 1
    assert explored["assignments_evaluated"] > 1


def test_req_ising_044_verdict_branches() -> None:
    """REQ-ISING-044: verdicts distinguish solved, partial, and timeout runs."""

    assert (
        exp._honest_verdict(rate=1.0, timeout_exceeded=False, assignments_evaluated=3)
        == "complete_neural_solver_found_satisfying_assignment_cpu_only"
    )
    assert (
        exp._honest_verdict(rate=0.91, timeout_exceeded=False, assignments_evaluated=3)
        == "complete_partial_constraint_satisfaction_cpu_only"
    )
    assert (
        exp._honest_verdict(rate=0.91, timeout_exceeded=True, assignments_evaluated=3)
        == "timeout_partial_constraint_satisfaction_cpu_only"
    )
    assert (
        exp._honest_verdict(rate=0.0, timeout_exceeded=True, assignments_evaluated=0)
        == "timeout_before_any_assignment_cpu_only"
    )


def test_scenario_ising_044_run_experiment_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-ISING-044: Exp 1927 writes the complete CPU-only artifact."""

    output_path = tmp_path / "experiment_1927_hard_csp_neural.json"
    config = exp.HardCspNeuralConfig(
        n_variables=6,
        n_clauses=26,
        attempts=2,
        max_steps=10,
        time_budget_s=1.0,
        seed=1927,
    )

    artifact = exp.run_experiment(output_path=output_path, config=config)

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1927
    assert artifact["spec_refs"] == ["REQ-ISING-044", "SCENARIO-ISING-044"]
    assert artifact["problem"]["n_variables"] == 6
    assert artifact["problem"]["n_clauses"] == 26
    assert 0.0 <= artifact["true_constraint_satisfaction_rate"] <= 1.0
    assert artifact["cpu_only"] is True
    assert artifact["hardware_execution_performed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert "cpu_only" in artifact["honest_verdict"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_scenario_ising_044_writes_in_progress_marker(tmp_path: Path) -> None:
    """SCENARIO-ISING-044: runner emits a bootstrap marker before solving."""

    marker_path = tmp_path / "experiment_1927_marker.json"

    marker = exp.write_in_progress_artifact(marker_path)

    assert marker["status"] == "in_progress"
    assert marker["experiment_id"] == 1927
    assert marker["spec_refs"] == ["REQ-ISING-044", "SCENARIO-ISING-044"]
    assert marker["cpu_only"] is True
    assert marker["hardware_execution_performed"] is False
    assert marker["hardware_claim_allowed"] is False
    assert json.loads(marker_path.read_text(encoding="utf-8")) == marker


def test_scenario_ising_044_main_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-ISING-044: CLI writes marker, runs, and prints a summary."""

    calls: list[str] = []

    def fake_marker(path: Path = exp.DEFAULT_RESULT_PATH) -> dict:
        calls.append(f"marker:{path.name}")
        return {"status": "in_progress"}

    def fake_run() -> dict:
        calls.append("run")
        return {
            "true_constraint_satisfaction_rate": 0.875,
            "best_satisfied_constraints": 21,
            "total_constraints": 24,
            "assignments_evaluated": 11,
            "timeout_exceeded": False,
            "honest_verdict": "complete_partial_constraint_satisfaction_cpu_only",
        }

    monkeypatch.setattr(exp, "write_in_progress_artifact", fake_marker)
    monkeypatch.setattr(exp, "run_experiment", fake_run)

    exp.main()

    assert calls == ["marker:experiment_1927_hard_csp_neural.json", "run"]
    assert "0.875 21 24 11 False complete_partial_constraint_satisfaction_cpu_only" in (
        capsys.readouterr().out
    )
