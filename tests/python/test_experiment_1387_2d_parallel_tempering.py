"""Tests for Exp 1387 2D parallel tempering CPU/KV260 estimate.

Spec traces: REQ-ISING-021, SCENARIO-ISING-031
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carnot.samplers.two_dimensional_parallel_tempering import (
    IsingConstraintProblem,
    ParallelTemperingConfig,
    TwoDParallelTemperingSampler,
    estimate_kv260_lut_budget,
    ising_energy,
    make_temperature_schedule,
    metropolis_swap_acceptance_probability,
    run_single_temperature_ising,
)
from scripts import experiment_1387_2d_parallel_tempering_kv260_fpga_estimate as exp


REQUIRED_FIELDS = {
    "status",
    "constraint_problems_tested",
    "replica_count",
    "temperature_schedule",
    "steps_to_convergence_standard_pt",
    "steps_to_convergence_2d_pt",
    "convergence_speedup_2d_pt",
    "sparsification_k_value",
    "estimated_kv260_lut_count_per_replica",
    "estimated_kv260_total_lut_count_15_replicas",
    "lut_budget_feasible",
    "hardware_claim_allowed",
    "kv260_claim_allowed",
    "honest_verdict",
}


def _tiny_planted_problem() -> IsingConstraintProblem:
    """Return a small low-energy all-ones problem for deterministic sampler tests."""
    n_spins = 6
    target = np.ones(n_spins, dtype=np.float64)
    biases = np.ones(n_spins, dtype=np.float64) * 1.2
    coupling_matrix = np.ones((n_spins, n_spins), dtype=np.float64) * 0.35
    np.fill_diagonal(coupling_matrix, 0.0)
    ground_energy = ising_energy(target, biases, coupling_matrix)
    return IsingConstraintProblem(
        name="tiny_planted",
        question_id="unit",
        label="synthetic",
        n_spins=n_spins,
        biases=biases,
        coupling_matrix=coupling_matrix,
        target_state=target,
        ground_energy=ground_energy,
        convergence_energy=ground_energy + 0.1,
    )


def test_temperature_schedule_and_swap_probability() -> None:
    """REQ-ISING-021: 15 temperatures cover [0.5, 5.0] and swaps use Metropolis."""
    schedule = make_temperature_schedule(
        replica_count=15,
        min_temperature=0.5,
        max_temperature=5.0,
    )

    assert len(schedule) == 15
    assert schedule[0] == 0.5
    assert schedule[-1] == 5.0

    cold_beta = 1.0 / schedule[0]
    hot_beta = 1.0 / schedule[-1]
    favorable = metropolis_swap_acceptance_probability(
        energy_left=5.0,
        energy_right=-5.0,
        beta_left=cold_beta,
        beta_right=hot_beta,
    )
    unfavorable = metropolis_swap_acceptance_probability(
        energy_left=-5.0,
        energy_right=5.0,
        beta_left=cold_beta,
        beta_right=hot_beta,
    )

    assert favorable == 1.0
    assert 0.0 < unfavorable < 0.001


def test_2d_parallel_tempering_runs_replicas_and_swaps() -> None:
    """REQ-ISING-021: PT runs multiple temperatures and attempts adjacent swaps."""
    problem = _tiny_planted_problem()
    config = ParallelTemperingConfig(
        replica_count=5,
        min_temperature=0.5,
        max_temperature=2.0,
        max_steps=24,
        swap_interval=1,
    )

    pt = TwoDParallelTemperingSampler(config).run(problem, seed=7)
    standard = run_single_temperature_ising(
        problem,
        seed=7,
        temperature=0.5,
        max_steps=24,
    )

    assert 1 <= pt.steps_to_convergence <= 24
    assert pt.best_energy <= problem.convergence_energy
    assert pt.swap_attempts > 0
    assert standard.steps_to_convergence >= 1


def test_kv260_lut_estimate_is_honest_about_15_replicas() -> None:
    """SCENARIO-ISING-031: 15 replicas exceed KV260 but three replicas fit."""
    estimate = estimate_kv260_lut_budget(
        replica_count=15,
        lut_count_per_replica=36_000,
        kv260_lut_budget=117_000,
    )

    assert estimate["estimated_kv260_total_lut_count_15_replicas"] == 540_000
    assert estimate["max_replicas_that_fit_kv260_budget"] == 3
    assert estimate["fits_15_replicas_kv260_budget"] is False
    assert estimate["lut_budget_feasible"] is True


def test_run_experiment_writes_required_cpu_only_artifact(tmp_path: Path) -> None:
    """SCENARIO-ISING-031: Exp 1387 writes required fields and no HW claim."""
    output_path = tmp_path / "experiment_1387.json"

    artifact = exp.run_experiment(
        output_path=output_path,
        n_problems=3,
        n_spins=12,
        max_steps=18,
        seeds=(0,),
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["status"] == "complete"
    assert artifact["replica_count"] == 15
    assert artifact["temperature_schedule"][0] == 0.5
    assert artifact["temperature_schedule"][-1] == 5.0
    assert len(artifact["constraint_problems_tested"]) == 3
    assert artifact["estimated_kv260_lut_count_per_replica"] == 36_000
    assert artifact["estimated_kv260_total_lut_count_15_replicas"] == 540_000
    assert artifact["lut_budget_feasible"] is True
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["kv260_claim_allowed"] is False

    persisted = json.loads(output_path.read_text())
    assert persisted["honest_verdict"] == artifact["honest_verdict"]
