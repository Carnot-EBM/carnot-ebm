"""Tests for Exp 1399 discrete SB CPU/KV260 estimate.

Spec traces: REQ-ISING-022, SCENARIO-ISING-032
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carnot.samplers.discrete_simulated_bifurcation import (
    DiscreteSBConfig,
    DiscreteSBConstraintProblem,
    bipolar_ising_energy,
    estimate_kv260_discrete_sb_resources,
    make_pressure_schedule,
    run_discrete_sb,
    run_gibbs_ising_baseline,
)
from scripts import experiment_1399_discrete_sb_kv260_cpu_simulation as exp


REQUIRED_FIELDS = {
    "status",
    "algorithm",
    "constraint_problems_tested",
    "n_variables",
    "steps_to_convergence_ising_baseline",
    "steps_to_convergence_discrete_sb",
    "convergence_speedup_discrete_sb",
    "bram_estimate_kb_for_256var",
    "kv260_bram_budget_kb",
    "bram_budget_feasible",
    "lut_estimate_per_update_unit",
    "kv260_lut_budget_fits",
    "hardware_claim_allowed",
    "kv260_claim_allowed",
    "honest_verdict",
}


def _tiny_bipolar_problem() -> DiscreteSBConstraintProblem:
    """Return a small planted problem for deterministic dSB tests."""

    target = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    coupling_matrix = np.zeros((target.size, target.size), dtype=np.int8)
    for i in range(target.size):
        for j in range(i + 1, target.size):
            coupling = int(target[i] * target[j])
            coupling_matrix[i, j] = coupling
            coupling_matrix[j, i] = coupling
    ground_energy = bipolar_ising_energy(target, coupling_matrix)
    return DiscreteSBConstraintProblem(
        name="tiny_bipolar",
        question_id="unit",
        label="synthetic",
        n_variables=target.size,
        coupling_matrix=coupling_matrix,
        target_state=target,
        ground_energy=ground_energy,
        convergence_energy=ground_energy + 0.5,
    )


def test_pressure_schedule_reaches_zero_and_one() -> None:
    """REQ-ISING-022: dSB pressure schedule is linear from 0 to 1."""

    schedule = make_pressure_schedule(5)

    assert schedule == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_discrete_sb_and_gibbs_run_same_problem() -> None:
    """REQ-ISING-022: dSB and Gibbs report convergence steps on one Ising problem."""

    problem = _tiny_bipolar_problem()
    dsb = run_discrete_sb(problem, seed=3, config=DiscreteSBConfig(max_steps=16, eta=0.2))
    gibbs = run_gibbs_ising_baseline(problem, seed=3, max_steps=16, beta=0.2)

    assert 1 <= dsb.steps_to_convergence <= 16
    assert dsb.best_energy <= problem.convergence_energy
    assert 1 <= gibbs.steps_to_convergence <= 16
    assert gibbs.energy_trace


def test_kv260_bram_and_lut_estimate_matches_exp1399_gate() -> None:
    """SCENARIO-ISING-032: 256-variable int8 J fits BRAM and one update unit fits LUTs."""

    estimate = estimate_kv260_discrete_sb_resources(
        n_variables=256,
        bits_per_coupling=8,
        bram36_blocks=144,
        lut_estimate_per_update_unit=2_000,
        kv260_lut_budget=117_000,
    )

    assert estimate["bram_estimate_kb_for_256var"] == 64.0
    assert estimate["kv260_bram_budget_kb"] == 648
    assert estimate["bram_budget_feasible"] is True
    assert estimate["lut_estimate_per_update_unit"] == 2_000
    assert estimate["kv260_lut_budget_fits"] is True


def test_run_experiment_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-ISING-032: Exp 1399 writes required fields and budget-gated claims."""

    output_path = tmp_path / "experiment_1399.json"

    artifact = exp.run_experiment(
        output_path=output_path,
        n_problems=3,
        n_variable_schedule=(8, 12, 16),
        max_steps=24,
        seeds=(0,),
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["status"] == "complete"
    assert len(artifact["constraint_problems_tested"]) == 3
    assert artifact["n_variables"] == [8, 12, 16]
    assert artifact["bram_estimate_kb_for_256var"] == 64.0
    assert artifact["kv260_bram_budget_kb"] == 648
    assert artifact["bram_budget_feasible"] is True
    assert artifact["lut_estimate_per_update_unit"] == 2_000
    assert artifact["kv260_lut_budget_fits"] is True
    assert artifact["hardware_claim_allowed"] is True
    assert artifact["kv260_claim_allowed"] is True

    persisted = json.loads(output_path.read_text())
    assert persisted["honest_verdict"] == artifact["honest_verdict"]
