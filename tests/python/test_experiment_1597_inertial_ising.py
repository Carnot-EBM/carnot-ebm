"""Tests for Exp 1597 inertial dSB Ising CPU ablation.

Spec traces: REQ-ISING-029, SCENARIO-ISING-039
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.samplers.discrete_simulated_bifurcation import (
    DiscreteSBConstraintProblem,
    InertialDiscreteSBConfig,
    bipolar_ising_energy,
    run_discrete_sb,
    run_inertial_discrete_sb,
)
from scripts import experiment_1597_inertial_ising as exp1597


REQUIRED_FIELDS = {
    "status",
    "experiment_id",
    "algorithm",
    "baseline_algorithm",
    "constraint_problems_tested",
    "n_variables",
    "seeds",
    "steps_to_convergence_gibbs_baseline",
    "steps_to_convergence_inertial_ising",
    "convergence_speedup_inertial_ising",
    "inertia_coefficient",
    "pressure_schedule",
    "eta",
    "cpu_only",
    "simulator_only",
    "hardware_execution_performed",
    "hardware_claim_allowed",
    "kv260_claim_allowed",
    "honest_verdict",
    "per_problem_results",
}


def _tiny_bipolar_problem() -> DiscreteSBConstraintProblem:
    """Return a planted dense Ising problem for deterministic ablation tests."""

    target = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    coupling_matrix = np.zeros((target.size, target.size), dtype=np.int8)
    for i in range(target.size):
        for j in range(i + 1, target.size):
            coupling = int(target[i] * target[j])
            coupling_matrix[i, j] = coupling
            coupling_matrix[j, i] = coupling
    ground_energy = bipolar_ising_energy(target, coupling_matrix)
    return DiscreteSBConstraintProblem(
        name="tiny_inertial_bipolar",
        question_id="unit1597",
        label="synthetic",
        n_variables=target.size,
        coupling_matrix=coupling_matrix,
        target_state=target,
        ground_energy=ground_energy,
        convergence_energy=ground_energy + 0.5,
    )


def test_req_ising_029_spec_anchor_exists() -> None:
    """REQ-ISING-029, SCENARIO-ISING-039: Exp 1597 work is spec-anchored."""

    spec = (exp1597.REPO_ROOT / "openspec/capabilities/ising-backend/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-ISING-029" in spec
    assert "SCENARIO-ISING-039" in spec
    assert "results/experiment_1597_inertial_ising.json" in spec


def test_req_ising_029_zero_inertia_matches_base_discrete_sb() -> None:
    """REQ-ISING-029: inertia coefficient 0.0 reduces to the base dSB update."""

    problem = _tiny_bipolar_problem()
    base = run_discrete_sb(
        problem,
        seed=11,
        config=InertialDiscreteSBConfig(max_steps=12, eta=0.2, inertia_coefficient=0.0),
    )
    inertial = run_inertial_discrete_sb(
        problem,
        seed=11,
        config=InertialDiscreteSBConfig(max_steps=12, eta=0.2, inertia_coefficient=0.0),
    )

    assert inertial.steps_to_convergence == base.steps_to_convergence
    assert inertial.best_energy == base.best_energy
    assert inertial.final_energy == base.final_energy
    assert inertial.best_state == base.best_state
    assert inertial.energy_trace == base.energy_trace


def test_req_ising_029_rejects_invalid_inertia_coefficient() -> None:
    """REQ-ISING-029: inertia coefficient must stay in the stable [0, 1) interval."""

    with pytest.raises(ValueError, match="inertia_coefficient"):
        InertialDiscreteSBConfig(inertia_coefficient=1.0)


def test_req_ising_029_default_inertial_config_runs() -> None:
    """REQ-ISING-029: the inertial simulator has a deterministic default config."""

    result = run_inertial_discrete_sb(_tiny_bipolar_problem(), seed=5)

    assert result.inertia_coefficient == 0.6
    assert result.energy_trace


def test_req_ising_029_nonzero_inertia_records_momentum_trace() -> None:
    """REQ-ISING-029: nonzero inertia records deterministic simulator momentum."""

    result = run_inertial_discrete_sb(
        _tiny_bipolar_problem(),
        seed=7,
        config=InertialDiscreteSBConfig(max_steps=8, eta=0.25, inertia_coefficient=0.7),
    )

    assert result.inertia_coefficient == 0.7
    assert len(result.momentum_norm_trace) == len(result.energy_trace)
    assert max(result.momentum_norm_trace) > 0.0
    assert result.as_dict()["inertia_coefficient"] == 0.7


def test_scenario_ising_039_writes_required_cpu_artifact(tmp_path: Path) -> None:
    """SCENARIO-ISING-039: Exp 1597 writes the CPU-only inertial ablation artifact."""

    output_path = tmp_path / "experiment_1597_inertial_ising.json"

    artifact = exp1597.run_experiment(
        output_path=output_path,
        n_problems=3,
        n_variable_schedule=(8, 12, 16),
        max_steps=24,
        seeds=(0,),
        inertia_coefficient=0.6,
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1597
    assert len(artifact["constraint_problems_tested"]) == 3
    assert artifact["n_variables"] == [8, 12, 16]
    assert artifact["seeds"] == [0]
    assert artifact["inertia_coefficient"] == 0.6
    assert artifact["pressure_schedule"] == {"start": 0.0, "end": 1.0, "steps": 24}
    assert artifact["cpu_only"] is True
    assert artifact["simulator_only"] is True
    assert artifact["hardware_execution_performed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["kv260_claim_allowed"] is False
    assert "hardware" not in artifact["honest_verdict"]

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == artifact


def test_scenario_ising_039_records_no_speedup_verdict(tmp_path: Path) -> None:
    """SCENARIO-ISING-039: equal capped step counts record a no-speedup verdict."""

    artifact = exp1597.run_experiment(
        output_path=tmp_path / "experiment_1597_no_speedup.json",
        n_problems=3,
        n_variable_schedule=(8, 12, 16),
        max_steps=1,
        seeds=(0,),
        inertia_coefficient=0.6,
    )

    assert artifact["convergence_speedup_inertial_ising"] == 1.0
    assert artifact["honest_verdict"] == (
        "complete: inertial_ising_no_speedup_observed_cpu_simulator_only"
    )


def test_scenario_ising_039_writes_in_progress_marker(tmp_path: Path) -> None:
    """SCENARIO-ISING-039: runner writes a CPU-only bootstrap marker."""

    marker_path = tmp_path / "experiment_1597_marker.json"

    marker = exp1597.write_in_progress_artifact(marker_path)

    assert marker["status"] == "in_progress"
    assert marker["experiment_id"] == 1597
    assert marker["cpu_only"] is True
    assert marker["simulator_only"] is True
    assert marker["hardware_execution_performed"] is False
    assert marker["hardware_claim_allowed"] is False
    assert marker["kv260_claim_allowed"] is False
    assert json.loads(marker_path.read_text(encoding="utf-8")) == marker


def test_scenario_ising_039_main_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-ISING-039: CLI main writes marker, runs ablation, and prints summary."""

    calls: list[str] = []

    def fake_marker(path: Path = exp1597.DEFAULT_RESULT_PATH) -> dict:
        calls.append(f"marker:{path.name}")
        return {"status": "in_progress"}

    def fake_run() -> dict:
        calls.append("run")
        return {
            "convergence_speedup_inertial_ising": 1.25,
            "hardware_claim_allowed": False,
            "honest_verdict": "complete: fake_cpu_only",
        }

    monkeypatch.setattr(exp1597, "write_in_progress_artifact", fake_marker)
    monkeypatch.setattr(exp1597, "run_experiment", fake_run)

    exp1597.main()

    assert calls == ["marker:experiment_1597_inertial_ising.json", "run"]
    assert "1.25 False complete: fake_cpu_only" in capsys.readouterr().out
