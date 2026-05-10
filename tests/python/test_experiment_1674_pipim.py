"""Tests for Exp 1674 CPU-only PIPIM dense Ising ablation.

Spec traces: REQ-ISING-041, SCENARIO-ISING-041
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import experiment_1674_pipim as exp


REQUIRED_FIELDS = {
    "status",
    "experiment_id",
    "spec_refs",
    "algorithm",
    "baseline_algorithm",
    "dense_problems_tested",
    "n_variables",
    "seeds",
    "time_to_energy_gibbs_baseline",
    "time_to_energy_pipim",
    "time_to_energy_delta_steps",
    "time_to_energy_speedup",
    "sample_quality_gibbs_baseline",
    "sample_quality_pipim",
    "sample_quality_delta",
    "cpu_only",
    "simulator_only",
    "hardware_execution_performed",
    "hardware_claim_allowed",
    "honest_verdict",
}


def _write_fover_rows(path: Path, count: int = 3) -> None:
    rows = [
        {
            "question_id": f"unit-{index}",
            "step_text": f"Dense p-bit unit row {index}: {index} + 1 is deterministic.",
            "label": "correct" if index % 2 == 0 else "incorrect",
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_req_ising_041_spec_anchor_exists() -> None:
    """REQ-ISING-041, SCENARIO-ISING-041: Exp 1674 work is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/ising-backend/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-ISING-041" in spec
    assert "SCENARIO-ISING-041" in spec
    assert "scripts/experiment_1674_pipim.py" in spec
    assert "results/experiment_1674_pipim.json" in spec


def test_req_ising_041_dense_problem_builder_is_deterministic() -> None:
    """REQ-ISING-041: FoVer rows deterministically define dense planted problems."""

    row = {"question_id": "unit", "step_text": "x + y = y + x", "label": "correct"}

    first = exp.build_dense_pbit_problem(row, row_index=0, n_variables=8)
    second = exp.build_dense_pbit_problem(row, row_index=0, n_variables=8)

    assert first.name == "fover_unit_row0_n8"
    assert first.n_variables == 8
    assert np.array_equal(first.coupling_matrix, second.coupling_matrix)
    assert np.array_equal(first.target_state, second.target_state)
    assert np.allclose(first.coupling_matrix, first.coupling_matrix.T)
    assert np.allclose(np.diag(first.coupling_matrix), 0.0)
    assert np.count_nonzero(np.triu(first.coupling_matrix, k=1)) == 28
    assert exp.bipolar_energy(first.target_state, first.coupling_matrix) == pytest.approx(
        first.target_energy
    )

    with pytest.raises(ValueError, match="n_variables"):
        exp.build_dense_pbit_problem(row, row_index=0, n_variables=1)


def test_req_ising_041_config_validation() -> None:
    """REQ-ISING-041: PIPIM controls reject unstable settings."""

    assert exp.PIPIMConfig(max_steps=3, beta=1.0, inertia_alpha=0.5).max_steps == 3

    with pytest.raises(ValueError, match="max_steps"):
        exp.PIPIMConfig(max_steps=0)
    with pytest.raises(ValueError, match="beta"):
        exp.PIPIMConfig(beta=0.0)
    with pytest.raises(ValueError, match="inertia_alpha"):
        exp.PIPIMConfig(inertia_alpha=1.0)


def test_req_ising_041_pipim_run_is_seeded_and_records_quality() -> None:
    """REQ-ISING-041: synchronous p-bit inertia runs deterministically by seed."""

    row = {"question_id": "pipim", "step_text": "seeded dense instance", "label": "correct"}
    problem = exp.build_dense_pbit_problem(row, row_index=1, n_variables=6)
    config = exp.PIPIMConfig(max_steps=10, beta=1.8, inertia_alpha=0.6)

    first = exp.run_pipim(problem, seed=7, config=config)
    second = exp.run_pipim(problem, seed=7, config=config)

    assert first.steps_to_energy == second.steps_to_energy
    assert first.energy_trace == second.energy_trace
    assert first.best_state == second.best_state
    assert len(first.energy_trace) == len(first.ema_norm_trace)
    assert max(first.ema_norm_trace) > 0.0
    assert 0.0 <= first.target_overlap <= 1.0
    assert first.as_dict()["sampler"] == "synchronous_pbit_inertia_pipim"

    unreachable = exp.DensePBitProblem(
        name=problem.name,
        question_id=problem.question_id,
        label=problem.label,
        n_variables=problem.n_variables,
        coupling_matrix=problem.coupling_matrix,
        target_state=problem.target_state,
        target_energy=problem.target_energy,
        convergence_energy=problem.target_energy - 1.0,
    )
    capped = exp.run_pipim(unreachable, seed=7, config=exp.PIPIMConfig(max_steps=3))
    assert capped.steps_to_energy == 3
    assert capped.reached_energy is False


def test_req_ising_041_problem_loader_contracts(tmp_path: Path) -> None:
    """REQ-ISING-041: loader enforces the 3-to-5 dense-problem experiment scope."""

    fover_path = tmp_path / "fover.jsonl"
    _write_fover_rows(fover_path, count=5)

    problems = exp.load_dense_pbit_problems(
        repo_root=tmp_path,
        limit=3,
        fover_path=fover_path,
    )

    assert [problem.n_variables for problem in problems] == [32, 48, 64]

    with pytest.raises(ValueError, match="between 3 and 5"):
        exp.load_dense_pbit_problems(repo_root=tmp_path, limit=2, fover_path=fover_path)
    with pytest.raises(ValueError, match="length"):
        exp.load_dense_pbit_problems(
            repo_root=tmp_path,
            limit=3,
            n_variable_schedule=(4, 5),
            fover_path=fover_path,
        )
    with pytest.raises(ValueError, match="at least 2"):
        exp.load_dense_pbit_problems(
            repo_root=tmp_path,
            limit=3,
            n_variable_schedule=(4, 1, 6),
            fover_path=fover_path,
        )

    short_path = tmp_path / "short_fover.jsonl"
    _write_fover_rows(short_path, count=2)
    with pytest.raises(ValueError, match="needed 3 FoVer rows"):
        exp.load_dense_pbit_problems(repo_root=tmp_path, limit=3, fover_path=short_path)


def test_req_ising_041_verdict_branches() -> None:
    """REQ-ISING-041: verdicts reflect observed time and quality deltas."""

    assert (
        exp._verdict(time_to_energy_delta=1.0, gap_reduction=1.0)
        == "complete_pipim_time_and_quality_improved_cpu_simulator_only"
    )
    assert (
        exp._verdict(time_to_energy_delta=1.0, gap_reduction=0.0)
        == "complete_pipim_time_improved_quality_not_improved_cpu_simulator_only"
    )
    assert (
        exp._verdict(time_to_energy_delta=0.0, gap_reduction=1.0)
        == "complete_pipim_quality_improved_time_not_improved_cpu_simulator_only"
    )
    assert (
        exp._verdict(time_to_energy_delta=0.0, gap_reduction=0.0)
        == "complete_pipim_no_improvement_observed_cpu_simulator_only"
    )


def test_scenario_ising_041_writes_required_cpu_artifact(tmp_path: Path) -> None:
    """SCENARIO-ISING-041: Exp 1674 writes the CPU-only PIPIM ablation artifact."""

    fover_path = tmp_path / "fover.jsonl"
    output_path = tmp_path / "experiment_1674_pipim.json"
    _write_fover_rows(fover_path)

    artifact = exp.run_experiment(
        output_path=output_path,
        n_problems=3,
        n_variable_schedule=(6, 8, 10),
        max_steps=14,
        seeds=(0,),
        beta=1.5,
        inertia_alpha=0.55,
        fover_path=fover_path,
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1674
    assert artifact["spec_refs"] == ["REQ-ISING-041", "SCENARIO-ISING-041"]
    assert len(artifact["dense_problems_tested"]) == 3
    assert artifact["n_variables"] == [6, 8, 10]
    assert artifact["seeds"] == [0]
    assert isinstance(artifact["time_to_energy_delta_steps"], float)
    assert isinstance(artifact["time_to_energy_speedup"], float)
    assert "best_energy_gap_reduction" in artifact["sample_quality_delta"]
    assert artifact["cpu_only"] is True
    assert artifact["simulator_only"] is True
    assert artifact["hardware_execution_performed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert "hardware" not in artifact["honest_verdict"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_scenario_ising_041_writes_in_progress_marker(tmp_path: Path) -> None:
    """SCENARIO-ISING-041: runner writes a CPU-only bootstrap marker."""

    marker_path = tmp_path / "experiment_1674_marker.json"

    marker = exp.write_in_progress_artifact(marker_path)

    assert marker["status"] == "in_progress"
    assert marker["experiment_id"] == 1674
    assert marker["cpu_only"] is True
    assert marker["simulator_only"] is True
    assert marker["hardware_execution_performed"] is False
    assert marker["hardware_claim_allowed"] is False
    assert json.loads(marker_path.read_text(encoding="utf-8")) == marker


def test_scenario_ising_041_main_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-ISING-041: CLI writes marker, runs ablation, and prints deltas."""

    calls: list[str] = []

    def fake_marker(path: Path = exp.DEFAULT_RESULT_PATH) -> dict:
        calls.append(f"marker:{path.name}")
        return {"status": "in_progress"}

    def fake_run() -> dict:
        calls.append("run")
        return {
            "time_to_energy_delta_steps": 3.0,
            "sample_quality_delta": {"best_energy_gap_reduction": 1.25},
            "hardware_claim_allowed": False,
            "honest_verdict": "complete_pipim_cpu_only",
        }

    monkeypatch.setattr(exp, "write_in_progress_artifact", fake_marker)
    monkeypatch.setattr(exp, "run_experiment", fake_run)

    exp.main()

    assert calls == ["marker:experiment_1674_pipim.json", "run"]
    assert "3.0 1.25 False complete_pipim_cpu_only" in capsys.readouterr().out
