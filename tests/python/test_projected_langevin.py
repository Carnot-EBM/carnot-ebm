"""Tests for Exp 2355 Projected-Langevin constraint sampler.

Spec coverage: REQ-SAMPLE-2355, REQ-SAMPLE-2355-1, REQ-SAMPLE-2355-2,
REQ-SAMPLE-2355-3, REQ-SAMPLE-2355-4, REQ-SAMPLE-2355-5,
SCENARIO-SAMPLE-2355.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carnot.samplers.projected_langevin import (
    BoxConstraint,
    LinearEqualityConstraint,
    ProjectedLangevinSampler,
    build_experiment_2355_artifact,
    validate_experiment_2355_artifact,
    write_experiment_2355_artifact,
)


class QuadraticEnergy:
    def __init__(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=float)

    def __call__(self, x: np.ndarray) -> float:
        residual = np.asarray(x, dtype=float) - self.target
        return float(0.5 * residual @ residual)

    def grad_energy(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) - self.target


def test_req_sample_2355_box_constraints_are_clamped() -> None:
    sampler = ProjectedLangevinSampler(seed=42)
    constraints = [BoxConstraint(i, -0.25, 0.25) for i in range(3)]
    energy = QuadraticEnergy(np.array([2.0, -2.0, 0.0]))

    final = sampler.sample(
        energy,
        constraints,
        init=np.zeros(3),
        n_steps=20,
        step_size=0.2,
        temperature=0.0,
    )

    assert np.all(final >= -0.25 - 1e-12)
    assert np.all(final <= 0.25 + 1e-12)


def test_req_sample_2355_linear_equality_uses_gradient_projection() -> None:
    sampler = ProjectedLangevinSampler(seed=42, projection_steps=4)
    constraints = [LinearEqualityConstraint(np.ones(4), 1.0)]
    energy = QuadraticEnergy(np.array([3.0, 2.0, 1.0, 0.0]))

    final = sampler.sample(
        energy,
        constraints,
        init=np.zeros(4),
        n_steps=12,
        step_size=0.05,
        temperature=0.0,
    )

    assert np.isclose(np.sum(final), 1.0, atol=1e-8)
    assert np.all(np.isfinite(final))


def test_scenario_sample_2355_writes_terminal_artifact(tmp_path: Path) -> None:
    output = tmp_path / "experiment_2355_projected_langevin.json"

    artifact = write_experiment_2355_artifact(output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    validate_experiment_2355_artifact(artifact)
    assert artifact["n_problems"] == 3
    assert artifact["random_seed"] == 42
    assert artifact["constraint_satisfaction_rate"] >= artifact["casal_satisfaction_rate"]
    assert artifact["projected_langevin_competitive"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_sample_2355_artifact_builder_is_deterministic() -> None:
    first = build_experiment_2355_artifact(random_seed=42)
    second = build_experiment_2355_artifact(random_seed=42)

    assert first == second
    assert first["langevin_vs_casal_delta"] == (
        first["constraint_satisfaction_rate"] - first["casal_satisfaction_rate"]
    )
