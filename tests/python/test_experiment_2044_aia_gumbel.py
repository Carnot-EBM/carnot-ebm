"""Tests for Exp 2044 AIA Gumbel sampler simulation.

Spec traces: REQ-SAMPLE-2044, SCENARIO-SAMPLE-2044.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from carnot.core.energy import AutoGradMixin
from carnot.samplers.continuous_gumbel import ContinuousGumbelSampler
import scripts.experiment_2044_aia_gumbel as exp


class QuadraticEnergy(AutoGradMixin):
    """Quadratic bowl for REQ-SAMPLE-2044 sampler checks."""

    @property
    def input_dim(self) -> int:
        return 3

    def energy(self, x: jax.Array) -> jax.Array:
        return jnp.sum(x**2)


def test_req_sample_2044_spec_entry_exists() -> None:
    """REQ-SAMPLE-2044 and SCENARIO-SAMPLE-2044 are anchored before code."""
    spec = Path("openspec/capabilities/samplers/spec.md").read_text()
    assert "REQ-SAMPLE-2044" in spec
    assert "SCENARIO-SAMPLE-2044" in spec


def test_req_sample_2044_hard_gumbel_move_reduces_energy() -> None:
    """REQ-SAMPLE-2044-2/3: categorical Gumbel moves update a continuous state."""
    model = QuadraticEnergy()
    init = jnp.array([0.5, -0.5, 0.0])
    sampler = ContinuousGumbelSampler(
        step_size=0.1,
        energy_temperature=0.05,
        gumbel_scale=0.0,
        hard=True,
    )

    chain = sampler.sample_chain(model, init, n_steps=1, key=jax.random.PRNGKey(2044))
    final = sampler.sample(model, init, n_steps=1, key=jax.random.PRNGKey(2044))

    assert chain.shape == (1, 3)
    assert jnp.allclose(chain[0], jnp.array([0.4, -0.4, 0.0]))
    assert jnp.allclose(final, chain[-1])
    assert float(model.energy(final)) < float(model.energy(init))


def test_req_sample_2044_relaxed_noise_and_gradient_clipping_paths() -> None:
    """REQ-SAMPLE-2044-2: relaxed Gumbel-Softmax path remains finite."""
    model = QuadraticEnergy()
    init = jnp.array([0.6, -0.4, 0.2])
    sampler = ContinuousGumbelSampler(
        step_size=0.04,
        energy_temperature=0.05,
        softmax_temperature=0.4,
        gumbel_scale=0.1,
        hard=False,
        clip_norm=1.0,
    )

    noise = sampler._gumbel_noise(jax.random.PRNGKey(9), (2, 3))
    clipped = sampler._clip_gradient(jnp.array([3.0, 4.0, 0.0]))
    chain = sampler.sample_chain(model, init, n_steps=8, key=jax.random.PRNGKey(10))

    assert noise.shape == (2, 3)
    assert jnp.all(jnp.isfinite(noise))
    assert jnp.allclose(jnp.linalg.norm(clipped), 1.0, atol=1e-6)
    assert chain.shape == (8, 3)
    assert jnp.all(jnp.isfinite(chain))


def test_req_sample_2044_rejects_negative_steps() -> None:
    """REQ-SAMPLE-2044-3: sampler validates step counts."""
    with pytest.raises(ValueError, match="n_steps"):
        ContinuousGumbelSampler().sample(QuadraticEnergy(), jnp.zeros(3), n_steps=-1)


def test_scenario_sample_2044_eqm_landscape_converges_faster_than_mh() -> None:
    """SCENARIO-SAMPLE-2044: Exp 2041 landscape comparison favors Gumbel."""
    landscape = exp.EqM2041Landscape()
    init = exp.initial_eqm_state()
    artifact = exp.run_simulation(n_steps=80, threshold=0.05)

    assert float(landscape.energy(init)) == artifact["problem"]["initial_energy"]
    assert artifact["metrics"]["continuous_gumbel"]["finite_chain"] is True
    assert artifact["metrics"]["metropolis_hastings"]["finite_chain"] is True
    assert artifact["metrics"]["continuous_gumbel"]["final_energy"] < artifact["problem"]["initial_energy"]
    assert artifact["metrics"]["metropolis_hastings"]["final_energy"] < artifact["problem"]["initial_energy"]
    assert artifact["metrics"]["continuous_gumbel"]["converged"] is True
    assert artifact["gumbel_faster_than_metropolis_hastings"] is True
    assert artifact["gumbel_speedup"] > 1.0


def test_scenario_sample_2044_experiment_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-2044: run_experiment writes the required JSON artifact."""
    output_path = tmp_path / "experiment_2044_aia_gumbel.json"
    artifact = exp.run_experiment(output_path=output_path)

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == artifact
    assert artifact["experiment"] == 2044
    assert artifact["status"] == "success"
    assert artifact["spec_refs"] == ["REQ-SAMPLE-2044", "SCENARIO-SAMPLE-2044"]
    assert artifact["deliverable"] == "results/experiment_2044_aia_gumbel.json"
    assert artifact["metrics"]["continuous_gumbel"]["converged"] is True
    assert artifact["gumbel_faster_than_metropolis_hastings"] is True
