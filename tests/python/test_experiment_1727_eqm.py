"""Tests for Equilibrium Matching and Exp 1727.

Spec traces: REQ-SAMPLE-1727, REQ-SAMPLE-1728,
SCENARIO-SAMPLE-1727, SCENARIO-SAMPLE-1728.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from carnot.core.energy import AutoGradMixin
from carnot.samplers.equilibrium_matching import EquilibriumMatchingSampler
import scripts.experiment_1727_eqm as exp


class SoftConstraintBowl(AutoGradMixin):
    """Small convex constraint energy used to test REQ-SAMPLE-1727."""

    def __init__(self, target: jax.Array) -> None:
        self.target = target

    @property
    def input_dim(self) -> int:
        return int(self.target.shape[0])

    def energy(self, x: jax.Array) -> jax.Array:
        residual = x - self.target
        return 0.5 * jnp.sum(residual**2)


def test_req_sample_1727_spec_entry_exists() -> None:
    spec = Path("openspec/capabilities/samplers/spec.md").read_text()
    assert "REQ-SAMPLE-1727" in spec
    assert "SCENARIO-SAMPLE-1727" in spec


def test_req_sample_1727_gradient_learning_and_clipping() -> None:
    sampler = EquilibriumMatchingSampler(
        step_size=0.2,
        learning_rate=0.25,
        matching_strength=0.5,
        clip_norm=2.0,
    )
    grad = jnp.array([3.0, 4.0])

    clipped = sampler._clip_gradient(grad)
    assert jnp.allclose(jnp.linalg.norm(clipped), 2.0, atol=1e-6)
    assert jnp.allclose(clipped / jnp.linalg.norm(clipped), grad / jnp.linalg.norm(grad))

    learned = sampler._update_learned_gradient(jnp.zeros_like(grad), clipped)
    assert jnp.allclose(learned, 0.25 * clipped)

    matched = sampler._matched_gradient(clipped, learned)
    expected = 0.5 * clipped + 0.5 * learned
    assert jnp.allclose(matched, expected)

    unclipped_sampler = EquilibriumMatchingSampler()
    assert jnp.allclose(unclipped_sampler._clip_gradient(grad), grad)
    assert jnp.allclose(sampler._clip_gradient(jnp.zeros(2)), jnp.zeros(2))


def test_scenario_sample_1727_eqm_converges_and_matches_chain_tail() -> None:
    model = SoftConstraintBowl(target=jnp.array([0.5, -0.25, 0.75]))
    init = jnp.array([4.0, -3.5, 2.5])
    sampler = EquilibriumMatchingSampler(
        step_size=0.35,
        learning_rate=0.6,
        matching_strength=0.8,
        momentum=0.0,
        clip_norm=10.0,
    )

    chain = sampler.sample_chain(model, init, n_steps=40)
    final = sampler.sample(model, init, n_steps=40)

    assert chain.shape == (40, 3)
    assert jnp.all(jnp.isfinite(chain))
    assert jnp.allclose(final, chain[-1])
    assert float(model.energy(final)) < float(model.energy(init))
    assert float(model.energy(final)) < 1e-3


def test_req_sample_1727_zero_step_sample_returns_initial_state() -> None:
    model = SoftConstraintBowl(target=jnp.zeros(2))
    init = jnp.array([1.0, -1.0])
    sampler = EquilibriumMatchingSampler()

    chain = sampler.sample_chain(model, init, n_steps=0)
    final = sampler.sample(model, init, n_steps=0)

    assert chain.shape == (0, 2)
    assert jnp.allclose(final, init)


def test_req_sample_1728_metric_helper_finds_first_threshold_step() -> None:
    assert exp.first_step_at_or_below([3.0, 1.0, 0.1], threshold=0.5) == 2
    assert exp.first_step_at_or_below([3.0, 1.0], threshold=0.5) is None


def test_scenario_sample_1728_experiment_writes_terminal_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "experiment_1727_eqm.json"
    artifact = exp.run_experiment(output_path=output_path)

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == artifact

    assert artifact["experiment_id"] == "1727"
    assert artifact["spec_refs"] == ["REQ-SAMPLE-1727", "REQ-SAMPLE-1728", "SCENARIO-SAMPLE-1728"]
    assert artifact["problem"]["dimension"] == 16
    assert artifact["metrics"]["eqm"]["converged"] is True
    assert artifact["metrics"]["eqm"]["finite_chain"] is True
    assert artifact["metrics"]["langevin"]["finite_chain"] is True
    assert artifact["metrics"]["eqm"]["final_energy"] < artifact["metrics"]["eqm"]["initial_energy"]
    assert artifact["eqm_faster_than_langevin"] is True
    assert artifact["honest_verdict"] == "eqm_converged_faster"
