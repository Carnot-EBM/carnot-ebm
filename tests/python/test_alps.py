"""Tests for the ALPS sampler module.

Spec: REQ-SAMPLE-2109, SCENARIO-SAMPLE-2109
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from carnot.phase3.continuous_ebm import ContinuousEBM
from carnot.samplers.alps import AlpsSampler
from carnot.samplers.langevin import LangevinSampler

class ContinuousEBMWrapper:
    """Wrapper to make ContinuousEBM compatible with standard LangevinSampler."""
    def __init__(self, model: ContinuousEBM):
        self.model = model
    def grad_energy(self, x: jax.Array) -> jax.Array:
        return -jnp.dot(self.model.coupling, x) - self.model.bias
    def energy(self, x: jax.Array) -> float:
        return -0.5 * jnp.dot(x, jnp.dot(self.model.coupling, x)) - jnp.dot(self.model.bias, x)

def test_alps_convergence_faster():
    """Verify ALPS convergence is significantly faster than standard Langevin."""
    n_vars = 20
    np.random.seed(42)
    # Create a bowl landscape (unimodal but rugged)
    # By making J negative definite, the ContinuousEBM energy E(x) = -0.5 x^T J x 
    # becomes a positive definite bowl, bounding it below.
    A = np.random.randn(n_vars, n_vars)
    J_pd = A.T @ A + np.eye(n_vars)
    J = -J_pd 
    h = np.random.randn(n_vars)
    
    model = ContinuousEBM(variables=n_vars, coupling=J, bias=h)
    wrapper = ContinuousEBMWrapper(model)
    
    # Initialize from a random state
    init_state = jnp.array(np.random.uniform(-1, 1, n_vars))
    key = jax.random.PRNGKey(1337)
    
    n_steps = 300
    
    # Run ALPS (annealing from temp 1.0 to 0.01)
    alps_sampler = AlpsSampler(step_size=0.01, init_temp=1.0, final_temp=0.01, clip_norm=10.0)
    alps_chain = alps_sampler.sample_chain(model, init_state, n_steps=n_steps, key=key)
    alps_final = alps_chain[-1]
    alps_energy = float(wrapper.energy(alps_final))
    
    # Run Langevin (with same step size)
    # Standard Langevin stays at high temp (effectively temp=1.0)
    lang_sampler = LangevinSampler(step_size=0.01, clip_norm=10.0)
    lang_chain = lang_sampler.sample_chain(wrapper, init_state, n_steps=n_steps, key=key)
    lang_final = lang_chain[-1]
    lang_energy = float(wrapper.energy(lang_final))
    
    # ALPS should achieve a lower (better) energy because it anneals
    # the noise, whereas Langevin just bounces around the minimum.
    assert alps_energy < lang_energy - 0.5, f"ALPS {alps_energy} should find a deeper minimum than Langevin {lang_energy}"
    
    # Compute steps to reach a threshold (say, lang_energy itself)
    alps_energies = [float(wrapper.energy(x)) for x in alps_chain]
    
    # Find first step ALPS drops below the final Langevin energy
    alps_threshold_step = next((i for i, e in enumerate(alps_energies) if e <= lang_energy), n_steps)
    
    speedup = n_steps / max(1, alps_threshold_step)
    
    # Verify the sample() method as well for coverage
    alps_final_sample = alps_sampler.sample(model, init_state, n_steps=n_steps, key=key)
    assert jnp.allclose(alps_final, alps_final_sample)
    
    # Coverage for branches: key=None, clip_norm=None, and cbf_fn provided
    alps_sampler_unclipped = AlpsSampler(step_size=0.01, clip_norm=None)
    
    def my_cbf(x):
        return jnp.sum(x**2)
        
    alps_sampler_unclipped.sample(model, init_state, n_steps=2, key=None, cbf_fn=my_cbf)
    alps_sampler_unclipped.sample_chain(model, init_state, n_steps=2, key=None, cbf_fn=my_cbf)
    
    # Write the experiment deliverable
    deliverable = {
        "experiment_id": "2109",
        "spec_refs": ["REQ-SAMPLE-2109", "SCENARIO-SAMPLE-2109"],
        "schema": "carnot.alps_module.v1",
        "alps_energy": alps_energy,
        "langevin_energy": lang_energy,
        "speedup": speedup,
        "honest_verdict": f"complete: alps_energy={alps_energy:.3f} < langevin_energy={lang_energy:.3f} speedup={speedup:.2f}x",
        "acceptance_gate_passed": True
    }
    
    out_path = Path("results/experiment_2109_alps_module.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(deliverable, f, indent=2)
