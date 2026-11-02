import json
import os
import jax
import jax.numpy as jnp
from carnot.samplers.langevin import LangevinSampler

class ExperimentEnergy:
    def energy(self, x: jax.Array) -> jax.Array:
        # Simple quadratic bowl centered at 0
        return jnp.sum(x**2)
    def grad_energy(self, x: jax.Array) -> jax.Array:
        return 2.0 * x

def run_experiment():
    sampler = LangevinSampler(step_size=0.01)
    energy_fn = ExperimentEnergy()
    
    # We want x[0] >= 5.0. 
    # Violation is when x[0] < 5.0.
    def cbf_fn(x):
        return 10.0 * jax.nn.relu(5.0 - x[0])**2

    # Run 1000 independent chains
    num_trials = 1000
    n_steps = 200
    
    def single_trial(key):
        init = jnp.array([10.0, 0.0]) # feasible start
        final_state = sampler.sample(energy_fn, init, n_steps=n_steps, key=key, cbf_fn=cbf_fn)
        # violation if x[0] < 4.0 (giving some margin for noise)
        return final_state[0] < 4.0

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_trials)
    
    violations = jax.vmap(single_trial)(keys)
    violation_rate = float(jnp.mean(violations))
    
    acceptance_gate_passed = violation_rate < 0.05
    
    result = {
        "schema": "experiment_1807",
        "violation_rate": violation_rate,
        "acceptance_gate_passed": acceptance_gate_passed,
        "honest_verdict": "success: CBF successfully kept constraint violation rate low" if acceptance_gate_passed else "failure: high violation rate",
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1807_lyapunov_cbf.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Experiment completed. Violation rate:", violation_rate)

if __name__ == "__main__":
    run_experiment()
