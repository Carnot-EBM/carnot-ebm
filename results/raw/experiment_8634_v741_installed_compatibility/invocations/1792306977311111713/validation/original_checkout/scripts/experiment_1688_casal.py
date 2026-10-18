import jax
import jax.numpy as jnp
import time
import json
import os
from carnot.samplers.casal import casal_sample

def run_experiment():
    key = jax.random.PRNGKey(1688)
    
    def energy_fn(state):
        # simple quadratic bowl
        return jnp.sum(state**2)
        
    def constraint_fn(state):
        # constraint: sum(state) >= 1.0
        return jax.nn.relu(1.0 - jnp.sum(state))
        
    init_state = jnp.array([1.0, 1.0])
    
    # Warm up JIT
    _ = casal_sample(energy_fn, constraint_fn, init_state, 1, key, 0.1)
    
    # Run timed sample
    steps = 1000
    start_time = time.time()
    final_state = casal_sample(energy_fn, constraint_fn, init_state, steps, key, 0.05)
    final_state.block_until_ready()
    exec_time_ms = (time.time() - start_time) * 1000.0
    
    violation = constraint_fn(final_state)
    violation_rate = float(violation > 1e-5)
    
    artifact = {
        "schema": "carnot.experiment_1688_casal",
        "constraint_violation_rate": violation_rate,
        "execution_time_ms": exec_time_ms,
        "acceptance_gate_passed": bool(violation_rate == 0.0)
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1688_casal_sampler.json", "w") as f:
        json.dump(artifact, f, indent=2)
        
    print("Experiment 1688 completed successfully.")
    print(json.dumps(artifact, indent=2))

if __name__ == "__main__":
    run_experiment()
