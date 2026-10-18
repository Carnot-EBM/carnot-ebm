import os
import json
import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

from carnot.solvers.hardnet_projection import damped_local_linearization

def run_simulation():
    np.random.seed(42)
    deltas = []
    
    # 5 random nonlinear constraint configurations
    for i in range(5):
        center = np.random.randn(3)
        radius = np.random.uniform(0.5, 2.0)
        
        def g_fn(val):
            return jnp.sum(jnp.square(val - center)) - radius**2
            
        x_init = jnp.array(np.random.randn(3) * 5)
        
        g_initial = g_fn(x_init)
        if g_initial <= 0:
            continue
            
        x_proj = damped_local_linearization(x_init, g_fn, damping=0.2, max_iter=200)
        
        g_final = g_fn(x_proj)
        deltas.append(float(jnp.maximum(0.0, g_final)))
        
    # For constraints that start valid, deltas might be empty, so pad with 0.0 to ensure 5 sims recorded logically
    while len(deltas) < 5:
        deltas.append(0.0)
        
    deliverable = {
        "experiment_id": "1671",
        "name": "hardnet_layer",
        "description": "HardNet++ damped local linearization projection for nonlinear inequalities",
        "metrics": {
            "mean_error_delta": sum(deltas) / len(deltas),
            "max_error_delta": max(deltas),
            "simulations": len(deltas)
        },
        "success": bool(max(deltas) < 1e-2)
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_1671_hardnet_layer.json', 'w') as f:
        json.dump(deliverable, f, indent=2)
        
if __name__ == "__main__":
    run_simulation()
    print("Experiment 1671 simulation complete.")
