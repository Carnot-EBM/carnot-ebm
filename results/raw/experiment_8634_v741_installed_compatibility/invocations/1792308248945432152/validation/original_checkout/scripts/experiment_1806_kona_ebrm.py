import json
import os
import jax
import jax.numpy as jnp
from carnot.models.kona_ebrm import KonaEBRM

def main():
    """Spec: REQ-KONA-040, SCENARIO-KONA-040"""
    model = KonaEBRM(trace_length=10, dim=4)
    key = jax.random.PRNGKey(0)
    init_trace = jax.random.normal(key, (10, 4))
    target = jnp.array([1.0, -1.0, 0.5, -0.5])
    
    initial_energy = float(model.energy(init_trace, target))
    
    refined_trace = model.refine_trace(init_trace, target, steps=200, lr=0.05)
    
    final_energy = float(model.energy(refined_trace, target))
    
    results = {
        "experiment_id": 1806,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "trace_length": 10,
        "dim": 4,
        "honest_verdict": "continuous_improved" if final_energy < initial_energy else "no_improvement",
        "spec_refs": ["REQ-KONA-040", "SCENARIO-KONA-040"]
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1806_kona_ebrm.json", "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    main()
