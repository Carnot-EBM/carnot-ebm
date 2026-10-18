import json
import os
import jax.numpy as jnp
from carnot.pipeline.continuous_self_learner import ContinuousSelfLearner

def main():
    learner = ContinuousSelfLearner("unsloth/Qwen3.6-35B-A3B-GGUF")
    
    # 5 unlabelled scenarios
    scenarios = [
        jnp.array([0.5, 0.5, 0.5]),
        jnp.array([-0.5, -0.5, -0.5]),
        jnp.array([0.1, 0.2, 0.3]),
        jnp.array([2.0, 2.0, 2.0]),
        jnp.array([0.0, 1.0, 0.0])
    ]
    
    deltas = learner.process_scenarios(scenarios)
    
    deliverable = {
        "experiment_id": "2062",
        "task": "Unsupervised Continuous Self-Learning",
        "model": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "energy_deltas": deltas,
        "final_constraints": learner.constraints.tolist(),
        "status": "success",
        "note": "Processed 5 unlabelled scenarios and updated EBM constraints via continuous self-learning."
    }
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/experiment_2062_unsupervised_learning.json"
    with open(out_path, "w") as f:
        json.dump(deliverable, f, indent=2)
        
    print(f"Successfully wrote deliverable to {out_path}")

if __name__ == "__main__":
    main()
