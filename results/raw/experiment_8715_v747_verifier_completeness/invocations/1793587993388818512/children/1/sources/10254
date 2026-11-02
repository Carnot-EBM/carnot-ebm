import json
import time
import os
import jax.numpy as jnp
from carnot.solvers.hybrid_verifier import HybridVerifier

def run_experiment():
    A = jnp.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    b = jnp.array([1.0, 1.0, 1.0])
    
    verifier = HybridVerifier(A, b)
    
    num_trials = 10
    passes = 0
    total_latency = 0.0
    
    for i in range(num_trials):
        is_verified, latency = verifier.run_pipeline(seed=i)
        if is_verified:
            passes += 1
        total_latency += latency
        
    pass_rate = passes / num_trials
    avg_latency = total_latency / num_trials
    
    results = {
        "experiment_id": 1673,
        "pass_rate": pass_rate,
        "validation_latency_sec": avg_latency,
        "honest_verdict": "verified" if pass_rate == 1.0 else "failed",
        "num_trials": num_trials
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1673_hybrid_constraint.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Experiment 1673 completed. Pass rate: {pass_rate*100}%, Avg latency: {avg_latency:.4f}s")

if __name__ == "__main__":
    run_experiment()
