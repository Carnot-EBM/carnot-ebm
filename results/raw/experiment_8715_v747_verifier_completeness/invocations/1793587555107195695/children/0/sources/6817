import json
import numpy as np
from carnot.pipeline.continual_memory import ContinualMemory

def run_experiment():
    memory = ContinualMemory()
    np.random.seed(42)
    
    for i in range(100):
        cluster_id = i % 5
        base_vector = np.zeros(10)
        base_vector[cluster_id * 2] = 5.0
        vec = base_vector + np.random.normal(0, 0.5, 10)
        memory.add_state(vec, {"id": i, "semantic_cluster": cluster_id})
        
    memory.distill(n_clusters=5)
    
    final_clusters = sorted(list({state["metadata"]["semantic_cluster"] for state in memory.memory_states}))
    
    result = {
        "experiment_id": "1754",
        "name": "semantic_distillation",
        "honest_verdict": "distillation_successful" if len(memory.memory_states) == 5 and len(final_clusters) == 5 else "distillation_failed",
        "n_states_before": 100,
        "n_states_after": len(memory.memory_states),
        "retained_clusters": final_clusters
    }
    
    with open("results/experiment_1754_distillation.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    run_experiment()
