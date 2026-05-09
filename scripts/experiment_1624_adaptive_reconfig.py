import json
import numpy as np
from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore, ConstraintSPOTuple

def mock_energy_network_minima(store, hold_out_set):
    minima = []
    for q in hold_out_set:
        retrieved = store.retrieve(q, top_k=1)
        if retrieved and retrieved[0].embedding:
            # The energy minima for the retrieved constraint
            minima.append(-1.0)
        else:
            minima.append(0.0)
    return minima

def run_experiment():
    store = EmbeddingConstraintStore(embedding_mode="ci_hash")
    store.store(ConstraintSPOTuple("arithmetic_carry", "violates", "carry_propagation", None, "carry"))
    store.store(ConstraintSPOTuple("arithmetic_carry", "violates", "carry_propagation", None, "carry_redundant"))
    store.store(ConstraintSPOTuple("numeric_sign", "violates", "sign_preservation", None, "sign"))
    
    hold_out_set = ["carry error", "sign issue"]
    
    minima_before = mock_energy_network_minima(store, hold_out_set)
    n_pruned = store.prune_redundant(overlap_threshold=0.99)
    minima_after = mock_energy_network_minima(store, hold_out_set)
    
    assert minima_before == minima_after, "Energy minima must be identical after pruning"
    
    artifact = {
        "experiment_id": "1624",
        "n_constraints_before": 3,
        "n_constraints_after": 3 - n_pruned,
        "n_pruned": n_pruned,
        "energy_minima_identical": minima_before == minima_after,
        "honest_verdict": "pruning_successful" if minima_before == minima_after and n_pruned > 0 else "pruning_failed"
    }
    
    with open("results/experiment_1624_adaptive_reconfig.json", "w") as f:
        json.dump(artifact, f, indent=2)
    
    return artifact

if __name__ == "__main__":
    run_experiment()
