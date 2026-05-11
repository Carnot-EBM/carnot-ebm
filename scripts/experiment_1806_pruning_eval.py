import json
import jax.numpy as jnp
import jax.random as jrandom
from carnot.training.replay_buffer import ReplayBuffer

def run_evaluation():
    # Evaluate pruning on a simulated 2000-trace verified buffer
    # We generate 1000 unique traces and 1000 noisy duplicates (semantically similar)
    key = jrandom.PRNGKey(42)
    key, subkey = jrandom.split(key)
    
    # 1000 unique semantic embeddings
    unique_traces = jrandom.normal(subkey, (1000, 128))
    
    # 1000 semantically redundant traces (noisy duplicates)
    key, subkey = jrandom.split(key)
    noise = jrandom.normal(subkey, (1000, 128)) * 0.05
    noisy_duplicates = unique_traces + noise
    
    all_traces = jnp.concatenate([unique_traces, noisy_duplicates], axis=0)
    
    # Shuffle the traces to simulate real-world arrival order
    key, subkey = jrandom.split(key)
    all_traces = jrandom.permutation(subkey, all_traces)
    
    # Initialize ReplayBuffer with Semantic Pruning active
    buffer = ReplayBuffer(max_size=5000, similarity_threshold=0.9)
    
    # Add states. Redundant states will be pruned out.
    buffer.add(all_traces)
    
    # Calculate metrics
    retained_ratio = len(buffer) / 2000.0
    
    results = {
        "retained_ratio": retained_ratio,
        "total_added": 2000,
        "total_retained": len(buffer),
        "threshold": 0.9,
        "description": "2000-trace verified buffer evaluation"
    }
    
    out_file = "/home/ianblenke/github.com/ianblenke/carnot/results/experiment_1806_pruning.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {out_file}")

if __name__ == "__main__":
    run_evaluation()
