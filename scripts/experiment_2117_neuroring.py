import json
import jax
import jax.numpy as jnp
from pathlib import Path

from carnot.samplers.parallel_ising import NeuroRingIsingSampler
from carnot.verify.sudoku import build_sudoku_energy

def main():
    # Instantiate Sudoku empty puzzle
    # Standard 9x9 sudoku has 81 cells, 9 values per cell = 729 variables
    # Let's instantiate a simple empty grid or a sample one
    energy = build_sudoku_energy([[0]*9 for _ in range(9)])
    
    # Normally we extract coupling matrix. For hardware accounting we just need n_spins.
    n_spins = 729
    total_sweeps = 1000
    
    sampler = NeuroRingIsingSampler(ring_size=4)
    metrics = sampler.hardware_accounting(n_spins=n_spins, total_sweeps=total_sweeps)
    
    # We could also actually sample to prove it works
    key = jax.random.PRNGKey(42)
    biases = jnp.zeros((n_spins,), dtype=jnp.float32)
    coupling_matrix = jnp.zeros((n_spins, n_spins), dtype=jnp.float32)
    
    # Short test run
    sampler_test = NeuroRingIsingSampler(ring_size=4, n_warmup=10, n_samples=2, steps_per_sample=2)
    samples = sampler_test.sample(key, biases, coupling_matrix, beta=1.0)
    
    results = {
        "metrics": metrics,
        "test_sample_shape": list(samples.shape)
    }
    
    out_path = Path("results/experiment_2117_neuroring.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
