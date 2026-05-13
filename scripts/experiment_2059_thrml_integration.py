import json
import numpy as np
from pathlib import Path

from carnot.samplers.backend import get_sampler_backend

def main():
    sampler = get_sampler_backend("thrml_tsu")
    assert sampler.backend_name == "thrml_tsu"

    n_spins = 10
    biases = np.zeros(n_spins)
    couplings = np.zeros((n_spins, n_spins))

    # Test minimize_energy
    samples_min = sampler.minimize_energy(biases, couplings, n_samples=5, n_steps=100, beta=1.0)
    assert samples_min.shape == (5, n_spins)

    # Test sample
    samples_fixed = sampler.sample(biases, couplings, n_samples=3, config={"beta": 1.0})
    assert samples_fixed.shape == (3, n_spins)

    deliverable = {
        "experiment_id": 2059,
        "spec_refs": ["REQ-SAMPLE-2059", "SCENARIO-SAMPLE-2059"],
        "backend_name": sampler.backend_name,
        "minimize_energy_shape": list(samples_min.shape),
        "sample_shape": list(samples_fixed.shape),
        "verdict": "pass",
        "hardware_execution_claim": False
    }

    out_path = Path("results/experiment_2059_thrml_integration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deliverable, indent=2))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
