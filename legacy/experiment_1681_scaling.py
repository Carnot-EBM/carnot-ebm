import json
import os
from carnot.phase4_scaling import run_scaling_experiment

def main():
    n_values = [8, 16, 32]
    mld_steps = 100
    n_samples = 100
    base_seed = 42
    
    # Run the experiment
    output = run_scaling_experiment(
        n_values=n_values,
        mld_steps=mld_steps,
        n_samples_per_n=n_samples,
        base_seed=base_seed
    )
    
    # Ensure results dir exists
    os.makedirs("results", exist_ok=True)
    out_path = "results/experiment_1681_phase4_scaling.json"
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Generated {out_path}")
    print(f"Collapse scale: {output['collapse_scale']}")
    print(f"Gate passed: {output['acceptance_gate_passed']}")

if __name__ == "__main__":
    main()
