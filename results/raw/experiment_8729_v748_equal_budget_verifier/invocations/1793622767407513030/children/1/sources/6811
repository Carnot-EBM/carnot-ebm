#!/usr/bin/env python3
"""Exp 1724: Continuous Online Updater Prototype.

Tests the online SGD/AdamW updater with a synthetic stream of errors.
Spec: REQ-LEARN-101
"""

import json
from pathlib import Path
from carnot.models.cikan_verifier import CIKAN
from carnot.training.online_updater import OnlineUpdater
from carnot.pipeline.verification_loop import VerificationLoop, Violation

def run_experiment():
    print("Setting up CIKAN and OnlineUpdater...")
    cikan = CIKAN(feature_names=["f1", "f2"], seed=42)
    updater = OnlineUpdater(optimizer="adamw", learning_rate=0.05)
    loop = VerificationLoop(cikan, updater)
    
    print("Generating synthetic stream...")
    # Synthetic stream: alternate violations and valid samples
    # For a synthetic stream, let's say:
    # 50 items. Valid samples have features around [0.1, 0.1]. 
    # Violations have features around [0.9, 0.9].
    stream = []
    for i in range(50):
        if i % 2 == 0:
            stream.append(Violation(features=[0.9, 0.9], label=0.0))
        else:
            stream.append(Violation(features=[0.1, 0.1], label=1.0))
            
    # Measure initial energies
    e_val_initial = cikan.energy([0.1, 0.1])
    e_viol_initial = cikan.energy([0.9, 0.9])
    
    print(f"Initial: Valid Energy={e_val_initial:.4f}, Violation Energy={e_viol_initial:.4f}")
    
    print("Running verification loop...")
    loop.run(stream)
    
    # Measure final energies
    e_val_final = cikan.energy([0.1, 0.1])
    e_viol_final = cikan.energy([0.9, 0.9])
    
    print(f"Final: Valid Energy={e_val_final:.4f}, Violation Energy={e_viol_final:.4f}")
    
    results = {
        "experiment_id": "1724",
        "n_processed": loop.n_processed,
        "n_updated": loop.n_updated,
        "metrics": {
            "initial_valid_energy": e_val_initial,
            "initial_violation_energy": e_viol_initial,
            "final_valid_energy": e_val_final,
            "final_violation_energy": e_viol_final,
            "energy_delta_valid": e_val_final - e_val_initial,
            "energy_delta_violation": e_viol_final - e_viol_initial
        },
        "success": bool(e_val_final < e_val_initial and e_viol_final > e_viol_initial)
    }
    
    output_path = Path("results/experiment_1724_online_updater.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    run_experiment()
