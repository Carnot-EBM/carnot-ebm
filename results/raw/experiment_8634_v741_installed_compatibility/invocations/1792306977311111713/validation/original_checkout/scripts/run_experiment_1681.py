#!/usr/bin/env python3
"""Run experiment 1681: ETS for policy evaluation in the FR-11 loop."""

import os
import json
from carnot.memory.ets_policy import EtsPolicyEvaluator

def main():
    print("Initializing ETS Policy Evaluator...")
    evaluator = EtsPolicyEvaluator(base_compute=100.0)
    
    candidates = [
        {
            "candidate": {"id": "policy_alpha"},
            "probs": [0.9, 0.85, 0.95],
            "uncertainty": 0.1
        },
        {
            "candidate": {"id": "policy_beta"},
            "probs": [0.4, 0.5, 0.45],
            "uncertainty": 0.05
        },
        {
            "candidate": {"id": "policy_gamma"},
            "probs": [0.7, 0.6, 0.8],
            "uncertainty": 0.8  # High uncertainty
        }
    ]
    
    results = []
    for c in candidates:
        decision = evaluator.promote_policy(c["candidate"], c["probs"], c["uncertainty"])
        results.append(decision)
        
    output_path = "results/experiment_1681_ets_policy.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    deliverable = {
        "experiment": "1681",
        "description": "ETS (Energy-Term Transition Probabilities) replacing RLHF in FR-11",
        "total_policies_evaluated": len(candidates),
        "promoted_policies": sum(1 for r in results if r["is_promoted"]),
        "evaluations": results
    }
    
    with open(output_path, "w") as f:
        json.dump(deliverable, f, indent=2)
        
    print(f"Successfully evaluated policies with ETS and wrote output to {output_path}")

if __name__ == "__main__":
    main()
