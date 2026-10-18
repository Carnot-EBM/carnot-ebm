#!/usr/bin/env python3
"""Experiment 3372: FR-11 CX Repair loop using Z3 Unsat Cores scaled to 100 cases."""

import json
from pathlib import Path
from carnot.pipeline.fr11_cx_repair import Z3ConstraintRepairAgent

def generate_synthetic_flawed_templates() -> list[dict]:
    """Generate 100 synthetic flawed templates."""
    templates = []
    for i in range(100):
        # A flawed template constraint set that is guaranteed to be UNSAT.
        # Core 1: x > i and x < i - 1
        templates.append({
            "variables": ["x", "y"],
            "constraints": [
                (f"c1_{i}", f"x > {i}"),
                (f"c2_{i}", f"x < {i - 1}"),
                (f"c3_{i}", f"y == {i}")
            ]
        })
    return templates

def main() -> float:
    """Run the repair experiment on 100 flawed templates."""
    agent = Z3ConstraintRepairAgent()
    templates = generate_synthetic_flawed_templates()
    
    successes = 0
    total = len(templates)
    
    for tmpl in templates:
        success, _ = agent.repair_template(tmpl["variables"], tmpl["constraints"])
        if success:
            successes += 1
            
    repair_success_rate = successes / total if total > 0 else 0.0
    
    results = {
        "experiment": "3372_fr11_cx_repair_scale",
        "repair_success_rate": repair_success_rate,
        "n_cases": total,
        "honest_verdict": "success" if repair_success_rate == 1.0 else "failure"
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    with open(results_dir / "experiment_3372_fr11_cx_repair_scale.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Repair success rate: {repair_success_rate:.2f}")
    return repair_success_rate

if __name__ == "__main__":
    main()
