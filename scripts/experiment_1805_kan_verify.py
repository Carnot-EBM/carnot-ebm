#!/usr/bin/env python3
"""Experiment 1805 — KAN MILP Formal Verification Smoke Test

Runs an end-to-end formal verification smoke test of the KAEM model on a 10-constraint task.
"""
from __future__ import annotations

import os
import sys
import json
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.random as jrandom
from carnot.models.kaem_energy import KAEMEnergy
from scripts.experiment_972_kan_milp_formal_verification import verify_monotonicity_milp, verify_output_range

RESULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "experiment_1805_smoke.json"
)

def run_smoke_test() -> dict:
    """Run a 10-constraint formal verification smoke test on KAEM."""
    n_vars = 10
    key = jrandom.PRNGKey(42)
    model = KAEMEnergy(n_vars=n_vars, n_hidden=8, key=key)
    layer = model.layer
    
    t_start = time.perf_counter()
    
    milp_results = []
    constraints_verified = []
    # Test 10 formal safety constraints (monotonicity on each of 10 variables)
    for var_idx in range(n_vars):
        res = verify_monotonicity_milp(layer, var_idx)
        milp_results.append(res)
        if res.get("verified", False):
            constraints_verified.append(f"milp_monotonicity_var_{var_idx}")
            
    range_res = verify_output_range(layer, n_spins=n_vars)
    
    bounds_soundness = bool(range_res.get("verified", False) and all(r.get("verified", False) for r in milp_results))
    
    t_end = time.perf_counter()
    verification_time_ms = int((t_end - t_start) * 1000)
    
    result = {
        "experiment_id": "exp1805",
        "verification_time_ms": verification_time_ms,
        "bounds_soundness": bounds_soundness,
        "constraints_verified": constraints_verified,
        "status": "complete",
        "honest_verdict": "success" if bounds_soundness else "violations_found"
    }
    return result

def main():
    result = run_smoke_test()
    out_path = os.path.abspath(RESULT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Smoke test complete in {result['verification_time_ms']}ms.")
    print(f"Bounds soundness: {result['bounds_soundness']}")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
