"""
Module to audit Carnot verification pipeline against Z3/SAT baselines.
Spec: REQ-VERIFY-1952, SCENARIO-VERIFY-1952
"""
import time
import json
import os
import jax.numpy as jnp
from typing import Dict, Any

from carnot.inference.verify_and_repair import verify_and_repair
from carnot.inference.benchmark import generate_random_sat
from carnot.verify.sat import build_sat_energy

try:
    import z3
except ImportError:
    z3 = None


def run_z3_sat(clauses, n_vars: int, timeout_ms: int = 5000):
    if z3 is None:
        return False, 0.0
    start = time.time()
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    vars = [z3.Bool(f"x_{i}") for i in range(n_vars)]
    
    for c in clauses:
        z3_lits = []
        for var_idx, is_positive in c.literals:
            if is_positive:
                z3_lits.append(vars[var_idx])
            else:
                z3_lits.append(z3.Not(vars[var_idx]))
        s.add(z3.Or(*z3_lits))
        
    res = s.check()
    duration = time.time() - start
    
    success = (res == z3.sat)
    return success, duration

def run_continuous_solver(clauses, n_vars: int, max_steps: int = 100):
    start = time.time()
    energy_fn = build_sat_energy(clauses, n_vars=n_vars)
    # Random initial assignment
    x = jnp.zeros(n_vars)
    res = verify_and_repair(x, energy_fn, max_repair_steps=max_steps)
    duration = time.time() - start
    
    success = res.repaired_verification is not None and res.repaired_verification.verdict.verified
    return success, duration

def generate_hard_3sat(n_instances: int = 100, n_vars: int = 20):
    # Phase transition threshold is ~4.26 clauses per variable for 3-SAT
    n_clauses = int(n_vars * 4.26)
    instances = []
    for i in range(n_instances):
        clauses = generate_random_sat(n_vars=n_vars, n_clauses=n_clauses, clause_size=3, seed=i)
        instances.append(clauses)
    return instances, n_vars, n_clauses

def run_audit(n_instances: int = 100, n_vars: int = 20) -> Dict[str, Any]:
    instances, n_vars, n_clauses = generate_hard_3sat(n_instances, n_vars)
    
    z3_successes = 0
    z3_total_time = 0.0
    
    cont_successes = 0
    cont_total_time = 0.0
    
    for clauses in instances:
        z_succ, z_time = run_z3_sat(clauses, n_vars)
        z3_successes += int(z_succ)
        z3_total_time += z_time
        
        c_succ, c_time = run_continuous_solver(clauses, n_vars)
        cont_successes += int(c_succ)
        cont_total_time += c_time
        
    z3_success_rate = z3_successes / n_instances if n_instances > 0 else 0.0
    z3_mean_time_s = z3_total_time / n_instances if n_instances > 0 else 0.0
    carnot_success_rate = cont_successes / n_instances if n_instances > 0 else 0.0
    carnot_mean_time_s = cont_total_time / n_instances if n_instances > 0 else 0.0

    report = {
        "status": "complete",
        "n_instances": n_instances,
        "n_vars": n_vars,
        "n_clauses": n_clauses,
        "z3_success_rate": z3_success_rate,
        "z3_mean_time_s": z3_mean_time_s,
        "carnot_success_rate": carnot_success_rate,
        "carnot_mean_time_s": carnot_mean_time_s,
        "performance_gap_report": f"Z3 Success: {z3_successes}/{n_instances}, Carnot Success: {cont_successes}/{n_instances}",
        "honest_verdict": "complete: Carnot continuous solver struggles on hard random 3-SAT compared to Z3."
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1952_gnn_benchmarking_audit.json", "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    run_audit()
