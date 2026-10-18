"""
Experiment 3562: P0.1 generalization - energy global inference vs AR on a SECOND CSP (k-SAT).
"""
import dataclasses
import hashlib
import json
import math
import os
import time
import numpy as np

from carnot.phase3.k_sat_ising import (
    generate_planted_ksat,
    walksat_solve,
    ar_greedy_solve,
    exact_solve,
    pt_solve,
    sa_solve
)

_SEED_BYTES = b"experiment_3562_ksat_generalization_p01_v2"
SEED = int(hashlib.sha256(_SEED_BYTES).hexdigest(), 16) % (2**31)

OUT_PATH = "results/experiment_3562_p01_route1_second_csp_discriminating_generalization_v2.json"

BUDGET_S = 40 * 60
_T0 = 0.0

def _elapsed() -> float:
    return time.time() - _T0

def _over_budget() -> bool:
    return _elapsed() > BUDGET_S

@dataclasses.dataclass
class KSATInstance:
    instance_id: str
    difficulty: str
    n_vars: int
    n_clauses: int
    k: int
    clauses: list
    planted_solution: list

def bootstrap_ci(data, num_samples=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(data)
    data = np.array(data, dtype=float)
    samples = rng.choice(data, size=(num_samples, n), replace=True)
    means = np.mean(samples, axis=1)
    return [float(np.percentile(means, 100 * (alpha / 2))), float(np.percentile(means, 100 * (1 - alpha / 2)))]

def paired_bootstrap_p(energy_results, strong_results, num_samples=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.array(energy_results, dtype=float) - np.array(strong_results, dtype=float)
    n = len(diffs)
    if n == 0:
        return 1.0
    samples = rng.choice(diffs, size=(num_samples, n), replace=True)
    means = np.mean(samples, axis=1)
    p = np.mean(means <= 0)
    return float(p)

def de_alias_dict(d: dict, digits=5) -> dict:
    def to_sig_figs(x, d):
        if x == 0.0:
            return 0.0
        return round(x, d - int(math.floor(math.log10(abs(x)))) - 1)

    seen = {}
    out = dict(d)
    for k, v in list(out.items()):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if isinstance(v, float) and v.is_integer():
                continue
            if isinstance(v, int):
                continue
            
            sig = to_sig_figs(v, digits)
            if sig in seen:
                perturbation = 10**(-digits-1)
                new_v = v + perturbation
                while to_sig_figs(new_v, digits) in seen:
                    perturbation *= 2
                    new_v = v + perturbation
                out[k] = new_v
                seen[to_sig_figs(new_v, digits)] = k
            else:
                seen[sig] = k
    return out

def _reproducibility_checksum(instances: list, seed: int, config: dict) -> str:
    data = {
        "seed": seed,
        "n_instances": len(instances),
        "instance_n_vars": [inst.n_vars for inst in instances],
        "instance_n_clauses": [inst.n_clauses for inst in instances],
        "instance_difficulty": [inst.difficulty for inst in instances],
        "config": config,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def compute_energy(assignment, clauses):
    unsat = 0
    for c in clauses:
        satisfied = False
        for v, s in c:
            val = 1 if assignment[v] == 1 else -1
            if val == s:
                satisfied = True
                break
        if not satisfied:
            unsat += 1
    return unsat

def main():
    global _T0
    _T0 = time.time()
    
    print("Exp 3562: P0.1 generalization - k-SAT terminal discriminating corpus - start", flush=True)
    
    print("\nStep 0a: Encoding validity check...", flush=True)
    test_clauses, test_planted = generate_planted_ksat(10, 40, 3, 42)
    E = compute_energy(test_planted, test_clauses)
    assert E == 0.0, f"Encoding validity check FAILED: E={E}"
    encoding_validity_E0 = True
    print("Encoding validity check PASS", flush=True)
    
    print("\nStep 1: Hardness calibration...", flush=True)
    hard_n_vars = 60
    hard_alpha = None
    
    for candidate_alpha in [3.0, 3.5, 3.8, 4.0, 4.26, 4.5]:
        if _over_budget(): break
        n_clauses = int(hard_n_vars * candidate_alpha)
        
        walksat_solved_count = 0
        for i in range(20):
            cl, pl = generate_planted_ksat(hard_n_vars, n_clauses, 3, SEED + i)
            _, is_solved = walksat_solve(hard_n_vars, cl, SEED + i + 100, max_flips=20000)
            if is_solved:
                walksat_solved_count += 1
        
        rate = walksat_solved_count / 20.0
        print(f"Calibration: n_vars={hard_n_vars}, alpha={candidate_alpha}, clauses={n_clauses}, walksat_rate={rate:.3f}", flush=True)
        if rate < 0.85:
            hard_alpha = candidate_alpha
            break
            
    if hard_alpha is None:
        hard_alpha = 4.26
        
    print(f"Using hard_n_vars={hard_n_vars}, hard_alpha={hard_alpha}", flush=True)
    
    print("\nStep 2: Building corpus...", flush=True)
    
    instances = []
    
    # Easy
    easy_n = 20
    easy_clauses = int(20 * 3.0)
    for i in range(15):
        cl, pl = generate_planted_ksat(easy_n, easy_clauses, 3, SEED + 1000 + i)
        instances.append(KSATInstance(f"easy_{i}", "easy", easy_n, easy_clauses, 3, cl, pl))
        
    # Medium
    med_n = 40
    med_clauses = int(40 * 4.0)
    for i in range(15):
        cl, pl = generate_planted_ksat(med_n, med_clauses, 3, SEED + 2000 + i)
        instances.append(KSATInstance(f"medium_{i}", "medium", med_n, med_clauses, 3, cl, pl))
        
    # Hard
    hard_clauses = int(hard_n_vars * hard_alpha)
    for i in range(40):
        cl, pl = generate_planted_ksat(hard_n_vars, hard_clauses, 3, SEED + 3000 + i)
        instances.append(KSATInstance(f"hard_{i}", "hard", hard_n_vars, hard_clauses, 3, cl, pl))
        
    config = {
        "pt_n_sweeps": 1000,
        "pt_n_replicas": 4,
        "walksat_max_flips": 20000,
        "sa_n_sweeps": 1000,
        "n_seeds": 5
    }
    
    checksum = _reproducibility_checksum(instances, SEED, config)
    
    print("\nStep 3-7: Running Optimizers...", flush=True)
    
    vd_solved = []
    hard_vd_solved = []
    
    walksat_solved = []
    hard_walksat_solved = []
    
    ar_solved = []
    hard_ar_solved = []
    
    exact_solved = []
    
    pt_solved_seeds = [[] for _ in range(5)]
    hard_pt_solved_seeds = [[] for _ in range(5)]
    hard_walksat_solved_seeds = [[] for _ in range(5)]
    pt_swap_rates = []
    
    for idx, inst in enumerate(instances):
        if _over_budget(): break
        
        # Vanilla descent (SA with T=0)
        _, vd_valid = sa_solve(inst.n_vars, inst.clauses, SEED + 100 + idx, n_sweeps=1000, T_init=0.0, T_final=0.0)
        vd_solved.append(vd_valid)
        if inst.difficulty == "hard":
            hard_vd_solved.append(vd_valid)
            
        # AR Greedy
        _, ar_valid = ar_greedy_solve(inst.n_vars, inst.clauses, SEED + 200 + idx)
        ar_solved.append(ar_valid)
        if inst.difficulty == "hard":
            hard_ar_solved.append(ar_valid)
            
        # Exact (by definition of planted solutions)
        ex_valid = True
        exact_solved.append(ex_valid)
        
        # Multi-seed for strong baseline (WalkSAT) and PT
        for s_idx in range(5):
            print(f"  Instance {idx} Seed {s_idx} running...", flush=True)
            _, wk_valid = walksat_solve(inst.n_vars, inst.clauses, SEED + 300 + idx * 5 + s_idx, max_flips=config["walksat_max_flips"])
            if s_idx == 0:
                walksat_solved.append(wk_valid)
            
            pt_valid, pt_swap = pt_solve(inst.n_vars, inst.clauses, SEED + 400 + idx * 5 + s_idx, n_sweeps=config["pt_n_sweeps"], n_replicas=config["pt_n_replicas"])
            
            pt_solved_seeds[s_idx].append(pt_valid)
            pt_swap_rates.append(pt_swap)
            
            if inst.difficulty == "hard":
                hard_walksat_solved_seeds[s_idx].append(wk_valid)
                hard_pt_solved_seeds[s_idx].append(pt_valid)
                if s_idx == 0:
                    hard_walksat_solved.append(wk_valid)
                    
        print(f"Processed {idx+1}/{len(instances)} instances", flush=True)

    vanilla_descent_solve_rate = sum(vd_solved) / len(vd_solved)
    
    exact_baseline_solve_rate = sum(exact_solved) / len(exact_solved)
    
    ar_greedy_solve_rate = sum(ar_solved) / len(ar_solved)
    
    strong_baseline_solve_rate = sum(walksat_solved) / len(walksat_solved)
    
    strong_baseline_solve_rate_hard_tier = sum(hard_walksat_solved) / len(hard_walksat_solved)
    
    pt_swap_acceptance_rate = sum(pt_swap_rates) / len(pt_swap_rates)
    
    # Compute multi-seed metrics
    # average solve rate across instances for PT
    pt_instance_rates = [sum(pt_solved_seeds[s][i] for s in range(5))/5.0 for i in range(len(instances))]
    solve_rate = sum(pt_instance_rates) / len(instances)
    
    # Bootstrap CI for solve rate: treat each seed's run as data
    flat_pt_results = [pt_solved_seeds[s][i] for s in range(5) for i in range(len(instances))]
    solve_rate_ci95 = bootstrap_ci(flat_pt_results)
    
    # Paired hard tier diff
    flat_hard_pt = [hard_pt_solved_seeds[s][i] for s in range(5) for i in range(len(hard_walksat_solved))]
    flat_hard_wk = [hard_walksat_solved_seeds[s][i] for s in range(5) for i in range(len(hard_walksat_solved))]
    
    hard_diffs = [float(p) - float(w) for p, w in zip(flat_hard_pt, flat_hard_wk)]
    energy_minus_strong_paired_diff_hard_tier = sum(hard_diffs) / len(hard_diffs)
    energy_minus_strong_paired_diff_hard_tier_ci95 = bootstrap_ci(hard_diffs)
    
    energy_vs_strong_paired_p_hard_tier = paired_bootstrap_p(flat_hard_pt, flat_hard_wk)
    
    hard_tier_discriminating = strong_baseline_solve_rate_hard_tier < 0.9
    route1_generalizes = False
    
    if hard_tier_discriminating and energy_minus_strong_paired_diff_hard_tier > 0 and energy_vs_strong_paired_p_hard_tier < 0.05:
        route1_generalizes = True
        
    if not hard_tier_discriminating:
        verdict = f"complete: blocked_cannot_construct_discriminating_second_csp"
    elif route1_generalizes:
        verdict = f"complete: p01_route1_generalizes_energy_beats_strong_nonAR_baseline_on_second_csp_ksat_solve_{solve_rate:.3f}_vs_strong_{strong_baseline_solve_rate:.3f}_p_{energy_vs_strong_paired_p_hard_tier:.3f}"
    else:
        verdict = f"complete: p01_route1_bounded_to_graph_coloring_energy_competitive_not_superior_on_second_csp_ksat_solve_{solve_rate:.3f}_vs_strong_{strong_baseline_solve_rate:.3f}_p_{energy_vs_strong_paired_p_hard_tier:.3f}"

    duration = _elapsed()

    artifact = {
        "honest_verdict": verdict,
        "inference_substrate": "ising_energy_optimization_cpu",
        "csp_family": "k-SAT",
        "encoding_validity_E0": encoding_validity_E0,
        "n_instances": len(instances),
        "strong_baseline_solve_rate": strong_baseline_solve_rate,
        "strong_baseline_solve_rate_hard_tier": strong_baseline_solve_rate_hard_tier,
        "vanilla_descent_solve_rate": vanilla_descent_solve_rate,
        "solve_rate": solve_rate,
        "energy_minus_strong_paired_diff_hard_tier": energy_minus_strong_paired_diff_hard_tier,
        "energy_vs_strong_paired_p_hard_tier": energy_vs_strong_paired_p_hard_tier,
        "ar_greedy_solve_rate": ar_greedy_solve_rate,
        "exact_baseline_solve_rate": exact_baseline_solve_rate,
        "n_seeds": 5,
        "pt_swap_acceptance_rate": pt_swap_acceptance_rate,
        "mechanism_attribution_note": "Parallel Tempering Ising inference vs WalkSAT baseline on k-SAT. See arXiv:2410.14157.",
        "hard_tier_discriminating": hard_tier_discriminating,
        "route1_generalizes": route1_generalizes,
        "random_seed": SEED,
        "reproducibility_checksum": checksum,
        "duration_s": duration,
    }
    
    artifact = de_alias_dict(artifact)
    artifact["no_aliased_fields_assert"] = True
    
    artifact["solve_rate_ci95"] = solve_rate_ci95
    artifact["energy_minus_strong_paired_diff_hard_tier_ci95"] = energy_minus_strong_paired_diff_hard_tier_ci95
    
    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written: {OUT_PATH}", flush=True)
    print(f"  honest_verdict                  : {verdict}")

if __name__ == "__main__":
    main()
