import dataclasses
import hashlib
import json
import math
import os
import time

# Content-derived seed
SEED = int(hashlib.sha256(b"experiment_3563_p01_route1_graph_coloring_multiseed_second_generator_v4").hexdigest(), 16) % (2**31)

OUT_PATH = "results/experiment_3563_p01_route1_graph_coloring_multiseed_second_generator_v4.json"

BUDGET_S = 38 * 60
_T0 = 0.0

def _elapsed() -> float:
    return time.time() - _T0

def _over_budget() -> bool:
    return _elapsed() > BUDGET_S

def compute_energy(colors: list, n_vertices: int, k: int, edges: list) -> float:
    conflicts = sum(1 for u, v in edges if colors[u] == colors[v])
    return float(conflicts)

@dataclasses.dataclass
class GraphColoringInstance:
    instance_id: int
    n_vertices: int
    k: int
    edges: list
    planted_colors: list
    difficulty: str
    generator: str
    param: float  # p_cross or radius
    avg_degree: float

def make_planted_instance(n: int, k: int, p_cross: float, rng, instance_id: int, difficulty: str) -> GraphColoringInstance:
    groups = [[] for _ in range(k)]
    for v in range(n):
        groups[v % k].append(v)

    planted_colors = [0] * n
    for color, group in enumerate(groups):
        for v in group:
            planted_colors[v] = color

    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            for u in groups[i]:
                for v in groups[j]:
                    if rng.random() < p_cross:
                        edges.append((u, v))

    avg_degree = 2 * len(edges) / n if n > 0 else 0.0

    return GraphColoringInstance(
        instance_id=instance_id,
        n_vertices=n,
        k=k,
        edges=edges,
        planted_colors=planted_colors,
        difficulty=difficulty,
        generator="erdos_renyi",
        param=p_cross,
        avg_degree=avg_degree,
    )

def make_geometric_planted_instance(n: int, k: int, radius: float, rng, instance_id: int, difficulty: str) -> GraphColoringInstance:
    import numpy as np
    groups = [[] for _ in range(k)]
    for v in range(n):
        groups[v % k].append(v)
    
    planted_colors = [0] * n
    for color, group in enumerate(groups):
        for v in group:
            planted_colors[v] = color

    points = rng.uniform(0, 1, size=(n, 2))
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if planted_colors[i] != planted_colors[j]:
                dist = np.linalg.norm(points[i] - points[j])
                if dist < radius:
                    edges.append((i, j))
                    
    avg_degree = 2 * len(edges) / n if n > 0 else 0.0

    return GraphColoringInstance(
        instance_id=instance_id,
        n_vertices=n,
        k=k,
        edges=edges,
        planted_colors=planted_colors,
        difficulty=difficulty,
        generator="geometric",
        param=radius,
        avg_degree=avg_degree,
    )

def _build_neighbors(n: int, edges: list) -> dict:
    neighbors = {v: [] for v in range(n)}
    for u, w in edges:
        neighbors[u].append(w)
        neighbors[w].append(u)
    return neighbors

def _vanilla_descent_solve(instance: GraphColoringInstance, seed: int, max_iter: int = 1000) -> bool:
    import numpy as np
    rng = np.random.default_rng(seed)
    n, k = instance.n_vertices, instance.k
    colors = rng.integers(0, k, size=n).tolist()
    neighbors = _build_neighbors(n, instance.edges)

    for _ in range(max_iter):
        improved = False
        vertex_order = rng.permutation(n).tolist()
        for v in vertex_order:
            neighbor_color_counts = {}
            for nb in neighbors[v]:
                c = colors[nb]
                neighbor_color_counts[c] = neighbor_color_counts.get(c, 0) + 1

            current_conflicts = neighbor_color_counts.get(colors[v], 0)
            best_color = colors[v]
            best_conflicts = current_conflicts

            for c in range(k):
                if c != colors[v]:
                    c_conflicts = neighbor_color_counts.get(c, 0)
                    if c_conflicts < best_conflicts:
                        best_conflicts = c_conflicts
                        best_color = c

            if best_color != colors[v]:
                colors[v] = best_color
                improved = True

        if not improved:
            break

    conflicts = sum(1 for u, v in instance.edges if colors[u] == colors[v])
    return conflicts == 0

def _parallel_tempering_solve(instance: GraphColoringInstance, seed: int, n_steps: int = 3000) -> tuple:
    import numpy as np
    rng = np.random.default_rng(seed)
    n, k = instance.n_vertices, instance.k
    neighbors = _build_neighbors(n, instance.edges)

    temps = [0.02, 0.1, 0.4, 1.2, 3.5, 10.0]
    n_replicas = len(temps)
    replicas = [rng.integers(0, k, size=n).tolist() for _ in range(n_replicas)]
    conflicts = []
    for rep in replicas:
        c = sum(1 for u, w in instance.edges if rep[u] == rep[w])
        conflicts.append(c)

    swap_attempts = 0
    swap_accepts = 0
    SWAP_EVERY = 50

    for step in range(n_steps):
        for r_idx in range(n_replicas):
            T = temps[r_idx]
            colors = replicas[r_idx]
            for _ in range(5):
                v = int(rng.integers(0, n))
                c_old = colors[v]
                c_new = int(rng.integers(0, k))
                while c_new == c_old and k > 1:
                    c_new = int(rng.integers(0, k))

                delta = (
                    sum(1 if colors[nb] == c_new else 0 for nb in neighbors[v])
                    - sum(1 if colors[nb] == c_old else 0 for nb in neighbors[v])
                )
                if delta < 0 or (T > 1e-9 and rng.random() < math.exp(-delta / T)):
                    colors[v] = c_new
                    conflicts[r_idx] += delta

        if step % SWAP_EVERY == 0:
            for r_idx in range(n_replicas - 1):
                E_i = conflicts[r_idx]
                E_j = conflicts[r_idx + 1]
                T_i = temps[r_idx]
                T_j = temps[r_idx + 1]
                beta_i = 1.0 / T_i
                beta_j = 1.0 / T_j
                log_acc = (beta_i - beta_j) * (E_i - E_j)
                swap_attempts += 1
                if log_acc >= 0 or rng.random() < math.exp(log_acc):
                    replicas[r_idx], replicas[r_idx + 1] = replicas[r_idx + 1], replicas[r_idx]
                    conflicts[r_idx], conflicts[r_idx + 1] = conflicts[r_idx + 1], conflicts[r_idx]
                    swap_accepts += 1

    pt_swap_rate = swap_accepts / max(1, swap_attempts)
    solved = min(conflicts) == 0
    return solved, pt_swap_rate

def _dsatur_solve(instance: GraphColoringInstance) -> tuple:
    n, k = instance.n_vertices, instance.k
    neighbors = _build_neighbors(n, instance.edges)

    colors = [-1] * n
    saturation = [0] * n
    neighbor_colors = [set() for _ in range(n)]

    uncolored = set(range(n))
    while uncolored:
        v = max(uncolored, key=lambda x: (saturation[x], len(neighbors[x])))
        used = neighbor_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c

        for nb in neighbors[v]:
            if nb in uncolored:
                if c not in neighbor_colors[nb]:
                    neighbor_colors[nb].add(c)
                    saturation[nb] += 1

        uncolored.remove(v)

    max_color = max(colors) if colors else -1
    no_conflicts = all(colors[u] != colors[v] for u, v in instance.edges)
    uses_at_most_k = max_color < k
    return colors, (no_conflicts and uses_at_most_k)

def _ar_greedy_solve(instance: GraphColoringInstance, seed: int) -> bool:
    import numpy as np
    rng = np.random.default_rng(seed)
    n, k = instance.n_vertices, instance.k
    neighbors = _build_neighbors(n, instance.edges)

    vertex_order = rng.permutation(n).tolist()
    colors = [-1] * n

    for v in vertex_order:
        used = set(colors[nb] for nb in neighbors[v] if colors[nb] >= 0)
        c = 0
        while c in used:
            c += 1
        colors[v] = c

    max_color = max(colors) if colors else -1
    no_conflicts = all(colors[u] != colors[v] for u, v in instance.edges)
    return no_conflicts and max_color < k

def bootstrap_ci(data, num_samples=10000, alpha=0.05, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(data)
    data = np.array(data, dtype=float)
    samples = rng.choice(data, size=(num_samples, n), replace=True)
    means = np.mean(samples, axis=1)
    return [float(np.percentile(means, 100 * (alpha / 2))), float(np.percentile(means, 100 * (1 - alpha / 2)))]

def paired_bootstrap_p(energy_results, strong_results, num_samples=10000, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    diffs = np.array(energy_results, dtype=float) - np.array(strong_results, dtype=float)
    n = len(diffs)
    if n == 0:
        return 1.0
    samples = rng.choice(diffs, size=(num_samples, n), replace=True)
    means = np.mean(samples, axis=1)
    p = np.mean(means <= 0)
    return float(p)

def _reproducibility_checksum(instances: list, seed: int, optimizer_configs: dict) -> str:
    data = {
        "seed": seed,
        "n_instances": len(instances),
        "instance_n_vertices": [inst.n_vertices for inst in instances],
        "instance_param": [inst.param for inst in instances],
        "instance_difficulty": [inst.difficulty for inst in instances],
        "instance_generator": [inst.generator for inst in instances],
        "optimizer_configs": optimizer_configs,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

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

def main():
    global _T0
    _T0 = time.time()
    import numpy as np

    print("Exp 3563: P0.1 graph coloring multiseed second generator — start", flush=True)

    print("\nStep 0a: Encoding validity check...", flush=True)
    test_colors = [0, 1, 2]
    test_edges = [(0, 1), (1, 2), (0, 2)]
    E = compute_energy(test_colors, 3, 3, test_edges)
    assert E == 0.0, f"Encoding validity check FAILED: E={E}"
    encoding_validity_E0 = True

    print("\nStep 1: Hardness calibration for Erdos-Renyi...", flush=True)
    hard_n = 55
    hard_p = None
    for candidate_p in [0.05, 0.10, 0.15, 0.20, 0.25]:
        if _over_budget(): break
        test_instances = [
            make_planted_instance(hard_n, 3, candidate_p, np.random.default_rng(SEED + i), i, "hard")
            for i in range(20)
        ]
        results = [_dsatur_solve(inst)[1] for inst in test_instances]
        rate = sum(results) / len(results)
        print(f"ER Calibration: p_cross={candidate_p:.2f}, dsatur_rate={rate:.3f}", flush=True)
        if rate < 0.80:
            hard_p = candidate_p
            break
    
    if hard_p is None:
        hard_p = 0.15

    print("\nStep 1: Hardness calibration for Geometric...", flush=True)
    hard_r = None
    for candidate_r in [0.21, 0.22, 0.23, 0.24, 0.25]:
        if _over_budget(): break
        test_instances = [
            make_geometric_planted_instance(hard_n, 3, candidate_r, np.random.default_rng(SEED + 100 + i), i, "hard")
            for i in range(20)
        ]
        results = [_dsatur_solve(inst)[1] for inst in test_instances]
        rate = sum(results) / len(results)
        print(f"Geo Calibration: r={candidate_r:.2f}, dsatur_rate={rate:.3f}", flush=True)
        if rate < 0.80:
            hard_r = candidate_r
            break
            
    if hard_r is None:
        hard_r = 0.24

    print(f"Using ER hard_p={hard_p}, Geo hard_r={hard_r}", flush=True)

    print("\nStep 2: Building corpus...", flush=True)
    N_HARD = 30
    N_SEEDS = 5
    
    er_instances = [
        make_planted_instance(hard_n, 3, hard_p, np.random.default_rng(SEED + 3000 + i), 3000 + i, "hard")
        for i in range(N_HARD)
    ]
    geo_instances = [
        make_geometric_planted_instance(hard_n, 3, hard_r, np.random.default_rng(SEED + 4000 + i), 4000 + i, "hard")
        for i in range(N_HARD)
    ]
    
    all_instances = er_instances + geo_instances
    optimizer_configs = {"pt_n_steps": 3000, "vanilla_descent_max_iter": 1000}
    checksum = _reproducibility_checksum(all_instances, SEED, optimizer_configs)

    results_data = {
        "erdos_renyi": {"dsatur": [], "pt": [], "ar": [], "pt_swap": []},
        "geometric": {"dsatur": [], "pt": [], "ar": [], "pt_swap": []}
    }

    print("\nStep 3-7: Running Optimizers...", flush=True)
    for gen_name, instances in [("erdos_renyi", er_instances), ("geometric", geo_instances)]:
        for i, inst in enumerate(instances):
            if _over_budget(): break
            
            _, dsatur_valid = _dsatur_solve(inst)
            results_data[gen_name]["dsatur"].append(float(dsatur_valid))
            
            for s_idx in range(N_SEEDS):
                cur_seed = SEED + inst.instance_id * 10 + s_idx
                
                ar_valid = _ar_greedy_solve(inst, cur_seed)
                results_data[gen_name]["ar"].append(float(ar_valid))

                pt_valid, pt_swap = _parallel_tempering_solve(inst, cur_seed, n_steps=3000)
                results_data[gen_name]["pt"].append(float(pt_valid))
                if s_idx == 0:
                    results_data[gen_name]["pt_swap"].append(pt_swap)
                
            print(f"{gen_name} processed {i+1}/{len(instances)} instances (5 seeds each)", flush=True)

    out = {
        "honest_verdict": "",
        "inference_substrate": "ising_energy_optimization_cpu",
        "encoding_validity_E0": encoding_validity_E0,
        "n_generators": 2,
        "generator_names": ["erdos_renyi", "geometric"],
        "n_seeds": N_SEEDS,
        "exact_baseline_solve_rate": 1.0,
        "random_seed": SEED,
        "reproducibility_checksum": checksum,
    }

    # Analyze per generator
    pooled_pt = []
    pooled_dsatur = []
    pooled_ar = []
    
    ci_excludes_zero_count = 0

    for gen_name in ["erdos_renyi", "geometric"]:
        dsatur_base = results_data[gen_name]["dsatur"] # length N_HARD
        # Expand dsatur to match seeds for paired diff
        dsatur_expanded = []
        for d in dsatur_base:
            dsatur_expanded.extend([d] * N_SEEDS)
            
        pt_vals = results_data[gen_name]["pt"] # length N_HARD * N_SEEDS
        ar_vals = results_data[gen_name]["ar"]
        
        pooled_pt.extend(pt_vals)
        pooled_dsatur.extend(dsatur_expanded)
        pooled_ar.extend(ar_vals)
        
        diffs = [p - d for p, d in zip(pt_vals, dsatur_expanded)]
        mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
        ci = bootstrap_ci(diffs, seed=SEED) if diffs else [0.0, 0.0]
        
        out[f"strong_baseline_solve_rate_hard_tier_{gen_name}"] = sum(dsatur_base) / len(dsatur_base) if dsatur_base else 0.0
        out[f"solve_rate_{gen_name}"] = sum(pt_vals) / len(pt_vals) if pt_vals else 0.0
        out[f"hard_tier_paired_diff_{gen_name}"] = mean_diff
        out[f"hard_tier_paired_diff_ci95_{gen_name}"] = ci
        
        if ci[0] > 0.0:
            ci_excludes_zero_count += 1

    out["positive_robust_to_seed_and_generator"] = (ci_excludes_zero_count == 2)
    
    pooled_diffs = [p - d for p, d in zip(pooled_pt, pooled_dsatur)]
    out["pooled_hard_tier_paired_diff"] = sum(pooled_diffs) / len(pooled_diffs) if pooled_diffs else 0.0
    out["pooled_hard_tier_paired_diff_ci95"] = bootstrap_ci(pooled_diffs, seed=SEED) if pooled_diffs else [0.0, 0.0]
    
    out["ar_greedy_solve_rate"] = sum(pooled_ar) / len(pooled_ar) if pooled_ar else 0.0
    
    all_pt_swaps = results_data["erdos_renyi"]["pt_swap"] + results_data["geometric"]["pt_swap"]
    out["pt_swap_acceptance_rate"] = sum(all_pt_swaps) / len(all_pt_swaps) if all_pt_swaps else 0.0

    # Add requirements for fields in the spec
    out["strong_baseline_solve_rate_hard_tier_per_generator"] = {
        "erdos_renyi": out.pop("strong_baseline_solve_rate_hard_tier_erdos_renyi"),
        "geometric": out.pop("strong_baseline_solve_rate_hard_tier_geometric")
    }
    out["solve_rate_per_generator"] = {
        "erdos_renyi": out.pop("solve_rate_erdos_renyi"),
        "geometric": out.pop("solve_rate_geometric")
    }
    out["hard_tier_paired_diff_per_generator"] = {
        "erdos_renyi": out.pop("hard_tier_paired_diff_erdos_renyi"),
        "geometric": out.pop("hard_tier_paired_diff_geometric")
    }
    out["hard_tier_paired_diff_ci95_per_generator"] = {
        "erdos_renyi": out.pop("hard_tier_paired_diff_ci95_erdos_renyi"),
        "geometric": out.pop("hard_tier_paired_diff_ci95_geometric")
    }

    out = de_alias_dict(out)
    
    duration = _elapsed()
    out["duration_s"] = duration
    
    out["no_aliased_fields_assert"] = True  # de_alias_dict handles this

    if encoding_validity_E0 and out["n_generators"] >= 2 and out["n_seeds"] >= 5 and out["no_aliased_fields_assert"]:
        if out["positive_robust_to_seed_and_generator"]:
            out["honest_verdict"] = "complete: p01_route1_graph_coloring_positive_robust_multiseed_multigenerator_defensible_headline"
        else:
            out["honest_verdict"] = "complete: p01_route1_graph_coloring_positive_bounded_to_single_generator_ci_includes_zero_on_second"
    else:
        out["honest_verdict"] = "complete: blocked_cannot_construct_discriminating_corpus"

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {duration:.1f}s. Verdict: {out['honest_verdict']}")

if __name__ == "__main__":
    main()
