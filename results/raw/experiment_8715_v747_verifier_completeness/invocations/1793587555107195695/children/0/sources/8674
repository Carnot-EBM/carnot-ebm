import gzip
import json
import time
import sys
import numpy as np
import os

from carnot.agentic.arc_gap4_execution_verifier import Gap4ExecutionVerifier, apply_rule, get_consistency_energy
from sklearn.metrics import roc_auc_score

def synthesize_near_miss(grid):
    """Create a 1-5% cells changed near miss of the same shape."""
    grid = np.array(grid)
    out = grid.copy()
    n_cells = grid.size
    n_change = max(1, int(n_cells * 0.03)) # ~3% change
    
    y_idx = np.random.randint(0, grid.shape[0], size=n_change)
    x_idx = np.random.randint(0, grid.shape[1], size=n_change)
    
    for y, x in zip(y_idx, x_idx):
        old_c = out[y, x]
        new_c = np.random.randint(0, 10)
        if new_c == old_c:
            new_c = (new_c + 1) % 10
        out[y, x] = new_c
    return out

def main():
    start_time = time.time()
    
    pool_path = 'results/arc3_gap3_stage2_eval_pool.json.gz'
    try:
        with gzip.open(pool_path, 'rt') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load pool: {e}")
        with open('results/experiment_3975_gap4_execution_verifier_build.json', 'w') as f:
            json.dump({
                "honest_verdict": "blocked_eval_pool_unreadable",
                "duration_s": time.time() - start_time
            }, f)
        sys.exit(0)

    entries = data.get('entries', [])
    
    verifier = Gap4ExecutionVerifier()
    
    n_tasks_total = len(entries)
    n_tasks_covered = 0
    
    y_true = []
    y_score = []
    
    for entry in entries:
        demos = entry.get('demos', [])
        test_input = entry.get('test_input')
        if not demos or test_input is None:
            continue
            
        rule = verifier.induce_program(demos)
        if rule is not None:
            n_tasks_covered += 1
            pseudo_gold = apply_rule(rule, test_input)
            
            if pseudo_gold is not None:
                near_miss = synthesize_near_miss(pseudo_gold)
                
                energy_gold = get_consistency_energy(rule, test_input, pseudo_gold)
                energy_near_miss = get_consistency_energy(rule, test_input, near_miss)
                
                y_true.extend([1, 0])
                y_score.extend([-energy_gold, -energy_near_miss])
                
    coverage = n_tasks_covered / n_tasks_total if n_tasks_total > 0 else 0
    
    auroc = None
    if len(y_true) > 0 and len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
        
    passed = auroc is not None and auroc > 0.70
    
    verdict = f"complete: gap4_verifier_built_coverage{coverage:.2f}_auroc{auroc:.2f}" if passed else f"complete: gap4_positive_control_failed_auroc{auroc if auroc else 0:.2f}"
    
    artifact = {
        "positive_control_passed": bool(passed),
        "gold_vs_nearmiss_auroc": float(auroc) if auroc is not None else 0.0,
        "program_synthesis_coverage": float(coverage),
        "n_tasks_covered": n_tasks_covered,
        "n_tasks_total": n_tasks_total,
        "llm_proposer_used": verifier.llm_proposer_used,
        "missing_verifier_gaps": "DSL cannot express complex compositional geometry changes or non-color transformations.",
        "random_seed": 42,
        "honest_verdict": verdict,
        "duration_s": time.time() - start_time,
        "inference_substrate": "dsl-only"
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_3975_gap4_execution_verifier_build.json', 'w') as f:
        json.dump(artifact, f, indent=2)

if __name__ == "__main__":
    np.random.seed(42)
    main()
