"""Empirical delta calculation for absorbing Markov chain verification."""
import json
from pathlib import Path

def compute_empirical_delta(results_dir: Path) -> float:
    """
    Computes Carnot's empirical delta from recent verify-repair runs.
    Delta is the single-step absorption probability.
    """
    total_iters = 0
    total_success = 0
    
    for p in results_dir.glob("*.json"):
        if not p.is_file():
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
            
            if "per_seed" in data:
                for s in data["per_seed"]:
                    iters = 0
                    for k, v in s.items():
                        if "repair_iterations" in k and isinstance(v, (int, float)):
                            iters = max(iters, v)
                    
                    success = False
                    for k, v in s.items():
                        if ("converged" in k and isinstance(v, bool) and v) or \
                           ("satisfaction" in k and isinstance(v, float) and v >= 0.99):
                            success = True
                            
                    if iters > 0:
                        total_iters += int(iters)
                        if success:
                            total_success += 1
        except Exception:
            pass
            
    if total_iters == 0:
        return 0.0
        
    return total_success / total_iters
