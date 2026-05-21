import json
from pathlib import Path

results_dir = Path("results")
files = [
    "experiment_1276_snarenet_repair_layer_gated.json",
    "experiment_163_results.json",
    "experiment_208_results.json",
    "experiment_226_results.json",
    "experiment_227_results.json"
]

total_iters = 0
total_success = 0

for fname in files:
    p = results_dir / fname
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        if "per_seed" in data:
            for s in data["per_seed"]:
                # find iterations
                iters = 0
                for k, v in s.items():
                    if "repair_iterations" in k and isinstance(v, (int, float)):
                        iters = max(iters, v)
                
                # find success
                success = False
                for k, v in s.items():
                    if "converged" in k or "satisfaction" in k:
                        if isinstance(v, bool) and v: success = True
                        if isinstance(v, float) and v >= 0.99: success = True
                
                if iters > 0:
                    total_iters += iters
                    if success:
                        total_success += 1

print(f"Total iters: {total_iters}, Total success: {total_success}")
if total_iters > 0:
    print(f"Empirical Delta (success / iters): {total_success / total_iters}")
