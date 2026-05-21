import json
import os
import time
import datetime
import subprocess
from carnot.phase4_n64 import run_n64_scaling_experiment

def main():
    start_time = time.time()
    
    git_rev = ""
    try:
        git_rev = subprocess.check_output(["git", "log", "-1", "--format=%H", "scripts/research_conductor.py"]).decode('utf-8').strip()
    except Exception:
        git_rev = "unknown"
        
    output_dict = run_n64_scaling_experiment(
        n_spins=64,
        mld_steps=100,
        n_seeds=30,
        random_seed=171193,
        git_rev=git_rev
    )
    
    end_time = time.time()
    duration_s = end_time - start_time
    run_date = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if not run_date.endswith('Z'):
        run_date = run_date.replace('+00:00', 'Z')
        
    output = {
        "schema": "carnot.phase4_active_inference_n64.v1",
        "experiment": 1693,
        "run_date": run_date,
        "duration_s": duration_s,
        "random_seed": 171193
    }
    output.update(output_dict)
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/experiment_1693_phase4_n64.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Generated {out_path}")

if __name__ == "__main__":
    main()
