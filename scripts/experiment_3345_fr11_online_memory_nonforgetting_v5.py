#!/usr/bin/env python3
import json
from pathlib import Path
from carnot.eval.fr11_online_verifier_memory_nonforgetting_v5 import run_experiment_3345

def main():
    artifact = run_experiment_3345()
    out_path = Path("results/experiment_3345_fr11_online_memory_nonforgetting_v5.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
