#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from carnot.eval.fr11_online_verifier_memory_nonforgetting_v4 import run_experiment_3334

def main():
    print("Running FR-11 Online Verifier Memory Nonforgetting Experiment 3334...")
    try:
        artifact = run_experiment_3334()
        
        output_path = Path("results/experiment_3334_fr11_online_verifier_memory_nonforgetting_v4.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)
            
        print(f"Artifact written to {output_path}")
        if artifact["fr11_nonforgetting_ready"]:
            print("Verdict: SUCCESS (fr11_nonforgetting_ready=True)")
            sys.exit(0)
        else:
            print("Verdict: BLOCKED")
            print(f"Reasons: {artifact.get('blocked_reasons', [])}")
            sys.exit(0)
            
    except Exception as e:
        print(f"Experiment failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()