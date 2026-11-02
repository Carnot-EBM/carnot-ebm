#!/usr/bin/env python3
import sys
from pathlib import Path

# Add python/ to path so we can import carnot
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from carnot.verification.experiment_3605_cross_domain_synthesis_v3 import run_experiment

def main():
    results_dir = Path("results")
    output_path = results_dir / "experiment_3605_cross_domain_synthesis_v3.json"
    
    upstream = {
        "3598": results_dir / "experiment_3598_diagnose_330_cascade_audit.json",
        "3599": results_dir / "experiment_3599_factual_corpus_v2_with_evidence.json",
        "3600": results_dir / "experiment_3600_real_nli_grounding_verifier.json",
        "3601": results_dir / "experiment_3601_corrected_cross_domain_remeasurement.json",
        "3602": results_dir / "experiment_3602_math_to_code_prm_transfer.json",
    }
    
    run_experiment(output_path, upstream)
    print(f"Wrote {output_path}")

if __name__ == "__main__":
    main()
