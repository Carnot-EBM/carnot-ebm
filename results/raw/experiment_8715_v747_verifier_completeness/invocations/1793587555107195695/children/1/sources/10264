#!/usr/bin/env python3
"""Run Experiment 3835: Formal Core 5-seed CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

def main() -> int:
    try:
        from carnot.eval.experiment_3835 import run_experiment_3835
    except ImportError as e:
        print(f"BLOCKED: cannot import experiment_3835 module: {e}")
        return 1
        
    try:
        result = run_experiment_3835(REPO_ROOT)
    except Exception as e:
        print(f"BLOCKED: {e}")
        return 1

    out_file = REPO_ROOT / "results" / "experiment_3835_formal_core_5seed_ci.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
        
    print(f"Wrote result to {out_file.relative_to(REPO_ROOT)}")
    print(f"Verdict: {result['honest_verdict']}")
    print(f"Full AUROC: {result['full_ensemble_auroc_mean']:.4f}")
    print(f"Formal AUROC: {result['formal_only_auroc_mean']:.4f}")
    print(f"Learned AUROC: {result['learned_only_auroc_mean']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
