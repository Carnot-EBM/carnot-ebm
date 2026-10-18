#!/usr/bin/env python3
"""Script to run the comparative evaluation of KAGNN vs MLP.

Spec references: REQ-KAN-2035, SCENARIO-KAN-2035.
"""

import sys
from pathlib import Path

# Ensure the python directory is in the path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.models.ising.kagnn_eval import run_evaluation

if __name__ == "__main__":
    run_evaluation()
    print("Evaluation completed. Results saved to results/exp2035_kagnn_eval.json")
