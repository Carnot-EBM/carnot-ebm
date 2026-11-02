#!/usr/bin/env python3
"""Run Capstone v313."""
import os
import sys
import json
import time
from pathlib import Path

# Add python dir to path
REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v313_3402 import run_capstone

def main():
    start_time = time.perf_counter()
    result = run_capstone()
    result["duration_s"] = round(time.perf_counter() - start_time, 3)
    
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_3402_capstone_v313.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

if __name__ == "__main__":
    main()
