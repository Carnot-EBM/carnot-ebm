#!/usr/bin/env python3
"""
Generate CARM Benchmark Cases
Traces to: REQ-BENCH-1771
"""

import os
import sys

# Ensure carnot package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from carnot.eval.experiment_1771_care_test_suite import generate_carm_benchmark

def main():
    os.makedirs("results", exist_ok=True)
    primary_path = "results/experiment_1771_care_test_suite.json"
    backup_path = "results/carm_benchmark_cases.json"
    generate_carm_benchmark(primary_path, backup_path)
    print(f"Generated benchmark suites at {primary_path} and {backup_path}")

if __name__ == "__main__":
    main()
