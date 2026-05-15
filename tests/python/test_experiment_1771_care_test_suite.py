"""
Tests for REQ-BENCH-1771 and SCENARIO-BENCH-1771.
Ensures CARM benchmark cases generation meets requirements.
"""

import json
import os
import pytest

from carnot.eval.experiment_1771_care_test_suite import generate_carm_benchmark

def test_carm_benchmark_generation(tmp_path):
    """
    Test that the CARM benchmark script generates valid cases and outputs.
    Traces to: REQ-BENCH-1771, SCENARIO-BENCH-1771
    """
    care_test_suite_path = tmp_path / "experiment_1771_care_test_suite.json"
    carm_cases_path = tmp_path / "carm_benchmark_cases.json"
    
    generate_carm_benchmark(str(care_test_suite_path), str(carm_cases_path))
    
    assert care_test_suite_path.exists()
    assert carm_cases_path.exists()
    
    with open(care_test_suite_path, "r") as f:
        data = json.load(f)
        
    assert data.get("schema") == "carnot.carm.benchmark.v1"
    assert data.get("num_cases") == 20
    assert "cases" in data
    assert len(data["cases"]) == 20
    
    cases = data["cases"]
    
    has_tool_use = any(c.get("constraint_type") == "tool-use" for c in cases)
    has_arithmetic = any(c.get("constraint_type") == "arithmetic" for c in cases)
    has_logic = any(c.get("constraint_type") == "logic" for c in cases)
    
    assert has_tool_use, "Must have tool-use constraints"
    assert has_arithmetic, "Must have arithmetic constraints"
    assert has_logic, "Must have logic constraints"
    
    for case in cases:
        assert "instruction" in case
        assert "ground_truth" in case
        assert "constraint_type" in case
