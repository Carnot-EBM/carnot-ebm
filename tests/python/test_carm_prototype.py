"""
Tests for CARM prototype.
References: REQ-CARM-1772-1, SCENARIO-CARM-1772-1
"""
import json
from pathlib import Path
from carnot.carm.prototype import CARMExtractor

def test_extract_all_care_suite_constraints():
    """Test extracting constraints for all cases in the CARE test suite. SCENARIO-CARM-1772-1"""
    extractor = CARMExtractor()
    assert extractor.model_spec == "unsloth/Qwen3.6-35B-A3B-GGUF"
    
    test_suite_path = Path("results/experiment_1771_care_test_suite.json")
    if test_suite_path.exists():
        test_suite = json.loads(test_suite_path.read_text())
        for case in test_suite["cases"]:
            result = extractor.extract_constraints(case["instruction"])
            assert result == case["ground_truth"], f"Failed on case {case['id']}"

def test_extract_unknown_constraints():
    """Test extracting unknown constraints."""
    extractor = CARMExtractor()
    result = extractor.extract_constraints("Do something unknown.")
    assert result == {}
