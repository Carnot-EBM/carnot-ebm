"""Tests for Exp 1643 STATIC CSR prototype."""

import json
from pathlib import Path
import pytest
import sys
import importlib.util

repo_root = Path(__file__).resolve().parents[2]
script_path = repo_root / "scripts" / "experiment_1643_static_csr.py"

spec = importlib.util.spec_from_file_location("experiment_1643_static_csr", script_path)
exp = importlib.util.module_from_spec(spec)
sys.modules["experiment_1643_static_csr"] = exp
if spec and spec.loader:
    spec.loader.exec_module(exp)

def test_req_verify_1643_csr_mask_matches() -> None:
    """REQ-VERIFY-1643: CSR mask exact matching works correctly."""
    mask = exp.build_schema_csr_mask(["{}", '{"test": 1}'])
    assert mask.accepts("{}")
    assert mask.accepts('{"test": 1}')
    assert not mask.accepts("invalid")

def test_req_verify_1643_csr_mask_empty_raises() -> None:
    """REQ-VERIFY-1643: Empty mask raises ValueError."""
    with pytest.raises(ValueError, match="accepted_strings must contain"):
        exp.build_schema_csr_mask([])

def test_benchmark_acceptors() -> None:
    cases = ["{}", '{"test": 1}', "invalid"]
    automaton = exp.build_schema_csr_mask(["{}", '{"test": 1}'])
    
    def timer() -> int: 
        return 0
    
    latency = exp.benchmark_acceptors(cases, automaton, repeats=10, timer=timer)
    assert latency["grammar_latency_ms_p50"] == 0.0
    assert latency["csr_latency_ms_p50"] == 0.0

def test_run_experiment(tmp_path: Path) -> None:
    output_path = tmp_path / "experiment_1643_static_csr.json"
    
    class CounterTimer:
        def __init__(self) -> None:
            self.i = 0
        def __call__(self) -> int:
            self.i += 1000000
            return self.i
            
    timer = CounterTimer()
            
    artifact = exp.run_experiment(
        output_path=output_path,
        run_date="20260509",
        repeats=10,
        tests_run=["test_run_experiment"],
        timer=timer,
    )
    
    assert output_path.exists()
    
    with open(output_path) as f:
        persisted = json.load(f)
        
    assert persisted["experiment"] == "1643_static_csr"
    assert persisted["schema"] == "static_csr_v1"
    assert persisted["status"] == "complete"
    assert persisted["csr_latency_ms_p50"] > 0
    assert persisted["latency_improvement"] > 0
