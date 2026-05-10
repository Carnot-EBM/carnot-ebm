"""
Tests for experiment 1731 FPGA audit latency benchmark.
"""

import json
import os
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.experiment_1731_fpga_audit import run_audit

# REQ-HW-051: CIKAN FPGA implementation MUST benchmark inference latency against baselines
# SCENARIO-HW-051: Audit measures inference latency for 1000 CIKAN evaluations

def test_run_audit():
    """Test running the latency audit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = os.path.join(tmpdir, "out.json")
        
        results = run_audit(output_json, batch_size=1000)
            
        assert os.path.exists(output_json)
        
        with open(output_json) as f:
            data = json.load(f)
            assert data["experiment"] == "1731_fpga_audit"
            assert data["batch_size"] == 1000
            assert "latency_ms" in data
            assert data["status"] == "success"
