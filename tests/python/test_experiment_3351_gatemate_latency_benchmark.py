"""Test for Exp 3351 GateMate Latency Benchmark.

Spec refs: REQ-HW-103, SCENARIO-HW-103.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.experiment_3351_gatemate_latency_benchmark import run_experiment, RESULT_PATH


def test_experiment_3351_gatemate_latency_benchmark_blocked():
    """Verify the GateMate latency benchmark records a blocked artifact."""
    artifact = run_experiment()
    
    assert artifact["experiment"] == 3351
    assert "honest_verdict" in artifact
    assert artifact["honest_verdict"] == "blocked_no_io_interface_in_rtl"
    assert artifact["gatemate_latency_us"] is None
    assert artifact["speedup_vs_cpu"] is None
    assert "GateMate n=16 RTL lacks a host communication interface" in artifact["blocked_reasons"][0]
    
    assert RESULT_PATH.exists()
    payload = json.loads(RESULT_PATH.read_text())
    assert payload["experiment"] == 3351
    assert payload["honest_verdict"] == "blocked_no_io_interface_in_rtl"


from carnot.experiment_3351_gatemate_latency_benchmark import main

def test_experiment_3351_main():
    """Verify main function handles arguments and returns 0."""
    result = main([])
    assert result == 0
