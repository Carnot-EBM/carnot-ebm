"""Tests for Exp 1868 LTLZinc evaluations."""

import pytest
from carnot.pipeline.experiment_1868_ltlzinc import run_experiment

def test_experiment_1868_ltlzinc_retention(tmp_path):
    """
    Test that LTLZinc memory retention evaluations pass for the pruned memory traces.
    Spec: REQ-LEARN-1868, SCENARIO-LEARN-1868
    """
    output_path = tmp_path / "experiment_1868_ltlzinc.json"
    result = run_experiment(output_path=output_path)
    
    assert result["experiment_id"] == 1868
    assert result["cerce_nonforgetting_rate"] == 1.0
    assert result["forgetting_rate"] == 0.0
    assert result["status"] == "complete"
    assert output_path.exists()
