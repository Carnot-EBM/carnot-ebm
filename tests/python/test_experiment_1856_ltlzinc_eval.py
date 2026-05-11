"""Tests for experiment 1856 LTLZinc eval."""

import tempfile
from pathlib import Path

from carnot.pipeline.experiment_1856_ltlzinc_eval import run_experiment

def test_run_experiment_1856_ltlzinc_eval():
    """Verify that the experiment 1856 script produces a valid artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out.json"
        res = run_experiment(output_path=out_path)
        
        assert res["experiment_id"] == 1856
        assert res["experiment"] == "1856_ltlzinc_eval"
        assert res["status"] == "complete"
        assert res["ltlzinc_adapter_ready"] is True
        assert res["cerce_nonforgetting_rate"] == 1.0
        assert res["schema"] == "carnot.pipeline_ltlzinc_adapter.v1"
        assert out_path.exists()
