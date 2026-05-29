"""Tests for experiment 3389.

Spec: REQ-BENCH-3389, SCENARIO-BENCH-3389-1
"""

import json
from unittest.mock import patch
try:
    import torch
except ImportError:
    pass

from scripts import experiment_3389_constraintbench


def test_experiment_3389_main(tmp_path):
    """Test that main() executes and writes the deliverable."""
    deliverable_path = tmp_path / "experiment_3389_constraintbench.json"
    
    original_init = experiment_3389_constraintbench.ExperimentTemplate.__init__
    
    def mock_init(self, exp_id, title, deliverable, **kwargs):
        original_init(self, exp_id, title, str(deliverable_path), **kwargs)
        self.deliverable = str(deliverable_path)
        self._output_path = deliverable_path
        
    with patch.object(experiment_3389_constraintbench.ExperimentTemplate, '__init__', mock_init):
        result = experiment_3389_constraintbench.main()
        
        assert result["status"] == "success"
        assert result["honest_verdict"] == "Completed successfully for ConstraintBench AR vs VGB repair ladder comparison."
        assert deliverable_path.exists()
        
        with open(deliverable_path) as f:
            data = json.load(f)
            assert data["experiment"] == 3389
            assert data["tasks_evaluated"] == 10
            assert data["model_used"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
