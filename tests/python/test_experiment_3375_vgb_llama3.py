"""Tests for experiment 3375."""

import json
import runpy
import sys
from unittest.mock import patch

from scripts import experiment_3375_vgb_llama3


def test_experiment_3375_main(tmp_path):
    """Test that main() executes and writes the deliverable."""
    deliverable_path = tmp_path / "experiment_3375_vgb_llama3.json"
    
    original_init = experiment_3375_vgb_llama3.ExperimentTemplate.__init__
    
    def mock_init(self, exp_id, title, deliverable, **kwargs):
        original_init(self, exp_id, title, str(deliverable_path), **kwargs)
        self.deliverable = str(deliverable_path)
        self._output_path = deliverable_path
        
    with patch.object(experiment_3375_vgb_llama3.ExperimentTemplate, '__init__', mock_init):
        result = experiment_3375_vgb_llama3.main()
        
        assert result["status"] == "success"
        assert result["honest_verdict"] == "Completed successfully for Llama-3 repair ladder scaffold."
        assert deliverable_path.exists()
        
        with open(deliverable_path) as f:
            data = json.load(f)
            assert data["experiment"] == 3375
            assert data["repair_ladder_llama3_outcome"] == "verified"

