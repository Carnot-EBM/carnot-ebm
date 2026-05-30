"""Tests for experiment 3393."""

import json
import numpy as np
from unittest.mock import patch

from scripts import experiment_3393_proximal_gradient_constraint
from carnot.verify.proximal_gradient_constraint_layer import (
    continuous_relaxation_penalty,
    proximal_descent_projection,
    measure_constraint_satisfaction_improvement
)

def test_proximal_gradient_constraint_layer():
    """Test the proximal gradient functions. (REQ-VERIFY-3393)"""
    def constraint_a(logits):
        return float(np.sum(logits**2))
        
    logits = np.array([1.0, -1.0])
    constraints = [constraint_a]
    
    penalty = continuous_relaxation_penalty(logits, constraints)
    assert penalty == 2.0
    
    projected = proximal_descent_projection(logits, constraints, step_size=0.1, num_steps=2)
    assert projected.shape == logits.shape
    
    improvement = measure_constraint_satisfaction_improvement(logits, projected, constraints)
    assert improvement > 0

def test_experiment_3393_main(tmp_path):
    """Test that main() executes and writes the deliverable. (SCENARIO-VERIFY-3393)"""
    deliverable_path = tmp_path / "experiment_3393_proximal_gradient_constraint.json"
    
    original_init = experiment_3393_proximal_gradient_constraint.ExperimentTemplate.__init__
    
    def mock_init(self, exp_id, title, deliverable, **kwargs):
        original_init(self, exp_id, title, str(deliverable_path), **kwargs)
        self.deliverable = str(deliverable_path)
        self._output_path = deliverable_path
        
    with patch.object(experiment_3393_proximal_gradient_constraint.ExperimentTemplate, '__init__', mock_init), \
         patch.object(experiment_3393_proximal_gradient_constraint.ExperimentTemplate, 'setup', return_value=None), \
         patch.object(experiment_3393_proximal_gradient_constraint, 'cached_sota_pair', return_value=[{"model_path": "dummy"}]):
        result = experiment_3393_proximal_gradient_constraint.main()
        
        assert result["status"] == "success"
        assert "Proximal-Gradient" in result["honest_verdict"]
        assert deliverable_path.exists()
        
        with open(deliverable_path) as f:
            data = json.load(f)
            assert data["experiment"] == 3393
            assert data["improvement"] > 0
