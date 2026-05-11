"""Tests for COCOM pipeline.

Spec: REQ-PIPELINE-1831, SCENARIO-PIPELINE-1831
"""

import json
import os
import tempfile
import numpy as np
import pytest

from carnot.pipeline.cocom import COCOMPipeline

def test_cocom_initialization():
    """Test initialization of COCOM.
    
    Spec: REQ-PIPELINE-1831
    """
    pipeline = COCOMPipeline(learning_rate=0.1, memory_size=5, parameter_dim=2)
    assert pipeline.learning_rate == 0.1
    assert pipeline.memory_size == 5
    assert len(pipeline.memory) == 0
    assert pipeline.parameters.shape == (2,)

def test_cocom_update():
    """Test the update step tracks memory constraints.
    
    Spec: SCENARIO-PIPELINE-1831
    """
    pipeline = COCOMPipeline(learning_rate=0.1, memory_size=5, parameter_dim=2)
    obj_grad = np.array([0.5, 0.5])
    const_grad = np.array([0.1, -0.1])
    
    pipeline.update(obj_grad, const_grad)
    assert len(pipeline.memory) == 1
    # Check that parameters are updated
    assert not np.allclose(pipeline.parameters, np.zeros(2))

def test_cocom_memory_limit():
    """Test memory limit is respected.
    
    Spec: SCENARIO-PIPELINE-1831
    """
    pipeline = COCOMPipeline(learning_rate=0.1, memory_size=2, parameter_dim=2)
    
    pipeline.update(np.array([0.5, 0.5]), np.array([0.1, -0.1]))
    pipeline.update(np.array([0.5, 0.5]), np.array([0.2, -0.2]))
    pipeline.update(np.array([0.5, 0.5]), np.array([0.3, -0.3]))
    
    assert len(pipeline.memory) == 2

def test_cocom_write_artifact():
    """Test writing the experiment artifact.
    
    Spec: REQ-PIPELINE-1831
    """
    pipeline = COCOMPipeline(learning_rate=0.1, memory_size=5, parameter_dim=2)
    pipeline.update(np.array([0.5, 0.5]), np.array([0.1, -0.1]))
    
    # Train oracle so oracle_weights is not None
    pipeline.estimate_hidden_constraint(np.array([1.0, 2.0]), 5.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "experiment_1831_cocom.json")
        pipeline.write_artifact(filepath)
        
        assert os.path.exists(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
            
        assert data["experiment_id"] == "1831"
        assert data["status"] == "complete"
        assert data["honest_verdict"] == "cocom_implemented"
        assert data["learning_rate"] == 0.1
        assert data["memory_size"] == 5
        assert data["oracle_weights"] is not None

def test_cocom_zero_violation_guarantee():
    """Test zero-constraint violation guarantee with safety margin correction.
    
    Spec: REQ-LEARN-1832, SCENARIO-LEARN-1832
    """
    pipeline = COCOMPipeline(learning_rate=0.1, memory_size=5, parameter_dim=2)
    obj_grad = np.array([0.0, 0.0])  # No objective gradient to isolate correction
    const_grad = np.array([1.0, 0.0])
    
    # constraint_value > safety_margin triggers correction
    pipeline.update(obj_grad, const_grad, constraint_value=0.5, safety_margin=0.1)
    
    # The correction added to v should be (0.5 - 0.1) * [1.0, 0.0] = [0.4, 0.0]
    # Parameters updated by: param - lr * v = [0.0, 0.0] - 0.1 * [0.4, 0.0] = [-0.04, 0.0]
    assert np.allclose(pipeline.parameters, np.array([-0.04, 0.0]))

def test_cocom_no_violation():
    """Test no correction when constraint is within safety margin."""
    pipeline = COCOMPipeline(learning_rate=0.1, memory_size=5, parameter_dim=2)
    obj_grad = np.array([0.5, 0.5])
    const_grad = np.array([1.0, 0.0])
    
    # Copy parameters before update
    params_before = np.copy(pipeline.parameters)
    
    # constraint_value <= safety_margin triggers NO correction
    pipeline.update(obj_grad, const_grad, constraint_value=0.1, safety_margin=0.1)
    
    # The normal objective gradient [0.5, 0.5] is projected onto null space of const_grad
    # const_grad is [1, 0]. Null space projection of [0.5, 0.5] onto [1, 0] leaves [0.0, 0.5]
    # Update is param - lr * v = [0.0, 0.0] - 0.1 * [0.0, 0.5] = [0.0, -0.05]
    assert np.allclose(pipeline.parameters, np.array([0.0, -0.05]))

def test_cocom_estimate_hidden_constraint():
    """Test online regression oracle for estimating hidden constraints.
    
    Spec: REQ-PIPELINE-1833, SCENARIO-PIPELINE-1833
    """
    pipeline = COCOMPipeline(learning_rate=0.1, memory_size=5, parameter_dim=2)
    features = np.array([1.0, 2.0])
    
    # Initially prediction is 0.0
    pred = pipeline.predict_hidden_constraint(features)
    assert pred == 0.0
    
    # Train oracle
    true_constraint = 5.0
    pipeline.estimate_hidden_constraint(features, true_constraint)
    
    # Check if prediction moves towards true_constraint
    pred_after = pipeline.predict_hidden_constraint(features)
    assert np.allclose(pred_after, 2.5)

def test_cocom_update_with_epsilon():
    """Test update with hard epsilon updates.
    
    Spec: REQ-PIPELINE-1843, SCENARIO-PIPELINE-1843
    """
    pipeline = COCOMPipeline(learning_rate=0.1, memory_size=5, parameter_dim=2)
    obj_grad = np.array([0.0, 0.0])  # Zero obj gradient to isolate epsilon effect
    const_grad = np.array([1.0, 0.0])
    epsilon = 0.5
    
    # Run the epsilon update
    pipeline.update_with_epsilon(obj_grad, const_grad, epsilon)
    
    # Expected update: v = obj_grad_proj + epsilon * (const_grad / norm)
    # v = [0.0, 0.0] + 0.5 * [1.0, 0.0] = [0.5, 0.0]
    # param = param - lr * v = [0.0, 0.0] - 0.1 * [0.5, 0.0] = [-0.05, 0.0]
    assert np.allclose(pipeline.parameters, np.array([-0.05, 0.0]))


