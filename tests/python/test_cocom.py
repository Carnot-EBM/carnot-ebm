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
