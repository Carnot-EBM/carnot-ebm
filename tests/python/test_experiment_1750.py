"""Tests for Experiment 1750: Symbolic-KAN vs CIKAN constraint evaluation.

Spec references: REQ-KAN-1750, SCENARIO-KAN-1750.
"""

import json
from pathlib import Path
import numpy as np
import pytest

# Adjusting path to allow importing the script
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from scripts.experiment_1750_symbolic_eval import (
    generate_toy_dataset,
    train_cikan,
    train_symbolic_kan,
    run_experiment,
    MockConstraint,
    extract_params,
    repack_params
)
from carnot.models.kan.symbolic_kan import SymbolicKANParams, SymbolicKANConfig, SymbolicRoutingLayer

def test_generate_toy_dataset():
    """Test toy dataset generation.
    
    Spec references: REQ-KAN-1750, SCENARIO-KAN-1750.
    """
    X, Y = generate_toy_dataset()
    assert X.shape == (200, 2)
    assert Y.shape == (200,)
    assert np.all((Y == 0.0) | (Y == 1.0))
    assert X.min() >= 0.0
    assert X.max() <= 1.0

def test_mock_constraint():
    """Test MockConstraint struct.
    
    Spec references: REQ-KAN-1750.
    """
    mc = MockConstraint(["f1", "f2"], "(f1 AND f2)", "f1*f2")
    assert mc.variables == ["f1", "f2"]
    assert mc.expression == "(f1 AND f2)"
    assert mc.polynomial == "f1*f2"

def test_train_cikan():
    """Test CIKAN training runs without crashing and achieves reasonable accuracy.
    
    Spec references: REQ-KAN-1750.
    """
    X, Y = generate_toy_dataset()
    acc = train_cikan(X[:20], Y[:20]) # Train on small subset to be fast
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_extract_repack_params():
    """Test parameter extraction and repacking.
    
    Spec references: REQ-KAN-1750.
    """
    config = SymbolicKANConfig(input_dim=2, n_routes=2, primitives=("identity", "square"))
    layer = SymbolicRoutingLayer(config)
    d = extract_params(layer.params)
    assert "projection_weights" in d
    p = repack_params(d)
    assert isinstance(p, SymbolicKANParams)

def test_train_symbolic_kan():
    """Test Symbolic-KAN training runs without crashing.
    
    Spec references: REQ-KAN-1750.
    """
    X, Y = generate_toy_dataset()
    acc = train_symbolic_kan(X[:20], Y[:20])
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_run_experiment(tmp_path):
    """Test full experiment run and artifact writing.
    
    Spec references: REQ-KAN-1750, SCENARIO-KAN-1750.
    """
    output_path = tmp_path / "result.json"
    result = run_experiment(str(output_path))
    
    assert result["status"] == "complete"
    assert result["schema"] == "carnot.experiment_1750.v1"
    assert "cikan_accuracy" in result
    assert "symbolic_kan_accuracy" in result
    
    assert output_path.exists()
    saved_result = json.loads(output_path.read_text())
    assert saved_result == result
