"""Tests for OnlineUpdater and VerificationLoop.

Spec: REQ-LEARN-101
"""

import numpy as np
import pytest
from unittest import mock

from carnot.models.cikan_verifier import CIKAN
from carnot.training.online_updater import OnlineUpdater, DeepSaDeUpdater, z3_available
from carnot.pipeline.verification_loop import VerificationLoop, Violation

def test_online_updater_sgd():
    """Test online SGD updates."""
    cikan = CIKAN(feature_names=["f1", "f2"], seed=42)
    updater = OnlineUpdater(optimizer="sgd", learning_rate=0.1)
    
    # Check initial energy
    x = [0.5, 0.5]
    initial_energy = cikan.energy(x)
    
    # Step with y=1 (should decrease energy)
    loss1 = updater.step(cikan, x, 1.0)
    energy_after_1 = cikan.energy(x)
    assert energy_after_1 < initial_energy
    
    # Step with y=0 (should increase energy)
    loss2 = updater.step(cikan, x, 0.0)
    energy_after_2 = cikan.energy(x)
    assert energy_after_2 > energy_after_1

def test_online_updater_adamw():
    """Test online AdamW updates."""
    cikan = CIKAN(feature_names=["f1", "f2"], seed=42)
    updater = OnlineUpdater(optimizer="adamw", learning_rate=0.1)
    
    x = [0.5, 0.5]
    initial_energy = cikan.energy(x)
    
    updater.step(cikan, x, 1.0)
    assert cikan.energy(x) < initial_energy

def test_online_updater_invalid_optimizer():
    """Test invalid optimizer name raises error."""
    with pytest.raises(ValueError, match="Unknown optimizer"):
        OnlineUpdater(optimizer="invalid")

def test_verification_loop():
    """Test the verification loop correctly triggers the updater."""
    cikan = CIKAN(feature_names=["f1", "f2"], seed=42)
    updater = OnlineUpdater(optimizer="adamw", learning_rate=0.1)
    loop = VerificationLoop(cikan, updater)
    
    stream = [
        Violation(features=[0.5, 0.5], label=0.0),
        Violation(features=[0.1, 0.9], label=1.0),
    ]
    
    assert loop.n_processed == 0
    assert loop.n_updated == 0
    
    loop.run(stream)
    
    assert loop.n_processed == 2
    assert loop.n_updated == 2

def test_deepsade_updater_valid_update():
    """Test DeepSaDeUpdater accepts a valid update within constraints."""
    cikan = CIKAN(feature_names=["f1", "f2"], seed=42)
    # Set bound high so the update passes
    updater = DeepSaDeUpdater(optimizer="sgd", learning_rate=0.1, constraint_bound=1.0)
    
    old_bias = cikan.bias
    old_ctrl = cikan.residual_control_points.copy()
    
    x = [0.5, 0.5]
    updater.step(cikan, x, 1.0)
    
    # Check that update was applied (weights changed)
    assert cikan.bias != old_bias or not np.allclose(cikan.residual_control_points, old_ctrl)

def test_deepsade_updater_invalid_update_rollback():
    """Test DeepSaDeUpdater rejects an invalid update and rolls back."""
    cikan = CIKAN(feature_names=["f1", "f2"], seed=42)
    # Give it an impossible constraint bound to force a MaxSMT violation/rollback
    updater = DeepSaDeUpdater(optimizer="sgd", learning_rate=0.1, constraint_bound=0.0)
    
    # We first artificially bump the weights so they violate constraint bound 0.0
    cikan.residual_control_points[:] = 0.5
    
    old_bias = cikan.bias
    old_ctrl = cikan.residual_control_points.copy()
    
    x = [0.5, 0.5]
    updater.step(cikan, x, 1.0)
    
    # Check that rollback occurred (weights stayed exactly the same as prior to step, ignoring the proposed SGD update)
    assert cikan.bias == old_bias
    assert np.allclose(cikan.residual_control_points, old_ctrl)

@mock.patch("carnot.training.online_updater.z3_available", True)
def test_deepsade_updater_z3_mock():
    """Test DeepSaDeUpdater with z3 logic mocked."""
    cikan = CIKAN(feature_names=["f1"], seed=42)
    updater = DeepSaDeUpdater(optimizer="sgd", learning_rate=0.01, constraint_bound=0.95)
    
    with mock.patch("carnot.training.online_updater.z3") as mock_z3:
        mock_solver = mock.MagicMock()
        # Mock sat response
        mock_solver.check.return_value = mock_z3.sat
        mock_z3.Optimize.return_value = mock_solver
        
        mock_real = mock.MagicMock()
        mock_real.__le__.return_value = True
        mock_real.__ge__.return_value = True
        mock_real.__eq__.return_value = True
        mock_z3.Real.return_value = mock_real
        
        mock_z3.sat = mock_z3.sat
        
        old_bias = cikan.bias
        updater.step(cikan, [0.5], 1.0)
        
        mock_solver.check.assert_called_once()
        assert cikan.bias != old_bias  # Update applied

@mock.patch("carnot.training.online_updater.z3_available", True)
def test_deepsade_updater_z3_mock_rollback():
    """Test DeepSaDeUpdater z3 rollback on unsat."""
    cikan = CIKAN(feature_names=["f1"], seed=42)
    updater = DeepSaDeUpdater(optimizer="sgd", learning_rate=0.1, constraint_bound=0.95)
    
    with mock.patch("carnot.training.online_updater.z3") as mock_z3:
        mock_solver = mock.MagicMock()
        # Mock unsat response
        mock_solver.check.return_value = mock_z3.unsat
        mock_z3.Optimize.return_value = mock_solver
        
        mock_real = mock.MagicMock()
        mock_real.__le__.return_value = True
        mock_real.__ge__.return_value = True
        mock_real.__eq__.return_value = True
        mock_z3.Real.return_value = mock_real
        
        mock_z3.sat = "sat"
        
        old_bias = cikan.bias
        old_ctrl = cikan.residual_control_points.copy()
        updater.step(cikan, [0.5], 1.0)
        
        mock_solver.check.assert_called_once()
        assert cikan.bias == old_bias
        assert np.allclose(cikan.residual_control_points, old_ctrl)
