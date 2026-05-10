"""Tests for OnlineUpdater and VerificationLoop.

Spec: REQ-LEARN-101
"""

import numpy as np
import pytest

from carnot.models.cikan_verifier import CIKAN
from carnot.training.online_updater import OnlineUpdater
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
