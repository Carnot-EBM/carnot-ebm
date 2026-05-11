"""Tests for Symbolic-KAN integration into ThreeTierPipeline.
Spec: REQ-SYMKAN-1751, SCENARIO-SYMKAN-1751.
"""

from unittest.mock import MagicMock

import pytest

from carnot.pipeline.symbolic_kan_tier3 import SymbolicKANTier3, step_to_features
from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline


def test_symbolic_kan_tier3_call():
    """Test SymbolicKANTier3.__call__ computes features and checks threshold.
    Spec: REQ-SYMKAN-1751
    """
    mock_model = MagicMock()
    mock_model.energy.return_value = -0.5
    tier3 = SymbolicKANTier3(mock_model, threshold=0.0)
    
    verified, energy = tier3("Step 1: 3 + 5 = 8", "What is 3+5?")
    assert verified is True
    assert energy == -0.5
    
    # Test failure case
    mock_model.energy.return_value = 1.0
    verified, energy = tier3("Step 1: 3 + 5 = 9", "What is 3+5?")
    assert verified is False
    assert energy == 1.0


def test_three_tier_pipeline_integration():
    """Test ThreeTierPipeline routes to SymbolicKANTier3 when wired as ising_pipeline.
    Spec: SCENARIO-SYMKAN-1751
    """
    mock_model = MagicMock()
    mock_model.energy.return_value = -1.2
    tier3 = SymbolicKANTier3(mock_model, threshold=0.0)
    
    mock_sink = MagicMock()
    mock_eorm = MagicMock()
    
    pipeline = ThreeTierPipeline(
        sink_probe=mock_sink,
        eorm_model=mock_eorm,
        ising_pipeline=tier3
    )
    
    # Bypass early tiers to ensure it hits Tier 3 (Ising)
    # EORM threshold 0.5, so if EORM returns 1.0, it goes to Tier 3.
    # SinkProbe is bypassed when attention_matrix is None.
    pipeline.eorm_threshold = 0.5
    mock_eorm.energy.return_value = 1.0 
    
    verified, tier_used, energy = pipeline.verify("Step 1: 3 + 5 = 8", question="What is 3+5?", attention_matrix=None)
    
    assert verified is True
    assert energy == -1.2
    assert tier_used == "ising"
    mock_model.energy.assert_called_once()
