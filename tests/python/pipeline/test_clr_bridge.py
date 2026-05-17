"""Tests for the CLR Verifier Bridge.

References: REQ-VERIFY-2139, SCENARIO-VERIFY-2139.
"""

import json
import os

import pytest
import numpy as np

from carnot.pipeline.clr_bridge import CLRVerifierBridge

def test_clr_bridge_maps_vectors_and_saves_json(tmp_path):
    """Test that CLR bridge maps latent vectors and saves JSON.
    
    Validates SCENARIO-VERIFY-2139.
    """
    json_path = tmp_path / "experiment_2139_clr_bridge.json"
    
    bridge = CLRVerifierBridge(output_path=str(json_path))
    
    # Mock some continuous EBM vectors
    ebm_vectors = np.array([[0.1, 0.9, -0.5], [0.8, -0.2, 0.4]])
    
    discrete_logic = bridge.map_to_discrete(ebm_vectors)
    
    assert len(discrete_logic) == 2
    assert discrete_logic[0] == [True, True, False]
    assert discrete_logic[1] == [True, False, True]
    
    bridge.save_results()
    
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
        
    assert data["status"] == "complete"
    assert data["honest_verdict"] == "success_mapped_vectors"
    assert "discrete_logic_formats" in data
    assert data["discrete_logic_formats"] == [[True, True, False], [True, False, True]]
