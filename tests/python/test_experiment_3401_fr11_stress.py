"""Tests for the FR-11 stress test (Experiment 3401).

Spec: REQ-LEARN-3401, SCENARIO-LEARN-3401
"""

import json
from pathlib import Path
from unittest.mock import patch

from scripts import experiment_3401_fr11_stress

def test_experiment_3401_fr11_stress_produces_deliverable(tmp_path):
    """Test that the script runs and produces the required deliverable.
    
    Spec: SCENARIO-LEARN-3401
    """
    deliverable_path = tmp_path / "experiment_3401_fr11_stress.json"
    
    with patch.object(experiment_3401_fr11_stress, "DELIVERABLE_PATH", deliverable_path):
        # We also want to reduce interactions to speed up the test, but the spec says 1000.
        # We can either mock the loop count or run it fast if it's lightweight.
        with patch.object(experiment_3401_fr11_stress, "INTERACTIONS", 10):
            experiment_3401_fr11_stress.main()
            
    assert deliverable_path.exists()
    
    with open(deliverable_path) as f:
        data = json.load(f)
        
    assert "honest_verdict" in data
    assert "fidelity" in data
    assert "interactions" in data
    assert data["interactions"] == 10

def test_experiment_3401_fr11_stress_cas_hopfield_used(tmp_path):
    """Test that CAS and Hopfield methods are used in the main loop.
    
    Spec: REQ-LEARN-3401
    """
    deliverable_path = tmp_path / "experiment_3401_fr11_stress.json"
    
    with patch("scripts.experiment_3401_fr11_stress.CASConstraintUpdater") as mock_cas, \
         patch("scripts.experiment_3401_fr11_stress.EBMCoTCalibratorV3") as mock_ebm:
             
        with patch.object(experiment_3401_fr11_stress, "DELIVERABLE_PATH", deliverable_path):
            with patch.object(experiment_3401_fr11_stress, "INTERACTIONS", 1):
                experiment_3401_fr11_stress.main()
                
        assert mock_cas.called
        assert mock_ebm.called
