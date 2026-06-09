import pytest
import numpy as np
from carnot.agentic.arc_pinductor import PinductorModel
from carnot.agentic.arc_pinductor_candidates import get_candidates

def test_pinductor_positive_control():
    s0 = np.zeros((2, 2), dtype=int)
    s1 = np.ones((2, 2), dtype=int)
    a = (6, 0, 0)
    
    # Trajectory where next state depends on alternating step counter
    # state doesn't give information about phase.
    traj1 = [
        (s0, a, s0),
        (s0, a, s1),
        (s1, a, s1),
        (s1, a, s0)
    ]
    
    candidates = get_candidates()
    best_energy = 1.0
    for name, fn, K in candidates:
        p_model = PinductorModel("test_game", fn, K)
        p_model.fit([traj1])
        energy_info = p_model.consistency_energy([traj1])
        energy = energy_info.get("energy")
        if energy is None:
            energy = 1.0
        if energy < best_energy:
            best_energy = energy
            
    assert best_energy == 0.0

def test_pinductor_fallback():
    # Test fallback conditions in prediction
    def mock_latent(L, s, a):
        return 0
        
    p_model = PinductorModel("test_game", mock_latent, 1)
    s0 = np.zeros((2,2), dtype=int)
    a = (6,0,0)
    
    # Predict without fit should fallback to input state
    pred = p_model.predict_belief(s0, a, {0: 1.0})
    assert np.array_equal(pred, s0)
    
    # Empty trajectories
    p_model.fit([])
    energy_info = p_model.consistency_energy([])
    assert energy_info.get("energy") is None
