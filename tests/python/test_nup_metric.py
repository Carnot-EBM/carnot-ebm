import pytest
import numpy as np
from carnot.pipeline.nup_metric import NUPMetric

def test_nup_metric_initialization():
    # REQ-VERIFY-3405
    metric = NUPMetric(threshold=0.5)
    assert metric.threshold == 0.5

def test_nup_metric_gradients_to_ising_energy():
    # REQ-VERIFY-3405
    metric = NUPMetric()
    grad1 = np.array([1.0, -1.0, 1.0])
    grad2 = np.array([1.0, 1.0, 1.0])
    grad3 = np.array([-1.0, -1.0, -1.0])
    
    energies = metric.gradients_to_ising_energy([grad1, grad2, grad3])
    assert len(energies) == 2
    # spin1: [1, -1, 1], spin2: [1, 1, 1]
    # dot: 1*1 + (-1)*1 + 1*1 = 1 - 1 + 1 = 1. energy = -1 / 3 = -0.333
    assert np.isclose(energies[0], -0.3333333333333333)
    # spin2: [1, 1, 1], spin3: [-1, -1, -1]
    # dot: -3. energy = -(-3) / 3 = 1.0
    assert np.isclose(energies[1], 1.0)

def test_nup_metric_detect_phase_transition():
    # SCENARIO-VERIFY-3405
    metric = NUPMetric(threshold=1.2)
    # From -0.33 to 1.0 is a shift of 1.33 >= 1.2
    assert metric.detect_phase_transition([-0.3333333, 1.0]) is True
    assert metric.detect_phase_transition([-0.3333333, 0.5]) is False

def test_nup_metric_evaluate():
    # SCENARIO-VERIFY-3405
    metric = NUPMetric(threshold=1.2)
    grad1 = np.array([1.0, -1.0, 1.0])
    grad2 = np.array([1.0, 1.0, 1.0])
    grad3 = np.array([-1.0, -1.0, -1.0])
    
    result = metric.evaluate([grad1, grad2, grad3])
    assert result["hallucination_detected"] is True
    assert len(result["energies"]) == 2
