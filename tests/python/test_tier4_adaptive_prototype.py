import pytest
from carnot.models.tier4_adaptive_prototype import SimpleAdaptiveKAN, detect_new_pattern, adapt_structure

def test_simple_adaptive_kan_initial_energy():
    kan = SimpleAdaptiveKAN()
    assert kan.num_knots == 3
    # With x=0.0 and knot at 0.0, energy should be roughly 1.0 (from that knot) + tiny bit from others
    energy = kan.energy(0.0)
    assert energy > 0.0

def test_detect_new_pattern():
    # Less than or equal to 3 occurrences: shouldn't trigger
    history_no_trigger = [0.5, 0.5, 0.5]
    assert len(detect_new_pattern(history_no_trigger)) == 0

    # More than 3 occurrences: should trigger
    history_trigger = [0.5, 0.5, 0.5, 0.5]
    assert 0.5 in detect_new_pattern(history_trigger)

def test_adapt_structure():
    kan = SimpleAdaptiveKAN()
    trigger_examples = [0.5, 0.5, 0.5, 0.5]
    
    before_energy = kan.energy(0.5)
    
    b_e, a_e = adapt_structure(kan, 0.5, trigger_examples)
    
    assert kan.num_knots == 4
    assert a_e < b_e
    assert kan.energy(0.5) < before_energy
