import pytest
from carnot.phase3.exp3575_discriminating_value import compute_conditional_lift

def test_compute_conditional_lift_empty():
    """SCENARIO-3575: Return zeros and p=1.0 when no errors are passed."""
    res = compute_conditional_lift([], [])
    assert res["n_errors"] == 0
    assert res["baseline_fraction"] == 0.0
    assert res["ensemble_fraction"] == 0.0
    assert res["conditional_catch_rate"] == 0.0
    assert res["mcnemar_p"] == 1.0

def test_compute_conditional_lift_mismatched_length():
    """SCENARIO-3575: Raise ValueError if lengths mismatch."""
    with pytest.raises(ValueError, match="Inputs must have the same length"):
        compute_conditional_lift([True], [])

def test_compute_conditional_lift_no_misses():
    """SCENARIO-3575: When baseline misses nothing, conditional catch rate is 0.0."""
    res = compute_conditional_lift([True, True], [True, False])
    assert res["n_errors"] == 2
    assert res["baseline_fraction"] == 1.0
    assert res["conditional_catch_rate"] == 0.0

def test_compute_conditional_lift_basic():
    """SCENARIO-3575: Calculate lift correctly."""
    # Baseline catches 2/4. Misses 2.
    # Ensemble catches 3/4. Out of the 2 baseline misses (last two), ensemble catches 1.
    baseline = [True, True, False, False]
    ensemble = [True, False, True, False]
    res = compute_conditional_lift(baseline, ensemble)
    
    assert res["n_errors"] == 4
    assert res["baseline_fraction"] == 0.5
    assert res["ensemble_fraction"] == 0.5
    # conditional catch rate: baseline missed index 2 and 3. Ensemble caught index 2.
    # So 1 / 2 = 0.5
    assert res["conditional_catch_rate"] == 0.5
    # mcnemar uses discordance.
    # baseline=True, ensemble=False -> 1
    # baseline=False, ensemble=True -> 1
    # Tied! p should be 1.0
    assert res["mcnemar_p"] == 1.0

def test_compute_conditional_lift_significant():
    """SCENARIO-3575: Ensure McNemar p < 0.05 when highly lopsided."""
    # Baseline misses all 10. Ensemble catches 10.
    baseline = [False] * 10
    ensemble = [True] * 10
    res = compute_conditional_lift(baseline, ensemble)
    assert res["conditional_catch_rate"] == 1.0
    assert res["mcnemar_p"] < 0.05
