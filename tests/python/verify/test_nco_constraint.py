"""
Tests for NCO constraint module.
"""
from carnot.verify.nco_constraint import compute_nco_rejection_rate

def test_compute_nco_rejection_rate():
    """Test standard rejection rate computation."""
    logprobs = [-1.0, -5.0, -11.0, -15.0]
    rate = compute_nco_rejection_rate(logprobs, threshold=-10.0)
    assert rate == 0.5  # 2 out of 4 rejected
    
def test_compute_nco_rejection_rate_empty():
    """Test behavior with empty logprobs list."""
    assert compute_nco_rejection_rate([], threshold=-10.0) == 0.0
