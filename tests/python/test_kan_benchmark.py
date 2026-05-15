"""Tests for KAN benchmark accounting.

Traces to:
- REQ-KAN-1782
- SCENARIO-KAN-1782
"""

from carnot.hardware.kan_benchmark import compute_bops, compute_nabs

def test_compute_bops():
    """Test BOPs calculation."""
    assert compute_bops(num_points=64, num_edges=1) == 512
    assert compute_bops(num_points=256, num_edges=1) == 2048

def test_compute_nabs():
    """Test NABS calculation."""
    assert compute_nabs(num_points=64, num_edges=1) == 256
    assert compute_nabs(num_points=256, num_edges=1) == 1024
