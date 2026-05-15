"""Tests for KANelE Look-Up Table (LUT) evaluations (REQ-KAN-1781)."""

import json
from pathlib import Path

import pytest
from carnot.hardware.kan_lut import convert_kan_to_lut


def test_kan_to_lut_conversion() -> None:
    """
    Test the KAN to LUT conversion.
    SCENARIO-KAN-1781: Transform a small KAN tier to LUT format.
    """
    # Mock a small KAN tier edge function: f(x) = x^2
    def mock_edge_fn(x: float) -> float:
        return x**2

    # Provide simple domain and points
    lut = convert_kan_to_lut(mock_edge_fn, domain=(-1.0, 1.0), num_points=5)
    
    # We expect 5 points from -1.0 to 1.0: -1.0, -0.5, 0.0, 0.5, 1.0
    # Expected LUT values: 1.0, 0.25, 0.0, 0.25, 1.0
    assert len(lut) == 5
    assert pytest.approx(lut[0], 0.01) == 1.0
    assert pytest.approx(lut[1], 0.01) == 0.25
    assert pytest.approx(lut[2], 0.01) == 0.0
    assert pytest.approx(lut[3], 0.01) == 0.25
    assert pytest.approx(lut[4], 0.01) == 1.0


def test_kan_to_lut_conversion_invalid_points() -> None:
    """Test that num_points < 2 raises ValueError."""
    def mock_edge_fn(x: float) -> float:
        return x**2

    with pytest.raises(ValueError, match="num_points must be at least 2."):
        convert_kan_to_lut(mock_edge_fn, domain=(-1.0, 1.0), num_points=1)

