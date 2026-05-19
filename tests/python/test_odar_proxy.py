"""Tests for ODAR free energy proxy.

Spec: REQ-TIER0-008-1
"""

import numpy as np
from carnot.pipeline.odar_proxy import compute_odar_energy_proxy


def test_compute_odar_energy_proxy_empty():
    """Test proxy with empty input."""
    energy, surprise, complexity = compute_odar_energy_proxy([])
    assert energy == 0.0
    assert surprise == 0.0
    assert complexity == 0.0


def test_compute_odar_energy_proxy_values():
    """Test proxy with values."""
    logprobs = [-1.0, -2.0, -3.0]
    energy, surprise, complexity = compute_odar_energy_proxy(logprobs)

    arr = np.array(logprobs)
    expected_surprise = -np.mean(arr)  # 2.0
    expected_complexity = np.var(arr)  # 0.6666...
    expected_energy = expected_surprise + 0.5 * expected_complexity

    np.testing.assert_allclose(surprise, expected_surprise)
    np.testing.assert_allclose(complexity, expected_complexity)
    np.testing.assert_allclose(energy, expected_energy)
