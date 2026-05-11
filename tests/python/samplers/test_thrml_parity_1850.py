"""Tests for experiment 1850 THRML parity.

Spec traces: REQ-SAMPLE-1850
"""
from carnot.samplers.thrml_parity_1850 import run_parity_n128, kl_divergence
import numpy as np

def test_kl_divergence():
    """Test KL divergence calculation."""
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert kl_divergence(p, q) < 1e-6
    
def test_run_parity_n128():
    """Test the n128 parity function with minimal samples."""
    result = run_parity_n128(seed=42, n_samples=2)
    assert "schema" in result
    assert result["n_spins"] == 128
    assert result["n_samples"] == 2
    assert "mean_energy_delta_abs" in result
    assert "kl_divergence" in result
    assert "ks_p_value" in result
    assert "acceptance_gate_passed" in result
def test_run_parity_n128_failed(monkeypatch):
    """Test the n128 parity function with mocked metrics to force a failure and cover lines."""
    import carnot.samplers.thrml_parity_1850 as mod
    
    # Mock to make energies identical to cover the `min_e == max_e` line
    # but then manipulate metrics to fail the gate to cover `verdict = "complete: ..."`
    original_histogram = np.histogram
    
    def mock_histogram(a, bins, density):
        return np.array([1.0, 0.0]), bins
        
    monkeypatch.setattr(np, "histogram", mock_histogram)
    
    # Actually just pass very few samples, but to ensure min_e == max_e we can mock energies
    def mock_run_parity():
        result = mod.run_parity_n128(seed=42, n_samples=1)
        return result
        
    # We can also just mock the backend sample to return all 0s
    class MockBackend:
        def __init__(self, seed): pass
        def sample(self, b, J, n, schedule): return np.zeros((n, 128))
    monkeypatch.setattr(mod, "CpuBackend", MockBackend)
    monkeypatch.setattr(mod, "ThrmlSamplerBackend", MockBackend)
    
    result = mod.run_parity_n128(seed=42, n_samples=1)
    
    # It might pass or fail, if it passes we need to ensure we have a test that fails.
    # To definitely fail, let's mock kl_divergence
    monkeypatch.setattr(mod, "kl_divergence", lambda p, q: 1.0) # > 0.05
    result2 = mod.run_parity_n128(seed=42, n_samples=1)
    assert result2["acceptance_gate_passed"] is False
    assert result2["honest_verdict"].startswith("complete:")
