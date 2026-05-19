import numpy as np
import pytest
from carnot.samplers.gatemate_onboard_sampler import (
    compute_analytic_distribution,
    get_empirical_distribution,
    compute_kl_divergence,
    run_gatemate_sampler,
    generate_gatemate_artifact
)

def test_analytic_distribution():
    h = np.zeros(2)
    J = np.zeros((2, 2))
    probs = compute_analytic_distribution(h, J)
    assert len(probs) == 4
    np.testing.assert_allclose(probs, np.array([0.25, 0.25, 0.25, 0.25]))

def test_empirical_distribution():
    samples = np.array([[-1, -1], [1, -1], [-1, -1], [1, 1]])
    probs = get_empirical_distribution(samples, 2)
    np.testing.assert_allclose(probs, np.array([0.5, 0.25, 0.0, 0.25]))

def test_kl_divergence():
    emp = np.array([0.5, 0.5])
    ana = np.array([0.5, 0.5])
    kl = compute_kl_divergence(emp, ana)
    assert np.isclose(kl, 0.0, atol=1e-5)

def test_run_gatemate_sampler():
    h = np.zeros(4)
    J = np.zeros((4, 4))
    samples, rate, duration = run_gatemate_sampler(
        n_samples=100, n_spins=4, h=h, J=J, seed=42
    )
    # Testing that it returns the expected shapes and bounds
    assert samples.shape == (100, 4)
    assert duration >= 10.0
    assert rate == 100 / duration

def test_generate_gatemate_artifact():
    artifact = generate_gatemate_artifact()
    assert artifact["gatemate_onboard_sampler_validated"] is True
    assert artifact["n_samples"] >= 10000
    assert artifact["duration_s"] >= 10.0
    assert artifact["sample_rate_hz"] > 0
    assert artifact["kl_divergence_vs_analytic"] >= 0.0
    assert artifact["random_seed"] == 42
    assert "thermal_note" in artifact
    assert "reproducibility_checksum" in artifact
