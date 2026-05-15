"""Tests for the Z1 DTM stub (Exp 2112).

Spec: REQ-SAMPLE-066, SCENARIO-SAMPLE-094
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.samplers.dtm_stub import DtmStub


# ---------------------------------------------------------------------------
# REQ-SAMPLE-066-2 — backend name
# ---------------------------------------------------------------------------


def test_backend_name():
    """REQ-SAMPLE-066-2: backend_name must be 'dtm-stub-z1'."""
    stub = DtmStub()
    assert stub.backend_name == "dtm-stub-z1"


# ---------------------------------------------------------------------------
# REQ-SAMPLE-066-1 — sample_thermodynamic output shape and dtype
# ---------------------------------------------------------------------------


def test_sample_thermodynamic_shape():
    """REQ-SAMPLE-066-1, SCENARIO-SAMPLE-094: output shape matches input."""
    stub = DtmStub(seed=0)
    n_samples, n_spins = 4, 8
    noisy = np.random.default_rng(42).uniform(0.0, 1.0, (n_samples, n_spins)).astype(np.float32)
    out = stub.sample_thermodynamic(noisy, beta=1.0, n_denoising_steps=5)
    assert out.shape == (n_samples, n_spins)


def test_sample_thermodynamic_dtype():
    """REQ-SAMPLE-066-1: output dtype is float32."""
    stub = DtmStub(seed=1)
    noisy = np.ones((3, 6), dtype=np.float32) * 0.5
    out = stub.sample_thermodynamic(noisy, beta=1.0)
    assert out.dtype == np.float32


def test_sample_thermodynamic_values_in_range():
    """REQ-SAMPLE-066-1, SCENARIO-SAMPLE-094: output values in [0, 1]."""
    stub = DtmStub(seed=2)
    noisy = np.random.default_rng(7).uniform(0.0, 1.0, (8, 16)).astype(np.float32)
    out = stub.sample_thermodynamic(noisy, beta=2.0, n_denoising_steps=10)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_sample_thermodynamic_continuous_not_boolean():
    """REQ-SAMPLE-066-1: thermodynamic output is continuous, not boolean."""
    stub = DtmStub(seed=3)
    noisy = np.random.default_rng(99).uniform(0.0, 1.0, (16, 32)).astype(np.float32)
    out = stub.sample_thermodynamic(noisy, beta=1.0, n_denoising_steps=20)
    # At least some values should be strictly between 0 and 1
    strictly_interior = np.logical_and(out > 0.0, out < 1.0)
    assert strictly_interior.any(), "Expected continuous (non-boolean) output"


def test_sample_thermodynamic_reproducible():
    """REQ-SAMPLE-066-1: same seed + input yields identical output."""
    noisy = np.random.default_rng(5).uniform(0.0, 1.0, (4, 8)).astype(np.float32)
    out_a = DtmStub(seed=10).sample_thermodynamic(noisy, beta=1.5, n_denoising_steps=5)
    out_b = DtmStub(seed=10).sample_thermodynamic(noisy, beta=1.5, n_denoising_steps=5)
    np.testing.assert_array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-066-3 — discrete SamplerBackend protocol compatibility
# ---------------------------------------------------------------------------


def test_sample_returns_boolean():
    """REQ-SAMPLE-066-3: sample() output must be boolean array."""
    stub = DtmStub(seed=20)
    n_spins = 6
    biases = np.zeros(n_spins)
    couplings = np.zeros((n_spins, n_spins))
    out = stub.sample(biases, couplings, n_samples=4, config={"beta": 1.0, "steps": 5})
    assert out.dtype == bool
    assert out.shape == (4, n_spins)


def test_minimize_energy_returns_boolean():
    """REQ-SAMPLE-066-3: minimize_energy() output must be boolean array."""
    stub = DtmStub(seed=21)
    n_spins = 4
    biases = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    couplings = np.zeros((n_spins, n_spins), dtype=np.float32)
    out = stub.minimize_energy(biases, couplings, n_samples=3, n_steps=5, beta=2.0)
    assert out.dtype == bool
    assert out.shape == (3, n_spins)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-066-4 — no hardware execution attributes
# ---------------------------------------------------------------------------


def test_no_hardware_attributes():
    """REQ-SAMPLE-066-4: stub has no hardware execution attributes."""
    stub = DtmStub()
    # These attributes must not exist on the simulator stub — their presence
    # would imply a hardware execution claim, which is forbidden for CPU-only runs.
    assert not hasattr(stub, "hardware_execution_performed")
    assert not hasattr(stub, "authenticated_access_proof")
