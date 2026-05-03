"""Tests for the THRML/Extropic sampler backend stub.

Spec refs: REQ-SAMPLE-040, SCENARIO-SAMPLE-066, SCENARIO-SAMPLE-067.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.samplers.backend import SamplerBackend
from carnot.samplers.thrml_backend import ThrmlSamplerBackend


def _ferromagnetic_problem(n_spins: int = 6) -> tuple[np.ndarray, np.ndarray]:
    biases = np.ones(n_spins, dtype=np.float32)
    couplings = np.ones((n_spins, n_spins), dtype=np.float32) * 0.25
    np.fill_diagonal(couplings, 0.0)
    return biases, couplings


def test_thrml_backend_conforms_to_sampler_protocol() -> None:
    """REQ-SAMPLE-040: ThrmlSamplerBackend satisfies SamplerBackend."""
    assert isinstance(ThrmlSamplerBackend(seed=0), SamplerBackend)


def test_thrml_backend_reports_cpu_fallback_when_device_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SAMPLE-066: unset CARNOT_TSU_DEVICE selects CPU Gibbs fallback."""
    monkeypatch.delenv("CARNOT_TSU_DEVICE", raising=False)

    backend = ThrmlSamplerBackend(seed=0)

    assert backend.backend_name == "thrml_cpu_fallback"
    assert backend.using_hardware is False


def test_thrml_backend_sample_routes_to_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-SAMPLE-066: sample returns CPU Gibbs fallback samples."""
    monkeypatch.delenv("CARNOT_TSU_DEVICE", raising=False)
    biases, couplings = _ferromagnetic_problem()

    samples = ThrmlSamplerBackend(seed=0).sample(
        biases,
        couplings,
        n_samples=4,
        config={"beta": 2.0, "n_warmup": 5, "steps_per_sample": 1},
    )

    assert samples.shape == (4, 6)
    assert samples.dtype == bool


def test_thrml_backend_minimize_routes_to_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-SAMPLE-066: minimize_energy returns CPU fallback samples."""
    monkeypatch.delenv("CARNOT_TSU_DEVICE", raising=False)
    biases, couplings = _ferromagnetic_problem()

    samples = ThrmlSamplerBackend(seed=0).minimize_energy(
        biases,
        couplings,
        n_samples=3,
        n_steps=5,
        beta=2.0,
    )

    assert samples.shape == (3, 6)
    assert samples.dtype == bool


def test_thrml_backend_hardware_path_raises_until_sdk_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SAMPLE-067: requested TSU hardware path raises NotImplementedError."""
    monkeypatch.setenv("CARNOT_TSU_DEVICE", "z1-devkit")
    biases, couplings = _ferromagnetic_problem()
    backend = ThrmlSamplerBackend(seed=0)

    assert backend.backend_name == "thrml_hardware:z1-devkit"
    assert backend.using_hardware is True
    with pytest.raises(NotImplementedError, match="Extropic TSU hardware"):
        backend.sample(biases, couplings, n_samples=2, config={"beta": 1.0})
    with pytest.raises(NotImplementedError, match="Extropic TSU hardware"):
        backend.minimize_energy(biases, couplings, n_samples=2, n_steps=5, beta=1.0)
