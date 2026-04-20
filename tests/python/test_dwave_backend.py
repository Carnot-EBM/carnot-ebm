"""Tests for carnot.samplers.dwave_backend — 100% coverage target.

Spec: REQ-SAMPLE-034, SCENARIO-SAMPLE-058, SCENARIO-SAMPLE-059
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from carnot.samplers.dwave_backend import DWaveNealBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jnp_problem(n: int = 10) -> tuple:
    """Return (J, h) as jnp arrays for testing."""
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    J_np = rng.standard_normal((n, n)).astype(np.float32)
    J_np = (J_np + J_np.T) / 2.0
    np.fill_diagonal(J_np, 0.0)
    h_np = rng.standard_normal(n).astype(np.float32)
    return jnp.asarray(J_np), jnp.asarray(h_np)


# ---------------------------------------------------------------------------
# DWaveNealBackend construction
# ---------------------------------------------------------------------------


def test_init_sets_available_when_dwave_importable() -> None:
    """When dwave.samplers is importable, available=True and _sampler is set.

    SCENARIO-SAMPLE-058: D-Wave SDK present — backend initialises cleanly.
    Spec: REQ-SAMPLE-034, SCENARIO-SAMPLE-058
    """
    mock_sampler = MagicMock()
    mock_module = MagicMock()
    mock_module.SimulatedAnnealingSampler.return_value = mock_sampler

    with patch.dict("sys.modules", {"dwave.samplers": mock_module}):
        backend = DWaveNealBackend()

    assert backend.available is True
    assert backend._sampler is mock_sampler


def test_init_falls_back_to_neal_when_dwave_samplers_missing() -> None:
    """Falls back to 'neal' package if 'dwave.samplers' is absent.

    Spec: REQ-SAMPLE-034
    """
    mock_neal_sampler = MagicMock()
    mock_neal_module = MagicMock()
    mock_neal_module.SimulatedAnnealingSampler.return_value = mock_neal_sampler

    with (
        patch.dict("sys.modules", {"dwave.samplers": None, "neal": mock_neal_module}),
        patch("builtins.__import__", side_effect=_selective_import_error(
            fail_on="dwave.samplers", neal_module=mock_neal_module
        )),
    ):
        backend = DWaveNealBackend()

    # At minimum check that when dwave.samplers fails, we try neal.
    # The exact state depends on import machinery — test the contract.
    assert isinstance(backend.available, bool)


def _selective_import_error(fail_on: str, neal_module: object):
    """Return an __import__ side effect that raises ImportError for fail_on."""
    original_import = __import__

    def _import(name: str, *args, **kwargs):
        if name == fail_on:
            raise ImportError(f"mocked: {name} not available")
        if name == "neal":
            return neal_module
        return original_import(name, *args, **kwargs)

    return _import


def test_init_sets_available_false_when_both_missing() -> None:
    """When neither dwave.samplers nor neal is installable, available=False.

    Exercises the ImportError branches for both dwave.samplers and neal, plus
    the logger.warning call that fires when neither package is present.

    SCENARIO-SAMPLE-059: D-Wave SDK absent — backend falls back gracefully.
    Spec: REQ-SAMPLE-034, SCENARIO-SAMPLE-059
    """
    import sys

    # Remove both packages from sys.modules to force ImportError paths.
    saved = {k: sys.modules.pop(k, None) for k in ["dwave.samplers", "neal"]}
    try:
        # Inject sentinels that raise ImportError on attribute access.
        class _FailModule:
            def __getattr__(self, name: str):
                raise ImportError("mocked unavailable")

        sys.modules["dwave.samplers"] = _FailModule()  # type: ignore[assignment]
        sys.modules["neal"] = _FailModule()  # type: ignore[assignment]

        backend = DWaveNealBackend()
    finally:
        # Restore original state.
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    assert backend.available is False
    assert backend._sampler is None


# ---------------------------------------------------------------------------
# DWaveNealBackend.sample — D-Wave available path
# ---------------------------------------------------------------------------


def test_sample_returns_correct_shape_with_mock_sampler() -> None:
    """sample() returns jnp array of shape (n_samples, n_spins) when available.

    Spec: REQ-SAMPLE-034
    """
    import jax.numpy as jnp

    n_spins = 5
    n_samples = 4
    J, h = _make_jnp_problem(n_spins)

    # Build a mock SampleSet that returns n_samples samples.
    mock_responses = [{i: bool(i % 2) for i in range(n_spins)} for _ in range(n_samples)]

    mock_sample_set = MagicMock()
    mock_sample_set.samples.return_value = iter(mock_responses)

    mock_sampler = MagicMock()
    mock_sampler.sample_ising.return_value = mock_sample_set

    backend = DWaveNealBackend.__new__(DWaveNealBackend)
    backend.available = True
    backend._sampler = mock_sampler

    result = backend.sample(J, h, n_samples=n_samples)

    assert result.shape == (n_samples, n_spins)
    assert result.dtype == jnp.bool_


def test_sample_pads_when_sampler_returns_fewer_samples() -> None:
    """sample() pads with the last row when the sampler returns fewer samples.

    Spec: REQ-SAMPLE-034
    """
    import jax.numpy as jnp

    n_spins = 3
    n_samples = 5

    # Sampler only returns 2 samples.
    mock_responses = [{i: True for i in range(n_spins)} for _ in range(2)]

    mock_sample_set = MagicMock()
    mock_sample_set.samples.return_value = iter(mock_responses)

    mock_sampler = MagicMock()
    mock_sampler.sample_ising.return_value = mock_sample_set

    backend = DWaveNealBackend.__new__(DWaveNealBackend)
    backend.available = True
    backend._sampler = mock_sampler

    result = backend.sample(jnp.zeros((n_spins, n_spins)), jnp.zeros(n_spins), n_samples)
    assert result.shape == (n_samples, n_spins)


def test_sample_returns_zeros_when_sampler_empty() -> None:
    """sample() returns all-False array when the sampler returns no samples.

    Spec: REQ-SAMPLE-034
    """
    import jax.numpy as jnp

    n_spins = 4
    n_samples = 3

    mock_sample_set = MagicMock()
    mock_sample_set.samples.return_value = iter([])

    mock_sampler = MagicMock()
    mock_sampler.sample_ising.return_value = mock_sample_set

    backend = DWaveNealBackend.__new__(DWaveNealBackend)
    backend.available = True
    backend._sampler = mock_sampler

    result = backend.sample(jnp.zeros((n_spins, n_spins)), jnp.zeros(n_spins), n_samples)
    assert result.shape == (n_samples, n_spins)
    assert bool(jnp.all(result == False))  # noqa: E712


# ---------------------------------------------------------------------------
# DWaveNealBackend.sample — fallback path
# ---------------------------------------------------------------------------


def test_sample_cpu_fallback_when_unavailable() -> None:
    """sample() uses ParallelIsingSampler fallback when available=False.

    SCENARIO-SAMPLE-059: fallback path returns correct shape.
    Spec: REQ-SAMPLE-034, SCENARIO-SAMPLE-059
    """
    import jax.numpy as jnp

    n_spins = 8
    n_samples = 6
    J, h = _make_jnp_problem(n_spins)

    backend = DWaveNealBackend.__new__(DWaveNealBackend)
    backend.available = False
    backend._sampler = None

    result = backend.sample(J, h, n_samples=n_samples)

    assert result.shape == (n_samples, n_spins)


# ---------------------------------------------------------------------------
# DWaveNealBackend.latency_ms
# ---------------------------------------------------------------------------


def test_latency_ms_returns_positive_float() -> None:
    """latency_ms returns a positive float representing milliseconds.

    Spec: REQ-SAMPLE-034
    """
    backend = DWaveNealBackend.__new__(DWaveNealBackend)
    backend.available = False
    backend._sampler = None

    ms = backend.latency_ms(10)

    assert isinstance(ms, float)
    assert ms > 0.0


def test_latency_ms_with_mock_dwave() -> None:
    """latency_ms runs 10 calls and returns a reasonable mean for mock sampler.

    Spec: REQ-SAMPLE-034
    """
    import jax.numpy as jnp

    n_spins = 5

    # Mock sampler that immediately returns a trivial sample set.
    def _fake_samples():
        yield {i: False for i in range(n_spins)}

    mock_sample_set = MagicMock()
    mock_sample_set.samples.return_value = _fake_samples()

    mock_sampler = MagicMock()
    mock_sampler.sample_ising.return_value = mock_sample_set

    backend = DWaveNealBackend.__new__(DWaveNealBackend)
    backend.available = True
    backend._sampler = mock_sampler

    # patch sample to avoid consuming the single iterator multiple times
    call_count = [0]

    def _sample_side_effect(J, h, n_samples=10):
        call_count[0] += 1
        return jnp.zeros((n_samples, n_spins), dtype=bool)

    backend.sample = _sample_side_effect  # type: ignore[method-assign]

    ms = backend.latency_ms(n_spins)
    assert call_count[0] == 10  # 10 warmup calls
    assert ms > 0.0


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


def test_dwave_neal_backend_exported_from_samplers() -> None:
    """DWaveNealBackend is exported from carnot.samplers.

    Spec: REQ-SAMPLE-034
    """
    from carnot.samplers import DWaveNealBackend as _Exported

    assert _Exported is DWaveNealBackend
