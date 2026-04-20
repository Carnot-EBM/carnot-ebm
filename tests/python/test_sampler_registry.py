"""Tests for get_sampler_backend() and backend_registry.

Covers the new production backend-selection API added in Exp 610.

Spec: REQ-SAMPLE-035, SCENARIO-SAMPLE-040, SCENARIO-SAMPLE-041
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_registry() -> None:
    """Clear backend_registry so it is repopulated on the next call.

    This is needed between tests that mutate CARNOT_SAMPLER because the
    registry is module-level and populated lazily — we want a clean state.
    """
    from carnot.samplers import backend_registry

    backend_registry.clear()


# ---------------------------------------------------------------------------
# REQ-SAMPLE-035-1: registry contains expected keys
# ---------------------------------------------------------------------------


def test_backend_registry_contains_cpu_and_dwave() -> None:
    """backend_registry maps 'cpu' and 'dwave' after first access.

    Spec: REQ-SAMPLE-035-1
    """
    _reset_registry()
    from carnot.samplers import backend_registry, get_sampler_backend

    # Trigger population.
    get_sampler_backend("cpu")

    assert "cpu" in backend_registry, "backend_registry must contain 'cpu'"
    assert "dwave" in backend_registry, "backend_registry must contain 'dwave'"


# ---------------------------------------------------------------------------
# REQ-SAMPLE-035-2 / SCENARIO-SAMPLE-041: cpu default
# ---------------------------------------------------------------------------


def test_get_sampler_backend_default_is_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_sampler_backend() without args returns CpuBackend when CARNOT_SAMPLER unset.

    Spec: REQ-SAMPLE-035-2, SCENARIO-SAMPLE-041
    """
    _reset_registry()
    monkeypatch.delenv("CARNOT_SAMPLER", raising=False)

    from carnot.samplers.backend import CpuBackend
    from carnot.samplers import get_sampler_backend

    instance = get_sampler_backend()
    assert isinstance(instance, CpuBackend), (
        f"Expected CpuBackend, got {type(instance).__name__}"
    )


# ---------------------------------------------------------------------------
# REQ-SAMPLE-035-3: dwave by name
# ---------------------------------------------------------------------------


def test_get_sampler_backend_dwave_by_name() -> None:
    """get_sampler_backend('dwave') returns DWaveNealBackend.

    Spec: REQ-SAMPLE-035-3
    """
    _reset_registry()
    from carnot.samplers.dwave_backend import DWaveNealBackend
    from carnot.samplers import get_sampler_backend

    instance = get_sampler_backend("dwave")
    assert isinstance(instance, DWaveNealBackend), (
        f"Expected DWaveNealBackend, got {type(instance).__name__}"
    )


def test_get_sampler_backend_cpu_by_name() -> None:
    """get_sampler_backend('cpu') returns CpuBackend.

    Spec: REQ-SAMPLE-035-3
    """
    _reset_registry()
    from carnot.samplers.backend import CpuBackend
    from carnot.samplers import get_sampler_backend

    instance = get_sampler_backend("cpu")
    assert isinstance(instance, CpuBackend), (
        f"Expected CpuBackend, got {type(instance).__name__}"
    )


# ---------------------------------------------------------------------------
# SCENARIO-SAMPLE-040: CARNOT_SAMPLER env var respected
# ---------------------------------------------------------------------------


def test_carnot_sampler_env_var_selects_dwave(monkeypatch: pytest.MonkeyPatch) -> None:
    """CARNOT_SAMPLER=dwave causes get_sampler_backend() to return DWaveNealBackend.

    Spec: SCENARIO-SAMPLE-040
    """
    _reset_registry()
    monkeypatch.setenv("CARNOT_SAMPLER", "dwave")

    from carnot.samplers.dwave_backend import DWaveNealBackend
    from carnot.samplers import get_sampler_backend

    instance = get_sampler_backend()
    assert isinstance(instance, DWaveNealBackend), (
        f"Expected DWaveNealBackend, got {type(instance).__name__}"
    )


# ---------------------------------------------------------------------------
# Error path: unknown name raises ValueError
# ---------------------------------------------------------------------------


def test_get_sampler_backend_unknown_name_raises() -> None:
    """get_sampler_backend raises ValueError for unknown backend names.

    Spec: REQ-SAMPLE-035-2
    """
    _reset_registry()
    from carnot.samplers import get_sampler_backend

    with pytest.raises(ValueError, match="Unknown CARNOT_SAMPLER backend"):
        get_sampler_backend("nonexistent_backend_xyz")


# ---------------------------------------------------------------------------
# REQ-SAMPLE-035-4: exports visible from carnot.samplers
# ---------------------------------------------------------------------------


def test_exports_visible_from_carnot_samplers() -> None:
    """get_sampler_backend and backend_registry are exported from carnot.samplers.

    Spec: REQ-SAMPLE-035-4
    """
    import carnot.samplers as samplers

    assert hasattr(samplers, "get_sampler_backend"), (
        "carnot.samplers must export get_sampler_backend"
    )
    assert hasattr(samplers, "backend_registry"), (
        "carnot.samplers must export backend_registry"
    )
    assert callable(samplers.get_sampler_backend)
    assert isinstance(samplers.backend_registry, dict)
