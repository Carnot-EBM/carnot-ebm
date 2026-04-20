"""Tests for LowRankKAEM cascade integration.

Verifies that get_kaem_energy() selects the right model type, that
VerificationResult.use_lowrank_kaem exists with the correct default,
and that VerifyRepairPipeline._build_kan_fast_path_model() honours
the n_vars <= 100 threshold.

Spec: REQ-SAMPLE-029, SCENARIO-SAMPLE-044, SCENARIO-SAMPLE-045
"""

from __future__ import annotations

import pytest

from carnot.models.kaem_energy import KAEMEnergy, get_kaem_energy
from carnot.models.lowrank_kaem import LowRankKAEMEnergy
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


# ---------------------------------------------------------------------------
# get_kaem_energy factory — REQ-SAMPLE-029
# ---------------------------------------------------------------------------


def test_get_kaem_energy_returns_lowrank_when_flag_true():
    """Factory returns LowRankKAEMEnergy with correct n_vars and k."""
    # SCENARIO-SAMPLE-044
    model = get_kaem_energy(n_vars=50, use_lowrank=True, k=2)
    assert isinstance(model, LowRankKAEMEnergy)
    assert model.n_vars == 50
    assert model.k == 2


def test_get_kaem_energy_returns_fullrank_when_flag_false():
    """Factory returns KAEMEnergy when use_lowrank=False."""
    # SCENARIO-SAMPLE-045
    model = get_kaem_energy(n_vars=200, use_lowrank=False)
    assert isinstance(model, KAEMEnergy)
    assert model.n_vars == 200


def test_get_kaem_energy_default_is_lowrank_k2():
    """Default call (no flags) returns LowRankKAEMEnergy with k=2."""
    model = get_kaem_energy(n_vars=10)
    assert isinstance(model, LowRankKAEMEnergy)
    assert model.k == 2


def test_get_kaem_energy_custom_k():
    """Custom k is forwarded to LowRankKAEMEnergy."""
    model = get_kaem_energy(n_vars=30, use_lowrank=True, k=5)
    assert isinstance(model, LowRankKAEMEnergy)
    assert model.k == 5


def test_get_kaem_energy_n_vars_100_lowrank():
    """n_vars=100 with use_lowrank=True returns LowRankKAEMEnergy (boundary)."""
    model = get_kaem_energy(n_vars=100, use_lowrank=True, k=2)
    assert isinstance(model, LowRankKAEMEnergy)


def test_get_kaem_energy_n_vars_101_fullrank():
    """n_vars=101 with use_lowrank=False returns KAEMEnergy (boundary + 1)."""
    model = get_kaem_energy(n_vars=101, use_lowrank=False)
    assert isinstance(model, KAEMEnergy)


# ---------------------------------------------------------------------------
# VerificationResult.use_lowrank_kaem field — REQ-SAMPLE-029
# ---------------------------------------------------------------------------


def test_verification_result_has_use_lowrank_kaem_field():
    """VerificationResult.use_lowrank_kaem defaults to False."""
    result = VerificationResult(
        verified=True,
        constraints=[],
        energy=0.0,
        violations=[],
    )
    assert hasattr(result, "use_lowrank_kaem")
    assert result.use_lowrank_kaem is False


def test_verification_result_use_lowrank_kaem_settable():
    """VerificationResult.use_lowrank_kaem can be set to True."""
    result = VerificationResult(
        verified=True,
        constraints=[],
        energy=0.0,
        violations=[],
        use_lowrank_kaem=True,
    )
    assert result.use_lowrank_kaem is True


# ---------------------------------------------------------------------------
# VerifyRepairPipeline._build_kan_fast_path_model — REQ-SAMPLE-029
# ---------------------------------------------------------------------------


def _make_pipeline() -> VerifyRepairPipeline:
    """Construct a minimal VerifyRepairPipeline (no model, no extractor)."""
    return VerifyRepairPipeline(
        model=None,
        domains=[],
        max_repairs=0,
        extractor=None,
        semantic_grounding_verifier=None,
        semantic_verifier_v2=None,
        timeout_seconds=30,
        memory=None,
        template_library=None,
        session_memory=None,
        constraint_memory=None,
    )


def test_build_kan_fast_path_small_uses_lowrank():
    """n_vars=10 -> LowRankKAEMEnergy, use_lowrank=True."""
    pipeline = _make_pipeline()
    model, use_lowrank = pipeline._build_kan_fast_path_model(10)
    assert isinstance(model, LowRankKAEMEnergy)
    assert use_lowrank is True


def test_build_kan_fast_path_100_uses_lowrank():
    """n_vars=100 (boundary) -> LowRankKAEMEnergy, use_lowrank=True."""
    pipeline = _make_pipeline()
    model, use_lowrank = pipeline._build_kan_fast_path_model(100)
    assert isinstance(model, LowRankKAEMEnergy)
    assert use_lowrank is True


def test_build_kan_fast_path_large_uses_fullrank():
    """n_vars=200 -> KAEMEnergy, use_lowrank=False."""
    pipeline = _make_pipeline()
    model, use_lowrank = pipeline._build_kan_fast_path_model(200)
    assert isinstance(model, KAEMEnergy)
    assert use_lowrank is False
