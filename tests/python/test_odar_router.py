"""Tests for ODAR-style free-energy routing in the verify-repair cascade.

Spec: REQ-ODAR-2243, SCENARIO-ODAR-2243
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.odar_router import FreeEnergyRouter, RoutingDecision
from carnot.pipeline.verify_repair import VerifyRepairPipeline


class _StableHalluField:
    """Tier 0e fixture that gives the ODAR gate a low-risk probe output."""

    def score(self, logits):
        return {
            "is_unstable": False,
            "risk_score": 0.1,
            "confidence": 0.9,
            "threshold": 0.5,
        }


class _UnstableHalluField:
    """Tier 0e fixture that gives the ODAR gate a high-risk probe output."""

    def score(self, logits):
        return {
            "is_unstable": True,
            "risk_score": 0.8,
            "confidence": 0.9,
            "threshold": 0.5,
        }


def test_low_efe_probe_outputs_route_fast_path() -> None:
    """REQ-ODAR-2243: low-risk Tier 0 outputs produce FAST_PATH."""
    router = FreeEnergyRouter(risk_threshold=0.5)

    decision = router.route(
        {
            "nup": {"risk_score": 0.1, "confidence": 0.9},
            "hallufield": {"risk_score": 0.2, "confidence": 0.8},
        }
    )

    assert decision is RoutingDecision.FAST_PATH
    assert router.evaluate(
        {
            "nup": {"risk_score": 0.1, "confidence": 0.9},
            "hallufield": {"risk_score": 0.2, "confidence": 0.8},
        }
    ).expected_free_energy == pytest.approx(0.1875)


def test_high_efe_probe_outputs_route_deliberative() -> None:
    """REQ-ODAR-2243: high-risk Tier 0 outputs produce DELIBERATIVE."""
    router = FreeEnergyRouter(risk_threshold=0.5)

    decision = router.route(
        {
            "hallufield": {"is_unstable": True, "risk_score": 0.9, "confidence": 0.9},
            "semantic_energy": {"risk_score": 0.7, "confidence": 0.6},
        }
    )

    assert decision is RoutingDecision.DELIBERATIVE


def test_risk_threshold_changes_route_for_same_efe() -> None:
    """REQ-ODAR-2243: threshold sensitivity is explicit and deterministic."""
    probe_outputs = {"tier0": {"risk_score": 0.4, "confidence": 1.0}}

    assert FreeEnergyRouter(risk_threshold=0.3).route(probe_outputs) is RoutingDecision.DELIBERATIVE
    assert FreeEnergyRouter(risk_threshold=0.5).route(probe_outputs) is RoutingDecision.FAST_PATH


def test_verify_use_odar_fast_path_skips_tier1_extraction() -> None:
    """SCENARIO-ODAR-2243: ODAR fast path returns before Tier 1 extraction."""
    pipeline = VerifyRepairPipeline()

    with patch.object(
        pipeline,
        "extract_constraints",
        side_effect=AssertionError("Tier 1 extraction should be skipped"),
    ):
        result = pipeline.verify(
            question="q",
            response="2 + 2 = 4",
            use_odar=True,
            odar_risk_threshold=0.5,
            hallufield_detector=_StableHalluField(),
        )

    assert result.verified is True
    assert result.mode == "ODAR_FAST_PATH"
    assert result.skipped is True
    assert result.certificate["odar_decision"] == "FAST_PATH"
    assert result.certificate["odar_expected_free_energy"] < 0.5


def test_verify_use_odar_deliberative_route_runs_tier1_extraction() -> None:
    """REQ-ODAR-2243: high-EFE ODAR route falls through to normal extraction."""
    pipeline = VerifyRepairPipeline()

    with patch.object(pipeline, "extract_constraints", return_value=[]) as extract_constraints:
        result = pipeline.verify(
            question="q",
            response="2 + 2 = 4",
            use_odar=True,
            odar_risk_threshold=0.5,
            hallufield_detector=_UnstableHalluField(),
        )

    extract_constraints.assert_called_once_with("2 + 2 = 4", None)
    assert result.mode != "ODAR_FAST_PATH"
    assert result.certificate["odar_decision"] == "DELIBERATIVE"
    assert result.certificate["odar_expected_free_energy"] >= 0.5


def test_verify_constructor_use_odar_default_remains_disabled() -> None:
    """REQ-ODAR-2243: use_odar defaults off for backward compatibility."""
    pipeline = VerifyRepairPipeline()
    detector = MagicMock()
    detector.score.return_value = {"is_unstable": False, "risk_score": 0.0, "confidence": 1.0}

    result = pipeline.verify(question="q", response="plain response", hallufield_detector=detector)

    assert result.mode != "ODAR_FAST_PATH"
    assert "odar_decision" not in result.certificate
