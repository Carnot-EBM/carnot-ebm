"""Tests for NUP Probe v6 Tier 0c wire-in to VerifyRepairPipeline.

Verifies that:
  - VerifyRepairPipeline accepts nup_probe and nup_probe_threshold constructor params.
  - When nup_probe.score(response) <= threshold, verify() returns NUP_PROBE_FAST_PATH.
  - When nup_probe.score(response) > threshold, verify() continues the full cascade.
  - When nup_probe is None (default), the pipeline is unchanged (backward compat).

Spec: REQ-VERIFY-146, REQ-VERIFY-147,
      SCENARIO-VERIFY-177, SCENARIO-VERIFY-178, SCENARIO-VERIFY-179
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.nup_probe_v4 import NUPProbeV4
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(nup_probe=None, nup_probe_threshold=0.5):
    """Construct a verify-only VerifyRepairPipeline with optional NUP probe."""
    return VerifyRepairPipeline(
        model=None,
        nup_probe=nup_probe,
        nup_probe_threshold=nup_probe_threshold,
    )


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-177: NUP fast-path fires on low-energy response
# ---------------------------------------------------------------------------


class TestNUPProbeFastPath:
    """REQ-VERIFY-146-3: low nup_score <= threshold → NUP_PROBE_FAST_PATH."""

    def test_fast_path_fires_when_score_below_threshold(self):
        # SCENARIO-VERIFY-177: score=0.1 < threshold=0.5 → fast-path
        mock_probe = MagicMock(spec=NUPProbeV4)
        mock_probe.score.return_value = 0.1

        pipeline = _make_pipeline(nup_probe=mock_probe, nup_probe_threshold=0.5)
        result = pipeline.verify(question="q", response="2 + 2 = 4", domain=None)

        assert result.verified is True
        assert result.mode == "NUP_PROBE_FAST_PATH"
        assert result.skipped is True
        mock_probe.score.assert_called_once_with("2 + 2 = 4")

    def test_fast_path_fires_when_score_exactly_equals_threshold(self):
        # Boundary: score == threshold should still trigger fast-path (<=)
        mock_probe = MagicMock(spec=NUPProbeV4)
        mock_probe.score.return_value = 0.5

        pipeline = _make_pipeline(nup_probe=mock_probe, nup_probe_threshold=0.5)
        result = pipeline.verify(question="q", response="4 = 4", domain=None)

        assert result.verified is True
        assert result.mode == "NUP_PROBE_FAST_PATH"
        assert result.skipped is True

    def test_certificate_contains_nup_fields(self):
        # Certificate must expose nup_score and nup_threshold for observability.
        mock_probe = MagicMock(spec=NUPProbeV4)
        mock_probe.score.return_value = 0.2

        pipeline = _make_pipeline(nup_probe=mock_probe, nup_probe_threshold=0.5)
        result = pipeline.verify(question="q", response="simple", domain=None)

        assert result.certificate["nup_score"] == pytest.approx(0.2)
        assert result.certificate["nup_threshold"] == pytest.approx(0.5)
        assert result.certificate["mode"] == "NUP_PROBE_FAST_PATH"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-178: NUP falls through on high-energy response
# ---------------------------------------------------------------------------


class TestNUPProbeFallThrough:
    """REQ-VERIFY-146-3: score > threshold → no fast-path, cascade continues."""

    def test_no_fast_path_when_score_above_threshold(self):
        # SCENARIO-VERIFY-178: score=0.9 > threshold=0.5 → no short-circuit
        mock_probe = MagicMock(spec=NUPProbeV4)
        mock_probe.score.return_value = 0.9

        pipeline = _make_pipeline(nup_probe=mock_probe, nup_probe_threshold=0.5)
        # verify() must NOT return NUP_PROBE_FAST_PATH (it continues to constraint extraction)
        result = pipeline.verify(question="q", response="some response", domain=None)

        assert result.mode != "NUP_PROBE_FAST_PATH"
        mock_probe.score.assert_called_once_with("some response")


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-179: nup_probe=None preserves backward compatibility
# ---------------------------------------------------------------------------


class TestNUPProbeBackwardCompat:
    """REQ-VERIFY-146-4: nup_probe=None → no change to existing behaviour."""

    def test_pipeline_without_nup_probe_accepts_construction(self):
        # SCENARIO-VERIFY-179: default nup_probe=None should construct without error
        pipeline = _make_pipeline(nup_probe=None)
        assert pipeline._nup_probe is None

    def test_pipeline_without_nup_probe_runs_verify(self):
        # verify() must succeed when nup_probe is None
        pipeline = _make_pipeline(nup_probe=None)
        result = pipeline.verify(question="q", response="test", domain=None)
        # Result must NOT be in NUP_PROBE_FAST_PATH (probe was not active)
        assert result.mode != "NUP_PROBE_FAST_PATH"

    def test_pipeline_stores_probe_and_threshold(self):
        # Constructor stores both fields for later use in verify().
        mock_probe = MagicMock(spec=NUPProbeV4)
        mock_probe.score.return_value = 0.0

        pipeline = _make_pipeline(nup_probe=mock_probe, nup_probe_threshold=0.75)
        assert pipeline._nup_probe is mock_probe
        assert pipeline._nup_probe_threshold == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Integration: real NUPProbeV4 instance (no mocks)
# ---------------------------------------------------------------------------


class TestNUPProbeIntegration:
    """Integration test with a real NUPProbeV4 (random-init weights).

    Verifies that the fast-path fires or not based on actual score() output,
    and that latency is well under the 5 ms budget (REQ-VERIFY-147).
    """

    def test_correct_step_may_short_circuit(self):
        # A formulaic correct step typically has low character entropy → low score.
        # With random-init weights the score is arbitrary, but the pipeline must not crash.
        probe = NUPProbeV4(energy_dim=32, random_seed=42)
        pipeline = _make_pipeline(nup_probe=probe, nup_probe_threshold=1e9)
        # Set threshold very high so fast-path always fires → verifies the code path.
        result = pipeline.verify(question="q", response="2 + 2 = 4", domain=None)
        assert result.verified is True
        assert result.mode == "NUP_PROBE_FAST_PATH"

    def test_latency_under_5ms(self):
        # REQ-VERIFY-147-1: mean latency < 5 ms over a small batch.
        import time  # noqa: PLC0415

        probe = NUPProbeV4(energy_dim=32, random_seed=42)
        # Force fast-path by using very high threshold so every call short-circuits.
        pipeline = _make_pipeline(nup_probe=probe, nup_probe_threshold=1e9)

        latencies = []
        for i in range(20):
            t0 = time.perf_counter()
            pipeline.verify(question="q", response=f"answer {i}", domain=None)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = sum(latencies) / len(latencies)
        assert mean_ms < 5.0, f"Mean latency {mean_ms:.3f} ms exceeded 5 ms budget"
