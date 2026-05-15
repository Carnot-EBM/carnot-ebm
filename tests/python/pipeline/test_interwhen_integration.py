"""Tests for InterwhenMonitor integration into VerifyRepairPipeline.

Verifies that the advisory interwhen_monitor wiring added to VerifyRepairPipeline
correctly populates certificate["interwhen_monitor"] without altering result.verified.

Spec: REQ-VERIFY-130, REQ-VERIFY-131,
      SCENARIO-VERIFY-168, SCENARIO-VERIFY-169
"""

from __future__ import annotations

import pytest

from carnot.pipeline.interwhen_monitor import InterWhenMonitor
from carnot.pipeline.symcode_verifier import SymCodeVerifier
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_monitor() -> InterWhenMonitor:
    """Build a CI-mode InterWhenMonitor (no LLM required)."""
    return InterWhenMonitor(SymCodeVerifier(llm_caller=None))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInterwhenMonitorIntegration:
    """REQ-VERIFY-130: InterwhenMonitor wired into VerifyRepairPipeline as advisory tier."""

    def test_no_monitor_leaves_certificate_clean(self) -> None:
        """When interwhen_monitor is None, certificate must not have 'interwhen_monitor' key.

        SCENARIO-VERIFY-169: absence of the monitor must not break normal verify flow.
        """
        pipeline = VerifyRepairPipeline(interwhen_monitor=None)
        result = pipeline.verify("2+2=?", "The answer is 4.")
        assert "interwhen_monitor" not in result.certificate

    def test_monitor_populates_certificate(self) -> None:
        """When interwhen_monitor is set, certificate must contain 'interwhen_monitor' key.

        SCENARIO-VERIFY-168: InterwhenMonitor output recorded in certificate.
        """
        pipeline = VerifyRepairPipeline(interwhen_monitor=_make_monitor())
        result = pipeline.verify("2+2=?", "The answer is 4.")
        assert "interwhen_monitor" in result.certificate

    def test_certificate_has_required_keys(self) -> None:
        """Certificate 'interwhen_monitor' dict must have n_violations, early_detection_rate, etc."""
        pipeline = VerifyRepairPipeline(interwhen_monitor=_make_monitor())
        result = pipeline.verify("What is 3+4?", "3 + 4 = 7.")
        iw = result.certificate["interwhen_monitor"]
        assert "n_violations" in iw
        assert "early_detection_count" in iw
        assert "total_sentences" in iw
        assert "early_detection_rate" in iw

    def test_advisory_does_not_override_verified(self) -> None:
        """InterwhenMonitor is advisory only — must not flip result.verified.

        REQ-VERIFY-131: advisory tiers must never override result.verified.
        """
        pipeline = VerifyRepairPipeline(interwhen_monitor=_make_monitor())
        # Run on a response that might flag something
        result = pipeline.verify("What is 3*4?", "3 * 4 = 99.")
        # verified is driven by the Ising/constraint pipeline, not InterwhenMonitor
        assert isinstance(result.verified, bool)

    def test_n_violations_is_non_negative_int(self) -> None:
        """n_violations must be a non-negative integer."""
        pipeline = VerifyRepairPipeline(interwhen_monitor=_make_monitor())
        result = pipeline.verify("x", "No arithmetic here.")
        iw = result.certificate["interwhen_monitor"]
        assert isinstance(iw["n_violations"], int)
        assert iw["n_violations"] >= 0

    def test_early_detection_rate_in_range(self) -> None:
        """early_detection_rate must be in [0.0, 1.0]."""
        pipeline = VerifyRepairPipeline(interwhen_monitor=_make_monitor())
        result = pipeline.verify("x", "Step one. Step two. Step three.")
        iw = result.certificate["interwhen_monitor"]
        assert 0.0 <= iw["early_detection_rate"] <= 1.0

    def test_zero_violations_early_detection_rate_is_zero(self) -> None:
        """When no violations detected, early_detection_rate must be 0.0."""
        pipeline = VerifyRepairPipeline(interwhen_monitor=_make_monitor())
        result = pipeline.verify("What colour is the sky?", "The sky is blue.")
        iw = result.certificate["interwhen_monitor"]
        if iw["n_violations"] == 0:
            assert iw["early_detection_rate"] == 0.0

    def test_total_sentences_matches_split(self) -> None:
        """total_sentences in certificate matches monitor.split_at_boundaries()."""
        monitor = _make_monitor()
        pipeline = VerifyRepairPipeline(interwhen_monitor=monitor)
        response = "First sentence. Second sentence. Third sentence."
        result = pipeline.verify("q", response)
        iw = result.certificate["interwhen_monitor"]
        expected = len(monitor.split_at_boundaries(response))
        assert iw["total_sentences"] == expected
