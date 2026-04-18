"""Tests for LiveThinkProbeResult.

100% coverage target for python/carnot/pipeline/live_think_probe_result.py.

Spec: REQ-PROBE-008, REQ-PROBE-009,
      SCENARIO-PROBE-013, SCENARIO-PROBE-014
"""

from __future__ import annotations

import pytest

from carnot.pipeline.live_think_probe_result import LiveThinkProbeResult
from carnot.pipeline.think_probe_v2 import ThinkProbeV2Result


class TestLiveThinkProbeResultInheritance:
    """LiveThinkProbeResult is a ThinkProbeV2Result subclass (REQ-PROBE-009)."""

    def test_is_subclass_of_think_probe_v2_result(self):
        r = LiveThinkProbeResult(n_completed=50, n_total=50, results=[])
        assert isinstance(r, ThinkProbeV2Result)

    def test_default_inference_mode_is_live_gpu(self):
        # REQ-PROBE-009: inference_mode defaults to 'live_gpu'
        r = LiveThinkProbeResult(n_completed=50, n_total=50, results=[])
        assert r.inference_mode == "live_gpu"

    def test_model_id_field_stored(self):
        r = LiveThinkProbeResult(
            n_completed=50,
            n_total=50,
            results=[],
            model_id="google/gemma-4-E4B-it",
        )
        assert r.model_id == "google/gemma-4-E4B-it"

    def test_gpu_used_field_stored(self):
        r = LiveThinkProbeResult(
            n_completed=50,
            n_total=50,
            results=[],
            gpu_used="cuda:0",
        )
        assert r.gpu_used == "cuda:0"

    def test_all_provenance_fields_explicit(self):
        r = LiveThinkProbeResult(
            n_completed=30,
            n_total=50,
            results=[],
            status="partial",
            inference_mode="live_gpu",
            model_id="google/gemma-4-E4B-it",
            gpu_used="cuda:0 (ROCm)",
        )
        assert r.inference_mode == "live_gpu"
        assert r.model_id == "google/gemma-4-E4B-it"
        assert r.gpu_used == "cuda:0 (ROCm)"


class TestLiveThinkProbeResultIsPartial:
    """SCENARIO-PROBE-014: is_partial semantics carry through from base class."""

    def test_is_partial_true_when_n_completed_lt_n_total(self):
        # SCENARIO-PROBE-014: partial run has is_partial=True
        r = LiveThinkProbeResult(
            n_completed=30,
            n_total=50,
            results=[],
            inference_mode="live_gpu",
            model_id="google/gemma-4-E4B-it",
            gpu_used="cuda:0",
        )
        assert r.is_partial is True
        assert r.inference_mode == "live_gpu"

    def test_is_partial_false_on_complete_run(self):
        r = LiveThinkProbeResult(n_completed=50, n_total=50, results=[])
        assert r.is_partial is False

    def test_completion_fraction_partial(self):
        r = LiveThinkProbeResult(n_completed=25, n_total=50, results=[], status="partial")
        assert r.completion_fraction == pytest.approx(0.5)

    def test_honest_verdict_partial(self):
        r = LiveThinkProbeResult(n_completed=25, n_total=50, results=[], status="partial")
        assert r.honest_verdict == "partial_25_of_50"

    def test_honest_verdict_complete(self):
        r = LiveThinkProbeResult(n_completed=50, n_total=50, results=[])
        assert r.honest_verdict == "complete"

    def test_honest_verdict_timeout_no_data(self):
        r = LiveThinkProbeResult(n_completed=0, n_total=50, results=[], status="empty")
        assert r.honest_verdict == "timeout_no_data"


class TestLiveThinkProbeResultDefaults:
    """Default field values are safe and well-defined."""

    def test_empty_model_id_default(self):
        r = LiveThinkProbeResult(n_completed=0, n_total=0, results=[])
        assert r.model_id == ""

    def test_empty_gpu_used_default(self):
        r = LiveThinkProbeResult(n_completed=0, n_total=0, results=[])
        assert r.gpu_used == ""

    def test_status_defaults_to_complete(self):
        r = LiveThinkProbeResult(n_completed=5, n_total=5, results=[])
        assert r.status == "complete"
