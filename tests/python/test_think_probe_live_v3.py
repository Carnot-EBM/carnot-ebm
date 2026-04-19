"""Tests for ThinkProbeLiveV3Result.

100% coverage target for python/carnot/pipeline/think_probe_live_v3_result.py.

Spec: REQ-PROBE-010, REQ-PROBE-011,
      SCENARIO-PROBE-015, SCENARIO-PROBE-016
"""

from __future__ import annotations

import pytest

from carnot.pipeline.think_probe_live_v3_result import ThinkProbeLiveV3Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(n_completed: int = 42, tp_rate: float = 0.80, fp_rate: float = 0.15, **kw) -> ThinkProbeLiveV3Result:
    """Construct a ThinkProbeLiveV3Result with sensible defaults for most tests."""
    defaults = {
        "inference_mode": "live_gpu",
        "model_id": "google/gemma-4-E4B-it",
        "n_completed": n_completed,
        "n_total": 50,
        "gpu_vram_gate_fired": True,
        "skip_rate": 0.0,
        "tp_rate": tp_rate,
        "fp_rate": fp_rate,
    }
    defaults.update(kw)
    return ThinkProbeLiveV3Result(**defaults)


# ---------------------------------------------------------------------------
# completion_fraction
# ---------------------------------------------------------------------------


class TestCompletionFraction:
    """REQ-PROBE-011: completion_fraction == n_completed / n_total."""

    def test_fraction_42_of_50(self):
        r = _make(n_completed=42)
        assert r.completion_fraction == pytest.approx(42 / 50)

    def test_fraction_full_run(self):
        r = _make(n_completed=50)
        assert r.completion_fraction == pytest.approx(1.0)

    def test_fraction_zero_completed(self):
        r = _make(n_completed=0)
        assert r.completion_fraction == pytest.approx(0.0)

    def test_fraction_zero_total(self):
        # Division-by-zero guard: returns 0.0 when n_total is 0.
        r = ThinkProbeLiveV3Result(
            inference_mode="live_gpu",
            model_id="google/gemma-4-E4B-it",
            n_completed=0,
            n_total=0,
            gpu_vram_gate_fired=False,
            skip_rate=0.0,
            tp_rate=0.0,
            fp_rate=0.0,
        )
        assert r.completion_fraction == 0.0


# ---------------------------------------------------------------------------
# is_viable — SCENARIO-PROBE-015
# ---------------------------------------------------------------------------


class TestIsViable:
    """REQ-PROBE-011, SCENARIO-PROBE-015/016: is_viable three-threshold logic."""

    def test_viable_when_42_completed_and_good_rates(self):
        # SCENARIO-PROBE-015: n_completed=42, tp_rate=0.80, fp_rate=0.15 → viable
        r = _make(n_completed=42, tp_rate=0.80, fp_rate=0.15)
        assert r.is_viable is True

    def test_not_viable_when_n_completed_30(self):
        # SCENARIO-PROBE-016: n_completed=30 → completion_fraction=0.60 < 0.80 → not viable
        r = _make(n_completed=30, tp_rate=0.80, fp_rate=0.10)
        assert r.is_viable is False

    def test_not_viable_at_exact_39_threshold(self):
        # 39/50 = 0.78 < 0.80, just below threshold
        r = _make(n_completed=39, tp_rate=0.90, fp_rate=0.05)
        assert r.is_viable is False

    def test_viable_at_exact_40_threshold(self):
        # 40/50 = 0.80, exactly at threshold
        r = _make(n_completed=40, tp_rate=0.70, fp_rate=0.20)
        assert r.is_viable is True

    def test_not_viable_when_tp_rate_below_threshold(self):
        # tp_rate=0.69 < 0.70 → not viable
        r = _make(n_completed=50, tp_rate=0.69, fp_rate=0.10)
        assert r.is_viable is False

    def test_viable_at_exact_tp_rate_threshold(self):
        # tp_rate=0.70 exactly at threshold
        r = _make(n_completed=50, tp_rate=0.70, fp_rate=0.20)
        assert r.is_viable is True

    def test_not_viable_when_fp_rate_above_threshold(self):
        # fp_rate=0.21 > 0.20 → not viable
        r = _make(n_completed=50, tp_rate=0.80, fp_rate=0.21)
        assert r.is_viable is False

    def test_viable_at_exact_fp_rate_threshold(self):
        # fp_rate=0.20 exactly at threshold
        r = _make(n_completed=50, tp_rate=0.80, fp_rate=0.20)
        assert r.is_viable is True

    def test_not_viable_all_thresholds_fail(self):
        # All three below threshold
        r = _make(n_completed=10, tp_rate=0.50, fp_rate=0.50)
        assert r.is_viable is False

    def test_not_viable_zero_completed(self):
        r = _make(n_completed=0, tp_rate=0.90, fp_rate=0.05)
        assert r.is_viable is False


# ---------------------------------------------------------------------------
# retro_036_closed
# ---------------------------------------------------------------------------


class TestRetro036Closed:
    """REQ-PROBE-010: retro_036_closed is always True for a live result."""

    def test_always_true_on_live_run(self):
        r = _make(n_completed=42)
        assert r.retro_036_closed is True

    def test_always_true_even_on_partial_run(self):
        # Even a partial run (n_completed=5) closes RETRO-036 because the
        # result object's existence proves the write path was reached.
        r = _make(n_completed=5)
        assert r.retro_036_closed is True

    def test_always_true_on_zero_completed(self):
        r = _make(n_completed=0)
        assert r.retro_036_closed is True


# ---------------------------------------------------------------------------
# Field storage (basic dataclass sanity)
# ---------------------------------------------------------------------------


class TestFieldStorage:
    """Verify all fields are stored and retrievable."""

    def test_all_fields_stored(self):
        r = ThinkProbeLiveV3Result(
            inference_mode="live_gpu",
            model_id="google/gemma-4-E4B-it",
            n_completed=45,
            n_total=50,
            gpu_vram_gate_fired=True,
            skip_rate=0.02,
            tp_rate=0.82,
            fp_rate=0.10,
        )
        assert r.inference_mode == "live_gpu"
        assert r.model_id == "google/gemma-4-E4B-it"
        assert r.n_completed == 45
        assert r.n_total == 50
        assert r.gpu_vram_gate_fired is True
        assert r.skip_rate == pytest.approx(0.02)
        assert r.tp_rate == pytest.approx(0.82)
        assert r.fp_rate == pytest.approx(0.10)

    def test_deferred_inference_mode(self):
        r = _make(n_completed=0, inference_mode="deferred")
        assert r.inference_mode == "deferred"
        # deferred is not viable (no completions)
        assert r.is_viable is False
