"""Tests for vg_search_scheduler.py — VGSearchScheduler and VGScheduleResult.

Covers:
  - VGScheduleResult dataclass fields
  - VGSearchScheduler.update() sliding window behaviour
  - VGSearchScheduler.should_skip() — insufficient_history case
  - VGSearchScheduler.should_skip() — low_variance_skip case (N stable values)
  - VGSearchScheduler.should_skip() — high_variance_run case
  - VGSearchScheduler.reset() clears history
  - ThreeTierPipeline integration with vg_scheduler (ADDITIVE, no regression)
  - ThreeTierPipeline vg_skip tier counts as skip in benchmark
  - Export from carnot.pipeline.__init__

Spec: REQ-VERIFY-171, REQ-VERIFY-172
SCENARIO-VERIFY-200
"""

from __future__ import annotations

import pytest
import numpy as np

from carnot.pipeline.vg_search_scheduler import VGScheduleResult, VGSearchScheduler
from carnot.pipeline import VGScheduleResult as _ExportedResult, VGSearchScheduler as _ExportedScheduler
from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _low_energy_sequence(n: int = 3, base: float = 0.5, noise: float = 0.001) -> list[float]:
    """Return n energies within a tiny range — variance will be << 0.05."""
    rng = np.random.default_rng(0)
    return [base + rng.uniform(-noise, noise) for _ in range(n)]


def _high_energy_sequence(n: int = 3) -> list[float]:
    """Return n energies spanning [0.1, 1.5] — high variance >> 0.05."""
    return [0.1, 1.5, 0.2][:n]


class _StubEORM:
    """Always returns energy 0.8 (above eorm_threshold=0.5) to reach Tier 3."""

    def energy(self, cot_input):
        return 0.8


class _StubSinkProbe:
    """Always returns zero sink score so Tier 1 never clears a response."""

    def score(self, attn, sink_positions):
        from carnot.pipeline.sink_probe import SinkConcentration

        return SinkConcentration(
            mean_sink_score=0.0,
            per_head_scores=[0.0],
            sink_positions=sink_positions,
        )


def _make_ising_stub(verified: bool, energy: float = 0.9):
    def _fn(response, question):
        return verified, energy

    return _fn


# ---------------------------------------------------------------------------
# VGScheduleResult dataclass tests
# ---------------------------------------------------------------------------


def test_vg_schedule_result_fields():
    """VGScheduleResult must expose all documented fields.  REQ-VERIFY-171."""
    result = VGScheduleResult(
        should_run_tier=True,
        energy_variance=0.1,
        variance_threshold=0.05,
        skip_reason="high_variance_run",
        honest_verdict="high_variance_run_tier",
    )
    assert result.should_run_tier is True
    assert result.energy_variance == pytest.approx(0.1)
    assert result.variance_threshold == pytest.approx(0.05)
    assert result.skip_reason == "high_variance_run"
    assert result.honest_verdict == "high_variance_run_tier"


# ---------------------------------------------------------------------------
# VGSearchScheduler — insufficient history
# ---------------------------------------------------------------------------


def test_should_skip_insufficient_history_empty():
    """should_skip returns should_run_tier=True when history is empty.  REQ-VERIFY-171-2."""
    scheduler = VGSearchScheduler(variance_threshold=0.05, window_size=3)
    result = scheduler.should_skip()
    assert result.should_run_tier is True
    assert result.skip_reason == "insufficient_history"
    assert result.energy_variance == 0.0


def test_should_skip_insufficient_history_partial():
    """should_skip returns should_run_tier=True when window not yet full.  REQ-VERIFY-171-2."""
    scheduler = VGSearchScheduler(variance_threshold=0.05, window_size=3)
    scheduler.update(0.5)
    scheduler.update(0.5)
    # Only 2 of 3 required readings — still insufficient.
    result = scheduler.should_skip()
    assert result.should_run_tier is True
    assert result.skip_reason == "insufficient_history"


# ---------------------------------------------------------------------------
# VGSearchScheduler — low variance skip
# ---------------------------------------------------------------------------


def test_should_skip_low_variance_after_stable_readings():
    """should_skip returns should_run_tier=False after N stable energies.  REQ-VERIFY-171-3.
    SCENARIO-VERIFY-200."""
    scheduler = VGSearchScheduler(variance_threshold=0.05, window_size=3)
    for e in _low_energy_sequence(3):
        scheduler.update(e)
    result = scheduler.should_skip()
    assert result.should_run_tier is False
    assert result.skip_reason == "low_variance_skip"
    assert result.energy_variance < 0.05


def test_should_skip_low_variance_honest_verdict():
    """honest_verdict matches skip_reason when variance is low.  REQ-VERIFY-171-3."""
    scheduler = VGSearchScheduler(variance_threshold=0.05, window_size=3)
    for e in [0.8, 0.8, 0.8]:
        scheduler.update(e)
    result = scheduler.should_skip()
    assert result.honest_verdict == "low_variance_skip_tier"
    assert result.should_run_tier is False


# ---------------------------------------------------------------------------
# VGSearchScheduler — high variance run
# ---------------------------------------------------------------------------


def test_should_skip_high_variance_runs_tier():
    """should_skip returns should_run_tier=True when variance >= threshold.  REQ-VERIFY-171-4."""
    scheduler = VGSearchScheduler(variance_threshold=0.05, window_size=3)
    for e in _high_energy_sequence(3):
        scheduler.update(e)
    result = scheduler.should_skip()
    assert result.should_run_tier is True
    assert result.skip_reason == "high_variance_run"
    assert result.energy_variance >= 0.05


# ---------------------------------------------------------------------------
# VGSearchScheduler — sliding window FIFO
# ---------------------------------------------------------------------------


def test_update_sliding_window_fifo():
    """update() maintains exactly window_size entries (oldest dropped).  REQ-VERIFY-171-1."""
    scheduler = VGSearchScheduler(variance_threshold=0.05, window_size=3)
    scheduler.update(0.1)
    scheduler.update(0.2)
    scheduler.update(0.3)
    # Window is now [0.1, 0.2, 0.3].
    assert len(scheduler._energy_history) == 3
    scheduler.update(0.4)
    # Oldest (0.1) should be evicted; window is [0.2, 0.3, 0.4].
    assert len(scheduler._energy_history) == 3
    assert scheduler._energy_history[0] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# VGSearchScheduler — reset
# ---------------------------------------------------------------------------


def test_reset_clears_history():
    """reset() empties the energy history window.  REQ-VERIFY-171."""
    scheduler = VGSearchScheduler()
    for e in [0.5, 0.6, 0.7]:
        scheduler.update(e)
    assert len(scheduler._energy_history) == 3
    scheduler.reset()
    assert len(scheduler._energy_history) == 0
    result = scheduler.should_skip()
    assert result.skip_reason == "insufficient_history"


# ---------------------------------------------------------------------------
# ThreeTierPipeline — vg_scheduler is ADDITIVE (no regression without it)
# ---------------------------------------------------------------------------


def test_three_tier_pipeline_no_vg_scheduler_baseline():
    """Omitting vg_scheduler leaves pipeline behaviour unchanged.  REQ-VERIFY-171-5."""
    pipeline = ThreeTierPipeline(
        sink_probe=_StubSinkProbe(),  # type: ignore[arg-type]
        eorm_model=_StubEORM(),  # type: ignore[arg-type]
        ising_pipeline=_make_ising_stub(True, 0.3),
        vg_scheduler=None,
    )
    verified, tier_used, energy = pipeline.verify("some response", question="q")
    assert tier_used == "ising"
    assert verified is True


def test_three_tier_pipeline_with_vg_scheduler_low_variance_skips_ising():
    """With vg_scheduler and pre-seeded low-variance window, Ising is skipped.  REQ-VERIFY-171-5.
    SCENARIO-VERIFY-200."""
    scheduler = VGSearchScheduler(variance_threshold=0.05, window_size=3)
    # Pre-seed with 3 stable energies so window is full.
    for e in [0.8, 0.8, 0.8]:
        scheduler.update(e)

    pipeline = ThreeTierPipeline(
        sink_probe=_StubSinkProbe(),  # type: ignore[arg-type]
        eorm_model=_StubEORM(),  # type: ignore[arg-type]
        ising_pipeline=_make_ising_stub(False, 0.9),  # Would return False if called.
        vg_scheduler=scheduler,
    )

    verified, tier_used, energy = pipeline.verify("stable response", question="q")
    # Ising stub would return False; vg_skip returns True (reuses EORM energy).
    assert tier_used == "vg_skip"
    assert verified is True  # VG skip declares verified=True.


def test_three_tier_pipeline_with_vg_scheduler_high_variance_runs_ising():
    """With vg_scheduler and high-variance window, Ising still runs.  REQ-VERIFY-171-5."""
    scheduler = VGSearchScheduler(variance_threshold=0.05, window_size=3)
    # Pre-seed with high-variance energies.
    for e in [0.1, 1.5, 0.2]:
        scheduler.update(e)

    ising_called = [False]

    def _ising(response, question):
        ising_called[0] = True
        return True, 0.3

    pipeline = ThreeTierPipeline(
        sink_probe=_StubSinkProbe(),  # type: ignore[arg-type]
        eorm_model=_StubEORM(),  # type: ignore[arg-type]
        ising_pipeline=_ising,
        vg_scheduler=scheduler,
    )

    _verified, tier_used, _energy = pipeline.verify("noisy response", question="q")
    assert tier_used == "ising"
    assert ising_called[0] is True


# ---------------------------------------------------------------------------
# Export test — carnot.pipeline.__init__ exposes the new symbols
# ---------------------------------------------------------------------------


def test_exported_from_pipeline_init():
    """VGScheduleResult and VGSearchScheduler are importable from carnot.pipeline.  REQ-VERIFY-171."""
    assert _ExportedResult is VGScheduleResult
    assert _ExportedScheduler is VGSearchScheduler
