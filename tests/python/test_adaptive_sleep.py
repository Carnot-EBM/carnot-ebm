"""Tests for the conductor's adaptive inter-iteration sleep logic.

The .71 operational retro found ~80 min of wasted wall time burned on
fixed-interval sleeps after doomed-rerun blocks. `compute_adaptive_sleep_min`
scales the sleep to how much real work the iteration did.

Tier definitions are documented in the function docstring; these tests pin
the boundary behaviour so future tuning has to make a deliberate decision.
Spec: REQ-INFRA-067
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from research_conductor import compute_adaptive_sleep_min  # noqa: E402

# ---------------------------------------------------------------------------
# Short tier: blocks and skips (iter_duration < 30 s)
# ---------------------------------------------------------------------------


def test_short_tier_for_doomed_rerun_block():
    """Doomed-rerun blocks complete in ~0.5 sec → short tier."""
    sleep_min, tier = compute_adaptive_sleep_min(0.5, interval_min=10)
    assert sleep_min == 1
    assert "short" in tier


def test_short_tier_for_deliverable_skip():
    """Deliverable-already-exists skips complete in <1 sec → short tier."""
    sleep_min, tier = compute_adaptive_sleep_min(0.05, interval_min=10)
    assert sleep_min == 1
    assert "short" in tier


def test_short_tier_floor_at_1_min():
    """Even with a tiny configured interval, short tier sleeps ≥ 1 min."""
    sleep_min, _ = compute_adaptive_sleep_min(1.0, interval_min=5)
    assert sleep_min == 1  # max(1, 5 // 10) = max(1, 0) = 1


def test_short_tier_scales_with_larger_interval():
    """A 30-min configured interval gets ~3 min short sleep."""
    sleep_min, _ = compute_adaptive_sleep_min(1.0, interval_min=30)
    assert sleep_min == 3


def test_short_tier_just_below_30s_boundary():
    """29.9 s iteration is still short tier."""
    sleep_min, tier = compute_adaptive_sleep_min(29.9, interval_min=10)
    assert sleep_min == 1
    assert "short" in tier


# ---------------------------------------------------------------------------
# Medium tier: CPU experiments (30 s ≤ iter_duration < 5 min)
# ---------------------------------------------------------------------------


def test_medium_tier_at_30s_boundary():
    """Exactly 30 s graduates to medium tier."""
    sleep_min, tier = compute_adaptive_sleep_min(30.0, interval_min=10)
    assert sleep_min == 5
    assert "medium" in tier


def test_medium_tier_for_cpu_experiment():
    """A 2-minute CPU experiment is medium tier."""
    sleep_min, tier = compute_adaptive_sleep_min(120.0, interval_min=10)
    assert sleep_min == 5
    assert "medium" in tier


def test_medium_tier_floor_at_2_min():
    """Even with a tiny configured interval, medium tier sleeps ≥ 2 min."""
    sleep_min, _ = compute_adaptive_sleep_min(60.0, interval_min=3)
    assert sleep_min == 2  # max(2, 3 // 2) = max(2, 1) = 2


def test_medium_tier_just_below_5_min_boundary():
    """4 min 59 s is still medium tier — the 5-min cache-window boundary."""
    sleep_min, tier = compute_adaptive_sleep_min(299.0, interval_min=10)
    assert sleep_min == 5
    assert "medium" in tier


# ---------------------------------------------------------------------------
# Long tier: GPU experiments and planner runs (iter_duration ≥ 5 min)
# ---------------------------------------------------------------------------


def test_long_tier_at_5_min_boundary():
    """Exactly 5 min graduates to long tier."""
    sleep_min, tier = compute_adaptive_sleep_min(300.0, interval_min=10)
    assert sleep_min == 10
    assert "long" in tier


def test_long_tier_for_gpu_experiment():
    """A 25-minute GPU experiment is long tier — full interval sleep."""
    sleep_min, tier = compute_adaptive_sleep_min(1500.0, interval_min=10)
    assert sleep_min == 10
    assert "long" in tier


def test_long_tier_for_planner_run():
    """A 15-minute planner Sonnet run is long tier — full interval sleep."""
    sleep_min, tier = compute_adaptive_sleep_min(900.0, interval_min=10)
    assert sleep_min == 10
    assert "long" in tier


def test_long_tier_returns_full_interval():
    """Long tier always returns the configured interval verbatim."""
    for interval in [5, 10, 15, 30, 60]:
        sleep_min, _ = compute_adaptive_sleep_min(600.0, interval_min=interval)
        assert sleep_min == interval


# ---------------------------------------------------------------------------
# Net savings — the .71 retro motivation
# ---------------------------------------------------------------------------


def test_71_retro_savings_estimate():
    """The .71 milestone had 8 doomed-rerun blocks (≤1 s each) and 4 real
    iterations (mix of CPU/GPU). With fixed 10-min sleep that's 110 min total.
    With adaptive: 8×1 + 2×5 + 2×10 = 38 min — saving ~72 min on .71's
    sleep budget. Pin the math here so future tuning can't silently regress."""
    interval = 10
    block_iters = [0.5] * 8
    cpu_iters = [120.0, 200.0]
    gpu_iters = [1500.0, 900.0]
    sleeps = [
        compute_adaptive_sleep_min(d, interval)[0] for d in block_iters + cpu_iters + gpu_iters
    ]
    adaptive_total = sum(sleeps)
    fixed_total = interval * len(sleeps)
    assert adaptive_total == 38
    assert fixed_total == 120
    assert fixed_total - adaptive_total == 82  # close to retro's ~80 min finding
