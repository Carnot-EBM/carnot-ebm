"""REQ-ARC-WMTE-6180: the incremental budget estimator equals the batch pair, step for step.

WHY (2026-08-17). The live explorer called `region_hud_evidence` +
`budget_exhaustion_estimate` over the WHOLE frame history on every action --
O(n^2) per run, measured at 77% of a profiled submission-gate game's wall
clock. The fix is `IncrementalBudgetExhaustionEstimator`. Its output feeds
frontier ordering, so ANY drift silently changes search trajectories. These
tests therefore assert EXACT dict equality between the incremental and batch
implementations at EVERY step of realistic sequences (level-up resets, episode
breaks, unusable frames, Frame-like wrappers, refuse paths), then assert the
per-call cost no longer grows with history length.

Equivalence here is bit-identical, not approximate: both implementations run
the same arithmetic on the same values in the same order, and the assertions
are plain `==` on the full result dicts.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from carnot.agentic.arc_hud_bar_detector import (
    IncrementalBudgetExhaustionEstimator,
    budget_exhaustion_estimate,
    region_hud_evidence,
)


class _FrameLike:
    """Minimal stand-in for an arcengine frame object (`.frame` attribute)."""

    def __init__(self, frame: Any) -> None:
        self.frame = frame


def _hud_mask(shape: tuple[int, int] = (24, 24), cells: int = 12) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :cells] = True
    return mask


def _admit_sequence(n: int, *, seed: int = 6180) -> list[Any]:
    """A realistic ADMIT-path sequence: a monotone HUD counter over a changing game area.

    Includes a level-up reset (the segment restart rule), episode breaks (None),
    a shape-mismatch frame, a Frame-like wrapper, and a 3-D frame, so every
    branch of the batch fold is exercised, not only the happy path.
    """

    rng = np.random.default_rng(seed)
    shape = (24, 24)
    frames: list[Any] = []
    base = rng.integers(0, 9, size=shape).astype(np.int16)
    counter = 0
    for i in range(n):
        if i in (37, 171):
            frames.append(None)  # episode break
            continue
        if i == 53:
            frames.append(np.zeros((5, 5), dtype=np.int16))  # shape mismatch -> unusable
            continue
        if i > 0 and i % 60 == 0:
            counter = 0  # level-up: the HUD counter returns to its reset value
        grid = base.copy()
        # The game area (complement) changes on ~70% of steps; the HUD region
        # ticks EVERY step -- the action-ubiquitous monotone signature.
        if rng.random() < 0.7:
            y = int(rng.integers(1, shape[0]))
            x = int(rng.integers(0, shape[1]))
            base = base.copy()
            base[y, x] = int(rng.integers(0, 9))
            grid = base.copy()
        counter += 1
        hud_value = min(counter, 200)
        grid[0, :12] = [min(9, max(0, hud_value - 9 * c)) for c in range(12)]
        if i == 88:
            frames.append(_FrameLike(grid))  # arcengine-style wrapper
        elif i == 121:
            frames.append(np.stack([np.zeros(shape, dtype=np.int16), grid]))  # 3-D: last plane
        else:
            frames.append(grid)
    return frames


def _refuse_sequence(n: int, *, seed: int = 6181) -> list[Any]:
    """A REFUSE-path sequence: the region revisits prior values (not monotone)."""

    rng = np.random.default_rng(seed)
    shape = (24, 24)
    frames: list[Any] = []
    base = rng.integers(0, 9, size=shape).astype(np.int16)
    for i in range(n):
        if i == 25:
            frames.append(None)
            continue
        grid = base.copy()
        grid[0, :12] = (i % 3) + 1  # oscillates -> in-episode revisits -> refuse
        if rng.random() < 0.5:
            base = base.copy()
            base[int(rng.integers(1, 24)), int(rng.integers(0, 24))] = int(rng.integers(0, 9))
        frames.append(grid)
    return frames


def _assert_step_equivalence(frames: list[Any], mask: np.ndarray | None) -> None:
    """Drive both implementations step for step and require identical dicts."""

    inc = IncrementalBudgetExhaustionEstimator(mask)
    for k in range(1, len(frames) + 1):
        inc.observe(frames[k - 1])
        batch_ev = region_hud_evidence(frames[:k], mask)
        batch_est = budget_exhaustion_estimate(frames[:k], mask, evidence=batch_ev)
        assert inc.evidence() == batch_ev, f"evidence diverged at step {k}"
        assert inc.estimate() == batch_est, f"estimate diverged at step {k}"


def test_equivalence_admit_path_step_for_step():
    """The headline test: 420 steps of the admit path, dict-identical at every step."""

    frames = _admit_sequence(420)
    mask = _hud_mask()
    _assert_step_equivalence(frames, mask)


def test_equivalence_matches_live_call_shape_without_evidence_kwarg():
    """The live call passes no `evidence=`; spot-check that shape agrees too."""

    frames = _admit_sequence(140)
    mask = _hud_mask()
    inc = IncrementalBudgetExhaustionEstimator(mask)
    for k in range(1, len(frames) + 1):
        inc.observe(frames[k - 1])
        if k in (1, 17, 61, 100, 140):
            assert inc.estimate() == budget_exhaustion_estimate(frames[:k], mask)


def test_equivalence_refuse_path_step_for_step():
    frames = _refuse_sequence(120)
    mask = _hud_mask()
    _assert_step_equivalence(frames, mask)


def test_equivalence_no_mask_and_empty_mask():
    frames = _admit_sequence(30)
    _assert_step_equivalence(frames, None)
    _assert_step_equivalence(frames, np.zeros((24, 24), dtype=bool))


def test_equivalence_non_bool_mask_input():
    """The batch functions asarray-coerce the mask; the class must match that path."""

    frames = _admit_sequence(60)
    mask = _hud_mask().astype(np.int16)  # non-bool input, same cells
    _assert_step_equivalence(frames, mask)


def test_ingest_call_site_equivalence_including_mask_swap():
    """Integration form at the REAL call site: `_ingest`'s adopted estimate equals the
    batch call over the buffer at every step, including a mid-run mask replacement
    (the deferred-activation pattern) which must trigger a rebuild, not stale state."""

    from carnot.agentic.arc_competition_agent import StepwiseExplorer

    frames = _admit_sequence(120)
    mask_a = _hud_mask()
    mask_b = _hud_mask(cells=10)  # a DIFFERENT admitted mask object mid-run
    explorer = StepwiseExplorer(budget_aware_search=True)
    explorer.hud_mask = mask_a
    for i, frame in enumerate(frames):
        if i == 70:
            explorer.hud_mask = mask_b
        explorer._ingest(frame)
        expected = budget_exhaustion_estimate(explorer._budget_frames, explorer.hud_mask)[
            "actions_remaining_estimate"
        ]
        assert explorer.actions_remaining_estimate == expected, f"call-site diverged at step {i}"


def test_per_call_cost_does_not_scale_with_history_length():
    """The complexity claim: per-call (observe + estimate) cost is flat in n.

    The batch pair costs O(n) per call, so its late/early chunk ratio grows
    roughly like n. The incremental estimator's ratio must stay near 1; the
    4x bound is deliberate slack so a loaded box cannot flake this test.
    """

    frames = _admit_sequence(2200)
    mask = _hud_mask()
    inc = IncrementalBudgetExhaustionEstimator(mask)

    def _chunk_seconds(start: int, stop: int) -> float:
        t0 = time.perf_counter()
        for i in range(start, stop):
            inc.observe(frames[i])
            inc.estimate()
        return time.perf_counter() - t0

    _chunk_seconds(0, 100)  # warm-up: first-touch allocation noise stays out of both chunks
    early = _chunk_seconds(100, 300)
    for i in range(300, 2000):
        inc.observe(frames[i])
    late = _chunk_seconds(2000, 2200)
    assert late < early * 4.0, (
        f"per-call cost grew with history length: early(100..300)={early:.4f}s "
        f"late(2000..2200)={late:.4f}s -- the O(n^2) regression is back"
    )
