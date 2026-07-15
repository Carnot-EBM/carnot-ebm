"""Regression test for the submission-prep pre-flight hang incident (2026-07-14):
ColorBlobSaliencePrior.score() recomputed the full-grid connected-component decomposition on
EVERY candidate action instead of reusing the one action_tier_rows() already computed, turning
an O(grid_cells) per-frame cost into an O(candidates x grid_cells) per-call one -- indistinguishable
from a hang on a large grid with many candidates. Found via a live faulthandler stack trace on a
real hung lp85 run, not guesswork.

Spec refs: REQ-ARC-FCP-5591-3, SCENARIO-ARC-FCP-5591-3-PER-FRAME-CACHE-NOT-PER-CANDIDATE.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from carnot.agentic import arc_color_blob_salience as mod
from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior


class _Frame:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame


def _click_candidates(n: int) -> list[dict]:
    """n distinct click candidates spread across a grid -- action_id=6 is the click action per
    _candidate_action_id's convention (matches score()'s own `!= 6` early-return check)."""
    return [{"action": 6, "data": {"x": i % 10, "y": (i // 10) % 10}} for i in range(n)]


def _grid_with_blobs() -> np.ndarray:
    grid = np.zeros((10, 10), dtype=np.int16)
    grid[1:4, 1:4] = 7  # a real salient-colored blob so score() takes the non-trivial path
    grid[6:9, 6:9] = 8
    return grid


def test_req_arc_fcp_5591_3_action_tier_rows_computes_blobs_once_not_per_candidate() -> None:
    """SCENARIO-ARC-FCP-5591-3: the exact bug -- N candidates must NOT trigger N calls to the
    expensive full-grid flood-fill; it must be computed once per action_tier_rows() invocation
    and threaded through every score() call."""

    prior = ColorBlobSaliencePrior()
    frame = _Frame(_grid_with_blobs())
    candidates = _click_candidates(25)

    with patch.object(mod, "connected_color_blobs", wraps=mod.connected_color_blobs) as spy:
        prior.action_tier_rows(frame, candidates)

    # Before the fix: 1 call inside action_tier_rows' own top-level computation + 1 call PER
    # candidate inside score() (25 candidates -> 26 total calls). After the fix: exactly 1.
    assert spy.call_count == 1, (
        f"expected exactly 1 connected_color_blobs() call for {len(candidates)} candidates, "
        f"got {spy.call_count} -- the per-candidate cache is not being reused"
    )


def test_req_arc_fcp_5591_3_score_recomputes_when_no_cache_given() -> None:
    """The public two-arg score(frame, candidate) protocol (used by every OTHER action-prior
    caller in this codebase) must still work standalone, computing its own blobs when no cache
    is passed -- the fix must not break the shared action-prior interface."""

    prior = ColorBlobSaliencePrior()
    frame = _Frame(_grid_with_blobs())
    candidate = {"action": 6, "data": {"x": 2, "y": 2}}

    with patch.object(mod, "connected_color_blobs", wraps=mod.connected_color_blobs) as spy:
        direct_score = prior.score(frame, candidate)

    assert spy.call_count == 1
    assert direct_score >= 0.0


def test_req_arc_fcp_5591_3_cached_and_uncached_scores_agree() -> None:
    """The cache is purely a performance optimization -- passing a pre-computed blobs/
    color_counts must produce the IDENTICAL score as letting score() compute them itself."""

    prior = ColorBlobSaliencePrior()
    frame = _Frame(_grid_with_blobs())
    candidate = {"action": 6, "data": {"x": 2, "y": 2}}

    uncached = prior.score(frame, candidate)

    grid = mod._as_grid(frame)
    blobs = mod.connected_color_blobs(
        grid,
        min_pixels=prior.min_pixels,
        max_component_fraction=prior.max_component_fraction,
    )
    from collections import Counter

    color_counts = Counter(int(value) for value in grid.flatten().tolist())
    cached = prior.score(frame, candidate, blobs=blobs, color_counts=color_counts)

    assert cached == uncached


def test_req_arc_fcp_5591_3_action_tier_rows_output_unchanged_by_the_fix() -> None:
    """The fix changes HOW blobs/color_counts are computed (once vs per-candidate), not WHAT
    action_tier_rows() returns -- rows must be identical to a reference computed via the
    original (pre-fix) per-candidate-recompute code path."""

    prior = ColorBlobSaliencePrior()
    frame = _Frame(_grid_with_blobs())
    candidates = _click_candidates(12) + [{"action": 3, "data": {}}]  # mix in a non-click action

    fixed_rows = prior.action_tier_rows(frame, candidates)

    # Reference: force score() to ALWAYS recompute (the pre-fix behavior) by never passing a
    # cache, replicating action_tier_rows' logic inline.
    grid = mod._as_grid(frame)
    reference_rows = []
    for index, candidate in enumerate(candidates):
        action_id = prior._candidate_action_id(candidate)
        data = prior._candidate_data(candidate)
        if action_id != 6 or "x" not in data or "y" not in data:
            reference_rows.append(
                {"index": index, "tier": None, "score": prior.score(frame, candidate)}
            )
            continue
        blobs = mod.connected_color_blobs(
            grid, min_pixels=prior.min_pixels, max_component_fraction=1.0
        )
        blob = prior._blob_for_click(blobs, int(data["x"]), int(data["y"]))
        tier = None if blob is None else int(prior.tier(blob))
        reference_rows.append(
            {"index": index, "tier": tier, "score": prior.score(frame, candidate)}
        )
    reference_rows.sort(
        key=lambda row: (
            99 if row["tier"] is None else row["tier"],
            -float(row["score"] or 0.0),
            row["index"],
        )
    )

    assert [(r["index"], r["tier"]) for r in fixed_rows] == [
        (r["index"], r["tier"]) for r in reference_rows
    ]
    for fixed, reference in zip(fixed_rows, reference_rows, strict=True):
        assert fixed["score"] == reference["score"]
