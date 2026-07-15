"""Regression test for the second submission-prep pre-flight incident (2026-07-15):
CrossGameDiscriminativeCandidateRouter.rank() called score() once per candidate action, and
score() independently recomputed cross_game_features_v3()'s expensive frame-level pieces
(_object_relational_features's O(components^2) greedy frame-matching loop, plus v2/frame-delta/
predicate-distance features) from scratch every time, even though only the cheap
_action_features(action_id) slice actually depends on the per-candidate action_id. Structurally
identical to the arc_color_blob_salience.py incident fixed earlier the same day
(REQ-ARC-FCP-5591-3) -- same anti-pattern, different module. Found via a live faulthandler
stack trace on a real slow lp85 run (not guesswork): the hang bottomed out in
arc_value_learner.py:_component_stats_from_grid via arc_discriminative_router.py:score/rank.

Spec refs: REQ-CAPSTONE-4556-2, SCENARIO-CAPSTONE-4556-2-PER-FRAME-CONTEXT-NOT-PER-CANDIDATE.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from carnot.agentic import arc_discriminative_router as router_mod
from carnot.agentic import arc_value_learner as learner_mod
from carnot.agentic.arc_discriminative_router import CrossGameDiscriminativeCandidateRouter
from carnot.agentic.arc_value_learner import cross_game_features_v3, cross_game_frame_context_v3


class _Frame:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame


class _Action:
    def __init__(self, action_id: int, x: int = 0, y: int = 0) -> None:
        self.action_id = action_id
        self.data = {"x": x, "y": y}


class _StubVerifier:
    def proba_features(self, features: list[float]) -> float:
        return 0.5


def _grid_with_components() -> np.ndarray:
    grid = np.zeros((12, 12), dtype=np.int16)
    grid[1:3, 1:3] = 4
    grid[5:8, 5:8] = 7
    grid[9:11, 9:11] = 2
    return grid


def _candidates(n: int) -> list[_Action]:
    return [_Action(6, x=i % 12, y=(i // 12) % 12) for i in range(n)]


def test_req_capstone_4556_2_cross_game_features_v3_cached_matches_uncached() -> None:
    """The cache is purely a performance optimization -- passing a pre-computed frame_context
    must produce the IDENTICAL feature vector as computing everything fresh."""

    frame = _Frame(_grid_with_components())
    prev = _Frame(_grid_with_components())

    uncached = cross_game_features_v3(frame, previous_frame=prev, action_id=6, goal_frame=None)
    ctx = cross_game_frame_context_v3(frame, prev, None)
    cached = cross_game_features_v3(
        frame, previous_frame=prev, action_id=6, goal_frame=None, frame_context=ctx
    )

    assert cached == uncached


def test_req_capstone_4556_2_frame_context_varies_only_by_frame_not_action() -> None:
    """Two different action_ids against the SAME frame must produce feature vectors that are
    identical everywhere EXCEPT the action-conditioned slice -- confirming the frame_context
    cache genuinely captures the action-independent part."""

    frame = _Frame(_grid_with_components())
    ctx = cross_game_frame_context_v3(frame, None, None)

    features_a = cross_game_features_v3(frame, action_id=1, frame_context=ctx)
    features_b = cross_game_features_v3(frame, action_id=6, frame_context=ctx)

    diff_indices = [i for i, (a, b) in enumerate(zip(features_a, features_b)) if a != b]
    assert diff_indices, "expected the action-conditioned slice to differ"
    # The action-conditioned slice is a small one-hot-style run (a fixed "is-set" flag plus a
    # 1-of-6 action-id bit) -- not scattered across the whole feature vector.
    n_action_features = len(learner_mod._action_features(1))
    assert max(diff_indices) - min(diff_indices) < n_action_features


def test_req_capstone_4556_2_router_rank_computes_frame_context_once_not_per_candidate() -> None:
    """SCENARIO-CAPSTONE-4556-2: the exact bug -- N candidates must NOT trigger N calls to the
    expensive component-stats computation; it must be computed once per rank() call."""

    router = CrossGameDiscriminativeCandidateRouter(_StubVerifier())
    frame = _Frame(_grid_with_components())
    candidates = _candidates(20)

    with patch.object(
        learner_mod, "_component_stats_from_grid", wraps=learner_mod._component_stats_from_grid
    ) as spy:
        router.rank(frame, candidates)

    # _component_stats_from_grid is called twice per frame_context computation (once for the
    # current frame via _object_relational_features, once inside _frame_delta_features for both
    # cur+prev -- with previous_frame=None here, _object_relational_features short-circuits after
    # one call and _frame_delta_features returns early too since previous_frame is None). The
    # precise count only matters relative to N: before the fix it scaled with len(candidates);
    # after the fix it's a small constant independent of candidate count.
    assert spy.call_count < len(candidates), (
        f"expected component-stats calls to NOT scale with {len(candidates)} candidates, "
        f"got {spy.call_count} calls -- the per-candidate cache is not being reused"
    )


def test_req_capstone_4556_2_router_rank_output_unchanged_by_the_fix() -> None:
    """The fix changes HOW frame-level features are computed (once vs per-candidate), not WHAT
    rank() returns -- ranked order must match a reference computed via the original (always-
    recompute) code path."""

    router = CrossGameDiscriminativeCandidateRouter(_StubVerifier())
    frame = _Frame(_grid_with_components())
    candidates = _candidates(9)

    fixed_order = [id(a) for a in router.rank(frame, candidates)]

    reference_scored = [
        (router.score(frame, action, previous_frame=None, frame_context=None), index, action)
        for index, action in enumerate(candidates)
    ]
    reference_order = [
        id(action)
        for _score, _index, action in sorted(reference_scored, key=lambda item: (-item[0], item[1]))
    ]

    assert fixed_order == reference_order
