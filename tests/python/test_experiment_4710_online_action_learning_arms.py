"""Tests for REQ-ARC-OAE-4710 / SCENARIO-ARC-OAE-4710 -- online action-effect scorer.

All tests use synthetic frames / canned data; no live ARC engine is required.
Every test has at least one assertion. No pytest.mark.skip allowed (CLAUDE.md Tests Must Run).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from carnot.agentic.arc_frame_change_predictor import (
    FrameChangeScorer,
    LiveActionEffectScorer,
    SmallFrameChangeCNN,
    frame_state_key,
)
from carnot.agentic.arc_online_action_effect_scorer import (
    OnlineActionEffectScorer,
    build_online_scorer,
)

# ---------------------------------------------------------------------------
# Synthetic test helpers
# ---------------------------------------------------------------------------


def _make_grid(value: int = 0, shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    """A tiny grid filled with a constant color value."""
    return np.full(shape, value, dtype=np.int16)


def _make_frame(value: int = 0, shape: tuple[int, int] = (4, 4)) -> Any:
    """A SimpleNamespace frame whose .frame attribute is a tiny grid."""
    ns = SimpleNamespace()
    ns.frame = _make_grid(value, shape)
    return ns


def _make_candidate(action_id: int, x: int | None = None, y: int | None = None) -> Any:
    """A minimal candidate object exposing .action_id and .data."""
    data = {}
    if x is not None:
        data["x"] = x
    if y is not None:
        data["y"] = y
    return SimpleNamespace(action_id=action_id, data=data)


def _fresh_scorer(hidden_channels: int = 8) -> OnlineActionEffectScorer:
    """Build an OnlineActionEffectScorer with no memory and a random CNN."""
    cnn = FrameChangeScorer(SmallFrameChangeCNN(num_colors=16, hidden_channels=hidden_channels))
    return OnlineActionEffectScorer(memory=None, cnn_scorer=cnn, train_enabled=True)


def _unchanged_frame() -> Any:
    """Two identical frames -- observing a transition between them signals no change."""
    return _make_frame(0), _make_frame(0)


def _changed_frame() -> Any:
    """Two frames that differ -- observing a transition signals a change."""
    return _make_frame(0), _make_frame(3)


# ---------------------------------------------------------------------------
# Test 1: candidate_score equals LiveActionEffectScorer's blend before any fit
# ---------------------------------------------------------------------------


def test_req_arc_oae_4710_candidate_score_matches_live_blend_before_fit() -> None:
    """REQ-ARC-OAE-4710: OnlineActionEffectScorer.candidate_score equals the LiveActionEffectScorer
    blend when the model has not been fitted yet (weights are identical to initialization).

    WHY: the shipped agent uses LiveActionEffectScorer(memory=None, cnn_scorer=scorer).
    OnlineActionEffectScorer must produce the same score before any online update so the
    arm="frozen" path is byte-identical and the arm="online-warm" starts from parity.
    """
    # REQ-ARC-OAE-4710 SCENARIO: pre-fit score parity

    torch.manual_seed(42)
    cnn_model = SmallFrameChangeCNN(num_colors=16, hidden_channels=8)

    # Shared weights -- both scorers reference the same model instance.
    shared_cnn = FrameChangeScorer(cnn_model)
    live = LiveActionEffectScorer(memory=None, cnn_scorer=shared_cnn, cnn_weight=0.05)
    online = OnlineActionEffectScorer(
        memory=None,
        cnn_scorer=shared_cnn,
        cnn_weight=0.05,
        train_enabled=False,  # no training -- pure parity check
    )

    frame = _make_frame(1)
    candidate_dir = _make_candidate(action_id=1)  # directional action
    candidate_click = _make_candidate(action_id=6, x=1, y=1)

    for candidate in (candidate_dir, candidate_click):
        score_live = live.candidate_score(frame, candidate)
        score_online = online.candidate_score(frame, candidate)
        assert abs(score_live - score_online) < 1e-6, (
            f"Scores differ for action {candidate.action_id}: live={score_live}, online={score_online}"
        )


# ---------------------------------------------------------------------------
# Test 2: after >= 5 CHANGED transitions, weights change AND cache is cleared
# ---------------------------------------------------------------------------


def test_req_arc_oae_4710_weights_change_and_cache_cleared_after_fit() -> None:
    """REQ-ARC-OAE-4710: after accumulating >=fit_every changed transitions, the model's
    weights change AND the CNN prediction cache is empty.

    WHY weight must change: if the optimizer is broken, no learning happens regardless of
    how many transitions we buffer. A weight change proves the backward pass ran.

    WHY cache must be empty: the _maybe_fit() function MUST clear cnn_scorer._cache after
    every gradient step so the explorer doesn't serve stale predictions (risk #4 per spec).
    """
    # REQ-ARC-OAE-4710 SCENARIO: online learning triggers a weight update

    torch.manual_seed(99)
    scorer = _fresh_scorer()

    # Capture a snapshot of one parameter BEFORE any training.
    before_param = copy.deepcopy(list(scorer.cnn_scorer.model.parameters())[0].data)

    # Pre-warm the cache so we can confirm it is cleared after fitting.
    frame_before, frame_after = _changed_frame()
    _ = scorer.candidate_score(frame_before, _make_candidate(1))
    assert len(scorer.cnn_scorer._cache) > 0, "Cache should be non-empty after a score call"

    # Observe fit_every=5 CHANGED transitions (frame_delta=1).
    for i in range(scorer.fit_every):
        before, after = _changed_frame()
        scorer.observe_transition(before, action_id=1 + (i % 5), data=None, after_frame=after)

    # Assert: weights changed.
    after_param = list(scorer.cnn_scorer.model.parameters())[0].data
    weight_changed = not torch.equal(before_param, after_param)
    assert weight_changed, "Model weights must change after at least one gradient step"

    # Assert: _fits counter incremented.
    assert scorer._fits >= 1, f"Expected at least 1 fit, got {scorer._fits}"

    # Assert: cache was cleared after fit.
    assert len(scorer.cnn_scorer._cache) == 0, (
        "CNN prediction cache must be empty after fit (stale-cache risk #4)"
    )

    # Assert: observed counter matches the number of observations we sent.
    assert scorer._observed == scorer.fit_every, (
        f"_observed={scorer._observed} != fit_every={scorer.fit_every}"
    )


# ---------------------------------------------------------------------------
# Test 3: build_online_scorer("frozen") returns a LiveActionEffectScorer (no _optimizer)
# ---------------------------------------------------------------------------


def test_req_arc_oae_4710_frozen_arm_is_live_action_effect_scorer(tmp_path: Any) -> None:
    """REQ-ARC-OAE-4710: build_online_scorer("frozen") returns a LiveActionEffectScorer or None
    (not an OnlineActionEffectScorer), so it has no _optimizer and no observe_transition.

    WHY: the frozen arm must be byte-identical to what the shipped E3AgentPolicy uses. The
    guarded hooks in arc_competition_agent.py use `hasattr(fcs, "observe_transition")` before
    calling online methods. A LiveActionEffectScorer has neither attribute so both hooks are
    no-ops -- guaranteed parity with the pre-experiment behavior.

    Note: if the transition corpus / checkpoint are absent in tmp_path the scorer may be None
    (load_live_action_effect_scorer returns None when both memory and CNN are unavailable).
    The assertion is that the result is NOT an OnlineActionEffectScorer, regardless of None.
    """
    # REQ-ARC-OAE-4710 SCENARIO: frozen arm parity guard

    scorer = build_online_scorer("frozen", tmp_path)

    # Must not be an OnlineActionEffectScorer.
    assert not isinstance(scorer, OnlineActionEffectScorer), (
        "frozen arm must NOT return an OnlineActionEffectScorer"
    )
    # If a scorer was returned, it must not have online methods.
    if scorer is not None:
        assert not hasattr(scorer, "observe_transition"), (
            "frozen arm scorer must not have observe_transition"
        )
        assert not hasattr(scorer, "_optimizer"), (
            "frozen arm scorer must not have an _optimizer"
        )
        # The parity hook check: getattr(fcs, "propose_enabled", False) must be False.
        assert getattr(scorer, "propose_enabled", False) is False, (
            "frozen arm scorer propose_enabled must be False"
        )


# ---------------------------------------------------------------------------
# Test 4: propose_coords returns <= k tuples within grid bounds
# ---------------------------------------------------------------------------


def test_req_arc_oae_4710_propose_coords_bounds() -> None:
    """REQ-ARC-OAE-4710: propose_coords returns at most k (x,y) tuples, all within grid bounds.

    WHY: if propose_coords returns out-of-bounds coordinates, the explorer will try to inject
    invalid click actions (e.g. x=-1) into the candidate set. The guarded hook appends these
    to the existing rows, so we must verify they are valid grid coords.
    """
    # REQ-ARC-OAE-4710 SCENARIO: propose_coords returns valid grid coordinates

    scorer = _fresh_scorer()
    scorer.propose_enabled = True

    frame = _make_frame(2, shape=(5, 7))  # 5 rows, 7 cols
    k = 4
    coords = scorer.propose_coords(frame, k=k)

    # At most k results.
    assert len(coords) <= k, f"Expected <= {k} coords, got {len(coords)}"

    # All coords are (x, y) tuples within [0, w-1] x [0, h-1].
    h, w = 5, 7
    for x, y in coords:
        assert 0 <= x < w, f"x={x} out of bounds [0, {w-1}]"
        assert 0 <= y < h, f"y={y} out of bounds [0, {h-1}]"

    # At least 1 coord returned (CNN heatmap always has some maximum).
    assert len(coords) >= 1, "propose_coords should return at least 1 coordinate"


# ---------------------------------------------------------------------------
# Test 5: CI delta computation from two canned measurement dicts
# ---------------------------------------------------------------------------


def test_req_arc_oae_4710_cross_arm_delta_ci_computes_correctly() -> None:
    """REQ-ARC-OAE-4710: the paired_first_win_delta_ci from exp4605 produces a non-trivial
    CI when we supply two canned measurement dicts with differing solve patterns.

    WHY: experiment 4710 uses mod.paired_first_win_delta_ci to compare arms. If the CI
    calculation is broken we would report wrong confidence intervals in the artifact. This
    test validates the function on known input.

    Canned data:
      arm A: games lp85~color01, vc33~color01, dc22~color01 all solved
      arm B: only lp85~color01 solved
      expected delta = (3/3) - (1/3) = 0.667 point estimate (approximately)
    """
    # REQ-ARC-OAE-4710 SCENARIO: cross-arm delta CI computed from canned measurements

    import carnot.experiment_4605_live_integration_scored_agent as mod

    def _attempt(sig: str, solved: bool) -> dict[str, Any]:
        return {
            "variant_signature": sig,
            "attempted": True,
            "first_win": solved,
            "solved": solved,
        }

    sigs = ["lp85~color01", "vc33~color01", "dc22~color01"]
    integrated_attempts = [_attempt(sig, True) for sig in sigs]
    bare_attempts = [_attempt(sigs[0], True), _attempt(sigs[1], False), _attempt(sigs[2], False)]

    ci_result = mod.paired_first_win_delta_ci(
        integrated_attempts, bare_attempts, random_seed=4710, n_bootstrap=200
    )

    assert "point" in ci_result, "CI result must have a 'point' key"
    assert "ci95" in ci_result, "CI result must have a 'ci95' key"
    assert isinstance(ci_result["ci95"], list) and len(ci_result["ci95"]) == 2, (
        "ci95 must be a 2-element list"
    )

    # Point estimate: arm A solves all 3, arm B solves 1. Deltas = [0, 1, 1]. Mean = 2/3.
    expected_point = 2.0 / 3.0
    assert abs(ci_result["point"] - expected_point) < 0.01, (
        f"Point estimate {ci_result['point']:.4f} != expected {expected_point:.4f}"
    )

    # CI lower bound must be >= 0 (n=3 is too small for a tight CI; both arms agree
    # on lp85, so the worst bootstrap replicate can be delta=0 even when the point is 2/3).
    # The important property is that ci95[0] <= point <= ci95[1].
    assert ci_result["ci95"][0] >= 0.0, (
        f"CI lower bound {ci_result['ci95'][0]} must be >= 0"
    )
    assert ci_result["ci95"][0] <= ci_result["point"] <= ci_result["ci95"][1], (
        f"Point estimate {ci_result['point']} must be within CI {ci_result['ci95']}"
    )

    # delta_arm_minus_frozen = integrated - bare first_win_rate.
    delta = (3.0 / 3.0) - (1.0 / 3.0)
    assert abs(delta - expected_point) < 0.01, (
        f"delta {delta:.4f} != expected_point {expected_point:.4f}"
    )
