"""Tests for the additive online coordinate-aware click router.

Spec refs: REQ-ARC-FCP-5904, SCENARIO-ARC-FCP-5904-COORDINATE-BLINDNESS-IS-REPAIRED,
SCENARIO-ARC-FCP-5904-COLD-START-IS-A-NO-OP, SCENARIO-ARC-FCP-5904-EPISODE-ISOLATION,
SCENARIO-ARC-FCP-5904-PER-FRAME-NOT-PER-CANDIDATE.

Two of these tests are the load-bearing ones:

* ``test_flag_off_reproduces_the_coordinate_blind_defect_exactly`` -- locks in live parity.
  It asserts the DEFECT is still present with the flag off (one distinct click score, input
  order preserved), which is what makes an accidental default flip a test failure rather than
  a silent live behaviour change.
* ``test_flag_on_with_a_fitted_head_discriminates_click_targets`` -- asserts the defect is
  actually repaired when the flag is on.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from carnot.agentic import arc_click_target_features as feat_mod
from carnot.agentic import arc_discriminative_router as router_mod
from carnot.agentic.arc_click_target_features import (
    CLICK_TARGET_FEATURE_DIM,
    OnlineClickTargetDiscriminator,
    click_target_features,
    click_target_frame_context,
)
from carnot.agentic.arc_discriminative_router import (
    CrossGameDiscriminativeCandidateRouter,
    OnlineClickTargetRouter,
    load_online_click_target_router,
)
from carnot.agentic.arc_value_learner import (
    cross_game_feature_slices_v3,
    cross_game_features_v3,
)


class _Frame:
    def __init__(self, grid: np.ndarray, game_id: str = "gm00-test", guid: str = "guid-a") -> None:
        self.frame = grid
        self.game_id = game_id
        self.guid = guid


class _Action:
    def __init__(self, action_id: int, x: int | None = None, y: int | None = None) -> None:
        self.action_id = action_id
        self.data = None if x is None else {"x": x, "y": y}


class _ConstantVerifier:
    """Stands in for the real v3 verifier: constant because v3 is coordinate-blind.

    This is not a convenience -- it is the measured behaviour. Feeding the real checkpoint any
    two clicks on the same frame yields the same probability, because
    ``cross_game_features_v3`` only ever sees the action TYPE integer.
    """

    def __init__(self, value: float = 0.42) -> None:
        self.value = float(value)
        self.seen: list[list[float]] = []

    def proba_features(self, features: list[float]) -> float:
        self.seen.append(list(features))
        return self.value


def _grid() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=np.int16)
    grid[2:14, 2:14] = 3
    for index, (y, x) in enumerate([(16, 2), (16, 8), (16, 14), (20, 2), (20, 8), (20, 14)]):
        grid[y : y + 2, x : x + 2] = 6 + index
    grid[0, :] = 16
    grid[6, 20] = 9
    return grid


# A curated target list covering EVERY region class in ``_grid()`` -- the six buttons, the
# flat field, the status strip, the rare pixel, and plain background. A naive row-major
# stride would sample only the top rows, miss the buttons, and make the discrimination
# assertions pass or fail for the wrong reason.
_CLICK_TARGETS: tuple[tuple[int, int], ...] = (
    (2, 16),
    (8, 16),
    (14, 16),
    (2, 20),
    (8, 20),
    (14, 20),  # the six buttons
    (4, 4),
    (6, 6),
    (8, 8),
    (10, 10),
    (12, 4),
    (4, 12),  # inside the flat field
    (0, 0),
    (12, 0),
    (22, 0),  # status strip
    (20, 6),  # the single rare pixel
    (20, 20),
    (22, 22),
    (18, 2),
    (22, 10),
    (16, 22),
    (20, 12),
    (22, 4),
    (18, 18),  # background
)


def _click_candidates(limit: int = 24) -> list[_Action]:
    return [_Action(6, x, y) for x, y in _CLICK_TARGETS[:limit]]


def _fit_head_on_frame(frame: _Frame, candidates: list[_Action]) -> OnlineClickTargetDiscriminator:
    """Fit a head with a label that depends on the CLICK (button-likeness of the target)."""

    head = OnlineClickTargetDiscriminator()
    ctx = click_target_frame_context(frame, use_cache=False)
    for action in candidates:
        vector = click_target_features(ctx, action.data["x"], action.data["y"])
        head.observe(vector, 1.0 if vector[3] > 0.5 else 0.0)
    assert head.fit() is True, "fixture must produce both classes"
    return head


# ------------------------------------------------------------------ default-off parity


def test_module_flag_defaults_off() -> None:
    """REQ-ARC-FCP-5904: shipping default-on would change live behaviour with no A/B."""

    assert router_mod.SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED is False
    assert OnlineClickTargetRouter(None).enabled is False
    assert load_online_click_target_router().enabled is False


def test_flag_off_reproduces_the_coordinate_blind_defect_exactly() -> None:
    """SCENARIO-ARC-FCP-5904-COORDINATE-BLINDNESS-IS-REPAIRED (the OFF half).

    With the flag off the wrapper must be indistinguishable from the incumbent router:
    exactly ONE distinct click score, and ``rank()`` a stable no-op. Asserting the defect is
    still present is deliberate -- it is what turns an accidental default flip into a test
    failure.
    """

    frame = _Frame(_grid())
    candidates = _click_candidates()
    base = CrossGameDiscriminativeCandidateRouter(_ConstantVerifier())
    wrapped = OnlineClickTargetRouter(base, enabled=False)

    base_scores = [base.score(frame, action) for action in candidates]
    wrapped_scores = [wrapped.score(frame, action) for action in candidates]
    assert len(set(base_scores)) == 1
    assert wrapped_scores == base_scores  # byte-identical floats

    assert [id(a) for a in wrapped.rank(frame, candidates)] == [
        id(a) for a in base.rank(frame, candidates)
    ]
    assert [id(a) for a in wrapped.rank(frame, candidates)] == [id(a) for a in candidates]


def test_flag_off_does_no_click_featurization_work() -> None:
    """REQ-ARC-FCP-5904: off must cost nothing, not merely produce the same numbers."""

    frame = _Frame(_grid())
    candidates = _click_candidates()
    wrapped = OnlineClickTargetRouter(
        CrossGameDiscriminativeCandidateRouter(_ConstantVerifier()), enabled=False
    )
    with patch.object(
        feat_mod, "connected_color_blobs", wraps=feat_mod.connected_color_blobs
    ) as spy:
        wrapped.rank(frame, candidates)
        for action in candidates:
            wrapped.score(frame, action)
        assert spy.call_count == 0


# ----------------------------------------------------------------------- cold start


def test_cold_start_is_an_exact_no_op_even_with_the_flag_on() -> None:
    """SCENARIO-ARC-FCP-5904-COLD-START-IS-A-NO-OP."""

    frame = _Frame(_grid())
    candidates = _click_candidates()
    base = CrossGameDiscriminativeCandidateRouter(_ConstantVerifier())
    router = OnlineClickTargetRouter(base, enabled=True)

    assert router.discriminator_for(frame).fitted is False
    for action in candidates:
        assert router.score(frame, action) == base.score(frame, action)
        assert router.click_delta(frame, action) == 0.0
    assert [id(a) for a in router.rank(frame, candidates)] == [
        id(a) for a in base.rank(frame, candidates)
    ]


def test_partial_observations_below_the_gate_stay_a_no_op() -> None:
    """SCENARIO-ARC-FCP-5904-COLD-START-IS-A-NO-OP: a couple of clicks must not be enough."""

    grid = _grid()
    frame_before = _Frame(grid)
    frame_after = _Frame(grid.copy())
    frame_after.frame[22, 22] = 5  # something changed -> a positive label

    base = CrossGameDiscriminativeCandidateRouter(_ConstantVerifier())
    router = OnlineClickTargetRouter(base, enabled=True)
    assert router.observe_click_outcome(frame_before, _Action(6, 2, 16), frame_after) is True
    assert router.observe_click_outcome(frame_before, _Action(6, 8, 16), frame_before) is True

    head = router.discriminator_for(frame_before)
    assert head.stats()["n_samples"] == 2
    assert head.gate_met is False
    assert head.fitted is False
    for action in _click_candidates():
        assert router.score(frame_before, action) == base.score(frame_before, action)


# ----------------------------------------------------------------- the repair itself


def test_flag_on_with_a_fitted_head_discriminates_click_targets() -> None:
    """SCENARIO-ARC-FCP-5904-COORDINATE-BLINDNESS-IS-REPAIRED (the ON half).

    The direct assertion that the defect is repaired: many distinct scores where the
    incumbent produces one, and a genuine reordering of the click candidates.
    """

    frame = _Frame(_grid())
    candidates = _click_candidates()
    base = CrossGameDiscriminativeCandidateRouter(_ConstantVerifier())
    head = _fit_head_on_frame(frame, candidates)
    router = OnlineClickTargetRouter(base, enabled=True, discriminator=head)

    scores = [router.score(frame, action) for action in candidates]
    assert len(set(scores)) > 1
    assert len(set(scores)) >= 0.5 * len(candidates), (len(set(scores)), len(candidates))

    ranked = router.rank(frame, candidates)
    assert [id(a) for a in ranked] != [id(a) for a in candidates]
    assert sorted(id(a) for a in ranked) == sorted(id(a) for a in candidates)


def test_online_fit_puts_button_like_targets_first() -> None:
    """REQ-ARC-FCP-5904: the ranking must reflect the learned signal, not just move."""

    frame = _Frame(_grid())
    candidates = _click_candidates()
    head = _fit_head_on_frame(frame, candidates)
    router = OnlineClickTargetRouter(
        CrossGameDiscriminativeCandidateRouter(_ConstantVerifier()),
        enabled=True,
        discriminator=head,
        weight=0.25,
    )
    ctx = click_target_frame_context(frame, use_cache=False)

    def is_positive(action: _Action) -> bool:
        return click_target_features(ctx, action.data["x"], action.data["y"])[3] > 0.5

    positives = [a for a in candidates if is_positive(a)]
    assert positives, "fixture must contain button-like targets"
    ranked = router.rank(frame, candidates)
    first_positive_rank = next(i for i, a in enumerate(ranked) if is_positive(a))
    baseline_rank = next(i for i, a in enumerate(candidates) if is_positive(a))
    assert first_positive_rank <= baseline_rank


def test_weight_zero_leaves_the_base_ordering_intact() -> None:
    """REQ-ARC-FCP-5904: the blend weight is the operator's safety dial."""

    frame = _Frame(_grid())
    candidates = _click_candidates()
    base = CrossGameDiscriminativeCandidateRouter(_ConstantVerifier())
    head = _fit_head_on_frame(frame, candidates)
    router = OnlineClickTargetRouter(base, enabled=True, discriminator=head, weight=0.0)
    for action in candidates:
        assert router.score(frame, action) == base.score(frame, action)
    assert [id(a) for a in router.rank(frame, candidates)] == [id(a) for a in candidates]


def test_non_click_candidates_contribute_exactly_zero() -> None:
    """REQ-ARC-FCP-5904: keyboard candidates keep their v3-governed placement."""

    frame = _Frame(_grid())
    candidates = _click_candidates()
    head = _fit_head_on_frame(frame, candidates)
    router = OnlineClickTargetRouter(
        CrossGameDiscriminativeCandidateRouter(_ConstantVerifier()),
        enabled=True,
        discriminator=head,
    )
    for action in (_Action(1), _Action(4), _Action(6), _Action(0)):
        assert router.click_delta(frame, action) == 0.0
    assert router.observe_click_outcome(frame, _Action(1), frame) is False


def test_score_is_self_sufficient_without_a_supplied_context() -> None:
    """REQ-ARC-FCP-5904: rank() precomputes contexts, but score() must still stand alone --
    the existing per-frame-cache regression test compares rank() against bare score()."""

    frame = _Frame(_grid())
    candidates = _click_candidates()
    head = _fit_head_on_frame(frame, candidates)
    router = OnlineClickTargetRouter(
        CrossGameDiscriminativeCandidateRouter(_ConstantVerifier()),
        enabled=True,
        discriminator=head,
    )
    ranked = router.rank(frame, candidates)
    reference = sorted(
        enumerate(candidates),
        key=lambda pair: (-router.score(frame, pair[1]), pair[0]),
    )
    assert [id(a) for a in ranked] == [id(a) for _i, a in reference]


def test_standalone_router_without_a_base_still_ranks() -> None:
    """REQ-ARC-FCP-5904: the online head needs no checkpoint, so base=None must work."""

    frame = _Frame(_grid())
    candidates = _click_candidates()
    head = _fit_head_on_frame(frame, candidates)
    cold = OnlineClickTargetRouter(None, enabled=True)
    assert [id(a) for a in cold.rank(frame, candidates)] == [id(a) for a in candidates]
    warm = OnlineClickTargetRouter(None, enabled=True, discriminator=head)
    assert [id(a) for a in warm.rank(frame, candidates)] != [id(a) for a in candidates]


# ------------------------------------------------------------------ episode isolation


def test_online_state_is_isolated_per_episode() -> None:
    """SCENARIO-ARC-FCP-5904-EPISODE-ISOLATION.

    ``scripts/arc_leaderboard_eval.py`` module-caches ONE router for a whole multi-game
    sweep, so this is the exact shape in which cross-game leakage would occur -- the
    direction retired by exclusion-manifest id
    ``cross_game_value_transfer_retired_exp4342_v401``.
    """

    grid = _grid()
    game_a = _Frame(grid, game_id="aa11-x", guid="guid-a")
    game_b = _Frame(grid, game_id="bb22-y", guid="guid-b")
    changed = _Frame(grid.copy(), game_id="aa11-x", guid="guid-a")
    changed.frame[22, 22] = 5

    router = OnlineClickTargetRouter(None, enabled=True)
    ctx = click_target_frame_context(game_a, use_cache=False)
    for action in _click_candidates():
        vector = click_target_features(ctx, action.data["x"], action.data["y"])
        router.observe_click_outcome(game_a, action, changed if vector[3] > 0.5 else game_a)
    head_a = router.discriminator_for(game_a)
    assert head_a.fitted is True

    head_b = router.discriminator_for(game_b)
    assert head_b is not head_a
    assert head_b.fitted is False
    assert head_b.proba([0.5] * CLICK_TARGET_FEATURE_DIM) == 0.5
    # game B's ranking is therefore the untouched input order: episode A's fitted head has
    # no reach into it at all.
    b_candidates = _click_candidates()
    assert [id(a) for a in router.rank(game_b, b_candidates)] == [id(a) for a in b_candidates]

    # And the store stays bounded, so a long sweep cannot accumulate per-game state.
    for index in range(6):
        router.discriminator_for(_Frame(grid, game_id=f"g{index}", guid=f"u{index}"))
    assert router.episode_count() <= router.max_episodes


def test_reset_episode_drops_all_online_state() -> None:
    """REQ-ARC-FCP-5904: nothing is persisted; state must be droppable on demand."""

    frame = _Frame(_grid())
    router = OnlineClickTargetRouter(None, enabled=True)
    router.discriminator_for(frame)
    assert router.episode_count() == 1
    router.reset_episode()
    assert router.episode_count() == 0
    assert router.stats()["episodes"] == 0


# --------------------------------------------------------------- v3 contract intact


def test_v3_feature_vector_stays_79_wide_with_the_flag_on() -> None:
    """REQ-ARC-FCP-5904: the shared v3 contract is load-bearing.

    ``models/arc_discriminative_verifier_v3.json`` bakes in 79 features + bias and
    ``DiscriminativeVerifier.load`` does NOT validate the shape, so a length change raises
    inside ``proba_features`` where ``score``'s blanket except turns it into a constant 0.5
    for every candidate -- silently re-creating the bug this work fixes.
    """

    frame = _Frame(_grid())
    candidates = _click_candidates()
    verifier = _ConstantVerifier()
    head = _fit_head_on_frame(frame, candidates)
    router = OnlineClickTargetRouter(
        CrossGameDiscriminativeCandidateRouter(verifier), enabled=True, discriminator=head
    )
    router.rank(frame, candidates)

    assert verifier.seen, "the base verifier must still be consulted"
    action_slice = cross_game_feature_slices_v3()["action_conditioned"]
    for features in verifier.seen:
        assert len(features) == 79
        assert sum(features[action_slice[0] : action_slice[1]]) > 0.0


def test_real_v3_checkpoint_still_loads_and_scores() -> None:
    """REQ-ARC-FCP-5904: a real guard against a silent length break in the shared contract."""

    from carnot.agentic.arc_value_learner import DiscriminativeVerifier

    path = router_mod.REPO_ROOT / router_mod.DEFAULT_CHECKPOINT_RELATIVE_PATH
    if not path.exists():  # pragma: no cover - the checkpoint is committed
        pytest.fail(f"expected the committed v3 checkpoint at {path}")
    verifier = DiscriminativeVerifier.load(path, cross_game_features_v3)
    value = verifier.proba_features(cross_game_features_v3(_Frame(_grid()), action_id=6))
    assert 0.0 <= float(value) <= 1.0


# -------------------------------------------------------- per-frame, not per-candidate


def test_rank_segmentation_cost_does_not_scale_with_candidate_count() -> None:
    """SCENARIO-ARC-FCP-5904-PER-FRAME-NOT-PER-CANDIDATE, at the router boundary."""

    frame = _Frame(_grid())
    small = _click_candidates(limit=8)
    large = _click_candidates(limit=24)
    head = _fit_head_on_frame(frame, large)

    def count_calls(candidates: list[_Action]) -> int:
        feat_mod.clear_click_target_frame_context_cache()
        router = OnlineClickTargetRouter(
            CrossGameDiscriminativeCandidateRouter(_ConstantVerifier()),
            enabled=True,
            discriminator=head,
        )
        with patch.object(
            feat_mod, "connected_color_blobs", wraps=feat_mod.connected_color_blobs
        ) as spy:
            router.rank(frame, candidates)
            return spy.call_count

    assert count_calls(small) == count_calls(large) == 1


# --------------------------------------------------------------------- observation


def test_observe_click_outcome_labels_from_the_observed_frame_pair() -> None:
    """REQ-ARC-FCP-5904: the label must be causally downstream of the CLICK (the exp5835
    defect was a label that was a pure function of the step index)."""

    grid = _grid()
    before = _Frame(grid)
    unchanged = _Frame(grid.copy())
    changed = _Frame(grid.copy())
    changed.frame[22, 22] = 5

    router = OnlineClickTargetRouter(None, enabled=True)
    assert router.observe_click_outcome(before, _Action(6, 2, 16), changed) is True
    assert router.observe_click_outcome(before, _Action(6, 8, 16), unchanged) is True
    head = router.discriminator_for(before)
    assert head.stats()["n_positives"] == 1
    assert head.stats()["n_negatives"] == 1

    # A level-up with no frame change is still a positive, and is counted separately.
    assert (
        router.observe_click_outcome(before, _Action(6, 14, 16), unchanged, leveled_up=True) is True
    )
    assert head.stats()["n_positives"] == 2
    assert head.stats()["n_level_up_labels"] == 1


def test_observe_click_outcome_is_a_no_op_when_disabled() -> None:
    """REQ-ARC-FCP-5904: the default-off guarantee covers the observation hook too."""

    frame = _Frame(_grid())
    router = OnlineClickTargetRouter(None, enabled=False)
    assert router.observe_click_outcome(frame, _Action(6, 2, 16), frame) is False
    assert router.stats()["episodes"] == 0


def test_grids_differ_copies_before_comparing() -> None:
    """REQ-ARC-FCP-5904: env.step() returns frames over a SHARED mutated-in-place buffer
    (documented at arc_solver_kit.py:5315-5325), so a naive comparison reads 'unchanged'."""

    grid = _grid()
    assert router_mod._grids_differ(_Frame(grid), _Frame(grid)) is False
    other = grid.copy()
    other[1, 1] = 11
    assert router_mod._grids_differ(_Frame(grid), _Frame(other)) is True
    assert router_mod._grids_differ(_Frame(grid), _Frame(np.zeros((4, 4), dtype=np.int16))) is True


def test_stats_exposes_per_episode_diagnostics() -> None:
    """REQ-ARC-FCP-5904: an artifact must be able to report what the head actually saw."""

    frame = _Frame(_grid(), game_id="zz99-a", guid="guid-z")
    router = OnlineClickTargetRouter(None, enabled=True)
    router.observe_click_outcome(frame, _Action(6, 2, 16), frame)
    stats = router.stats()
    assert stats["enabled"] is True
    assert stats["episodes"] == 1
    assert "zz99-a/guid-z" in stats["per_episode"]
    assert stats["per_episode"]["zz99-a/guid-z"]["n_samples"] == 1


def test_verifier_is_oracle_stays_false() -> None:
    """REQ-ARC-FCP-5904 / Circularity discipline: the online head scores from PERCEPTION
    features and never executes the candidate, so it is not the executable oracle."""

    assert OnlineClickTargetRouter(None).verifier_is_oracle is False


def test_real_v3_router_collapses_every_click_to_one_score() -> None:
    """SCENARIO-ARC-FCP-5904-COORDINATE-BLINDNESS-IS-REPAIRED: the DEFECT, on the real thing.

    The sibling flag-off test uses a constant stub verifier, which cannot by itself prove the
    SHIPPED router is coordinate-blind -- it could pass even if v3 were coordinate-aware. This
    test loads the REAL committed checkpoint and asserts the collapse directly: many distinct
    click targets, many distinct ``candidate_action_key`` values, exactly ONE distinct score,
    and ``rank()`` a stable no-op.

    If this test ever starts FAILING because the count exceeds one, that is good news and the
    signal to re-scope this work -- but it must be an explicit, noticed change, not a silent
    one.
    """

    from carnot.agentic.arc_discriminative_router import (
        candidate_action_key,
        load_cross_game_discriminative_router,
    )

    router = load_cross_game_discriminative_router()
    if router is None:  # pragma: no cover - the checkpoint is committed
        pytest.fail("the committed v3 checkpoint must load for this defect lock-in to be real")

    frame = _Frame(_grid())
    candidates = _click_candidates()
    assert len({candidate_action_key(a) for a in candidates}) == len(candidates)

    scores = [router.score(frame, action) for action in candidates]
    assert len(set(scores)) == 1, (
        "v3 became coordinate-aware; re-scope REQ-ARC-FCP-5904 deliberately"
    )
    assert [id(a) for a in router.rank(frame, candidates)] == [id(a) for a in candidates]

    # And the repair, on the same real router: wrapping it with a fitted online head separates
    # the targets the real checkpoint cannot tell apart.
    head = _fit_head_on_frame(frame, candidates)
    wrapped = OnlineClickTargetRouter(router, enabled=True, discriminator=head)
    wrapped_scores = [wrapped.score(frame, action) for action in candidates]
    assert len(set(wrapped_scores)) > 1


def test_injected_discriminator_is_bound_to_the_first_episode_only() -> None:
    """SCENARIO-ARC-FCP-5904-EPISODE-ISOLATION: close the injection leak path.

    The constructor's ``discriminator`` argument exists so tests and the offline experiment can
    hold a handle on the head. If it were honoured for EVERY episode, one fitted head would
    follow a cached router across games -- the cross-game value transfer retired by
    exclusion-manifest id ``cross_game_value_transfer_retired_exp4342_v401``, arriving through
    the exact shape that makes it likely (``scripts/arc_leaderboard_eval.py`` caches one
    router per sweep). So the injection is CONSUMED on first use.
    """

    grid = _grid()
    game_a = _Frame(grid, game_id="aa11-x", guid="guid-a")
    game_b = _Frame(grid, game_id="bb22-y", guid="guid-b")
    head = _fit_head_on_frame(game_a, _click_candidates())

    router = OnlineClickTargetRouter(None, enabled=True, discriminator=head)
    assert router.discriminator_for(game_a) is head
    second = router.discriminator_for(game_b)
    assert second is not head
    assert second.fitted is False
    assert second.proba([0.5] * CLICK_TARGET_FEATURE_DIM) == 0.5
    # Re-asking for episode A still returns the same head (the store, not the injection).
    assert router.discriminator_for(game_a) is head
