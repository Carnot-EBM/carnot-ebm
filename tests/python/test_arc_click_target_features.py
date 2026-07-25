"""Tests for the coordinate-aware click-target featurization.

Spec refs: REQ-ARC-FCP-5904, SCENARIO-ARC-FCP-5904-COORDINATE-BLINDNESS-IS-REPAIRED,
SCENARIO-ARC-FCP-5904-PER-FRAME-NOT-PER-CANDIDATE,
SCENARIO-ARC-FCP-5904-SATURATION-GUARD.

The load-bearing test here is ``test_two_targets_in_the_same_frame_differ``: it is the direct
regression test for the verified defect that motivated the module. The incumbent router
featurizes a click purely by its action TYPE, so 37-51 distinct click targets collapse to ONE
score; a featurization that could not separate two targets in the same frame would be the
same bug wearing a new name.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from carnot.agentic import arc_click_target_features as feat_mod
from carnot.agentic.arc_click_target_features import (
    CLICK_TARGET_FEATURE_DIM,
    CLICK_TARGET_FEATURE_NAMES,
    ClickEpisodeState,
    OnlineClickTargetDiscriminator,
    click_coordinates,
    click_target_features,
    click_target_frame_context,
    click_target_object_identity,
    clear_click_target_frame_context_cache,
    settled_grid_of,
)


class _Frame:
    """Minimal stand-in for ``FrameDataRaw``: a settled grid plus episode identity."""

    def __init__(self, grid: np.ndarray, game_id: str = "gm00-test", guid: str = "guid-a") -> None:
        self.frame = grid
        self.game_id = game_id
        self.guid = guid


class _Action:
    def __init__(self, action_id: int, x: int | None = None, y: int | None = None) -> None:
        self.action_id = action_id
        self.data = None if x is None else {"x": x, "y": y}


def _multi_blob_grid() -> np.ndarray:
    """A 24x24 frame with a dull background, a big flat field, and several small buttons.

    Deliberately heterogeneous: the small salient squares should look very different from the
    large flat region and from the status strip, which is exactly the distinction the
    incumbent featurization is blind to.
    """

    grid = np.zeros((24, 24), dtype=np.int16)
    grid[2:14, 2:14] = 3  # large flat field
    for index, (y, x) in enumerate([(16, 2), (16, 8), (16, 14), (20, 2), (20, 8), (20, 14)]):
        grid[y : y + 2, x : x + 2] = 6 + index  # small salient buttons, distinct colours
    grid[0, :] = 16  # status strip
    grid[6, 20] = 9  # single rare pixel
    return grid


# A curated target list that deliberately covers EVERY region class in the fixture grid --
# buttons, the flat field, the status strip, the rare pixel, and plain background. A naive
# row-major stride would sample only the top rows and miss the buttons entirely, which would
# make a "does it discriminate?" assertion pass or fail for the wrong reason.
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


def _click_targets(grid: np.ndarray, limit: int = 24) -> list[tuple[int, int]]:
    """Distinct (x, y) targets: one per distinct colour region plus a spread of background."""

    del grid  # the curated list is tied to _multi_blob_grid()'s layout
    return list(_CLICK_TARGETS[:limit])


# --------------------------------------------------------------------------- contract


def test_feature_names_and_dim_agree_and_are_unique() -> None:
    """REQ-ARC-FCP-5904: the vector's order is a contract an online head depends on."""

    assert CLICK_TARGET_FEATURE_DIM == 21
    assert len(CLICK_TARGET_FEATURE_NAMES) == CLICK_TARGET_FEATURE_DIM
    assert len(set(CLICK_TARGET_FEATURE_NAMES)) == CLICK_TARGET_FEATURE_DIM


def test_vector_length_and_finiteness_on_every_target() -> None:
    """REQ-ARC-FCP-5904: every entry must be finite and roughly in [0, 1]."""

    ctx = click_target_frame_context(_Frame(_multi_blob_grid()), use_cache=False)
    for x, y in _click_targets(_multi_blob_grid()):
        vector = click_target_features(ctx, x, y)
        assert len(vector) == CLICK_TARGET_FEATURE_DIM
        assert all(np.isfinite(value) for value in vector)
        assert all(-0.01 <= value <= 1.01 for value in vector), (x, y, vector)


# ------------------------------------------------------- the defect regression test


def test_two_targets_in_the_same_frame_differ() -> None:
    """SCENARIO-ARC-FCP-5904-COORDINATE-BLINDNESS-IS-REPAIRED.

    The direct regression test for the verified defect: the incumbent featurization gives
    every click in a frame the SAME vector (action type only). This asserts that the new
    featurization separates targets, including two structurally different regions.
    """

    grid = _multi_blob_grid()
    ctx = click_target_frame_context(_Frame(grid), use_cache=False)

    button = tuple(click_target_features(ctx, 2, 16))  # small salient button
    flat = tuple(click_target_features(ctx, 7, 7))  # large flat field
    status = tuple(click_target_features(ctx, 12, 0))  # status strip
    rare = tuple(click_target_features(ctx, 20, 6))  # single rare pixel

    assert len({button, flat, status, rare}) == 4
    # And not merely different in the positional tail -- the blob-morphology head differs too.
    assert button[:11] != flat[:11]
    assert button[:11] != status[:11]

    targets = _click_targets(grid, limit=24)
    vectors = {tuple(click_target_features(ctx, x, y)) for x, y in targets}
    assert len(targets) >= 20
    assert len(vectors) > 1
    # The vast majority of distinct targets must be distinguishable, not just a lucky pair.
    assert len(vectors) >= 0.5 * len(targets), (len(vectors), len(targets))


def test_same_blob_differs_only_in_positional_and_local_features() -> None:
    """REQ-ARC-FCP-5904: indices 0..10 describe the OBJECT, so two clicks on the same
    object share them; 11..20 are local/positional and may differ."""

    grid = np.zeros((16, 16), dtype=np.int16)
    grid[4:9, 4:9] = 7  # one 5x5 object, uniform inside
    ctx = click_target_frame_context(_Frame(grid), use_cache=False)
    a = click_target_features(ctx, 5, 5)
    b = click_target_features(ctx, 7, 7)
    assert a[:11] == b[:11]
    assert a[19:] != b[19:]


def test_off_blob_click_falls_back_to_nearest_blob_without_raising() -> None:
    """REQ-ARC-FCP-5904: a click can land on a background gap; the fallback is documented
    and must be exercised, not merely present."""

    grid = np.zeros((12, 12), dtype=np.int16)
    grid[2:4, 2:4] = 6
    ctx = click_target_frame_context(_Frame(grid), use_cache=False)
    # The 0-coloured background is a single blob only if it is under the size cap; on this
    # grid it is 136/144 = 94% > max_component_fraction, so it is suppressed and (10, 10) is
    # off every blob.
    assert int(ctx.blob_index[10, 10]) == -1
    vector = click_target_features(ctx, 10, 10)
    assert vector[0] == 0.0  # on_blob false
    assert any(value != 0.0 for value in vector[1:11])  # nearest-blob morphology filled in
    assert click_target_object_identity(ctx, 10, 10) is not None


def test_empty_blob_frame_is_handled() -> None:
    """REQ-ARC-FCP-5904: a uniform frame yields zero blobs; features must still be valid."""

    ctx = click_target_frame_context(_Frame(np.zeros((8, 8), dtype=np.int16)), use_cache=False)
    assert ctx.blobs == ()
    vector = click_target_features(ctx, 3, 4)
    assert len(vector) == CLICK_TARGET_FEATURE_DIM
    assert all(np.isfinite(v) for v in vector)
    assert click_target_object_identity(ctx, 3, 4) is None


def test_out_of_bounds_click_does_not_raise() -> None:
    """REQ-ARC-FCP-5904: a malformed candidate must never crash the live ranking path."""

    ctx = click_target_frame_context(_Frame(_multi_blob_grid()), use_cache=False)
    vector = click_target_features(ctx, 999, -5)
    assert len(vector) == CLICK_TARGET_FEATURE_DIM
    assert all(np.isfinite(v) for v in vector)


# ------------------------------------------------------------------- candidate shapes


def test_mapping_and_object_candidate_shapes_agree() -> None:
    """REQ-ARC-FCP-5904: both candidate shapes really reach the ranking path."""

    assert click_coordinates(_Action(6, 4, 9)) == (4, 9)
    assert click_coordinates({"action": 6, "data": {"x": 4, "y": 9}}) == (4, 9)
    assert click_coordinates({"action_id": 6, "data": {"x": 4, "y": 9}}) == (4, 9)

    ctx = click_target_frame_context(_Frame(_multi_blob_grid()), use_cache=False)
    from_obj = click_target_features(ctx, *click_coordinates(_Action(6, 2, 16)))
    from_map = click_target_features(
        ctx, *click_coordinates({"action": 6, "data": {"x": 2, "y": 16}})
    )
    assert from_obj == from_map


def test_non_click_candidates_have_no_coordinates() -> None:
    """REQ-ARC-FCP-5904: keyboard actions must yield None, not a zero vector."""

    assert click_coordinates(_Action(1)) is None
    assert click_coordinates(_Action(6)) is None  # click with no data
    assert click_coordinates({"action": 6}) is None
    assert click_coordinates({"action": 6, "data": {"x": 1}}) is None
    assert click_coordinates({"action": 6, "data": {"x": "nope", "y": 1}}) is None
    assert click_coordinates(_Action(0)) is None


# ------------------------------------------------------------ episode-local novelty


def test_episode_state_changes_only_the_novelty_features() -> None:
    """REQ-ARC-FCP-5904: novelty is in-episode state, discarded at episode end."""

    ctx = click_target_frame_context(_Frame(_multi_blob_grid()), use_cache=False)
    state = ClickEpisodeState()
    before = click_target_features(ctx, 2, 16, episode_state=state)
    identity = click_target_object_identity(ctx, 2, 16)
    for _ in range(3):
        state.observe_click(2, 16, identity)
    after = click_target_features(ctx, 2, 16, episode_state=state)
    assert before[17] == 0.0 and after[17] == 1.0
    assert before[18] == 0.0 and after[18] == 1.0
    assert before[:17] == after[:17]
    assert before[19:] == after[19:]


# --------------------------------------------------------- per-frame, not per-candidate


def test_segmentation_cost_does_not_scale_with_candidate_count() -> None:
    """SCENARIO-ARC-FCP-5904-PER-FRAME-NOT-PER-CANDIDATE.

    Encodes the measured 8176-segmentations-for-500-actions incident as a contract: the
    per-frame context is built once and every candidate is pure indexing afterwards.
    """

    grid = _multi_blob_grid()
    with patch.object(
        feat_mod, "connected_color_blobs", wraps=feat_mod.connected_color_blobs
    ) as spy:
        ctx = click_target_frame_context(_Frame(grid), use_cache=False)
        calls_after_context = spy.call_count
        for x, y in _click_targets(grid, limit=24):
            click_target_features(ctx, x, y)
        assert calls_after_context == 1
        assert spy.call_count == 1


def test_frame_context_cache_is_content_keyed_and_bounded() -> None:
    """REQ-ARC-FCP-5904: candidate generation re-wraps the same grid in new objects, so the
    cache must key on content; and it must stay bounded."""

    clear_click_target_frame_context_cache()
    grid = _multi_blob_grid()
    with patch.object(
        feat_mod, "connected_color_blobs", wraps=feat_mod.connected_color_blobs
    ) as spy:
        first = click_target_frame_context(_Frame(grid))
        second = click_target_frame_context(_Frame(grid.copy()))  # different object, same bytes
        assert spy.call_count == 1
    assert first is second

    clear_click_target_frame_context_cache()
    for shift in range(feat_mod._CONTEXT_CACHE_MAX_SIZE + 4):
        variant = grid.copy()
        variant[10, 10] = shift + 1
        click_target_frame_context(_Frame(variant))
    assert len(feat_mod._context_cache) <= feat_mod._CONTEXT_CACHE_MAX_SIZE


def test_blob_topology_is_not_used_anywhere_in_the_module() -> None:
    """REQ-ARC-FCP-5904: measured 322 ms/frame on bp35 (~100x the rest of the context
    combined, ~161 s per 500-action episode). Encoded as a contract, not a comment.

    The docstring mentions the name to explain the exclusion, so the assertion is scoped to
    the executable body below the docstring.
    """

    source = Path(feat_mod.__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]  # everything after the module docstring
    assert "blob_topology" not in body


def test_settled_grid_of_handles_animation_stacks() -> None:
    """REQ-ARC-FCP-5904: ARC frames arrive as layer stacks whose LAST layer is settled."""

    stack = np.zeros((3, 6, 6), dtype=np.int16)
    stack[-1, 2, 2] = 7
    grid = settled_grid_of(_Frame(stack))
    assert grid.shape == (6, 6)
    assert int(grid[2, 2]) == 7
    with pytest.raises(ValueError):
        settled_grid_of(np.zeros((4,), dtype=np.int16))


# --------------------------------------------------------------- the online head


def _separable_samples(n: int = 24) -> list[tuple[list[float], float]]:
    """Feature vectors where index 3 (is_button_like) perfectly predicts the label."""

    samples = []
    for i in range(n):
        vector = [0.0] * CLICK_TARGET_FEATURE_DIM
        vector[3] = 1.0 if i % 2 == 0 else 0.0
        vector[19] = (i % 5) / 5.0
        vector[20] = (i % 7) / 7.0
        samples.append((vector, vector[3]))
    return samples


def test_unfitted_head_returns_exactly_one_half() -> None:
    """SCENARIO-ARC-FCP-5904-COLD-START-IS-A-NO-OP: cold start must be an exact no-op."""

    head = OnlineClickTargetDiscriminator()
    vector = [0.5] * CLICK_TARGET_FEATURE_DIM
    assert head.proba(vector) == 0.5
    assert head.fitted is False
    assert head.gate_met is False
    # Below the gate a fit attempt is refused rather than producing a junk model.
    head.observe(vector, 1.0)
    head.observe(vector, 0.0)
    assert head.fit() is False
    assert head.proba(vector) == 0.5


def test_sample_gate_requires_both_classes() -> None:
    """REQ-ARC-FCP-5904: >= 3 positives, >= 3 negatives, >= 8 total before any opinion."""

    head = OnlineClickTargetDiscriminator()
    for vector, _label in _separable_samples(10):
        head.observe(vector, 1.0)  # all positives
    assert head.gate_met is False
    assert head.maybe_fit() is False
    for vector, _label in _separable_samples(6):
        head.observe(vector, 0.0)
    assert head.gate_met is True
    assert head.maybe_fit() is True
    assert head.fitted is True


def test_online_fit_separates_a_synthetic_separable_case() -> None:
    """REQ-ARC-FCP-5904: the head must actually learn, not merely run."""

    head = OnlineClickTargetDiscriminator()
    for vector, label in _separable_samples(24):
        head.observe(vector, label)
    assert head.fit() is True
    positive = [0.0] * CLICK_TARGET_FEATURE_DIM
    positive[3] = 1.0
    negative = [0.0] * CLICK_TARGET_FEATURE_DIM
    assert head.proba(positive) > head.proba(negative)
    assert head.proba(positive) > 0.5 > head.proba(negative)


def test_saturation_guard_never_returns_exactly_one() -> None:
    """SCENARIO-ARC-FCP-5904-SATURATION-GUARD.

    Reproduces the measured failure of ``sd = std + 1e-8``: 19 zeros and one 1e-6 gives
    sd = 2.28e-7, so a later value of 0.75 yields z = 3.29e6 and proba EXACTLY 1.0 --
    collapsing the ranking back into a tie and emitting IMPLAUSIBLE_PERFECT-shaped values.
    """

    head = OnlineClickTargetDiscriminator()
    for index in range(20):
        vector = [0.0] * CLICK_TARGET_FEATURE_DIM
        vector[6] = 1e-6 if index == 19 else 0.0
        vector[3] = 1.0 if index % 2 == 0 else 0.0
        head.observe(vector, vector[3])
    assert head.fit() is True

    probe = [0.0] * CLICK_TARGET_FEATURE_DIM
    probe[6] = 0.75  # far outside the fitted column's range
    value = head.proba(probe)
    assert 0.0 < value < 1.0, value
    assert head.stats()["saturation_clips"] >= 1


def test_observation_buffer_is_bounded() -> None:
    """REQ-ARC-FCP-5904: a long episode must not grow memory without limit."""

    head = OnlineClickTargetDiscriminator(max_samples=16)
    for vector, label in _separable_samples(60):
        head.observe(vector, label)
    assert head.stats()["n_samples"] == 16


def test_observe_rejects_a_wrong_length_vector() -> None:
    """REQ-ARC-FCP-5904: a silent dimension mismatch is how a fitted head goes junk."""

    head = OnlineClickTargetDiscriminator()
    with pytest.raises(ValueError):
        head.observe([0.0] * (CLICK_TARGET_FEATURE_DIM - 1), 1.0)


def test_proba_rejects_a_wrong_length_vector_without_raising() -> None:
    """REQ-ARC-FCP-5904: scoring is on the live path, so it degrades to neutral instead."""

    head = OnlineClickTargetDiscriminator()
    for vector, label in _separable_samples(24):
        head.observe(vector, label)
    head.fit()
    assert head.proba([0.0] * 3) == 0.5


def test_stats_reports_level_up_labels_separately() -> None:
    """REQ-ARC-FCP-5904: level-ups are the scarce label; report them, do not conflate."""

    head = OnlineClickTargetDiscriminator()
    for index, (vector, label) in enumerate(_separable_samples(12)):
        head.observe(vector, label, leveled_up=(index == 0))
    stats = head.stats()
    assert stats["n_level_up_labels"] == 1
    assert stats["n_positives"] + stats["n_negatives"] == stats["n_samples"]


def test_rare_anchor_features_are_populated_when_a_rare_colour_exists() -> None:
    """REQ-ARC-FCP-5904: relational distance to a rare-colour landmark is a real signal."""

    ctx = click_target_frame_context(_Frame(_multi_blob_grid()), use_cache=False)
    assert ctx.rare_anchors.shape[0] > 0
    near = click_target_features(ctx, 20, 6)
    far = click_target_features(ctx, 2, 20)
    assert near[15] != far[15]
    assert isinstance(ctx.color_counts, Counter)
