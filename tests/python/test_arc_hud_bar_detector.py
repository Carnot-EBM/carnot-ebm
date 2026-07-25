"""Regression tests for the REPAIRED orientation-complete HUD detector + its collapse guard.

Spec: REQ-ARC-WMTE-5960,
SCENARIO-ARC-WMTE-5960-VERTICAL-EDGE-COUNTER-DETECTED,
SCENARIO-ARC-WMTE-5960-SUPERSET-OF-SHIPPED-CLASSIFIER,
SCENARIO-ARC-WMTE-5960-OVER-BROAD-MASK-REFUSED,
SCENARIO-ARC-WMTE-5960-COLLAPSE-GUARD-HARD-REFUSAL,
SCENARIO-ARC-WMTE-5960-DEFAULT-OFF-PARITY.

WHY THIS FILE EXISTS. The shipped classifier
(`ColorBlobSaliencePrior.is_status_bar_like`) is orientation-blind: its geometric branch
requires `width >= 0.75*frame_width AND height <= 2`, a horizontal-bar template no VERTICAL
bar can satisfy at any frame size, and its edge test reads only the top/bottom rows. r11l's
monotone step counter renders into frame COLUMN 0, so the mask resolved to None there and the
live search's dedup never fired (measured: 1956 actions -> 1392 graph nodes over 31 true game
states on arm A; a single wall-blocked, game-state-inert click re-popped 1371 of them).

A DELIBERATE NOTE ON THE PRE-EXISTING TEST FILE (`test_arc_hud_mask.py`): every one of its
positive fixtures writes `grid[0, :] = 16`, and colour 16 IS `status_bar_color`, so those
tests pass through the classifier's colour SHORTCUT and never exercise the geometric branch at
all. The horizontal geometric rule was therefore effectively untested. The fixtures here use
NON-status colours precisely so the geometry is what is under test, and
`test_horizontal_geometric_bar_still_detected` is the horizontal regression guard that file
was missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior, connected_color_blobs
from carnot.agentic.arc_competition_agent import (
    SUBMITTED_AGENT_CONFIG,
    SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED,
    SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED,
    SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED,
    StepwiseExplorer,
    _assert_hud_flag_coupling,
    _compute_hud_mask_from_frame,
)
from carnot.agentic.arc_hud_bar_detector import (
    HUD_MASK_GUARD_REVOCATION_GLOBAL_HASH_FLIP,
    HUD_MASK_GUARD_REVOCATION_LOCAL,
    DeferredMaskActivation,
    EdgeBarThresholds,
    MaskCollapseGuard,
    edge_bar_hud_mask,
    is_edge_bar_like,
    mask_cell_digest,
    mask_summary,
    region_hud_evidence,
)


class _FakeFrame:
    """Minimal stand-in for an arcengine frame: only `.frame` is read by `grid_of`."""

    def __init__(self, grid: np.ndarray) -> None:
        self.frame = grid
        self.state = "NOT_FINISHED"
        self.levels_completed = 0


# ---------------------------------------------------------------------------
# Fixtures -- all NON-status colours, so the geometry is what is under test
# ---------------------------------------------------------------------------


def _r11l_like_grid(filled: int = 0) -> np.ndarray:
    """A 64x64 board with a VERTICAL monotone counter in column 0, r11l's shape.

    Column 0 is a single 4-connected colour-0 blob of 64 cells; `filled` of its cells carry
    the "consumed" colour, exactly as r11l's counter fills one cell per action.
    """

    grid = np.full((64, 64), 3, dtype=int)
    grid[:, 0] = 0
    if filled:
        grid[:filled, 0] = 5
    grid[30:34, 30:34] = 7  # a compact interior board object -- must never be masked
    return grid


def _static_edge_stripe_grid() -> np.ndarray:
    """An edge-adjacent stripe that is DECORATIVE: geometry alone cannot tell it apart."""

    grid = np.full((32, 32), 3, dtype=int)
    grid[:, 31] = 8
    grid[10:14, 10:14] = 7
    return grid


# ---------------------------------------------------------------------------
# Stage 1 -- single-frame geometry
# ---------------------------------------------------------------------------


def test_vertical_edge_counter_is_detected() -> None:
    """SCENARIO-ARC-WMTE-5960-VERTICAL-EDGE-COUNTER-DETECTED.

    The exact shape the shipped classifier misses. 64 cells, column 0 only.
    """

    grid = _r11l_like_grid()
    shipped = _compute_hud_mask_from_frame(_FakeFrame(grid))
    repaired = edge_bar_hud_mask(grid)

    assert shipped is None, "precondition: the shipped classifier must MISS a vertical bar"
    assert repaired is not None
    assert int(repaired.sum()) == 64
    assert bool(repaired[:, 0].all())
    assert not bool(repaired[:, 1:].any())
    assert not bool(repaired[30:34, 30:34].any())


def test_horizontal_geometric_bar_still_detected() -> None:
    """The horizontal regression guard the pre-existing test file could not provide.

    Uses colour 8 (NOT 16), so the classifier's colour shortcut cannot answer for it and the
    geometric path is genuinely exercised on both detectors.
    """

    grid = np.full((32, 32), 3, dtype=int)
    grid[0, :] = 8
    grid[10:14, 10:14] = 7

    shipped = _compute_hud_mask_from_frame(_FakeFrame(grid))
    repaired = edge_bar_hud_mask(grid)

    assert shipped is not None and int(shipped.sum()) == 32
    assert repaired is not None and int(repaired.sum()) == 32
    assert bool(repaired[0, :].all())


def test_repaired_mask_is_a_superset_of_the_shipped_mask() -> None:
    """SCENARIO-ARC-WMTE-5960-SUPERSET-OF-SHIPPED-CLASSIFIER.

    Superset BY CONSTRUCTION, so an A/B difference can only come from newly-detected cells --
    never from cells that silently stopped being masked.
    """

    grid = np.full((40, 40), 3, dtype=int)
    grid[0, :] = 8  # horizontal bar the shipped rule already finds
    grid[:, 39] = 9  # vertical bar only the repair finds
    grid[20:24, 20:24] = 7

    shipped = _compute_hud_mask_from_frame(_FakeFrame(grid))
    repaired = edge_bar_hud_mask(grid)

    assert shipped is not None and repaired is not None
    assert not bool((shipped & ~repaired).any()), "repair must not DROP a shipped cell"
    assert bool((repaired & ~shipped).any()), "repair must ADD the vertical bar"


def test_interior_bar_not_touching_any_edge_is_not_detected() -> None:
    """Elongation alone is not enough: a status bar has to hug a frame edge."""

    grid = np.full((32, 32), 3, dtype=int)
    grid[:, 16] = 8  # long and vertical, but dead centre
    assert edge_bar_hud_mask(grid) is None


def test_edge_adjacent_but_not_elongated_is_not_detected() -> None:
    """Edge adjacency alone is not enough either: a corner button must survive."""

    grid = np.full((32, 32), 3, dtype=int)
    grid[0:4, 0:4] = 8  # touches two edges, aspect ratio 1.0
    assert edge_bar_hud_mask(grid) is None


def test_horizontal_bar_on_a_vertical_edge_is_not_detected() -> None:
    """Orientation is coupled to the edge on purpose (the reference's looser rule is not ported)."""

    grid = np.full((32, 32), 3, dtype=int)
    grid[0:2, 0:20] = 8  # long HORIZONTALLY and hugging the LEFT edge...
    grid[2:32, 0:2] = 3  # ...but nothing vertical there
    mask = edge_bar_hud_mask(grid)
    # It IS admitted -- via the TOP edge, which it also touches, with long_horizontal true.
    # The point of the assertion is that admission comes from the top/bottom branch: a purely
    # left-hugging horizontal strip that does NOT reach the top must be refused, below.
    assert mask is not None

    grid2 = np.full((32, 32), 3, dtype=int)
    grid2[10:12, 0:20] = 8  # left-hugging, horizontal, rows 10-11 -> no top/bottom adjacency
    assert edge_bar_hud_mask(grid2) is None


def test_over_broad_mask_is_refused_wholesale() -> None:
    """SCENARIO-ARC-WMTE-5960-OVER-BROAD-MASK-REFUSED.

    A mask above the area ceiling returns None (today's behaviour) rather than being
    truncated: a partially-applied over-broad mask is exactly the correctness hazard.
    """

    grid = np.full((32, 32), 3, dtype=int)
    for col in range(0, 3):
        grid[:, col] = 8 + col  # three distinct full-height edge bars = 96 of 1024 cells = 9.4%

    assert edge_bar_hud_mask(grid) is None, "9.4% of the frame must exceed the 5% ceiling"
    generous = edge_bar_hud_mask(grid, thresholds=EdgeBarThresholds(max_mask_area_fraction=0.5))
    assert generous is not None and int(generous.sum()) >= 64


def _blob_at(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int]):
    """Build one ColorBlob directly, so tolerance is tested WITHOUT segmentation artifacts.

    Testing tolerance through `edge_bar_hud_mask` on a synthetic grid does not isolate the
    predicate: a bar inset by one cell SPLITS the background, and the 1-cell background strip
    it leaves behind is ITSELF geometrically an edge bar, so the union doubles (found by this
    test's own first version, which failed on the 5% area ceiling for that reason). That is
    real, correct, conservative behaviour of the area cap -- it is asserted separately in
    `test_inset_bar_leaves_a_background_strip_that_is_also_bar_like` -- but it is not what a
    tolerance test should be measuring.
    """

    from carnot.agentic.arc_color_blob_salience import ColorBlob

    y0, x0, y1, x1 = bbox
    cells = frozenset((y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1))
    return ColorBlob(
        color=8,
        pixel_count=len(cells),
        bbox=bbox,
        centroid=((y0 + y1) / 2.0, (x0 + x1) / 2.0),
        cells=cells,
        frame_shape=frame_shape,
    )


def test_edge_tolerance_is_symmetric_across_all_four_edges() -> None:
    """All four edges, same tolerance on each -- both independent shipped gaps, in one test.

    The shipped classifier tests only bbox[0]/bbox[2] (both Y coordinates), so LEFT/RIGHT are
    structurally unreachable for it; and it requires zero edge slack, which is why tn36's bar
    at row 1 is missed too.
    """

    shape = (64, 64)
    within = {
        "top": (1, 0, 1, 63),
        "bottom": (62, 0, 62, 63),
        "left": (0, 1, 63, 1),
        "right": (0, 62, 63, 62),
    }
    beyond = {
        "top": (2, 0, 2, 63),
        "bottom": (61, 0, 61, 63),
        "left": (0, 2, 63, 2),
        "right": (0, 61, 63, 61),
    }
    for placement, bbox in within.items():
        assert is_edge_bar_like(_blob_at(bbox, shape)) is True, f"{placement} within tolerance"
    for placement, bbox in beyond.items():
        assert is_edge_bar_like(_blob_at(bbox, shape)) is False, f"{placement} beyond tolerance"


def test_inset_bar_leaves_a_background_strip_that_is_also_bar_like() -> None:
    """A real, documented property of pure geometry -- and why the area cap has to be a cap.

    A bar one cell in from the top splits the background, and the leftover 1-row background
    strip at row 0 is itself edge-adjacent and elongated, so BOTH are masked. On a 64x64 frame
    that is 128 cells (3.1%) and admissible; on a 32x32 frame it is 64 cells (6.25%) and the
    ceiling refuses the whole mask rather than shipping half of it.
    """

    small = np.full((32, 32), 3, dtype=int)
    small[1, :] = 8
    assert edge_bar_hud_mask(small) is None, "6.25% exceeds the ceiling -> refuse wholesale"

    large = np.full((64, 64), 3, dtype=int)
    large[1, :] = 8
    mask = edge_bar_hud_mask(large)
    assert mask is not None and int(mask.sum()) == 128
    assert bool(mask[0, :].all()) and bool(mask[1, :].all())


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (1, 32), (32, 1), (5, 64), (64, 5), (17, 31)],
)
def test_degenerate_and_non_square_frames_do_not_crash(shape: tuple[int, int]) -> None:
    """Single row, single column, non-square, tiny -- must return a mask or None, never raise."""

    rng = np.random.default_rng(5960)
    grid = rng.integers(0, 10, size=shape)
    mask = edge_bar_hud_mask(grid)
    assert mask is None or (mask.shape == shape and mask.dtype == bool)


def test_empty_and_malformed_inputs_return_none() -> None:
    assert edge_bar_hud_mask(None) is None
    assert edge_bar_hud_mask(np.zeros((0, 0), dtype=int)) is None
    assert edge_bar_hud_mask(np.zeros((2, 2, 2, 2), dtype=int)) is None


def test_three_dimensional_frame_uses_the_last_layer() -> None:
    """Frame stacks are legal inputs; the settled layer is the one that gets masked."""

    stack = np.full((2, 64, 64), 3, dtype=int)
    stack[0, 8, 8] = 9
    stack[1, :, 0] = 8
    mask = edge_bar_hud_mask(stack)
    assert mask is not None
    assert int(mask.sum()) == 64
    assert bool(mask[:, 0].all())


def test_is_edge_bar_like_refuses_a_blob_without_frame_shape() -> None:
    """A blob with no frame shape cannot be tested for edge adjacency, so it must not fire."""

    grid = _r11l_like_grid()
    blobs = [b for b in connected_color_blobs(grid) if b.bbox == (0, 0, 63, 0)]
    assert blobs, "fixture must contain the column-0 blob"
    assert is_edge_bar_like(blobs[0]) is True

    detached = type(blobs[0])(
        color=blobs[0].color,
        pixel_count=blobs[0].pixel_count,
        bbox=blobs[0].bbox,
        centroid=blobs[0].centroid,
        cells=blobs[0].cells,
        frame_shape=None,
    )
    assert is_edge_bar_like(detached) is False


def test_is_edge_bar_like_refuses_zero_sized_frame_shape() -> None:
    """A malformed blob shape cannot be an edge bar even if the bbox is bar-like."""

    blob = _blob_at((0, 0, 4, 0), (0, 8))
    assert is_edge_bar_like(blob) is False


def test_shipped_classifier_predicate_is_untouched_by_the_repair() -> None:
    """The co-change guard: candidate RANKING and the trained CNN feature must not move.

    `is_status_bar_like` feeds `tier()`, `button_likelihood()` and feature 5 of the already
    trained click-target feature table. The repair is a NEW predicate precisely so those stay
    byte-identical; this test fails if someone later "simplifies" it by widening in place.
    """

    prior = ColorBlobSaliencePrior()
    grid = _r11l_like_grid()
    col0 = [b for b in connected_color_blobs(grid) if b.bbox == (0, 0, 63, 0)][0]

    assert prior.is_status_bar_like(col0) is False, "shipped predicate must still MISS this"
    assert is_edge_bar_like(col0) is True, "the repaired predicate is the one that catches it"
    assert prior.tier(col0) != 4, "tier must not have been reclassified as status-bar"


# ---------------------------------------------------------------------------
# Stage 2 -- multi-frame behavioural confirmation
# ---------------------------------------------------------------------------


def _column0_mask(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:, 0] = True
    return mask


def test_region_evidence_admits_a_monotone_edge_counter() -> None:
    """A counter that ticks on every action while the board sits still: ADMIT."""

    grids = [_r11l_like_grid(filled=i) for i in range(20)]
    actions = [None] + [(6, i % 3) for i in range(1, 20)]
    ev = region_hud_evidence(grids, _column0_mask(), actions=actions)

    assert ev["verdict"] == "admit", ev
    assert ev["independent_tick_rate"] == 1.0
    assert ev["ubiquity"] == 1.0
    assert ev["revisits"] == 0
    assert ev["n_distinct_complement_values"] == 1


def test_region_evidence_refuses_a_static_edge_stripe() -> None:
    """Geometry cannot tell a decorative border from a counter; this stage can. REFUSE."""

    grids = []
    for step in range(20):
        g = _static_edge_stripe_grid()
        g[5, 5 + (step % 8)] = 9  # only the BOARD changes; the stripe never does
        grids.append(g)
    mask = np.zeros((32, 32), dtype=bool)
    mask[:, 31] = True

    ev = region_hud_evidence(grids, mask, actions=[(1, 0)] * 20)
    assert ev["verdict"] == "refuse"
    assert ev["reason"] == "region_not_action_ubiquitous"
    assert ev["ubiquity"] == 0.0


def test_region_evidence_refuses_a_region_carrying_game_state() -> None:
    """A region that is REVISITED is reversible, so it is game state, not a counter. REFUSE."""

    # A token OSCILLATING between two column-0 positions: ubiquitous (it moves on every action)
    # but REVERSIBLE, so non-reciprocity is the only conjunct that can catch it.
    #
    # NOTE the deliberate shape: the oscillation must NOT pass back through the segment's FIRST
    # region value on every lap. `region_hud_evidence` treats a return to the first value as a
    # counter RESTART and opens a new segment (the fix that stopped r11l false-refusing on
    # seed 0), so a cycle whose phase 0 coincides with the reset value is invisible to the
    # revisit test. That is a real limitation, recorded in the module's HONEST LIMITS; this
    # fixture tests the case the rule can actually see. Found by this test's own first version.
    grids = [_r11l_like_grid()]
    for step in range(1, 24):
        g = _r11l_like_grid()
        g[10 + (step % 2), 0] = 9
        grids.append(g)
    ev = region_hud_evidence(grids, _column0_mask(), actions=[(1, 0)] * 24)

    assert ev["verdict"] == "refuse"
    assert ev["reason"] == "region_value_revisited_in_episode"
    assert ev["revisits"] > 0


def test_region_evidence_refuses_when_one_action_class_never_moves_the_region() -> None:
    """su15's measured shape: one action class moves it, another never does. Per-class MINIMUM."""

    grids = [_r11l_like_grid(filled=0)]
    actions = [None]
    filled = 0
    for step in range(1, 25):
        if step % 2:  # action class 6 advances the counter
            filled += 1
            actions.append((6, 0))
        else:  # action class 1 never touches it
            actions.append((1, 0))
        grids.append(_r11l_like_grid(filled=filled))

    ev = region_hud_evidence(grids, _column0_mask(), actions=actions)
    assert ev["verdict"] == "refuse"
    assert ev["ubiquity"] == 0.0
    assert ev["per_action_change_rate"]["(6, 0)"] == 1.0


def test_region_evidence_abstains_below_the_transition_floor() -> None:
    """An abstain is NOT a pass -- the caller must keep its previous behaviour."""

    grids = [_r11l_like_grid(filled=i) for i in range(4)]
    ev = region_hud_evidence(grids, _column0_mask(), actions=[(6, 0)] * 4)
    assert ev["verdict"] == "abstain"
    assert ev["reason"] == "insufficient_transitions"


def test_region_evidence_handles_no_mask_and_terminal_frames() -> None:
    assert region_hud_evidence([], None)["verdict"] == "abstain"
    assert region_hud_evidence([], np.zeros((4, 4), bool))["reason"] == "no_mask"

    # A None entry is an episode break, not a crash, and it resets the revisit segment.
    grids: list = [_r11l_like_grid(filled=i) for i in range(10)]
    grids.append(None)
    grids += [_r11l_like_grid(filled=i) for i in range(10)]
    ev = region_hud_evidence(grids, _column0_mask(), actions=[(6, 0)] * len(grids))
    assert ev["revisits"] == 0, "a counter restarting after a break is not a revisit"


def test_region_evidence_marks_a_pooled_ubiquity_as_pooled() -> None:
    """Without action labels the statistic degrades; it must SAY so rather than look per-class."""

    grids = [_r11l_like_grid(filled=i) for i in range(20)]
    ev = region_hud_evidence(grids, _column0_mask())
    assert ev["ubiquity_is_pooled"] is True
    assert ev["ubiquity"] == 1.0


def test_region_evidence_abstains_when_no_action_class_repeats() -> None:
    """Enough transitions are still not enough if no action class has a rate estimate."""

    grids = [_r11l_like_grid(filled=i) for i in range(20)]
    actions = [None] + [(6, i) for i in range(1, 20)]
    ev = region_hud_evidence(grids, _column0_mask(), actions=actions)
    assert ev["verdict"] == "abstain"
    assert ev["reason"] == "no_action_class_tried_twice"
    assert ev["ubiquity"] is None


# ---------------------------------------------------------------------------
# Stage 3 -- the runtime collapse guard
# ---------------------------------------------------------------------------


def _prove_a_collapse(guard: MaskCollapseGuard, node: str, action_key=(6, None)) -> bool:
    """Drive `guard` through the ONE observation sequence that yields a PROVEN collapse.

    Proving a collapse needs the unmasked control to have POWER on the DECIDING observation,
    i.e. that raw antecedent must already have been seen behaving deterministically. So:
    raw state `Ub` -> `S2`; a DIFFERENT raw state `Ua` sharing the same masked hash -> `S1`
    (branching, but `Ua` is new, so unproven); then `Ub` -> `S2` again, which repeats
    deterministically and turns the branching into a proof.
    """

    guard.observe(
        origin_masked=node,
        origin_unmasked=f"{node}_Ub",
        action_key=action_key,
        successor_masked=f"{node}_S2",
        successor_unmasked=f"{node}_SUb",
    )
    guard.observe(
        origin_masked=node,
        origin_unmasked=f"{node}_Ua",
        action_key=action_key,
        successor_masked=f"{node}_S1",
        successor_unmasked=f"{node}_SUa",
    )
    return guard.observe(
        origin_masked=node,
        origin_unmasked=f"{node}_Ub",
        action_key=action_key,
        successor_masked=f"{node}_S2",
        successor_unmasked=f"{node}_SUb",
    )


def test_collapse_guard_refuses_a_mask_that_aliases_distinct_states() -> None:
    """SCENARIO-ARC-WMTE-5960-COLLAPSE-GUARD-HARD-REFUSAL.

    Same masked node + same concrete action -> two DIFFERENT masked successors, while the
    unmasked control does not veto. The guard un-masks that node.

    NOTE THE OPT-IN. By default the guard acts only on PROVEN collapses -- the corpus measured
    that acting on unproven branchings costs whole wins (tu93/dc22/su15 each 1 level -> 0). This
    test pins `act_on_unproven_branchings=True` because it is about the REFUSAL MECHANICS, not
    the default policy; `test_collapse_guard_ignores_an_unproven_branching_by_default` covers
    the default, and `_prove_a_collapse` covers the proven path.
    """

    guard = MaskCollapseGuard(act_on_unproven_branchings=True)
    common = dict(action_key=(6, {"x": 3, "y": 4}))

    # THE SHAPE OF A REAL COLLAPSE, and it matters: the two observations share a MASKED origin
    # (that is the collapse) but have DIFFERENT unmasked origins (they are genuinely different
    # true states). So each unmasked key has exactly ONE successor and the control does not
    # veto. Passing the same `origin_unmasked` twice would instead describe an environment that
    # branches from one identical raw frame -- non-determinism, not a mask fault -- which the
    # guard correctly declines to blame on the mask (see the next test).
    first = guard.observe(
        origin_masked="M1",
        origin_unmasked="U1a",
        successor_masked="S1",
        successor_unmasked="SU1",
        **common,
    )
    assert first is False, "one observation can never prove anything"
    assert guard.refusals == 0

    second = guard.observe(
        origin_masked="M1",
        origin_unmasked="U1b",
        successor_masked="S2",
        successor_unmasked="SU2",
        **common,
    )
    assert second is True
    assert guard.refusals == 1
    assert guard.violations == 1
    assert guard.is_split("M1") is True
    assert guard.is_split("M2") is False
    assert guard.observable_key_count() == 1
    assert guard.globally_revoked is False
    # THE HONEST CLASSIFICATION. Neither unmasked antecedent repeated, so the control could not
    # have exonerated: the guard acted (because this instance opted in) but must NOT call it a
    # proof.
    diagnostics = guard.diagnostics()
    assert diagnostics["unproven_masked_branchings"] == 1
    assert diagnostics["proven_collapses"] == 0
    assert diagnostics["keys_with_repeated_unmasked_antecedent"] == 0
    assert diagnostics["control_had_power_on_any_key"] is False
    assert diagnostics["refusals_are_all_proven"] is False
    assert diagnostics["acted_on_unproven_branchings"] is True


def test_collapse_guard_ignores_an_unproven_branching_by_default() -> None:
    """REGRESSION for the measured harm of acting on branchings the control could not judge.

    Full 25-game corpus, 3 seeds, budget 2000: with unproven branchings acted on, the guard --
    the SAFETY mechanism -- became the run's largest source of lost wins, on games whose mask no
    repair touched. tu93 60 nodes / 1 level / 361 actions -> 578 nodes / 0 levels / 1957 actions
    (28 refusals, 20 of them unproven); dc22 1 level -> 0 (35 refusals, 32 unproven); su15
    1 level -> 0 (12 refusals, 10 unproven). On a monotone-counter region the unmasked antecedent
    never repeats by construction, so most branchings there can never be proven -- and treating
    each as a reason to un-mask reverses the dedup the mask exists to provide.
    """

    guard = MaskCollapseGuard()
    common = dict(action_key=(6, None))
    guard.observe(
        origin_masked="M1",
        origin_unmasked="U1a",
        successor_masked="S1",
        successor_unmasked="SU1",
        **common,
    )
    acted = guard.observe(
        origin_masked="M1",
        origin_unmasked="U1b",
        successor_masked="S2",
        successor_unmasked="SU2",
        **common,
    )
    assert acted is False
    assert guard.refusals == 0
    assert guard.is_split("M1") is False, "identity must be unchanged on an unproven branching"
    diagnostics = guard.diagnostics()
    # The branching is still COUNTED and ATTRIBUTED -- it is evidence, just not a proof.
    assert diagnostics["unproven_masked_branchings"] == 1
    assert diagnostics["proven_collapses"] == 0
    assert diagnostics["acted_on_unproven_branchings"] is False
    # And a PROVEN collapse on another node still fires under the same default.
    assert _prove_a_collapse(guard, "M2") is True
    assert guard.is_split("M2") is True
    assert guard.diagnostics()["proven_collapses"] == 1


def test_collapse_guard_separates_a_proven_collapse_from_an_unproven_branching() -> None:
    """REGRESSION for the control-POWER defect (2026-07-25 adversarial review).

    ``control_live`` only says an unmasked antecedent was SUPPLIED. The exoneration branch
    additionally needs that unmasked antecedent to REPEAT -- and if the masked region is a
    monotone counter it never does, so ``non_deterministic_keys_excluded_by_control: 0`` on such
    a key is a CONSTRUCTIONAL zero rather than evidence of determinism.

    The distinction is not cosmetic: it decides whether the guard acts. Under the proven-only
    default the deciding observation must land on a raw antecedent already shown to behave
    deterministically -- which is exactly what `_prove_a_collapse` constructs.
    """

    common = dict(action_key=(6, None))

    # UNPROVEN: two different raw antecedents, neither ever repeating. The control is LIVE but
    # could not have fired, so the branching is counted and not acted on.
    unproven = MaskCollapseGuard()
    unproven.observe(
        origin_masked="M1",
        origin_unmasked="Ub",
        successor_masked="S2",
        successor_unmasked="SUb",
        **common,
    )
    unproven.observe(
        origin_masked="M1",
        origin_unmasked="Ua",
        successor_masked="S1",
        successor_unmasked="SUa",
        **common,
    )
    diagnostics = unproven.diagnostics()
    assert diagnostics["control_live"] is True, "an antecedent WAS supplied"
    assert diagnostics["control_had_power_on_any_key"] is False, "...but it could not have fired"
    assert diagnostics["unproven_masked_branchings"] == 1
    assert diagnostics["proven_collapses"] == 0
    assert diagnostics["non_deterministic_keys_excluded_by_control"] == 0, (
        "a CONSTRUCTIONAL zero -- the exoneration branch was unreachable, not passed"
    )
    assert unproven.is_split("M1") is False

    # PROVEN: the deciding observation lands on a raw antecedent that repeated deterministically,
    # so environment non-determinism is genuinely ruled out at that key.
    proven = MaskCollapseGuard()
    assert _prove_a_collapse(proven, "P") is True
    diagnostics = proven.diagnostics()
    assert diagnostics["proven_collapses"] == 1
    assert diagnostics["unproven_masked_branchings"] == 1, (
        "the middle observation was itself an unproven branching, and is still counted"
    )
    assert diagnostics["keys_with_repeated_unmasked_antecedent"] == 1
    assert diagnostics["control_had_power_on_any_key"] is True
    assert proven.is_split("P") is True


def test_collapse_guard_does_not_fire_on_environment_nondeterminism() -> None:
    """The MANDATORY control: if the UNMASKED key also branches, the mask is not at fault.

    Measured on sc25, whose masked violations are matched 1:1 by unmasked-control violations.
    """

    guard = MaskCollapseGuard()
    for successor in ("S1", "S2"):
        guard.observe(
            origin_masked="M1",
            origin_unmasked="U1",
            action_key=(6, None),
            successor_masked=successor,
            successor_unmasked="unmasked_" + successor,
        )
    assert guard.refusals == 0
    assert guard.violations == 0
    assert guard.non_deterministic_keys == 1
    assert guard.is_split("M1") is False


def test_collapse_guard_declines_to_act_without_a_live_control() -> None:
    """REGRESSION for a real bug this module's own smoke run found (2026-07-25).

    The first version treated a MISSING unmasked control as a PASSED control, so it un-masked 6
    nodes on tu93 and 4 on lf52 on zero evidence -- destroying both wins -- while reporting
    `non_deterministic_keys_excluded_by_control: 0`, i.e. a dead control channel that read as a
    clean one. Root cause: the antecedent frame was read from the graph node, and a bare
    explorer only retains node frames when certain optional components are enabled, so it was
    None on 1952 of 1952 transitions. No control means no proof.
    """

    guard = MaskCollapseGuard()
    for successor in ("S1", "S2"):
        guard.observe(
            origin_masked="M1",
            origin_unmasked=None,
            action_key=(6, None),
            successor_masked=successor,
            successor_unmasked=None,
        )
    assert guard.refusals == 0, "an uncontrolled branching is NOT a proof"
    assert guard.violations == 0
    assert guard.is_split("M1") is False
    assert guard.uncontrolled_branchings_declined == 1
    assert guard.uncontrolled_observations == 2
    assert guard.diagnostics()["control_live"] is False


def test_collapse_guard_control_liveness_is_reported() -> None:
    """`control_live` must be the thing a reader checks before trusting a zero-refusal cell."""

    dead = MaskCollapseGuard()
    dead.observe(
        origin_masked="M",
        origin_unmasked=None,
        action_key=1,
        successor_masked="S",
        successor_unmasked=None,
    )
    assert dead.diagnostics()["control_live"] is False

    live = MaskCollapseGuard()
    live.observe(
        origin_masked="M",
        origin_unmasked="U",
        action_key=1,
        successor_masked="S",
        successor_unmasked="SU",
    )
    assert live.diagnostics()["control_live"] is True
    assert live.diagnostics()["controlled_keys"] == 1


def test_explorer_supplies_a_live_control_to_the_guard() -> None:
    """The wiring half of the same regression: the antecedent must not come from the graph node.

    Node frames are only retained when certain optional components are enabled, so the guard's
    control has to be sourced from the previously OBSERVED frame instead. Two ingests are enough
    to prove the channel is live.
    """

    # Stage 2 pinned OFF: the guard only observes while a mask is APPLIED, and with Stage 2 armed
    # no mask is applied until it has 16 transitions of evidence. This test is about the control
    # CHANNEL, so it uses the immediate-application configuration.
    explorer = StepwiseExplorer(
        edge_bar_hud_mask=True, hud_mask_collapse_guard=True, hud_mask_stage2_confirm=False
    )
    explorer._ingest(_FakeFrame(_r11l_like_grid(filled=0)))
    assert explorer._last_unmasked_hash is not None
    first = explorer._last_unmasked_hash

    # Simulate an issued action, then the resulting observation, exactly as `_next` -> `_ingest`.
    explorer.awaiting = {
        "origin": explorer._hash(_FakeFrame(_r11l_like_grid(filled=0))),
        "action": 6,
        "data": {"x": 1, "y": 2},
        "grid": None,
        "level_before": 0,
        "previous_frame": None,  # the state that made the control dead in the first version
    }
    explorer._ingest(_FakeFrame(_r11l_like_grid(filled=1)))

    guard = explorer._hud_collapse_guard.diagnostics()
    assert guard["observations"] == 1
    assert guard["control_live"] is True, "control must be live even with previous_frame=None"
    assert guard["uncontrolled_observations"] == 0
    assert explorer._last_unmasked_hash != first


def _split_n_nodes(guard: MaskCollapseGuard, count: int) -> None:
    """Split `count` nodes via PROVEN collapses, so these tests are about revocation only.

    Uses the proven path deliberately: the default guard ignores unproven branchings (see
    `test_collapse_guard_ignores_an_unproven_branching_by_default`), so an unproven driver here
    would make the revocation tests silently vacuous.
    """

    for index in range(count):
        assert _prove_a_collapse(guard, f"M{index}") is True


def test_collapse_guard_never_globally_revokes_by_default() -> None:
    """REGRESSION for the measured graph corruption (2026-07-25 adversarial review).

    Global revocation was NOT a fallback to the unmasked baseline. Once it fired, ``is_split``
    returned True unconditionally and ``_hash`` emitted the compound ``masked|u:unmasked`` key for
    every subsequent frame, while nodes created BEFORE it kept their plain masked keys -- the same
    true state present twice under two identity conventions. Instrumented on tu93 seed 20260724:
    72 hashes pre-revocation, 1927 post, 58 of 658 distinct raw frames holding BOTH key forms, and
    640 of 655 graph nodes (97.7%) on the far side of the switch. Its measured effect was strictly
    WORSE than shipping no guard at all: on tu93 -- where the repaired mask is IDENTICAL to the
    shipped one, so arming the guard was the only difference -- 1 level / 361 actions became 0
    levels / 1953 actions on 3 of 3 seeds.

    Splits are therefore LOCAL and unbounded: one identity convention per node, degrading
    gracefully toward unmasked identity instead of flipping the whole graph.
    """

    guard = MaskCollapseGuard()
    _split_n_nodes(guard, 8)
    assert guard.refusals == 8
    assert guard.globally_revoked is False
    assert guard.is_split("M0") is True, "the nodes that branched ARE un-masked"
    assert guard.is_split("never_seen_node") is False, "and nothing else is"
    diagnostics = guard.diagnostics()
    assert diagnostics["globally_revoked"] is False
    assert diagnostics["revocation_mode"] == HUD_MASK_GUARD_REVOCATION_LOCAL
    assert diagnostics["max_split_nodes"] is None
    # Reporting-only signal: past the threshold the mask is probably wrong wholesale, and the
    # artifact should say so -- but it changes NO behaviour.
    assert diagnostics["split_budget_exceeded"] is True
    assert diagnostics["split_node_count"] == 8


def test_legacy_global_hash_flip_mode_is_opt_in_and_flips_every_node() -> None:
    """The measured-harmful mode survives ONLY as a fixture, and it must stay opt-in.

    Kept so the corruption it causes is demonstrable rather than only described: past the cap it
    un-masks nodes it has never even seen, which is what re-keyed 97.7% of a real graph.
    """

    guard = MaskCollapseGuard(
        max_split_nodes=2,
        revocation_mode=HUD_MASK_GUARD_REVOCATION_GLOBAL_HASH_FLIP,
    )
    _split_n_nodes(guard, 3)
    assert guard.refusals == 3
    assert guard.globally_revoked is True
    assert guard.is_split("never_seen_node") is True, "this is exactly the corruption"
    assert guard.diagnostics()["globally_revoked"] is True

    # The DEFAULT-mode guard with the same cap does not flip, which is the point of the fix.
    default_mode = MaskCollapseGuard(max_split_nodes=2)
    _split_n_nodes(default_mode, 3)
    assert default_mode.globally_revoked is False
    assert default_mode.is_split("never_seen_node") is False


def test_collapse_guard_ignores_incomplete_observations() -> None:
    guard = MaskCollapseGuard()
    assert (
        guard.observe(
            origin_masked=None,
            origin_unmasked="U",
            action_key=1,
            successor_masked="S",
            successor_unmasked="SU",
        )
        is False
    )
    assert (
        guard.observe(
            origin_masked="M",
            origin_unmasked="U",
            action_key=1,
            successor_masked=None,
            successor_unmasked="SU",
        )
        is False
    )
    assert guard.observations == 0


def test_collapse_guard_handles_unhashable_action_payloads() -> None:
    """Click payloads arrive as dicts; a guard that raises on one would take the search down."""

    # Opted in because this test is about the ACTION-KEY coercion, not the refusal policy: with
    # the proven-only default a two-observation driver would never act and the assertion below
    # would be vacuous rather than exercising `_hashable`.
    guard = MaskCollapseGuard(act_on_unproven_branchings=True)
    payload = {"x": 1, "y": 2, "nested": [1, 2, {"z": 3}]}
    for index, successor in enumerate(("S1", "S2")):
        guard.observe(
            origin_masked="M",
            origin_unmasked=f"U{index}",
            action_key=(6, payload),
            successor_masked=successor,
            successor_unmasked="U" + successor,
        )
    assert guard.violations == 1


def test_collapse_guard_can_measure_without_acting_on_unproven_branchings() -> None:
    """The proven-only variant is explicit and still attributes the counted branching once."""

    guard = MaskCollapseGuard(act_on_unproven_branchings=False)
    for successor, unmasked in (("S1", "U1"), ("S2", "U2")):
        guard.observe(
            origin_masked="M",
            origin_unmasked=unmasked,
            action_key=(6, None),
            successor_masked=successor,
            successor_unmasked="US" + successor,
        )
    diagnostics = guard.diagnostics()
    assert diagnostics["acted_on_unproven_branchings"] is False
    assert diagnostics["unproven_masked_branchings"] == 1
    assert diagnostics["collapse_refusals"] == 0
    assert guard.is_split("M") is False
    assert diagnostics["attribution"]["attribution_unavailable"] == 1


def test_collapse_guard_hashes_sets_and_repr_only_payloads() -> None:
    """Set payloads and repr-only objects must be stable enough for repeated action keys."""

    class ReprOnly:
        __hash__ = None

        def __repr__(self) -> str:
            return "repr-only-action"

    # Opted in for the same reason as the sibling payload test: this is about `_hashable`
    # coercion of the ACTION KEY, not about the refusal policy. With the proven-only default a
    # two-observation driver would never act and the assertion below would be vacuous.
    guard = MaskCollapseGuard(act_on_unproven_branchings=True)
    payload = {"choices": {3, 1, 2}, "object": ReprOnly()}
    for successor, unmasked in (("S1", "U1"), ("S2", "U2")):
        guard.observe(
            origin_masked="M",
            origin_unmasked=unmasked,
            action_key=payload,
            successor_masked=successor,
            successor_unmasked="US" + successor,
        )
    assert guard.violations == 1
    assert guard.observable_key_count() == 1


def test_mask_summary_describes_a_mask_and_a_none() -> None:
    assert mask_summary(None) == {
        "resolved": False,
        "cell_count": 0,
        "rows": [],
        "cols": [],
        "digest": None,
    }
    summary = mask_summary(_column0_mask((8, 8)))
    assert summary["cell_count"] == 8
    assert summary["cols"] == [0]
    assert summary["rows"] == list(range(8))
    # The mask's IDENTITY, not just its size: two masks must be compared on this digest, because
    # equal cell COUNTS do not imply the same cells (the harness previously compared counts and
    # would therefore have exonerated a repair that MOVED a mask instead of widening it).
    assert isinstance(summary["digest"], str) and len(summary["digest"]) == 16
    assert summary["digest"] == mask_cell_digest(_column0_mask((8, 8)))
    moved = np.zeros((8, 8), dtype=bool)
    moved[:, 1] = True  # same 8 cells, different column
    assert mask_cell_digest(moved) != summary["digest"]
    assert mask_summary(None)["digest"] is None


def test_mask_cell_digest_treats_none_and_empty_masks_as_unresolved() -> None:
    assert mask_cell_digest(None) is None
    assert mask_cell_digest(np.zeros((3, 3), dtype=bool)) is None


# ---------------------------------------------------------------------------
# Wiring -- default-off parity, and the flag actually taking effect
# ---------------------------------------------------------------------------


def test_defaults_are_off_and_published_in_the_submitted_config() -> None:
    """SCENARIO-ARC-WMTE-5960-DEFAULT-OFF-PARITY (part 1)."""

    # FLIPPED ON 2026-07-25 (operator decision) after the full-corpus per-seed A/B: arm G3
    # (detector + Stage 2 + collapse guard) gains r11l on EVERY seed and loses nothing, while
    # detector-alone (G) and detector+guard (G2) both REGRESS lp85. See the flag block in
    # arc_competition_agent.py. These assertions pinned the shipped default and existed to force
    # a conscious update before any flip -- which is what happened.
    assert SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED is True
    assert SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED is True
    assert SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED is True
    assert SUBMITTED_AGENT_CONFIG["edge_bar_hud_mask_enabled"] is True
    assert SUBMITTED_AGENT_CONFIG["hud_mask_collapse_guard_enabled"] is True
    assert SUBMITTED_AGENT_CONFIG["hud_mask_stage2_confirm_enabled"] is True

    explorer = StepwiseExplorer()
    assert explorer.edge_bar_hud_mask_enabled is True
    assert explorer.hud_mask_collapse_guard_enabled is True
    assert explorer.hud_mask_stage2_confirm_enabled is True

    # the OFF behaviour is still a real, load-bearing property (it is what arm B2 measures against),
    # so it is now requested EXPLICITLY rather than inherited from the module defaults
    off = StepwiseExplorer(
        edge_bar_hud_mask=False, hud_mask_collapse_guard=False, hud_mask_stage2_confirm=False
    )
    assert off.edge_bar_hud_mask_enabled is False
    assert off.hud_mask_collapse_guard_enabled is False
    assert off.hud_mask_stage2_confirm_enabled is False
    assert off._hud_collapse_guard is None


def test_default_off_leaves_node_identity_byte_identical() -> None:
    """SCENARIO-ARC-WMTE-5960-DEFAULT-OFF-PARITY (part 2).

    The r11l-shaped frame is the sharpest possible case: with the repair ON its column-0
    counter leaves node identity, with the repair OFF it must NOT -- so two frames differing
    only in the counter must still hash DIFFERENTLY by default.
    """

    frame_a = _FakeFrame(_r11l_like_grid(filled=0))
    frame_b = _FakeFrame(_r11l_like_grid(filled=7))

    control = StepwiseExplorer(edge_bar_hud_mask=False, hud_mask_stage2_confirm=False)
    control._ingest(frame_a)
    control._ingest(frame_b)
    assert control.hud_mask is None
    assert control._hash(frame_a) != control._hash(frame_b), "default must NOT dedup these"

    # Stage 2 is pinned OFF here because this test is about STAGE 1 taking effect on the very
    # first frame. With Stage 2 armed (the default whenever the detector is on) the mask is
    # deliberately NOT applied until >=16 transitions have been observed -- see
    # `test_stage2_defers_activation_until_the_evidence_exists`.
    treatment = StepwiseExplorer(edge_bar_hud_mask=True, hud_mask_stage2_confirm=False)
    treatment._ingest(frame_a)
    assert treatment.hud_mask is not None
    assert int(treatment.hud_mask.sum()) == 64
    assert treatment._hash(frame_a) == treatment._hash(frame_b), "repair MUST dedup these"


def test_env_override_resolves_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The A/B harness flips arms by env so module globals never leak across arms in-process."""

    monkeypatch.setenv("CARNOT_ARC_EDGE_BAR_HUD_MASK", "1")
    monkeypatch.setenv("CARNOT_ARC_HUD_MASK_COLLAPSE_GUARD", "1")
    explorer = StepwiseExplorer()
    assert explorer.edge_bar_hud_mask_enabled is True
    assert explorer.hud_mask_collapse_guard_enabled is True

    # An explicit kwarg still outranks the env, so a test can pin an arm exactly.
    pinned = StepwiseExplorer(edge_bar_hud_mask=False)
    assert pinned.edge_bar_hud_mask_enabled is False


def test_hud_mask_source_records_which_detector_resolved_the_mask() -> None:
    """A reader must never have to infer the treatment from a cell count."""

    treatment = StepwiseExplorer(edge_bar_hud_mask=True, hud_mask_stage2_confirm=False)
    treatment._ingest(_FakeFrame(_r11l_like_grid()))
    assert treatment._hud_mask_source == "edge_bar_detector_req5960"

    # With Stage 2 armed (the coupled default) the SAME frame yields a PENDING source, because the
    # candidate is deliberately not applied yet. That distinction has to be visible in the row, or
    # "Stage 2 is still deciding" reads as "the detector found nothing".
    deferred = StepwiseExplorer(edge_bar_hud_mask=True)
    deferred._ingest(_FakeFrame(_r11l_like_grid()))
    assert deferred._hud_mask_source == "edge_bar_detector_req5960_stage2_pending"
    assert deferred.hud_mask is None

    # the PRE-FLIP control must now be requested explicitly (all three flags ship True since
    # 2026-07-25); with Stage 2 left armed this would report stage2_pending, not "no bar found"
    control = StepwiseExplorer(edge_bar_hud_mask=False, hud_mask_stage2_confirm=False)
    control._ingest(_FakeFrame(_r11l_like_grid()))
    assert control._hud_mask_source == "unresolved_no_bar_detected"

    horizontal = np.full((32, 32), 3, dtype=int)
    horizontal[0, :] = 8
    shipped = StepwiseExplorer(edge_bar_hud_mask=False, hud_mask_stage2_confirm=False)
    shipped._ingest(_FakeFrame(horizontal))
    assert shipped._hud_mask_source == "status_bar_classifier_req5583"


def test_diagnostics_report_dedup_and_guard_activity() -> None:
    """Every arm must emit the mask/dedup/guard fields, so a broken arm cannot read as a null."""

    # Stage 2 pinned OFF: this test asserts the DEDUP counters, which only move once a mask is
    # applied. The Stage-2-armed configuration's own diagnostics are covered by
    # `test_stage2_verdict_is_reported_in_the_diagnostics`.
    explorer = StepwiseExplorer(
        edge_bar_hud_mask=True, hud_mask_collapse_guard=True, hud_mask_stage2_confirm=False
    )
    for filled in range(6):
        explorer._ingest(_FakeFrame(_r11l_like_grid(filled=filled)))

    diag = explorer.hud_mask_diagnostics()
    assert diag["edge_bar_hud_mask_enabled"] is True
    assert diag["hud_mask_resolved"] is True
    assert diag["hud_mask_cell_count"] == 64
    assert diag["hud_mask_cols"] == [0]
    assert diag["unique_frames"] == 6, "six distinct RAW frames were observed"
    assert diag["graph_nodes"] == 1, "all six collapse to one node once the counter is masked"
    assert diag["node_inflation_vs_unique_frames"] == round(1 / 6, 4)
    assert diag["collapse_guard"] is not None
    assert diag["collapse_guard_refusals"] == 0


def test_split_node_hash_reverts_to_unmasked_identity() -> None:
    """The guard's decision has to actually change `_hash`, not just increment a counter."""

    explorer = StepwiseExplorer(
        edge_bar_hud_mask=True, hud_mask_collapse_guard=True, hud_mask_stage2_confirm=False
    )
    frame_a = _FakeFrame(_r11l_like_grid(filled=0))
    frame_b = _FakeFrame(_r11l_like_grid(filled=7))
    explorer._ingest(frame_a)
    assert explorer._hash(frame_a) == explorer._hash(frame_b)

    masked = explorer._hash(frame_a)
    explorer._hud_collapse_guard.split_hashes.add(masked)

    assert explorer._hash(frame_a) != explorer._hash(frame_b), "split node must un-mask"
    assert explorer._hash(frame_a).startswith(masked + "|u:")


# ---------------------------------------------------------------------------
# Stage 2b -- DEFERRED ACTIVATION (the ar25 over-mask fix), and the flag coupling
# ---------------------------------------------------------------------------


def _fill_gauge_grid(height: int) -> np.ndarray:
    """An ar25-shaped FILL-LEVEL GAUGE in column 63: an edge bar that is NOT a clock.

    This is the shape that makes Stage 1 alone unsafe. Geometrically it is indistinguishable
    from r11l's counter -- edge-adjacent, 64x1, one 4-connected blob -- but its value is a
    decision-relevant state variable, and on the real game masking it collapsed 1554 distinct
    raw frames into 233 graph nodes with the collapse guard proving aliasing keys.
    """

    grid = np.full((64, 64), 3, dtype=int)
    grid[:, 63] = 5
    grid[64 - height :, 63] = 11
    grid[30:34, 30:34] = 7
    return grid


def _fill_gauge_sequence(n: int) -> tuple[list[np.ndarray], list[int]]:
    """A gauge sequence with the REAL measured signature: one action class never moves it.

    ar25's newly-masked column 63 was measured at ubiquity 0.0 -- action classes 6 and 7 never
    move it while 1/2/4/5 do -- which is precisely how "the game state changed" differs from "a
    clock ticked". A fixture where EVERY action moves the region would instead land in the
    periodic blind spot Stage 2's own docstring records (a region that cycles back through its
    segment's first value reads as a series of restarts, not revisits) and would be ADMITTED --
    which is exactly why Stage 3 remains mandatory on top of Stage 2.
    """

    frames: list[np.ndarray] = []
    actions: list[int] = []
    height = 10
    for index in range(n):
        action = 6 if index % 2 else 4
        if index and action == 6:
            height = 10 + ((height - 10 + 2) % 40)
        frames.append(_fill_gauge_grid(height))
        actions.append(action)
    return frames, actions


def test_stage1_alone_fires_on_a_fill_gauge_which_is_why_stage2_exists() -> None:
    """The CARDINAL SIN, demonstrated: single-frame geometry cannot refuse a fill gauge.

    Stage 1 must fire here -- that is not a bug in Stage 1, it is the provable limit of a shape
    prior. The test exists so nobody "fixes" it by tightening the geometry (which would also
    lose r11l, whose counter has the identical shape) instead of gating it behind Stage 2.
    """

    r11l_like = _r11l_like_grid(filled=3)
    gauge = _fill_gauge_grid(height=20)

    assert edge_bar_hud_mask(r11l_like) is not None
    gauge_mask = edge_bar_hud_mask(gauge)
    assert gauge_mask is not None, "geometry alone CANNOT tell a gauge from a clock"
    assert bool(gauge_mask[:, 63].all())


def test_stage2_admits_a_monotone_counter_and_refuses_a_fill_gauge() -> None:
    """The statistic that separates them, on the two shapes Stage 1 conflates.

    A monotone counter ticks on EVERY action and never revisits a value. A fill gauge tracks
    game state: it moves only on some actions and its value recurs as the state recurs.
    """

    counter_mask = _column0_mask((64, 64))
    counter_frames = [_r11l_like_grid(filled=i) for i in range(20)]
    counter = region_hud_evidence(counter_frames, counter_mask, actions=[6] * 20)
    assert counter["verdict"] == "admit"
    assert counter["ubiquity"] == 1.0
    assert counter["revisits"] == 0

    # The gauge's measured signature on the real game: it responds to SOME action classes and
    # not others (ar25 ubiquity 0.0), which is what "the game state moved" looks like as opposed
    # to "a clock ticked". Built with `_fill_gauge_sequence` so the fixture matches that.
    gauge_mask = np.zeros((64, 64), dtype=bool)
    gauge_mask[:, 63] = True
    gauge_frames, gauge_actions = _fill_gauge_sequence(20)
    gauge = region_hud_evidence(gauge_frames, gauge_mask, actions=gauge_actions)
    assert gauge["verdict"] == "refuse"
    assert gauge["reason"] == "region_not_action_ubiquitous"
    assert gauge["ubiquity"] == 0.0


def test_stage2_defers_activation_until_the_evidence_exists() -> None:
    """Identity stays UNMASKED -- i.e. exactly today's shipped behaviour -- while Stage 2 waits."""

    activation = DeferredMaskActivation(min_transitions=4)
    activation.propose(_column0_mask((64, 64)))
    assert activation.pending is True
    assert activation.verdict == "pending"

    activated = None
    for index in range(6):
        result = activation.observe(_r11l_like_grid(filled=index), 6)
        if result is not None:
            activated = result
            break
    assert activated is not None, "an admitted candidate must be returned exactly once"
    assert activation.verdict == "admitted"
    assert activation.activated_after_transitions >= 4
    # And the buffer is released, so a decided instance does not keep holding frames.
    assert activation.diagnostics()["buffered_frames"] == 0
    # A second observe on a decided instance returns None rather than re-activating.
    assert activation.observe(_r11l_like_grid(filled=9), 6) is None


def test_stage2_refusal_never_applies_the_candidate() -> None:
    """A refused candidate must leave the caller on unmasked identity, permanently."""

    activation = DeferredMaskActivation(min_transitions=4)
    gauge_mask = np.zeros((64, 64), dtype=bool)
    gauge_mask[:, 63] = True
    activation.propose(gauge_mask)
    frames, actions = _fill_gauge_sequence(8)
    for frame, action in zip(frames, actions):
        assert activation.observe(frame, action) is None
    assert activation.verdict == "refused"
    assert activation.pending is False
    assert activation.observe(_fill_gauge_grid(30), 6) is None


def test_stage2_discards_rather_than_guesses_at_the_buffer_cap() -> None:
    """Abstaining forever is a memory leak; guessing is an over-mask. Discard is the third option."""

    activation = DeferredMaskActivation(min_transitions=100, max_buffered_frames=6)
    activation.propose(_column0_mask((64, 64)))
    for index in range(6):
        assert activation.observe(_r11l_like_grid(filled=index), 6) is None
    assert activation.verdict == "discarded"
    assert activation.diagnostics()["buffered_frames"] == 0


def test_stage2_discards_when_evidence_still_abstains_at_the_cap() -> None:
    """The buffer cap also applies after the transition floor when evidence has no opinion."""

    activation = DeferredMaskActivation(min_transitions=2, max_buffered_frames=4)
    activation.propose(_column0_mask((64, 64)))
    for index in range(4):
        assert activation.observe(_r11l_like_grid(filled=index), ("unique", index)) is None
    assert activation.verdict == "discarded"
    assert activation.reason == "buffer_cap_reached_while_stage2_still_abstaining"
    assert activation.evidence["reason"] == "no_action_class_tried_twice"


def test_stage2_no_candidate_is_not_a_refusal() -> None:
    """Stage 1 finding nothing and Stage 2 refusing something are different facts."""

    activation = DeferredMaskActivation()
    activation.propose(None)
    assert activation.verdict == "no_candidate"
    assert activation.pending is False
    assert activation.observe(_r11l_like_grid(), 6) is None


def test_stage2_normalizes_empty_baseline_and_short_circuits_when_nothing_was_added() -> None:
    empty = np.zeros((64, 64), dtype=bool)

    pending = DeferredMaskActivation()
    pending.propose(_column0_mask((64, 64)), empty)
    assert pending.pending is True
    assert pending.baseline is None

    unchanged = DeferredMaskActivation()
    mask = _column0_mask((64, 64))
    unchanged.propose(mask, mask.copy())
    assert unchanged.pending is False
    assert unchanged.verdict == "no_added_region"
    assert unchanged.fallback_mask() is not None
    diag = unchanged.diagnostics()
    assert diag["repair_added_cell_count"] == 0
    assert diag["baseline_cell_count"] == 64


def test_explorer_defers_the_mask_then_activates_it_on_a_real_counter() -> None:
    """The wiring: the explorer must not mask until Stage 2 admits, then must mask."""

    explorer = StepwiseExplorer(edge_bar_hud_mask=True)
    assert explorer.hud_mask_stage2_confirm_enabled is True
    for index in range(3):
        explorer.awaiting = None
        explorer._ingest(_FakeFrame(_r11l_like_grid(filled=index)))
    assert explorer.hud_mask is None, "identity must stay unmasked while Stage 2 decides"
    assert explorer._hud_mask_source == "edge_bar_detector_req5960_stage2_pending"

    for index in range(3, 30):
        explorer.awaiting = None
        explorer._ingest(_FakeFrame(_r11l_like_grid(filled=index % 60)))
    assert explorer.hud_mask is not None, "Stage 2 must eventually admit a real counter"
    assert explorer._hud_mask_source == "edge_bar_detector_req5960_stage2_confirmed"
    assert int(explorer.hud_mask.sum()) == 64


def test_explorer_never_activates_a_mask_stage2_refuses() -> None:
    """The ar25 case end to end: the gauge candidate is proposed and never applied."""

    explorer = StepwiseExplorer(edge_bar_hud_mask=True)
    frames, actions = _fill_gauge_sequence(40)
    for frame, action in zip(frames, actions):
        explorer.awaiting = {
            "origin": "x",
            "action": action,
            "data": None,
            "grid": None,
            "level_before": 0,
            "previous_frame": None,
        }
        explorer._ingest(_FakeFrame(frame))
    assert explorer.hud_mask is None, "a Stage-2-refused gauge must NEVER reach node identity"
    stage2 = explorer.hud_mask_diagnostics()["stage2"]
    assert stage2["stage2_verdict"] == "refused"
    assert stage2["candidate_cell_count"] > 0, "a candidate WAS proposed -- it was refused"


def test_stage2_verdict_is_reported_in_the_diagnostics() -> None:
    """A row must be able to say 'refused' distinctly from 'the detector found nothing'."""

    explorer = StepwiseExplorer(edge_bar_hud_mask=True)
    explorer._ingest(_FakeFrame(_r11l_like_grid()))
    diag = explorer.hud_mask_diagnostics()
    assert diag["hud_mask_stage2_confirm_enabled"] is True
    assert diag["stage2"]["stage2_verdict"] == "pending"
    assert diag["hud_mask_resolved"] is False

    off = StepwiseExplorer(edge_bar_hud_mask=False, hud_mask_stage2_confirm=False)
    assert off.hud_mask_diagnostics()["stage2"] is None


def test_the_detector_cannot_be_enabled_without_both_safety_stages() -> None:
    """REGRESSION for the fatal finding: the flip candidate had no guard (2026-07-25).

    The three flags were INDEPENDENT, and the arm the artifact reported as passing the gate had
    BOTH safety stages off. Flipping that reported-passing configuration would have shipped
    Stage 1's ar25 over-mask with nothing able to refuse it.
    """

    # The runtime default: enabling the detector arms both stages.
    explorer = StepwiseExplorer(edge_bar_hud_mask=True)
    assert explorer.hud_mask_collapse_guard_enabled is True
    assert explorer.hud_mask_stage2_confirm_enabled is True
    assert explorer.hud_mask_safety_stages_explicitly_disabled == []

    # An EXPERIMENT can still isolate a stage, but it is recorded as such so the gate can refuse
    # to treat that arm as flip-eligible.
    isolation = StepwiseExplorer(
        edge_bar_hud_mask=True, hud_mask_collapse_guard=False, hud_mask_stage2_confirm=False
    )
    assert isolation.hud_mask_collapse_guard_enabled is False
    assert isolation.hud_mask_safety_stages_explicitly_disabled == [
        "collapse_guard",
        "stage2_confirm",
    ]

    # And the SUBMITTED-flag combination that would ship the detector bare is refused at import.
    assert _assert_hud_flag_coupling() is None  # the SHIPPED (all-three-on) configuration is legal
    import carnot.agentic.arc_competition_agent as agent_module

    # UPDATED 2026-07-25: all three flags now ship True, so setting the DETECTOR true is the legal
    # shipped configuration and raises nothing. The property under test is unchanged -- the detector
    # must never ship without BOTH safety stages -- so provoke it by turning a SAFETY stage off.
    original_guard = agent_module.SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED
    original_stage2 = agent_module.SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED
    try:
        agent_module.SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED = False
        with pytest.raises(AssertionError, match="requires BOTH"):
            agent_module._assert_hud_flag_coupling()
        agent_module.SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED = True
        agent_module.SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED = False
        with pytest.raises(AssertionError, match="requires BOTH"):
            agent_module._assert_hud_flag_coupling()
    finally:
        agent_module.SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED = original_guard
        agent_module.SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED = original_stage2


def test_guard_attributes_a_branching_to_the_repair_added_region() -> None:
    """WHOSE mask is aliasing -- answered from the guard's own antecedents, not from lost wins.

    The harness's previous attribution iterated only games where a guard-armed arm LOST a win, so
    ar25's proven collapses (an unrun game, and one where no control-held win was lost) sat
    entirely outside its window while its empty output read as "the repair is clean".
    """

    shipped = np.zeros((8, 8), dtype=bool)
    shipped[0, :] = True  # the already-shipped horizontal bar
    applied = shipped.copy()
    applied[:, 7] = True  # the repair-added vertical bar

    guard = MaskCollapseGuard(applied_mask=applied, shipped_mask=shipped)
    first = np.zeros((8, 8), dtype=int)
    first[:, 7] = 4
    second = first.copy()
    second[3:, 7] = 9  # the two antecedents differ ONLY inside the repair-added column

    common = dict(action_key=(6, None))
    guard.observe(
        origin_masked="M",
        origin_unmasked="U1",
        successor_masked="S1",
        successor_unmasked="SU1",
        origin_grid=first,
        **common,
    )
    guard.observe(
        origin_masked="M",
        origin_unmasked="U2",
        successor_masked="S2",
        successor_unmasked="SU2",
        origin_grid=second,
        **common,
    )
    attribution = guard.diagnostics()["attribution"]
    assert attribution["branchings_differing_in_repair_added_region"] == 1
    assert attribution["branchings_differing_in_already_shipped_region"] == 0
    assert attribution["attribution_unavailable"] == 0
    assert attribution["regions_supplied"] is True


def test_guard_attributes_shipped_outside_missing_and_shape_mismatch_cases() -> None:
    """Every attribution bucket must be reachable and distinguishable in diagnostics."""

    def branch(guard: MaskCollapseGuard, first=None, second=None) -> dict:
        common = dict(action_key=(6, None))
        guard.observe(
            origin_masked="M",
            origin_unmasked="U1",
            successor_masked="S1",
            successor_unmasked="SU1",
            origin_grid=first,
            **common,
        )
        guard.observe(
            origin_masked="M",
            origin_unmasked="U2",
            successor_masked="S2",
            successor_unmasked="SU2",
            origin_grid=second,
            **common,
        )
        return guard.diagnostics()["attribution"]

    shipped = np.zeros((8, 8), dtype=bool)
    shipped[0, :] = True
    first = np.zeros((8, 8), dtype=int)

    shipped_changed = first.copy()
    shipped_changed[0, 3] = 9
    shipped_attr = branch(
        MaskCollapseGuard(applied_mask=shipped, shipped_mask=shipped), first, shipped_changed
    )
    assert shipped_attr["branchings_differing_in_already_shipped_region"] == 1

    applied = np.zeros((8, 8), dtype=bool)
    applied[:, 7] = True
    outside_changed = first.copy()
    outside_changed[4, 4] = 9
    outside_attr = branch(MaskCollapseGuard(applied_mask=applied), first, outside_changed)
    assert outside_attr["branchings_differing_outside_the_mask"] == 1

    missing_attr = branch(MaskCollapseGuard(applied_mask=applied), None, None)
    assert missing_attr["attribution_unavailable"] == 1

    wrong_shape = np.zeros((4, 4), dtype=bool)
    shape_attr = branch(MaskCollapseGuard(applied_mask=wrong_shape), first, outside_changed)
    assert shape_attr["attribution_unavailable"] == 1


def test_guard_counts_missing_attribution_rather_than_reporting_a_clean_zero() -> None:
    """An absent region/grid must never read as 'no added-region aliasing'."""

    guard = MaskCollapseGuard()  # no regions supplied
    common = dict(action_key=(6, None))
    for successor, unmasked in (("S1", "U1"), ("S2", "U2")):
        guard.observe(
            origin_masked="M",
            origin_unmasked=unmasked,
            successor_masked=successor,
            successor_unmasked="X" + successor,
            **common,
        )
    attribution = guard.diagnostics()["attribution"]
    assert attribution["regions_supplied"] is False
    assert attribution["attribution_unavailable"] == 1
    assert attribution["branchings_differing_in_repair_added_region"] == 0


def _proven_pair_with_grids(guard, node, first, second, action_key=(6, None)):
    """Drive a PROVEN collapse whose two antecedent GRIDS are supplied, so it is attributable."""

    guard.observe(
        origin_masked=node,
        origin_unmasked=f"{node}_Ub",
        action_key=action_key,
        successor_masked=f"{node}_S2",
        successor_unmasked=f"{node}_SUb",
        origin_grid=first,
    )
    guard.observe(
        origin_masked=node,
        origin_unmasked=f"{node}_Ua",
        action_key=action_key,
        successor_masked=f"{node}_S1",
        successor_unmasked=f"{node}_SUa",
        origin_grid=second,
    )
    return guard.observe(
        origin_masked=node,
        origin_unmasked=f"{node}_Ub",
        action_key=action_key,
        successor_masked=f"{node}_S2",
        successor_unmasked=f"{node}_SUb",
        origin_grid=second,
    )


def test_guard_does_not_retract_a_cell_the_live_configuration_already_masks() -> None:
    """REGRESSION for the measured over-reach of the guard onto the SHIPPED mask (2026-07-25).

    Restricting the guard to PROVEN collapses was not enough. On the full corpus the ALREADY-
    SHIPPED mask aliases heavily on its own -- 441 proven collapses with the guard armed -- and
    splitting those nodes cost the run dc22, su15, tu93 and lf52: wins the LIVE configuration
    holds BECAUSE the shipped mask collapses those states. Acting on them would mean this
    experiment regresses the baseline to fix a defect it did not introduce.

    So: a proven branching whose differing antecedent cells lie in the SHIPPED region is
    COUNTED and ATTRIBUTED but NOT acted on; one in the REPAIR-ADDED region is acted on.
    """

    shipped = np.zeros((8, 8), dtype=bool)
    shipped[0, :] = True
    applied = shipped.copy()
    applied[:, 7] = True  # the repair-added column

    # (1) The differing cells are in the ALREADY-SHIPPED row.
    base = np.zeros((8, 8), dtype=int)
    moved_shipped = base.copy()
    moved_shipped[0, 2:5] = 6
    shipped_guard = MaskCollapseGuard(applied_mask=applied, shipped_mask=shipped)
    acted = _proven_pair_with_grids(shipped_guard, "M", base, moved_shipped)
    assert acted is False, "the baseline's own mask is out of scope for this repair"
    assert shipped_guard.is_split("M") is False
    diagnostics = shipped_guard.diagnostics()
    assert diagnostics["proven_collapses"] >= 1, "still PROVEN -- just not acted on"
    assert diagnostics["branchings_in_shipped_region_not_acted_on"] >= 1
    assert diagnostics["attribution"]["branchings_differing_in_already_shipped_region"] >= 1
    assert diagnostics["restricted_action_to_repair_added_region"] is True

    # (2) The differing cells are in the REPAIR-ADDED column -- this IS in scope.
    moved_added = base.copy()
    moved_added[3:6, 7] = 9
    added_guard = MaskCollapseGuard(applied_mask=applied, shipped_mask=shipped)
    assert _proven_pair_with_grids(added_guard, "N", base, moved_added) is True
    assert added_guard.is_split("N") is True
    assert added_guard.diagnostics()["branchings_in_shipped_region_not_acted_on"] == 0


def test_guard_acts_on_an_unattributable_branching_when_nothing_was_shipped() -> None:
    """With no shipped mask every masked cell is repair-added, so the restriction cannot bite.

    This is the r11l/tn36 shape -- the shipped classifier resolves nothing there -- and it is the
    case where the guard has to remain effective, or the repair would ship unguarded on exactly
    the games it changes.
    """

    applied = np.zeros((8, 8), dtype=bool)
    applied[:, 0] = True
    guard = MaskCollapseGuard(applied_mask=applied, shipped_mask=None)
    # No `origin_grid` supplied at all -> unattributable, but nothing was shipped.
    assert _prove_a_collapse(guard, "M") is True
    assert guard.is_split("M") is True
    assert guard.diagnostics()["branchings_in_shipped_region_not_acted_on"] == 0


def test_guard_restriction_can_be_disabled_for_measurement() -> None:
    """The aggressive variant stays reachable so the measurement above is reproducible."""

    shipped = np.zeros((8, 8), dtype=bool)
    shipped[0, :] = True
    applied = shipped.copy()
    applied[:, 7] = True
    base = np.zeros((8, 8), dtype=int)
    moved = base.copy()
    moved[0, 2:5] = 6
    guard = MaskCollapseGuard(
        applied_mask=applied,
        shipped_mask=shipped,
        restrict_action_to_repair_added_region=False,
    )
    assert _proven_pair_with_grids(guard, "M", base, moved) is True
    assert guard.is_split("M") is True
    assert guard.diagnostics()["restricted_action_to_repair_added_region"] is False
