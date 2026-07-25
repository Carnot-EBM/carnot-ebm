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
    StepwiseExplorer,
    _compute_hud_mask_from_frame,
)
from carnot.agentic.arc_hud_bar_detector import (
    EdgeBarThresholds,
    MaskCollapseGuard,
    edge_bar_hud_mask,
    is_edge_bar_like,
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


# ---------------------------------------------------------------------------
# Stage 3 -- the runtime collapse guard
# ---------------------------------------------------------------------------


def test_collapse_guard_refuses_a_mask_that_aliases_distinct_states() -> None:
    """SCENARIO-ARC-WMTE-5960-COLLAPSE-GUARD-HARD-REFUSAL.

    Same masked node + same concrete action -> two DIFFERENT masked successors, while the
    unmasked control branches only once. That is a causal proof of collapse.
    """

    guard = MaskCollapseGuard()
    common = dict(action_key=(6, {"x": 3, "y": 4}))

    # THE SHAPE OF A REAL COLLAPSE, and it matters: the two observations share a MASKED origin
    # (that is the collapse) but have DIFFERENT unmasked origins (they are genuinely different
    # true states). So each unmasked key has exactly ONE successor and the control does not
    # veto. Passing the same `origin_unmasked` twice would instead describe an environment that
    # branches from one identical raw frame -- non-determinism, not a mask fault -- which the
    # guard correctly declines to blame on the mask (see the next test).
    first = guard.observe(
        origin_masked="M1", origin_unmasked="U1a", successor_masked="S1",
        successor_unmasked="SU1", **common,
    )
    assert first is False, "one observation can never prove anything"
    assert guard.refusals == 0

    second = guard.observe(
        origin_masked="M1", origin_unmasked="U1b", successor_masked="S2",
        successor_unmasked="SU2", **common,
    )
    assert second is True
    assert guard.refusals == 1
    assert guard.violations == 1
    assert guard.is_split("M1") is True
    assert guard.is_split("M2") is False
    assert guard.observable_key_count() == 1
    assert guard.globally_revoked is False


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
            origin_masked="M1", origin_unmasked=None, action_key=(6, None),
            successor_masked=successor, successor_unmasked=None,
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
        origin_masked="M", origin_unmasked=None, action_key=1,
        successor_masked="S", successor_unmasked=None,
    )
    assert dead.diagnostics()["control_live"] is False

    live = MaskCollapseGuard()
    live.observe(
        origin_masked="M", origin_unmasked="U", action_key=1,
        successor_masked="S", successor_unmasked="SU",
    )
    assert live.diagnostics()["control_live"] is True
    assert live.diagnostics()["controlled_keys"] == 1


def test_explorer_supplies_a_live_control_to_the_guard() -> None:
    """The wiring half of the same regression: the antecedent must not come from the graph node.

    Node frames are only retained when certain optional components are enabled, so the guard's
    control has to be sourced from the previously OBSERVED frame instead. Two ingests are enough
    to prove the channel is live.
    """

    explorer = StepwiseExplorer(edge_bar_hud_mask=True, hud_mask_collapse_guard=True)
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


def test_collapse_guard_escalates_to_global_revocation_past_the_cap() -> None:
    """Local splits are bounded; enough of them means the mask itself is wrong."""

    guard = MaskCollapseGuard(max_split_nodes=2)
    for index in range(3):
        node = f"M{index}"
        guard.observe(
            origin_masked=node, origin_unmasked=f"U{index}a", action_key=(6, None),
            successor_masked="A", successor_unmasked="UA",
        )
        guard.observe(
            origin_masked=node, origin_unmasked=f"U{index}b", action_key=(6, None),
            successor_masked="B", successor_unmasked="UB",
        )
    assert guard.refusals == 3
    assert guard.globally_revoked is True
    assert guard.is_split("never_seen_node") is True, "revoked means EVERY node is un-masked"
    assert guard.diagnostics()["globally_revoked"] is True


def test_collapse_guard_ignores_incomplete_observations() -> None:
    guard = MaskCollapseGuard()
    assert guard.observe(
        origin_masked=None, origin_unmasked="U", action_key=1,
        successor_masked="S", successor_unmasked="SU",
    ) is False
    assert guard.observe(
        origin_masked="M", origin_unmasked="U", action_key=1,
        successor_masked=None, successor_unmasked="SU",
    ) is False
    assert guard.observations == 0


def test_collapse_guard_handles_unhashable_action_payloads() -> None:
    """Click payloads arrive as dicts; a guard that raises on one would take the search down."""

    guard = MaskCollapseGuard()
    payload = {"x": 1, "y": 2, "nested": [1, 2, {"z": 3}]}
    for index, successor in enumerate(("S1", "S2")):
        guard.observe(
            origin_masked="M", origin_unmasked=f"U{index}", action_key=(6, payload),
            successor_masked=successor, successor_unmasked="U" + successor,
        )
    assert guard.violations == 1


def test_mask_summary_describes_a_mask_and_a_none() -> None:
    assert mask_summary(None) == {"resolved": False, "cell_count": 0, "rows": [], "cols": []}
    summary = mask_summary(_column0_mask((8, 8)))
    assert summary["cell_count"] == 8
    assert summary["cols"] == [0]
    assert summary["rows"] == list(range(8))


# ---------------------------------------------------------------------------
# Wiring -- default-off parity, and the flag actually taking effect
# ---------------------------------------------------------------------------


def test_defaults_are_off_and_published_in_the_submitted_config() -> None:
    """SCENARIO-ARC-WMTE-5960-DEFAULT-OFF-PARITY (part 1)."""

    assert SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED is False
    assert SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED is False
    assert SUBMITTED_AGENT_CONFIG["edge_bar_hud_mask_enabled"] is False
    assert SUBMITTED_AGENT_CONFIG["hud_mask_collapse_guard_enabled"] is False

    explorer = StepwiseExplorer()
    assert explorer.edge_bar_hud_mask_enabled is False
    assert explorer.hud_mask_collapse_guard_enabled is False
    assert explorer._hud_collapse_guard is None


def test_default_off_leaves_node_identity_byte_identical() -> None:
    """SCENARIO-ARC-WMTE-5960-DEFAULT-OFF-PARITY (part 2).

    The r11l-shaped frame is the sharpest possible case: with the repair ON its column-0
    counter leaves node identity, with the repair OFF it must NOT -- so two frames differing
    only in the counter must still hash DIFFERENTLY by default.
    """

    frame_a = _FakeFrame(_r11l_like_grid(filled=0))
    frame_b = _FakeFrame(_r11l_like_grid(filled=7))

    control = StepwiseExplorer()
    control._ingest(frame_a)
    control._ingest(frame_b)
    assert control.hud_mask is None
    assert control._hash(frame_a) != control._hash(frame_b), "default must NOT dedup these"

    treatment = StepwiseExplorer(edge_bar_hud_mask=True)
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

    treatment = StepwiseExplorer(edge_bar_hud_mask=True)
    treatment._ingest(_FakeFrame(_r11l_like_grid()))
    assert treatment._hud_mask_source == "edge_bar_detector_req5960"

    control = StepwiseExplorer()
    control._ingest(_FakeFrame(_r11l_like_grid()))
    assert control._hud_mask_source == "unresolved_no_bar_detected"

    horizontal = np.full((32, 32), 3, dtype=int)
    horizontal[0, :] = 8
    shipped = StepwiseExplorer()
    shipped._ingest(_FakeFrame(horizontal))
    assert shipped._hud_mask_source == "status_bar_classifier_req5583"


def test_diagnostics_report_dedup_and_guard_activity() -> None:
    """Every arm must emit the mask/dedup/guard fields, so a broken arm cannot read as a null."""

    explorer = StepwiseExplorer(edge_bar_hud_mask=True, hud_mask_collapse_guard=True)
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

    explorer = StepwiseExplorer(edge_bar_hud_mask=True, hud_mask_collapse_guard=True)
    frame_a = _FakeFrame(_r11l_like_grid(filled=0))
    frame_b = _FakeFrame(_r11l_like_grid(filled=7))
    explorer._ingest(frame_a)
    assert explorer._hash(frame_a) == explorer._hash(frame_b)

    masked = explorer._hash(frame_a)
    explorer._hud_collapse_guard.split_hashes.add(masked)

    assert explorer._hash(frame_a) != explorer._hash(frame_b), "split node must un-mask"
    assert explorer._hash(frame_a).startswith(masked + "|u:")
