"""REQ-ARC-WMTE-5950 / SCENARIO-ARC-WMTE-5950-A..H -- per-object click-pixel sampling.

Every test here asserts. There are no skips: the sampler is pure (grid in, coordinates
out), so nothing in it needs a GPU, a network, a game environment, or a model.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from carnot.agentic import arc_component_sampling as cps
from carnot.agentic import arc_frontier_discipline as fd
from carnot.agentic.arc_graph_explore import rich_action_candidates
from carnot.agentic.arc_solver_kit import object_centric_digest


def _digest_signature(grid) -> list[tuple]:
    return sorted(
        (
            int(c["color"]),
            int(c["area"]),
            int(c["bbox"][0]),
            int(c["bbox"][1]),
            int(c["bbox"][2]),
            int(c["bbox"][3]),
        )
        for c in object_centric_digest(grid)["components"]
    )


def _partition_signature(part: cps.ComponentPartition) -> list[tuple]:
    out = []
    for color, cells in zip(part.colors, part.cells):
        ys = [y for _x, y in cells]
        xs = [x for x, _y in cells]
        out.append((int(color), len(cells), min(ys), min(xs), max(ys), max(xs)))
    return sorted(out)


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-A: the partition must be the LIVE partition
# ---------------------------------------------------------------------------


def test_partition_is_component_for_component_identical_to_object_centric_digest():
    """The guard against this becoming a silent SET-MEMBERSHIP change.

    The live click generator's objects come from ``object_centric_digest`` (most-common
    colour excluded wholesale). ``connected_color_blobs`` uses a different suppression rule.
    If this sampler drew from a different partition, an A/B on the coordinate would silently
    also be an A/B on WHICH objects are clickable -- an uninterpretable experiment. So the
    equivalence is asserted, not assumed.
    """

    rng = random.Random(1234)
    for _ in range(40):
        h, w, ncol = rng.randint(2, 14), rng.randint(2, 14), rng.randint(1, 6)
        grid = np.array([[rng.randrange(ncol) for _ in range(w)] for _ in range(h)])
        part = cps.component_partition(grid)
        assert _partition_signature(part) == _digest_signature(grid)


def test_background_tie_break_matches_the_digest_exactly():
    """A TIED most-common colour must resolve the same way in both implementations.

    ``object_centric_digest`` uses ``np.unique(...)[argmax(counts)]``, and np.unique is
    sorted ascending, so a tie resolves to the SMALLEST colour value. ``Counter.most_common``
    would resolve it by insertion order instead -- a real divergence this asserts against.
    """

    grid = np.array([[3, 3, 1, 1], [3, 3, 1, 1]])  # colours 1 and 3 both appear 4 times
    part = cps.component_partition(grid)
    assert part.background_color == object_centric_digest(grid)["background_color"] == 1
    assert _partition_signature(part) == _digest_signature(grid)


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-B: uniform WITHIN the object, and with replacement
# ---------------------------------------------------------------------------


def test_sampling_is_uniform_over_member_cells_and_never_leaves_the_object():
    grid = np.zeros((6, 6), dtype=int)
    grid[1:4, 1:4] = 7  # a 3x3 block of a salient colour; background stays 0
    part = cps.component_partition(grid)
    assert len(part) == 1
    cells = set(part.cells[0])
    assert len(cells) == 9

    rng = random.Random(0)
    draws = [part.sample_pixel(0, rng) for _ in range(2000)]
    # (a) every draw is a real member pixel -- the whole point of the mechanism
    assert set(draws) <= cells
    # (b) every member pixel is reachable: the object is not collapsed to a subset
    assert set(draws) == cells
    # (c) roughly uniform. 2000 draws over 9 cells => ~222 each; a 2x/0.5x band is a
    # generous bound that a non-uniform (e.g. centroid-biased) sampler would still fail.
    counts = {c: draws.count(c) for c in cells}
    assert min(counts.values()) > 111, counts
    assert max(counts.values()) < 444, counts


def test_draws_are_with_replacement_the_sampler_keeps_no_memory():
    """The reference redraws on every revisit; it records objects, never pixels.

    So two independent calls must be free to return the same pixel, and a long run of
    calls must not exhaust the object.
    """

    grid = np.zeros((5, 5), dtype=int)
    grid[0:2, 0:2] = 4
    part = cps.component_partition(grid)
    rng = random.Random(7)
    draws = [part.sample_pixel(0, rng) for _ in range(200)]
    assert len(draws) == 200  # never exhausted
    assert len(set(draws)) == 4  # all four cells seen
    # WITH replacement: at least one coordinate repeats (probability of no repeat over
    # 200 draws from 4 cells is astronomically small, so this is a real assertion).
    assert len(set(draws)) < len(draws)


def test_sample_pixels_dedupes_but_never_pads_beyond_the_objects_own_cells():
    grid = np.zeros((4, 4), dtype=int)
    grid[0, 0] = 9
    grid[0, 1] = 9  # a 2-cell object
    part = cps.component_partition(grid)
    got = part.sample_pixels(0, random.Random(3), k=8)
    assert len(got) == 2  # only 2 distinct clicks exist; no invented coordinates
    assert set(got) == set(part.cells[0])


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-C: the centroid defect this exists to fix
# ---------------------------------------------------------------------------


def test_a_hollow_objects_centroid_is_not_its_own_pixel_and_sampling_fixes_it():
    """The measured defect: 204 of 204 real r11l states had >=1 such object.

    A ring's centroid is its hole, which is BACKGROUND. So "click that object" today
    clicks something else entirely. This asserts the sampler resolves the centroid back to
    the object that produced it and returns a genuine member pixel.
    """

    grid = np.zeros((7, 7), dtype=int)
    grid[1:6, 1:6] = 8
    grid[2:5, 2:5] = 0  # hollow it out -> a ring of colour 8 around background
    part = cps.component_partition(grid)
    ring = next(i for i, c in enumerate(part.colors) if c == 8)
    centroid_key = next(k for k, v in part.index_by_centroid.items() if v == ring)
    # the defect itself
    assert centroid_key not in part.cells[ring]

    sampled, diag = cps.sample_component_click_points(grid, [centroid_key], rng=random.Random(0))
    assert len(sampled) == 1
    assert sampled[0] in part.cells[ring]  # a real member pixel now
    assert diag.resolved_via_centroid == 1
    assert diag.centroid_outside_own_component == 1
    assert diag.unresolved == 0


def test_centroid_is_resolved_by_provenance_not_by_containment():
    """Resolution order matters: centroid FIRST, containment second.

    A centroid that lands inside a DIFFERENT object must still resolve to the object it
    came from, otherwise the mechanism would quietly retarget the click.
    """

    grid = np.zeros((7, 7), dtype=int)
    grid[1:6, 1:6] = 8
    grid[2:5, 2:5] = 0
    grid[3, 3] = 5  # a foreign 1-cell object sitting exactly on the ring's centroid
    part = cps.component_partition(grid)
    ring = next(i for i, c in enumerate(part.colors) if c == 8)
    assert part.index_by_centroid[(3, 3)] == ring
    assert part.index_by_cell[(3, 3)] != ring  # containment would say the colour-5 dot
    assert part.index_for_point(3, 3) == ring  # provenance wins


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-D: degenerate inputs fail OPEN, never raise, never drop
# ---------------------------------------------------------------------------


def test_single_pixel_object_always_returns_that_pixel():
    grid = np.zeros((3, 3), dtype=int)
    grid[1, 2] = 6
    part = cps.component_partition(grid)
    rng = random.Random(0)
    assert {part.sample_pixel(0, rng) for _ in range(25)} == {(2, 1)}


def test_uniform_frame_has_no_objects_and_points_pass_through_unchanged():
    """A single-colour frame is ALL background under the digest rule -> zero objects."""

    grid = np.full((5, 5), 3, dtype=int)
    part = cps.component_partition(grid)
    assert len(part) == 0
    sampled, diag = cps.sample_component_click_points(grid, [(2, 2), (0, 0)], rng=random.Random(0))
    assert sampled == [(2, 2), (0, 0)]  # unchanged, not dropped
    assert diag.unresolved == 2
    assert diag.points_out == 2


def test_out_of_bounds_and_negative_coordinates_are_left_alone():
    grid = np.zeros((4, 4), dtype=int)
    grid[0, 0] = 2
    sampled, diag = cps.sample_component_click_points(
        grid, [(99, 99), (-3, 1)], rng=random.Random(0)
    )
    assert sampled == [(99, 99), (-3, 1)]
    assert diag.unresolved == 2


def test_an_unsegmentable_grid_degrades_to_todays_coordinates_and_counts_the_error():
    sampled, diag = cps.sample_component_click_points(
        "not a grid at all", [(1, 2)], rng=random.Random(0)
    )
    assert sampled == [(1, 2)]
    assert diag.errors == 1
    assert diag.unresolved == 1


def test_redraw_declines_when_the_coordinate_cannot_be_attributed():
    grid = np.zeros((4, 4), dtype=int)
    grid[0, 0] = 2
    index, point = cps.redraw_component_pixel(grid, 3, 3, rng=random.Random(0))
    assert index is None and point is None
    index, point = cps.redraw_component_pixel("garbage", 0, 0, rng=random.Random(0))
    assert index is None and point is None


def test_empty_grid_raises_a_clear_error_rather_than_silently_returning_nothing():
    with pytest.raises(ValueError):
        cps.component_partition(np.zeros((2, 2, 2, 2)))


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-E: order is preserved; k>1 expands in place
# ---------------------------------------------------------------------------


def test_order_is_preserved_and_k_expands_each_object_in_place():
    grid = np.zeros((9, 9), dtype=int)
    grid[0:2, 0:2] = 7  # object A
    grid[6:8, 6:8] = 9  # object B
    part = cps.component_partition(grid)
    keys = [k for k, _ in sorted(part.index_by_centroid.items(), key=lambda kv: kv[1])]

    sampled, diag = cps.sample_component_click_points(
        grid, keys, rng=random.Random(0), samples_per_component=3
    )
    assert diag.points_in == 2
    assert len(sampled) == 6  # 2 objects x 3 draws, expanded in place
    # first three belong to object 0, last three to object 1 -> order preserved
    assert all(p in part.cells[0] for p in sampled[:3])
    assert all(p in part.cells[1] for p in sampled[3:])


def test_diagnostics_as_dict_is_json_safe_ints():
    _sampled, diag = cps.sample_component_click_points(
        np.zeros((3, 3), dtype=int), [(0, 0)], rng=random.Random(0)
    )
    payload = diag.as_dict()
    # SUPERSET, not equality: an exact-set assertion here would fail every time a new counter
    # is added, which is a pressure AGAINST instrumenting the mechanism -- the opposite of what
    # this test exists to protect. What matters is that the load-bearing counters are present
    # and JSON-safe.
    assert {
        "points_in",
        "points_out",
        "resolved_via_centroid",
        "resolved_via_cell",
        "unresolved",
        "centroid_outside_own_component",
        "errors",
        "coordinates_changed",
        "contested_centroid_points",
    } <= set(payload)
    assert all(isinstance(v, int) for v in payload.values())


def test_centroid_outside_component_rate_measures_the_motivating_defect():
    hollow = np.zeros((7, 7), dtype=int)
    hollow[1:6, 1:6] = 8
    hollow[2:5, 2:5] = 0
    solid = np.zeros((7, 7), dtype=int)
    solid[1:4, 1:4] = 8
    stats = cps.centroid_outside_component_rate([hollow, solid])
    assert stats["n_grids"] == 2.0
    assert stats["n_centroid_outside"] == 1.0
    assert stats["frac_grids_with_any_outside"] == 0.5


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-F: the tier-map co-change (the barrier must not be nulled)
# ---------------------------------------------------------------------------


def _ring_grid():
    grid = np.zeros((9, 9), dtype=int)
    grid[1:8, 1:8] = 3  # non-salient colour {0..5}, medium bbox -> tier 1
    grid[2:7, 2:7] = 0
    return grid


def test_sampled_pixels_miss_the_centroid_only_tier_map_and_hit_the_per_cell_map():
    """This IS the measured co-change: 100% map-miss on sampled pixels without it.

    A miss reads back as DEFAULT_TIER=0 = always eligible, so the tier barrier silently
    becomes a no-op on click games -- the games it was flipped ON for.
    """

    grid = _ring_grid()
    part = cps.component_partition(grid)
    ring = next(i for i, c in enumerate(part.colors) if c == 3)
    pixel = part.sample_pixel(ring, random.Random(0))
    row = {"action": 6, "data": {"x": pixel[0], "y": pixel[1]}}

    centroid_only = fd.click_tier_map(grid)
    per_cell = fd.click_tier_map(grid, include_cells=True)

    assert fd.row_tier(row, centroid_only) == fd.DEFAULT_TIER == 0  # the silent nulling
    assert fd.row_tier(row, per_cell) == 1  # the real tier, recovered
    # the centroid-only map is a strict subset -> include_cells is purely additive
    assert set(centroid_only) <= set(per_cell)
    for key, tier in centroid_only.items():
        assert per_cell[key] <= tier  # fail-open direction only


def test_include_cells_default_off_leaves_the_tier_map_byte_identical():
    grid = _ring_grid()
    assert fd.click_tier_map(grid) == fd.click_tier_map(grid, include_cells=False)
    assert fd.tier_map_for_frame(grid) == fd.click_tier_map(grid)


def test_sampling_within_an_object_is_tier_invariant():
    """The redraw must not be able to change a row's tier.

    Guaranteed by construction (same object -> same colour and bbox -> same predicate),
    and asserted here because the whole barrier depends on it.
    """

    grid = _ring_grid()
    part = cps.component_partition(grid)
    ring = next(i for i, c in enumerate(part.colors) if c == 3)
    per_cell = fd.click_tier_map(grid, include_cells=True)
    tiers = {
        fd.row_tier({"action": 6, "data": {"x": x, "y": y}}, per_cell)
        for (x, y) in part.cells[ring]
    }
    assert tiers == {1}


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-G: DEFAULT-OFF live parity
# ---------------------------------------------------------------------------


class _FakeFrame:
    """Minimal live-frame stand-in: ``available_actions`` + a nested-list ``frame``."""

    def __init__(self, grid, actions=(6,)):
        self.frame = [np.asarray(grid).tolist()]
        self.available_actions = list(actions)


def _rows(cands):
    return [(int(c.action_id), None if c.data is None else dict(c.data)) for c in cands]


def test_default_off_leaves_live_click_generation_byte_identical():
    grid = np.zeros((12, 12), dtype=int)
    grid[1:4, 1:4] = 7
    grid[8:11, 6:9] = 9
    grid[5, 5] = 2
    frame = _FakeFrame(grid)

    baseline = _rows(rich_action_candidates(frame))
    explicit_off = _rows(rich_action_candidates(frame, click_pixel_sampling=False))
    assert baseline == explicit_off
    # and the coordinates really are the truncated centroids (i.e. the baseline is the
    # behaviour this experiment means to vary, not something already sampled)
    part = cps.component_partition(grid)
    clicks = {(r[1]["x"], r[1]["y"]) for r in baseline if r[0] == 6}
    assert clicks <= set(part.index_by_centroid)


def test_flag_on_moves_every_click_onto_a_real_member_pixel():
    grid = np.zeros((12, 12), dtype=int)
    grid[1:8, 1:8] = 3
    grid[2:7, 2:7] = 0  # a hollow object: its centroid is background
    frame = _FakeFrame(grid)
    part = cps.component_partition(grid)
    all_cells = set().union(*[set(c) for c in part.cells])

    off = _rows(rich_action_candidates(frame))
    on = _rows(
        rich_action_candidates(frame, click_pixel_sampling=True, click_pixel_rng=random.Random(0))
    )
    assert off != on  # the mechanism actually did something
    assert len(on) == len(off)  # k=1 -> one row per object, same budget
    for action, data in on:
        if action == 6:
            assert (data["x"], data["y"]) in all_cells
    # keyboard rows (if any) are untouched
    assert [r for r in on if r[0] != 6] == [r for r in off if r[0] != 6]


def test_k_greater_than_one_expands_rows_without_shrinking_object_coverage():
    """Guards the confound lane 2 flagged: capping by POINT would divide objects by k."""

    grid = np.zeros((20, 20), dtype=int)
    # five 1-pixel objects (each has exactly ONE distinct click), all clear of the block
    single_coords = [(0, 0), (4, 4), (8, 8), (16, 16), (18, 18)]
    for i, (sx, sy) in enumerate(single_coords):
        grid[sy, sx] = 7 + i
    grid[10:14, 10:14] = 4  # one 16-pixel object
    frame = _FakeFrame(grid)

    k1 = _rows(
        rich_action_candidates(frame, click_pixel_sampling=True, click_pixel_rng=random.Random(0))
    )
    k3 = _rows(
        rich_action_candidates(
            frame,
            click_pixel_sampling=True,
            click_pixel_samples_per_component=3,
            click_pixel_rng=random.Random(0),
        )
    )
    # More rows overall (the multi-pixel object contributes 3), and coverage of the
    # 1-pixel objects is not lost to de-duplication or to a point-based cap.
    assert len(k3) > len(k1)
    singles = set(single_coords)
    got = {(d["x"], d["y"]) for a, d in k3 if a == 6}
    assert singles <= got


def test_the_fallback_rng_advances_instead_of_restarting_each_call():
    """A caller that enables the flag via env (no rng) must still get VARYING pixels.

    A fresh ``Random(0)`` per call would restart the same stream every time, silently
    turning a with-replacement sampler back into a fixed-point one -- the exact defect the
    mechanism exists to remove.
    """

    from carnot.agentic import arc_graph_explore as age

    grid = np.zeros((12, 12), dtype=int)
    grid[1:8, 1:8] = 8  # one large object -> many distinct pixels to draw from
    frame = _FakeFrame(grid)
    seen = set()
    for _ in range(12):
        rows = _rows(rich_action_candidates(frame, click_pixel_sampling=True))
        seen |= {(d["x"], d["y"]) for a, d in rows if a == 6}
    assert len(seen) > 1, seen
    # and it IS a seeded module-level stream, not an unseeded one
    assert isinstance(age._CLICK_PIXEL_FALLBACK_RNG, random.Random)


def test_sampling_is_reproducible_from_its_own_seed():
    grid = np.zeros((10, 10), dtype=int)
    grid[1:6, 1:6] = 8
    frame = _FakeFrame(grid)
    a = _rows(
        rich_action_candidates(frame, click_pixel_sampling=True, click_pixel_rng=random.Random(42))
    )
    b = _rows(
        rich_action_candidates(frame, click_pixel_sampling=True, click_pixel_rng=random.Random(42))
    )
    c = _rows(
        rich_action_candidates(frame, click_pixel_sampling=True, click_pixel_rng=random.Random(43))
    )
    assert a == b  # same seed -> same coordinates (the reference is NOT reproducible; we are)
    assert a != c  # different seed -> different coordinates


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-H: the explorer's bounded WITH-REPLACEMENT change
# ---------------------------------------------------------------------------


def _explorer(**kwargs):
    from carnot.agentic.arc_competition_agent import StepwiseExplorer

    return StepwiseExplorer(**kwargs)


def test_pop_appends_a_bounded_redraw_for_the_same_object_when_the_flag_is_on():
    grid = np.zeros((10, 10), dtype=int)
    grid[1:6, 1:6] = 8
    frame = _FakeFrame(grid)
    part = cps.component_partition(grid)
    cells = set(part.cells[0])
    start = sorted(cells)[0]

    ex = _explorer(click_pixel_sampling=True, click_pixel_redraw_budget=3)
    node = {
        "untested": [{"action": 6, "data": {"x": start[0], "y": start[1]}, "tier": 1}],
        "frame": frame,
    }
    popped = ex._pop_untested(node)
    assert popped["data"] == {"x": start[0], "y": start[1]}
    # one redraw was appended, for the SAME object, at a real member pixel
    assert len(node["untested"]) == 1
    fresh = node["untested"][0]
    assert (fresh["data"]["x"], fresh["data"]["y"]) in cells
    assert fresh["tier"] == 1  # tier carried over -- invariant within an object
    assert ex.frontier_discipline_diagnostics()["click_pixel_redraws"] == 1

    # budget=3 => at most 3 draws total per (node, object): the 3rd pop appends nothing.
    ex._pop_untested(node)
    assert len(node["untested"]) == 1
    ex._pop_untested(node)
    assert node["untested"] == []
    diag = ex.frontier_discipline_diagnostics()
    assert diag["click_pixel_redraws"] == 2
    assert diag["click_pixel_redraws_declined_budget"] == 1


def test_redraw_budget_of_one_is_pure_one_shot_no_reappend():
    grid = np.zeros((8, 8), dtype=int)
    grid[1:5, 1:5] = 8
    frame = _FakeFrame(grid)
    ex = _explorer(click_pixel_sampling=True, click_pixel_redraw_budget=1)
    node = {"untested": [{"action": 6, "data": {"x": 1, "y": 1}}], "frame": frame}
    ex._pop_untested(node)
    assert node["untested"] == []
    assert ex.frontier_discipline_diagnostics()["click_pixel_redraws"] == 0


def test_flag_off_pop_is_byte_identical_and_never_appends():
    grid = np.zeros((8, 8), dtype=int)
    grid[1:5, 1:5] = 8
    frame = _FakeFrame(grid)
    ex = _explorer(click_pixel_sampling=False)
    node = {"untested": [{"action": 6, "data": {"x": 1, "y": 1}}], "frame": frame}
    popped = ex._pop_untested(node)
    assert popped == {"action": 6, "data": {"x": 1, "y": 1}}
    assert node["untested"] == []
    diag = ex.frontier_discipline_diagnostics()
    assert diag["click_pixel_sampling_enabled"] is False
    assert diag["click_pixel_redraws"] == 0
    assert diag["click_pixel_rows_sampled"] == 0


def test_keyboard_rows_and_frameless_nodes_decline_the_redraw_and_are_counted():
    ex = _explorer(click_pixel_sampling=True, click_pixel_redraw_budget=3)
    node = {"untested": [{"action": 2, "data": None}], "frame": None}
    ex._pop_untested(node)
    assert node["untested"] == []
    assert ex.frontier_discipline_diagnostics()["click_pixel_redraws_declined_no_frame"] == 0

    node2 = {"untested": [{"action": 6, "data": {"x": 0, "y": 0}}], "frame": None}
    ex._pop_untested(node2)
    assert ex.frontier_discipline_diagnostics()["click_pixel_redraws_declined_no_frame"] == 1


def test_explorer_rng_is_a_separate_stream_from_the_tier_draw_rng():
    """Coupling the two would make an A/B unable to attribute a delta to either arm."""

    ex = _explorer(click_pixel_sampling=True, frontier_discipline_seed=99)
    assert ex._cps_rng is not ex._fd_rng
    before = ex._fd_rng.getstate()
    ex._cps_rng.random()
    assert ex._fd_rng.getstate() == before  # drawing a pixel does not perturb tier draws


def test_env_override_and_kwarg_precedence_match_the_frontier_discipline_pattern(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_CLICK_PIXEL_SAMPLING", "1")
    assert _explorer().click_pixel_sampling_enabled is True
    assert _explorer(click_pixel_sampling=False).click_pixel_sampling_enabled is False
    monkeypatch.setenv("CARNOT_ARC_CLICK_PIXEL_SAMPLING", "0")
    assert _explorer().click_pixel_sampling_enabled is False


def test_submitted_default_is_off_so_the_shipped_agent_is_unchanged():
    from carnot.agentic import arc_competition_agent as aca

    assert aca.SUBMITTED_CLICK_PIXEL_SAMPLING_ENABLED is False
    assert aca.SUBMITTED_AGENT_CONFIG["click_pixel_sampling"] is False
    assert _explorer().click_pixel_sampling_enabled is False


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-I: contested centroids must not starve an object
# (regression, 2026-07-25 adversarial review -- the sampler previously converted
#  a harmless duplicate-row dedup into a LOSS OF OBJECT REACHABILITY)
# ---------------------------------------------------------------------------


def _ring_with_dot_on_its_centroid():
    """A colour-8 ring whose truncated centroid is occupied by a 1-pixel colour-5 object.

    Both objects therefore emit the SAME generated click coordinate (3, 3). Flag OFF, the one
    de-duplicated row lands on (3, 3), which IS the dot -- so the dot is clickable today.
    """

    grid = np.ones((7, 7), dtype=int)  # colour 1 = the most-common colour = background
    grid[1:6, 1:6] = 8
    grid[2:5, 2:5] = 1
    grid[3, 3] = 5
    return grid


def test_every_claimant_of_a_contested_centroid_is_recorded():
    part = cps.component_partition(_ring_with_dot_on_its_centroid())
    assert len(part) == 2
    # the collision is real: one key, two objects
    assert part.centroid_claimants[(3, 3)] == (0, 1)
    # the single-answer map is unchanged (first claimant), so provenance-order callers are safe
    assert part.index_by_centroid[(3, 3)] == 0
    assert part.claimants_for_centroid(3, 3) == (0, 1)


def test_a_one_pixel_object_on_another_objects_centroid_stays_reachable():
    """THE REGRESSION. Both objects generate (3, 3); each must still get a candidate.

    Before the fix, ``index_by_centroid`` kept only the first claimant and resolution consulted
    it before containment, so BOTH points resolved to the ring and the dot received ZERO click
    candidates on every seed -- strictly worse than flag-off, which reached the dot.
    """

    grid = _ring_with_dot_on_its_centroid()
    part = cps.component_partition(grid)
    dot = next(i for i, c in enumerate(part.colors) if c == 5)
    ring = next(i for i, c in enumerate(part.colors) if c == 8)
    frame = _FakeFrame(grid)

    for seed in range(8):
        cands = rich_action_candidates(
            frame, click_pixel_sampling=True, click_pixel_rng=random.Random(seed)
        )
        clicks = {(int(c.data["x"]), int(c.data["y"])) for c in cands if int(c.action_id) == 6}
        reached = set()
        for point in clicks:
            owner = part.index_by_cell.get(point)
            if owner is not None:
                reached.add(owner)
        assert dot in reached, f"seed {seed}: the 1-pixel object became unreachable {clicks}"
        assert ring in reached, f"seed {seed}: the ring became unreachable {clicks}"


def test_contested_centroid_occurrences_are_handed_out_one_per_claimant():
    grid = _ring_with_dot_on_its_centroid()
    sampled, diag = cps.sample_component_click_points(grid, [(3, 3), (3, 3)], rng=random.Random(0))
    assert diag.points_in == 2 and diag.points_out == 2
    assert diag.contested_centroid_points == 2
    assert diag.unresolved == 0
    part = cps.component_partition(grid)
    owners = {part.index_by_cell.get(p) for p in sampled}
    assert owners == {0, 1}  # one slot per colliding object, neither starved


def test_no_object_is_starved_on_any_real_offline_reset_frame():
    """The measurement that motivated the fix, run for real rather than taken on trust.

    Before the fix this asserted-away silently: 54 of 867 objects across the 25 offline games
    (6.2%), concentrated in small objects, lost all reachability the moment the flag was on.
    """

    from carnot.agentic.arc_solver_kit import offline_arcade

    arc = offline_arcade()
    total = starved = 0
    for env in arc.get_environments():
        gid = str(getattr(env, "game_id", ""))
        try:
            part = cps.component_partition(arc.make(gid).reset())
        except Exception:  # pragma: no cover - a game that will not construct is not the SUT
            continue
        points = []
        for index in range(len(part)):
            key = next((k for k, v in part.centroid_claimants.items() if index in v), None)
            if key is not None:
                points.append(key)
        sampled, _diag = cps.sample_component_click_points(
            part_grid := arc.make(gid).reset(), points, rng=random.Random(0)
        )
        assert part_grid is not None
        reached = {part.index_by_cell.get(p) for p in sampled}
        total += len(part)
        starved += len(part) - len(reached & set(range(len(part))))
    assert total > 100, "the offline corpus did not load -- this test would be vacuous"
    assert starved == 0, f"{starved} of {total} objects unreachable under the sampler"


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-5950-J: a DEAD sampler must be mechanically detectable
# ---------------------------------------------------------------------------


def test_a_dead_sampler_reports_zero_activity_and_a_generation_error(monkeypatch):
    """The counter-defect: a dead mechanism used to report itself active and error-free.

    ``click_pixel_rows_sampled`` counts click rows PRESENT, so it is IDENTICAL for a working
    and a dead sampler. ``coordinates_changed`` / ``generation_errors`` are the fields that can
    tell them apart, which is why the generation-path diagnostics may never be discarded again.
    """

    grid = np.zeros((9, 9), dtype=int)
    grid[1:5, 1:5] = 8

    def _boom(*_a, **_k):
        raise RuntimeError("segmentation unavailable")

    monkeypatch.setattr(cps, "component_partition", _boom)
    diag_out: dict = {}
    frame = _FakeFrame(grid)
    cands = rich_action_candidates(
        frame,
        click_pixel_sampling=True,
        click_pixel_rng=random.Random(0),
        click_pixel_diagnostics_out=diag_out,
    )
    clicks = [(int(c.data["x"]), int(c.data["y"])) for c in cands if int(c.action_id) == 6]
    assert clicks, "a dead sampler must fail OPEN to today's coordinates, not drop them"
    assert diag_out["coordinates_changed"] == 0
    assert diag_out["errors"] > 0


def test_a_live_sampler_reports_nonzero_activity():
    grid = np.zeros((9, 9), dtype=int)
    grid[1:5, 1:5] = 8  # a 4x4 solid block: 15 of its 16 pixels differ from its centroid
    diag_out: dict = {}
    rich_action_candidates(
        _FakeFrame(grid),
        click_pixel_sampling=True,
        click_pixel_rng=random.Random(1),
        click_pixel_diagnostics_out=diag_out,
    )
    assert diag_out["points_in"] > 0
    assert diag_out["errors"] == 0
    # not asserted > 0 on a single draw (a uniform draw CAN land on the centroid); asserted over
    # enough draws that landing on the same single cell every time is not a realistic outcome
    total_changed = 0
    for seed in range(12):
        out: dict = {}
        rich_action_candidates(
            _FakeFrame(grid),
            click_pixel_sampling=True,
            click_pixel_rng=random.Random(seed),
            click_pixel_diagnostics_out=out,
        )
        total_changed += out["coordinates_changed"]
    assert total_changed > 0


def test_explorer_accumulates_the_activity_witness_into_its_diagnostics():
    grid = np.zeros((10, 10), dtype=int)
    grid[1:6, 1:6] = 8
    ex = _explorer(click_pixel_sampling=True, frontier_discipline_seed=3)
    ex._candidates(_FakeFrame(grid))
    diag = ex.frontier_discipline_diagnostics()
    assert diag["click_pixel_sampling_enabled"] is True
    assert diag["click_pixel_points_in"] > 0
    assert diag["click_pixel_generation_errors"] == 0
    assert "click_pixel_coordinates_changed" in diag
    assert diag["click_pixel_rows_sampled_is_not_an_activity_counter"] is True
