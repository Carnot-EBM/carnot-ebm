"""Tests for the just-explore frontier-discipline graft (arXiv:2512.24156).

REQ-ARC-WMTE-5836 / SCENARIO-ARC-WMTE-5836-{A,B,C,D,E,F}

Two layers are tested:

  * the PURE policies in ``carnot.agentic.arc_frontier_discipline`` -- no environment, no
    frames, no LLM; hand-built graphs and hand-built candidate rows only. This is where the
    load-bearing semantics live (cumulative eligibility, GLOBAL rather than per-node
    exhaustion, multi-source reverse BFS).
  * the WIRING into ``StepwiseExplorer`` -- specifically that the flags default OFF and that
    OFF is byte-identical to the historical behaviour, because the whole graft's premise is
    that the submitted agent is unchanged until an A/B says otherwise.

The single most important test here is
``test_exhaustion_is_global_not_per_node``: a per-node implementation of the barrier would
pass a naive single-node test while completely failing to reproduce the mechanism. That test
is constructed so a per-node implementation fails it.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.agentic import arc_frontier_discipline as fd  # noqa: E402
from carnot.agentic.arc_competition_agent import StepwiseExplorer  # noqa: E402


def _row(action: int, tier: int | None = None, x: int | None = None, y: int | None = None):
    row: dict = {"action": action, "data": None if x is None else {"x": x, "y": y}}
    if tier is not None:
        row["tier"] = tier
    return row


# ---------------------------------------------------------------------------
# MECHANISM (a) -- the pure tier-exhaustion barrier
# ---------------------------------------------------------------------------


def test_eligibility_is_cumulative_not_equal_to_active_tier():
    """SCENARIO-ARC-WMTE-5836-A: tier <= p, never tier == p.

    The reference's has_open_group loops range(group_id + 1). An `== p` reading would make
    already-admitted lower tiers unreachable the moment the barrier advanced, permanently
    orphaning candidates -- the exact failure mode this asserts against.
    """
    policy = fd.TierExhaustionPolicy(tier_count=5)
    rows = [_row(6, tier=0, x=1, y=1), _row(6, tier=2, x=2, y=2), _row(6, tier=4, x=3, y=3)]

    assert policy.eligible_indices(rows, 0) == [0]
    assert policy.eligible_indices(rows, 1) == [0]
    assert policy.eligible_indices(rows, 2) == [0, 1]
    assert policy.eligible_indices(rows, 4) == [0, 1, 2]


def test_barrier_never_descends_before_active_tier_is_empty():
    """SCENARIO-ARC-WMTE-5836-A: a tier-2 action is unreachable at p=1 and reachable at p=2."""
    policy = fd.TierExhaustionPolicy(tier_count=5)
    rows = [_row(6, tier=2, x=9, y=9)]

    # p = 0 and p = 1: the tier-2 row is NOT selectable at all.
    assert policy.select_index(rows, 0) is None
    assert policy.select_index(rows, 1) is None
    assert policy.node_has_open_tier(rows, 1) is False
    # p = 2: admitted.
    assert policy.select_index(rows, 2) == 0
    assert policy.node_has_open_tier(rows, 2) is True


def test_exhaustion_is_global_not_per_node():
    """SCENARIO-ARC-WMTE-5836-B: the barrier is GLOBAL. A per-node impl fails this.

    Node A is locally exhausted at tier 0 (its only remaining row is tier 3) while node B
    still has a tier-0 row. A per-node barrier would advance A's tier and let it spend a
    tier-3 action; the global barrier must NOT advance, because tier 0 is still open
    somewhere in the graph.
    """
    policy = fd.TierExhaustionPolicy(tier_count=5)
    node_a = [_row(6, tier=3, x=1, y=1)]
    node_b = [_row(6, tier=0, x=2, y=2)]

    # A is locally out of admitted work...
    assert policy.node_has_open_tier(node_a, 0) is False
    # ...but the GLOBAL barrier stays at 0 because B still has tier-0 work.
    assert policy.next_active_tier([node_a, node_b], 0) == 0
    # And A's tier-3 row remains un-selectable while the barrier is at 0.
    assert policy.select_index(node_a, 0) is None

    # Drain B; only now may the barrier move.
    assert policy.next_active_tier([node_a, []], 0) == 3


def test_barrier_can_skip_multiple_empty_tiers_in_one_step():
    """SCENARIO-ARC-WMTE-5836-B: advancement is a `while`, not an `if`.

    Tiers 1 and 2 are empty everywhere, so stopping at p=1 would stall the search for one
    whole decision per empty tier.
    """
    policy = fd.TierExhaustionPolicy(tier_count=5)
    nodes = [[_row(6, tier=3, x=1, y=1)], [_row(6, tier=4, x=2, y=2)]]

    assert policy.next_active_tier(nodes, 0) == 3
    # Already at the ceiling: never advance past tier_count - 1.
    assert policy.next_active_tier([[], []], 4) == 4
    assert policy.next_active_tier([[], []], 0) == 4


def test_barrier_accepts_a_generator_of_node_rows():
    """next_active_tier needs several passes; a generator must not be silently consumed once."""
    policy = fd.TierExhaustionPolicy(tier_count=5)
    nodes = [[_row(6, tier=4, x=1, y=1)], [_row(6, tier=4, x=2, y=2)]]
    assert policy.next_active_tier((n for n in nodes), 0) == 4


def test_within_tier_draw_is_first_by_default_and_uniform_with_an_rng():
    """SCENARIO-ARC-WMTE-5836-C: both within-tier draws are available.

    The reference draws UNIFORMLY among admitted candidates; Carnot's live pop(0) is the
    fully-greedy opposite, and three prior experiments found Carnot-scored reorderings of that
    uniform draw lost solves. So the draw must be varyable independently of the barrier or the
    A/B cannot separate "the barrier helped" from "breaking the static tie helped".
    """
    policy = fd.TierExhaustionPolicy(tier_count=5)
    rows = [_row(6, tier=3, x=0, y=0)] + [_row(6, tier=0, x=i, y=i) for i in range(1, 6)]

    # Deterministic arm: always the first ELIGIBLE row (index 1, since index 0 is deferred).
    assert all(policy.select_index(rows, 0) == 1 for _ in range(20))

    # Uniform arm: spreads over every eligible index, and NEVER selects the deferred one.
    seen = {policy.select_index(rows, 0, rng=random.Random(s)) for s in range(200)}
    assert seen == {1, 2, 3, 4, 5}
    assert 0 not in seen

    # top_k bounds the uniform pool to the first k ELIGIBLE rows.
    seen_k = {policy.select_index(rows, 0, rng=random.Random(s), top_k=2) for s in range(200)}
    assert seen_k == {1, 2}


def test_unclassified_rows_fail_open_at_tier_zero():
    """An un-stamped row, a non-click action, or an unmapped coordinate must stay SELECTABLE.

    Failing open costs only ordering. Failing closed would make part of the action space
    permanently unreachable -- a silent capability loss, which is strictly worse.
    """
    tier_map = {(5, 5): 4}
    assert fd.row_tier({"action": 6, "data": {"x": 5, "y": 5}}, tier_map) == 4
    # keyboard/simple action -> tier 0 (the reference forces these into group 0)
    assert fd.row_tier({"action": 2, "data": None}, tier_map) == 0
    # click on a coordinate that is not a detected object centroid -> DEFAULT_TIER
    assert fd.row_tier({"action": 6, "data": {"x": 63, "y": 63}}, tier_map) == fd.DEFAULT_TIER
    assert fd.DEFAULT_TIER == 0
    # malformed data
    assert fd.row_tier({"action": 6, "data": {"nope": 1}}, tier_map) == 0
    # an un-stamped row reads back as tier 0 through the policy
    policy = fd.TierExhaustionPolicy()
    assert policy.node_has_open_tier([{"action": 6, "data": {"x": 1, "y": 1}}], 0) is True


def test_annotate_tiers_stamps_without_reordering_and_without_mutating_input():
    original = [_row(6, x=1, y=1), _row(6, x=2, y=2), _row(3)]
    stamped = fd.annotate_tiers(original, None, tier_by_xy={(1, 1): 4, (2, 2): 0})

    assert [r["tier"] for r in stamped] == [4, 0, 0]
    # order preserved exactly (the barrier gates eligibility; the existing rankers own order)
    assert [(r["action"], r["data"]) for r in stamped] == [
        (r["action"], r["data"]) for r in original
    ]
    # inputs untouched
    assert all("tier" not in r for r in original)


def test_click_tier_map_reproduces_the_reference_five_tier_predicate():
    """The tiers MUST come from the same predicate the already-nulled SORT used.

    If this module derived its own tiers, an A/B difference could be a predicate difference
    rather than a discipline difference -- an uninterpretable experiment.
    """
    np = pytest.importorskip("numpy")
    grid = np.zeros((64, 64), dtype=int)
    grid[2:6, 2:6] = 9  # salient (9 in {6..15}) AND medium (4x4) -> tier 0
    grid[10:14, 10:14] = 3  # non-salient, medium -> tier 1
    grid[20:60, 20:60] = 7  # salient, too wide (40 > 32) -> tier 2
    grid[40:44, 2:6] = 16  # status-bar colour -> last tier

    tiers = set(fd.click_tier_map(grid).values())
    assert 0 in tiers and 1 in tiers and 2 in tiers
    assert fd.STATUS_BAR_TIER in tiers
    assert fd.STATUS_BAR_TIER == fd.TIER_COUNT - 1


# ---------------------------------------------------------------------------
# MECHANISM (b) -- the pure multi-source frontier-distance gradient
# ---------------------------------------------------------------------------


def _edge(action_id: int):
    return {"action": action_id, "data": None}


def test_multi_source_reverse_bfs_labels_hops_to_nearest_open_node():
    """SCENARIO-ARC-WMTE-5836-D: one pass seeded at ALL open nodes labels the whole graph.

    Graph:  ROOT -1-> A -2-> B   and   ROOT -3-> C
    Open nodes: {B, C}. From ROOT, C is 1 hop away and B is 2 -- so ROOT's field value is 1
    and its next hop is the edge toward C.
    """
    forward = {"ROOT": [(_edge(1), "A"), (_edge(3), "C")], "A": [(_edge(2), "B")]}
    rev = fd.reverse_adjacency(forward)
    field_ = fd.frontier_distance_field(rev, ["B", "C"])

    assert field_.distance["B"] == 0
    assert field_.distance["C"] == 0
    assert field_.distance["A"] == 1
    assert field_.distance["ROOT"] == 1
    assert field_.next_hop["A"] == (_edge(2), "B")
    assert field_.next_hop["ROOT"] == (_edge(3), "C")
    assert field_.is_open("C") is True
    assert field_.is_open("ROOT") is False


def test_gradient_prefers_the_navigation_nearest_frontier_not_the_shallowest():
    """SCENARIO-ARC-WMTE-5836-E: the gradient and depth-from-root DISAGREE here.

    ROOT -1-> SHALLOW_OPEN            (depth 1 from root, but NOT reachable forward from CUR)
    ROOT -2-> MID -3-> CUR -4-> NEAR_OPEN   (NEAR_OPEN is depth 3, one hop from CUR)

    Carnot's historical depth-primary ordering picks SHALLOW_OPEN and pays a full RESET+replay
    to get there. The gradient must pick NEAR_OPEN, which is one known-working hop away. If
    this test ever starts returning SHALLOW_OPEN, the graft has silently reverted to depth
    ordering and the mechanism is not actually installed.
    """
    forward = {
        "ROOT": [(_edge(1), "SHALLOW_OPEN"), (_edge(2), "MID")],
        "MID": [(_edge(3), "CUR")],
        "CUR": [(_edge(4), "NEAR_OPEN")],
    }
    target = fd.gradient_frontier_target(forward, None, ["SHALLOW_OPEN", "NEAR_OPEN"], "CUR")
    assert target == "NEAR_OPEN"

    # From ROOT the nearest open node IS the shallow one, so the two criteria agree there.
    assert fd.gradient_frontier_target(forward, None, ["SHALLOW_OPEN", "NEAR_OPEN"], "ROOT") == (
        "SHALLOW_OPEN"
    )


def test_gradient_returns_none_rather_than_vetoing_when_it_has_no_opinion():
    """The gradient is a PREFERENCE. It must never make a reachable node unreachable."""
    forward = {"ROOT": [(_edge(1), "A")]}
    # nothing open at all
    assert fd.gradient_frontier_target(forward, None, [], "ROOT") is None
    # the only open node is not forward-reachable from CUR over known-working edges
    assert fd.gradient_frontier_target(forward, None, ["A"], "ISLAND") is None
    # no current node
    assert fd.gradient_frontier_target(forward, None, ["A"], None) is None
    # empty graph
    assert fd.gradient_frontier_target({}, None, ["A"], "A") == "A"  # already open -> itself


def test_gradient_returns_current_node_when_current_is_itself_open():
    forward = {"CUR": [(_edge(1), "X")]}
    assert fd.gradient_frontier_target(forward, None, ["CUR", "X"], "CUR") == "CUR"


def test_reverse_adjacency_matches_an_explorers_incremental_index():
    """The explorer maintains radj incrementally; it must agree with the pure inversion.

    A drift between the two indices would make the gradient promise routes that do not exist,
    which is precisely the failure the "known-working edges only" rule exists to prevent.
    """
    exp = StepwiseExplorer(frontier_gradient=True)
    exp._record_forward_edge("ROOT", {"action": 1, "data": None}, "A")
    exp._record_forward_edge("A", {"action": 2, "data": None}, "B")
    exp._record_forward_edge("ROOT", {"action": 1, "data": None}, "A")  # dedup: no new edge

    assert fd.reverse_adjacency(exp.adj) == exp.radj
    assert exp.radj["A"] == [({"action": 1, "data": None}, "ROOT")]

    # self-edges (no state change) are NOT known-working edges and must not enter either index
    exp._record_forward_edge("B", {"action": 3, "data": None}, "B")
    assert "B" not in exp.radj or all(o != "B" for _a, o in exp.radj.get("B", []))
    assert fd.reverse_adjacency(exp.adj) == exp.radj


# ---------------------------------------------------------------------------
# WIRING -- default-off parity, and the wired barrier actually gating
# ---------------------------------------------------------------------------


def test_flags_default_off_and_pop_is_byte_identical_to_pop_zero():
    """SCENARIO-ARC-WMTE-5836-F: OFF must be the historical behaviour exactly.

    With the flags OFF, _pop_untested must still be an unconditional pop(0) even over rows whose
    stamped tiers would otherwise defer them. That escape hatch is what makes the A/B's B2_nofix arm
    a valid attribution control, so it stays load-bearing.

    UPDATED 2026-07-25: TIER_EXHAUSTION and TIER_UNIFORM_RANDOM were flipped ON as the shipped
    default (see the flag block in arc_competition_agent.py). The off-behaviour must therefore be
    requested EXPLICITLY here rather than inherited from the module defaults. The shipped-default
    values are asserted separately, just below.
    """
    exp = StepwiseExplorer(
        tier_exhaustion=False, tier_uniform_random=False, frontier_gradient=False
    )
    assert exp.tier_exhaustion_enabled is False
    assert exp.tier_uniform_random_enabled is False
    assert exp.frontier_gradient_enabled is False

    # and the SHIPPED default is now ON for the two flipped mechanisms, gradient still off
    shipped = StepwiseExplorer()
    assert shipped.tier_exhaustion_enabled is True
    assert shipped.tier_uniform_random_enabled is True
    assert shipped.tier_click_vocab_only is True
    assert shipped.frontier_gradient_enabled is False

    node = {
        "path": [],
        "untested": [_row(6, tier=4, x=1, y=1), _row(6, tier=0, x=2, y=2)],
        "value": None,
    }
    # tier 4 first in the list -> flags off means it is STILL taken first (pop(0) semantics)
    assert exp._pop_untested(node)["data"] == {"x": 1, "y": 1}
    assert exp._node_has_open_tier(node) is True
    assert exp._node_is_tier_deferred(node) is False


def test_wired_barrier_defers_a_high_tier_row_then_admits_it_after_advance():
    # tier_click_vocab_only=False: this test drives the barrier's state machine DIRECTLY with
    exp = StepwiseExplorer(tier_click_vocab_only=False, tier_exhaustion=True)
    node_a = {"path": [], "untested": [_row(6, tier=3, x=1, y=1)], "value": None}
    node_b = {"path": [], "untested": [_row(6, tier=0, x=2, y=2)], "value": None}
    exp.graph = {"A": node_a, "B": node_b}

    assert exp._node_has_open_tier(node_a) is False
    assert exp._node_is_tier_deferred(node_a) is True

    # GLOBAL barrier: B still has tier-0 work, so no advance.
    exp._maybe_advance_tier()
    assert exp._active_tier == 0
    assert exp._node_has_open_tier(node_a) is False

    # Drain B, then the barrier advances and A's tier-3 row becomes selectable.
    exp._pop_untested(node_b)
    exp._maybe_advance_tier()
    assert exp._active_tier == 3
    assert exp._tier_advances == 3
    assert exp._node_has_open_tier(node_a) is True
    assert exp._pop_untested(node_a)["data"] == {"x": 1, "y": 1}


def test_wired_frontier_advances_the_barrier_instead_of_reporting_explored_out():
    """A tier gate must never produce a spurious explored_out while deferred work remains.

    This is the trap that would make the whole A/B read as a null for a mechanical reason:
    _frontier returning None because everything is merely DEFERRED, not exhausted.
    """
    # tier_click_vocab_only=False: this test drives the barrier's state machine DIRECTLY with
    exp = StepwiseExplorer(tier_click_vocab_only=False, tier_exhaustion=True)
    exp.graph = {"A": {"path": [], "untested": [_row(6, tier=4, x=1, y=1)], "value": None}}
    exp.cur = "A"

    chosen = exp._frontier()
    assert chosen == "A", "deferred-only work must be admitted by an advance, not dropped"
    assert exp._active_tier == 4
    assert exp.explored_out is False


def test_wired_frontier_does_not_record_deferred_nodes_as_exhausted_negatives():
    """A DEFERRED node is not a dead end; labelling it 0 would poison the discriminator.

    The barrier must not silently corrupt an unrelated learned component.
    """
    exp = StepwiseExplorer(tier_exhaustion=True)
    recorded: list[str] = []
    exp._record_discriminative_features = (  # type: ignore[method-assign]
        lambda *a, **kw: recorded.append(kw.get("node_hash", "?"))
    )
    exp.graph = {
        "DEFERRED": {
            "path": [],
            "untested": [_row(6, tier=4, x=1, y=1)],
            "value": None,
            "discriminative_features": [0.0],
        },
        "EMPTY": {"path": [], "untested": [], "value": None, "discriminative_features": [0.0]},
        "OPEN": {"path": [], "untested": [_row(6, tier=0, x=2, y=2)], "value": None},
    }
    exp.cur = "OPEN"
    exp._frontier()

    assert "EMPTY" in recorded, "a genuinely empty node is still recorded as before"
    assert "DEFERRED" not in recorded, "a merely-deferred node must NOT be a negative sample"


def test_wired_frontier_batch_only_pops_tier_admitted_rows():
    """A deferred row popped here would silently defeat the barrier (batches expand blindly)."""
    # tier_click_vocab_only=False: this test drives the barrier's state machine DIRECTLY with
    exp = StepwiseExplorer(
        tier_click_vocab_only=False, tier_exhaustion=True, frontier_batch_size="all"
    )
    node = {
        "path": [],
        "untested": [
            _row(6, tier=4, x=1, y=1),
            _row(6, tier=0, x=2, y=2),
            _row(6, tier=0, x=3, y=3),
        ],
        "value": None,
    }
    batch = exp._pop_frontier_batch(node)

    assert [r["data"] for r in batch] == [{"x": 2, "y": 2}, {"x": 3, "y": 3}]
    assert [r["tier"] for r in node["untested"]] == [4], "the deferred row stays queued"
    assert batch, "must never return an empty batch when the node has admitted work"


def test_wired_gradient_target_prefers_the_nearest_open_frontier_node():
    exp = StepwiseExplorer(frontier_gradient=True)
    exp.graph = {
        "SHALLOW": {"path": [], "untested": [_row(6, x=1, y=1)], "value": None},
        "MID": {"path": [{"action": 1, "data": None}], "untested": [], "value": None},
        "CUR": {"path": [{"action": 1, "data": None}] * 2, "untested": [], "value": None},
        "NEAR": {
            "path": [{"action": 1, "data": None}] * 3,
            "untested": [_row(6, x=2, y=2)],
            "value": None,
        },
    }
    exp.cur = "CUR"
    exp._record_forward_edge("ROOT", {"action": 1, "data": None}, "SHALLOW")
    exp._record_forward_edge("ROOT", {"action": 2, "data": None}, "MID")
    exp._record_forward_edge("MID", {"action": 3, "data": None}, "CUR")
    exp._record_forward_edge("CUR", {"action": 4, "data": None}, "NEAR")

    assert exp._gradient_frontier_target(["SHALLOW", "NEAR"]) == "NEAR"
    assert exp.frontier_discipline_diagnostics()["gradient_targets_chosen"] == 1
    # a gradient with no route from here has NO opinion (falls back, never vetoes)
    exp.cur = "ISLAND"
    assert exp._gradient_frontier_target(["SHALLOW", "NEAR"]) is None
    assert exp.frontier_discipline_diagnostics()["gradient_misses"] == 1


def test_gradient_never_returns_a_depth_capped_current_node():
    """REGRESSION (adversarial review 2026-07-24): MECHANISM (b) must not cancel max_depth.

    ``next_move`` step 1 rides the current node only while ``len(path) < max_depth``; step 2's
    ``th == self.cur`` branch then expands a returned current node IN PLACE with NO depth test.
    Since ``self.cur`` is normally one of the gradient's own seeds, it sits at distance 0 and
    ``nearest_open_node`` returned it EVERY time -- so on deep-graph games the "gradient" was
    really re-enabling a branch the backtrack cap had abandoned (measured: 140/140 picks on r11l
    at budget 600, 195/195 at budget 800). Arms C/D were therefore confounded with the removal of
    max_depth rather than measuring frontier distance.

    The minimal reproduction is asserted directly: with a depth-capped ``cur``, the flag-ON
    explorer must choose what the flag-OFF explorer chooses, not the capped node.
    """

    def _build(**kw):
        exp = StepwiseExplorer(max_depth=2, **kw)
        exp.graph = {
            "ROOT": {"path": [], "untested": [_row(6, x=1, y=1)], "value": None},
            "DEEP": {
                "path": [{"action": 1, "data": None}] * 5,
                "untested": [_row(6, x=2, y=2)],
                "value": None,
            },
        }
        exp.cur = "DEEP"
        exp._record_forward_edge("ROOT", {"action": 1, "data": None}, "DEEP")
        return exp

    off = _build(frontier_gradient=False)
    on = _build(frontier_gradient=True)
    assert off._frontier() == "ROOT"
    assert on._frontier() == "ROOT", "a depth-capped cur must never be handed back as the target"
    diag = on.frontier_discipline_diagnostics()
    assert diag["gradient_cur_at_depth_cap_excluded"] == 1
    assert diag["gradient_pick_current_node"] == 0
    assert diag["max_depth"] == 2

    # ...and the mechanism still works: with cur UNDER the cap it may legitimately pick cur,
    # which is what best_first mode relies on (step 1 never runs there).
    under = StepwiseExplorer(max_depth=45, frontier_gradient=True)
    under.graph = {
        "CUR": {
            "path": [{"action": 1, "data": None}],
            "untested": [_row(6, x=1, y=1)],
            "value": None,
        }
    }
    under.cur = "CUR"
    assert under._gradient_frontier_target(["CUR"]) == "CUR"
    d2 = under.frontier_discipline_diagnostics()
    assert d2["gradient_pick_current_node"] == 1 and d2["gradient_pick_other_node"] == 0


def test_gradient_picks_another_node_when_cur_is_depth_capped_and_a_route_exists():
    """The real mechanism: with cur refused, the gradient still chooses the NEAREST other node.

    Distinguishes the fix "cur is excluded as a seed" from a bogus fix "the gradient is disabled
    whenever cur is capped": a reachable, nearer open node must still be selected, and it must be
    counted as an OTHER pick so the artifact can show what the gradient actually did.
    """
    exp = StepwiseExplorer(max_depth=3, frontier_gradient=True)
    exp.graph = {
        "ROOT": {"path": [], "untested": [_row(6, x=9, y=9)], "value": None},
        "CUR": {
            "path": [{"action": 1, "data": None}] * 5,  # over the cap
            "untested": [_row(6, x=1, y=1)],
            "value": None,
        },
        "NEXT": {
            "path": [{"action": 1, "data": None}] * 6,
            "untested": [_row(6, x=2, y=2)],
            "value": None,
        },
    }
    exp.cur = "CUR"
    exp._record_forward_edge("ROOT", {"action": 1, "data": None}, "CUR")
    exp._record_forward_edge("CUR", {"action": 2, "data": None}, "NEXT")

    # ROOT is shallowest-from-root; NEXT is nearest-from-here and forward-reachable.
    assert exp._gradient_frontier_target(["ROOT", "CUR", "NEXT"]) == "NEXT"
    diag = exp.frontier_discipline_diagnostics()
    assert diag["gradient_pick_other_node"] == 1
    assert diag["gradient_pick_current_node"] == 0
    assert diag["gradient_cur_at_depth_cap_excluded"] == 1


def test_within_tier_uniform_draw_is_unrestricted_not_the_diversity_topk():
    """REGRESSION: arm B2's draw must be the reference's UNRESTRICTED uniform choice.

    The wiring used to pass ``self._div_topk`` (default 8, the knob of the unrelated
    hybrid-diversity feature), which (a) made the arm a top-8 draw rather than the reference's
    ``random.choice`` over all untested edges in groups 0..p, and (b) coupled an A/B arm to a
    foreign env var, so an operator tuning CARNOT_ARC_EXPLORE_DIV_TOPK would silently change the
    experiment. Asserted by construction (top_k is None) AND behaviourally (a row beyond index 8
    is reachable).
    """
    exp = StepwiseExplorer(tier_exhaustion=True, tier_uniform_random=True)
    assert exp._fd_draw_topk is None
    assert exp.frontier_discipline_diagnostics()["tier_draw_top_k"] is None
    # The foreign knob must NOT be what the draw reads.
    exp._div_topk = 2
    node = {"path": [], "untested": [_row(6, tier=0, x=i, y=i) for i in range(20)], "value": None}
    exp.graph = {"N": node}
    seen = set()
    for _ in range(200):
        if not node["untested"]:
            break
        seen.add(exp._pop_untested(node)["data"]["x"])
    assert max(seen) > 8, f"draw is restricted to a top-k slice: only reached {sorted(seen)}"
    assert len(seen) == 20


def test_explicit_draw_topk_env_is_honoured_and_reported_as_a_deviation(monkeypatch):
    """A top-k restriction is measurable, but only as an EXPLICIT, self-reported deviation."""
    monkeypatch.setenv("CARNOT_ARC_FRONTIER_TIER_DRAW_TOPK", "3")
    exp = StepwiseExplorer(tier_exhaustion=True, tier_uniform_random=True)
    assert exp._fd_draw_topk == 3
    assert exp.frontier_discipline_diagnostics()["tier_draw_top_k"] == 3
    node = {"path": [], "untested": [_row(6, tier=0, x=i, y=i) for i in range(10)], "value": None}
    first = {
        exp._pop_untested(dict(node, untested=list(node["untested"])))["data"]["x"]
        for _ in range(60)
    }
    assert max(first) <= 2, f"top_k=3 must draw only among the first 3 rows, saw {sorted(first)}"


def test_env_flags_toggle_the_mechanisms_without_mutating_module_globals(monkeypatch):
    """The A/B harness flips arms by env var; module globals must stay at their OFF defaults."""
    import carnot.agentic.arc_competition_agent as agent_mod

    monkeypatch.setenv("CARNOT_ARC_FRONTIER_TIER_EXHAUSTION", "1")
    monkeypatch.setenv("CARNOT_ARC_FRONTIER_GRADIENT", "1")
    exp = StepwiseExplorer()
    assert exp.tier_exhaustion_enabled is True
    assert exp.frontier_gradient_enabled is True
    # TIER_EXHAUSTION was flipped ON 2026-07-25 (see the flag block in arc_competition_agent.py).
    # The point this test actually protects is unchanged: setting an env var must not MUTATE the
    # module global. So assert the global still equals its declared shipped value, whatever that is,
    # rather than hardcoding False.
    assert agent_mod.SUBMITTED_FRONTIER_TIER_EXHAUSTION_ENABLED is True
    assert agent_mod.SUBMITTED_FRONTIER_DISTANCE_GRADIENT_ENABLED is False  # gradient stays OFF
    # an explicit kwarg still wins over the env var
    assert StepwiseExplorer(tier_exhaustion=False).tier_exhaustion_enabled is False


# ---------------------------------------------------------------------------
# The A/B harness's own logic (pure; no games are run here)
# ---------------------------------------------------------------------------


def _hrow(arm, game, cond, seed, levels, first=None, states=None):
    return {
        "arm": arm,
        "game": game,
        "condition": cond,
        "seed": seed,
        "ran": True,
        "levels": levels,
        "reached": levels,
        "actions": 10,
        "efficiency": 0.0,
        "actions_to_first_levelup": first,
        "states_expanded": states,
        "duration_s": 0.1,
        "frontier_discipline": None,
    }


def _harness():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "carnot"
        / "experiment_5836_frontier_discipline_ab.py"
    )
    spec = importlib.util.spec_from_file_location("_exp5836_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_exp5836_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_harness_declares_all_six_arms_including_the_uniform_draw_and_the_control():
    """The uniform-within-tier arm and the reference positive control are BOTH mandatory.

    Without B2 a null on B is confounded with the greedy-draw hypothesis; without E a flat
    result cannot be told apart from a broken harness.
    """
    m = _harness()
    # B2_nofix added 2026-07-25: B2 now inherits the click-vocabulary gate from its shipped default,
    # so an arm with the gate explicitly OFF is required to attribute the delta to the GATE rather
    # than to drift in the barrier. It is as mandatory as B2 and E for the same reason.
    # F / F1 added 2026-07-25 (REQ-ARC-WMTE-5950): the per-object click-pixel SAMPLING arms.
    # They sit on top of B2's flags (the CURRENT live configuration) and differ from it by the
    # sampler alone, so B2 -- not A -- is their matched control. F1 pins redraw budget 1
    # (coordinate correction only) and F pins 3 (bounded with-replacement), which separates the
    # mechanism's two halves. Asserted as an EXACT set because a silently-dropped arm is how a
    # control goes missing.
    # G / G2 / G3 added 2026-07-25 (REQ-ARC-WMTE-5960): the REPAIRED orientation-complete HUD
    # status-bar detector arms. They also sit on B2's flags and differ from it by the detector
    # alone, so B2 is their matched control too. G is detection-only; G2 additionally arms the
    # runtime collapse guard, so G2 - G isolates the guard's cost; G3 additionally arms Stage 2's
    # pre-activation behavioural confirmation and is the ONLY flip candidate (G ships the Stage-1
    # geometry bare, which was measured over-masking a decision-relevant fill gauge on ar25, and
    # G2 applies a bad mask first and retracts it only after nodes exist under it). Still asserted
    # as an EXACT set for the reason stated above -- a silently-dropped arm is how a control goes
    # missing.
    assert set(m.ARMS) == {"A", "B", "B2", "B2_nofix", "C", "D", "E", "F", "F1", "G", "G2", "G3"}
    assert m.HUD_MASK_FLIP_CANDIDATE_ARMS == ("G3",)
    assert m.ARMS["G"]["kwargs"]["hud_mask_stage2_confirm"] is False
    assert m.ARMS["G3"]["kwargs"]["hud_mask_stage2_confirm"] is True
    assert m.ARMS["F"]["kwargs"]["click_pixel_sampling"] is True
    assert m.ARMS["F"]["kwargs"]["click_pixel_redraw_budget"] == 3
    assert m.ARMS["F1"]["kwargs"]["click_pixel_redraw_budget"] == 1
    assert m.CLICK_PIXEL_CONTROL_ARM == "B2"
    assert m.ARMS["B2_nofix"]["kwargs"]["tier_click_vocab_only"] is False
    assert m.ARMS["B2_nofix"]["kwargs"]["tier_uniform_random"] is True
    assert m.ARMS["B"]["kwargs"]["tier_uniform_random"] is False
    assert m.ARMS["B2"]["kwargs"]["tier_uniform_random"] is True
    assert m.ARMS["B2"]["deterministic"] is False
    assert m.ARMS["C"]["kwargs"]["frontier_gradient"] is True
    # UPDATED 2026-07-25: every explorer arm now pins all seven gated flags explicitly (see
    # test_every_explorer_arm_pins_all_seven_gated_flags for why), so exact-dict equality on one
    # arm's kwargs is brittle to that pinning. Assert D's DEFINING values instead -- tier barrier on
    # with the GREEDY draw, plus the gradient -- which is what distinguishes it from B2 and C.
    assert m.ARMS["D"]["kwargs"]["tier_exhaustion"] is True
    assert m.ARMS["D"]["kwargs"]["tier_uniform_random"] is False
    assert m.ARMS["D"]["kwargs"]["frontier_gradient"] is True
    assert m.ARMS["D"]["kwargs"]["edge_bar_hud_mask"] is False
    # Deterministic arms must NOT be given a fake replication axis.
    assert m._seeds_for("A", 3) == [m.RANDOM_SEED]
    assert len(m._seeds_for("B2", 3)) == 3
    # Budget 200 was measured degenerate (0/25 wins) -> the default must be well above it.
    assert m.DEFAULT_BUDGET >= 2000


def test_harness_ci_is_none_for_a_single_run_never_zero_width():
    """A width-zero interval would read as certainty. n<2 must report None."""
    m = _harness()
    assert m._mean_ci95([5.0])["ci95"] is None
    assert m._mean_ci95([])["mean"] is None
    ci = m._mean_ci95([10.0, 20.0, 30.0])
    assert ci["n"] == 3 and ci["mean"] == 20.0 and ci["ci95"][0] < 20.0 < ci["ci95"][1]


def test_harness_aggregate_and_regression_guard_detect_a_lost_baseline_win():
    """A new win bought by losing an existing one must be surfaced, not netted to zero."""
    m = _harness()
    rows = [
        _hrow("A", "lp85", "real", 1, 1, first=20, states=5),
        _hrow("A", "vc33", "real", 1, 1, first=60, states=6),
        _hrow("A", "bp35", "real", 1, 0, states=9),
        _hrow("B", "lp85", "real", 1, 0, states=5),  # LOST a baseline win
        _hrow("B", "vc33", "real", 1, 1, first=22, states=6),
        _hrow("B", "bp35", "real", 1, 1, first=88, states=9),  # gained a new one
    ]
    agg = m.aggregate(rows, ["lp85", "vc33", "bp35"])
    assert agg["A|real"]["n_games_won_any_seed"] == 2
    assert agg["B|real"]["n_games_won_any_seed"] == 2
    cmp_ = m.compare_to_baseline(agg, ["lp85", "vc33", "bp35"], rows)
    assert cmp_["regression_guard_provenance"].startswith("derived_from_arm_A")
    assert cmp_["B"]["new_wins"] == ["bp35"]
    assert cmp_["B"]["lost_wins"] == ["lp85"]
    assert cmp_["B"]["regressed_baseline_win"] is True, "a lost baseline win must not net to zero"
    assert cmp_["B"]["n_win_delta"] == 0


def test_harness_recolor_control_only_asserts_inertness_where_it_is_predicted():
    """Recolour is inert for the baseline order but NOT for the colour-keyed tier arms.

    Measured in this experiment's own smoke: arm B's mean actions-to-first-win went 13.5 (real)
    -> 168.5 (recoloured), because just-explore's tier predicate keys on ABSOLUTE colour values.
    That is a real limitation of the mechanism; flagging it as a control violation would
    mislabel it as a harness bug, and skipping the check for arm A would hide genuine leakage.
    """
    m = _harness()
    rows = [
        _hrow("A", "vc33", "real", 1, 1, first=60, states=6),
        _hrow("A", "vc33", "recolor_negative_control", 1, 1, first=60, states=6),
        _hrow("B", "vc33", "real", 1, 1, first=22, states=6),
        _hrow("B", "vc33", "recolor_negative_control", 1, 1, first=303, states=17),
    ]
    agg = m.aggregate(rows, ["vc33"])
    ctl = m.compare_to_baseline(agg, ["vc33"])["recolor_control"]
    assert ctl["A"]["expected_inert"] is True and ctl["A"]["control_violated"] is False
    assert ctl["B"]["expected_inert"] is False
    assert ctl["B"]["colour_dependent_by_construction"] is True
    assert ctl["B"]["control_violated"] is False, "a colour-keyed arm's delta is not a harness bug"
    assert ctl["B"]["mean_actions_real"] != ctl["B"]["mean_actions_recolor"], "delta is reported"

    # ...but genuine colour leakage on a baseline-order arm IS flagged.
    leak = [
        _hrow("A", "vc33", "real", 1, 1, first=60, states=6),
        _hrow("A", "vc33", "recolor_negative_control", 1, 1, first=999, states=6),
    ]
    ctl2 = m.compare_to_baseline(m.aggregate(leak, ["vc33"]), ["vc33"])
    assert ctl2["recolor_control"]["A"]["control_violated"] is True
    assert ctl2["recolor_control_violations"] == ["A"]


def test_harness_pooled_ci_is_small_sample_corrected_clamped_and_non_inferential():
    """REGRESSION: the pooled CI must not fake certainty, understate width, or go negative.

    Three defects in one function, all measured in this experiment's own smoke run:
      * two identical values produced ci95 [1.0, 1.0] -- a zero-width interval, i.e. exactly the
        fake certainty the function's docstring promised to avoid;
      * a normal-approximation 1.96 at n=2 understates the interval ~6.5x (t=12.706);
      * an action-count interval reported a physically impossible negative lower bound (-3.16).
    """
    m = _harness()
    zero_var = m._mean_ci95([1.0, 1.0])
    assert zero_var["ci95"] is None
    assert "zero_variance" in zero_var["ci95_absent_reason"]

    two = m._mean_ci95([20.0, 60.0])
    assert two["ci95_method"] == "t_distribution_small_sample"
    half = (two["ci95"][1] - two["ci95"][0]) / 2
    normal_half = 1.96 * two["sd"] / (2**0.5)
    assert half > 4 * normal_half, "small-n t correction must widen the interval, not fake it"

    clamped = m._mean_ci95([20.0, 60.0], clamp_min=0.0)
    assert clamped["ci95"][0] == 0.0 and clamped["ci95_clamped_at_min"] is True

    agg = m.aggregate([_hrow("A", "lp85", "real", 1, 1, first=20, states=5)], ["lp85"])
    assert agg["A|real"]["mean_actions_to_first_win_is_inferential"] is False


def test_harness_paired_statistic_sees_a_unanimous_effect_the_pooled_ci_cannot():
    """REGRESSION (fatal): the primary efficiency statistic must be PAIRED, with a p-value.

    The exact smoke data: arm A wins lp85 in 20 and vc33 in 60 actions; arm B wins them in 5 and
    22. Pooled, A's mean is 40.0 and B's is 13.5 with heavily overlapping intervals -- the
    statistic cannot see the effect. Paired, both deltas are positive (+15, +38): unanimous. The
    pairing is free (same games, same seeds, policy is the only difference), so discarding it was
    pure loss of power. This test asserts the paired table exists, is signed the documented way,
    and carries a real sign-test p (which at n=2 is 0.5 -- i.e. the '40.0 -> 13.5' headline was
    NOT significant, and the artifact now says so).
    """
    m = _harness()
    rows = [
        _hrow("A", "lp85", "real", 1, 1, first=20, states=5),
        _hrow("A", "vc33", "real", 1, 1, first=60, states=6),
        _hrow("B", "lp85", "real", 1, 1, first=5, states=5),
        _hrow("B", "vc33", "real", 1, 1, first=22, states=6),
    ]
    paired = m.paired_efficiency_vs_baseline(rows)
    real = paired["B"]["real"]
    assert real["n_paired_games"] == 2
    assert [p["delta"] for p in real["pairs"]] == [20 - 5, 60 - 22]
    assert real["sign_test"]["n_favouring_arm"] == 2
    assert real["sign_test"]["p_value"] == 0.5, "n=2 cannot be significant -- say so"
    assert real["median_paired_delta"]["ci95"] is None, "n<3 bootstrap must not look informative"

    # A game only one arm wins is a CAPABILITY difference and must NOT enter the paired table.
    rows_cap = rows + [_hrow("B", "bp35", "real", 1, 1, first=99), _hrow("A", "bp35", "real", 1, 0)]
    assert m.paired_efficiency_vs_baseline(rows_cap)["B"]["real"]["n_paired_games"] == 2

    # Six unanimous games DO reach significance -- the test can fire when the data supports it.
    six = []
    for i, g in enumerate(["cd82", "lf52", "lp85", "sp80", "su15", "vc33"]):
        six.append(_hrow("A", g, "real", 1, 1, first=100 + i))
        six.append(_hrow("B", g, "real", 1, 1, first=50 + i))
    st = m.paired_efficiency_vs_baseline(six)["B"]["real"]["sign_test"]
    assert st["p_value"] < 0.05 and st["n_pairs_nonzero"] == 6


def test_harness_publishes_its_power_ceiling():
    """An underpowered null must be labelled underpowered, not reported as evidence of absence."""
    m = _harness()
    p6 = m.power_ceiling(
        ["cd82", "lf52", "lp85", "sp80", "su15", "vc33"], m.ASSUMED_BASELINE_WIN_GAMES
    )
    assert p6["max_paired_deltas"] == 6
    assert p6["smallest_attainable_two_sided_p"] == 0.03125
    assert p6["clears_0.05_only_if_unanimous"] is True
    p2 = m.power_ceiling(["lp85", "vc33"], m.ASSUMED_BASELINE_WIN_GAMES)
    assert p2["smallest_attainable_two_sided_p"] == 0.5
    assert p2["clears_0.05_only_if_unanimous"] is False

    # The ceiling is computed on the CLICK stratum: the barrier/draw cannot move a nav-only game,
    # so its paired delta is a tie the sign test drops, and counting it would overstate power.
    mixed = m.power_ceiling(["lp85", "vc33", "tu93"], ["lp85", "vc33", "tu93"])
    assert "tu93" in m.NAV_ONLY_GAMES
    assert mixed["n_baseline_win_games_in_corpus"] == 3
    assert mixed["n_baseline_win_click_games"] == 2
    assert mixed["smallest_attainable_two_sided_p"] == 0.5
    assert mixed["smallest_attainable_two_sided_p_pooled_all_strata"] == 0.25


def test_harness_headline_is_the_capability_result_not_the_efficiency_delta():
    """REGRESSION (fatal): a zero-capability graft beside a winning control must be the headline.

    Measured reality: every grafted arm won the IDENTICAL game set as the baseline (n_win_delta
    = 0 in all three conditions) while the just-explore control won a game no Carnot arm won.
    The first write-up led with an unreplicated efficiency delta on games already solved. These
    fields exist so no downstream capstone can aggregate the efficiency number without the
    capability null attached.
    """
    m = _harness()
    rows = [
        _hrow("A", "lp85", "real", 1, 1, first=20),
        _hrow("A", "vc33", "real", 1, 1, first=60),
        _hrow("A", "r11l", "real", 1, 0),
        _hrow("B", "lp85", "real", 1, 1, first=5),
        _hrow("B", "vc33", "real", 1, 1, first=22),
        _hrow("B", "r11l", "real", 1, 0),
        _hrow("E", "lp85", "real", 1, 2, first=30),
        _hrow("E", "vc33", "real", 1, 1, first=64),
        _hrow("E", "r11l", "real", 1, 1, first=14),  # the game NO Carnot arm wins
    ]
    games = ["lp85", "vc33", "r11l"]
    agg = m.aggregate(rows, games)
    cmp_ = m.compare_to_baseline(agg, games, rows)
    cap = m.capability_summary(agg, cmp_)
    assert cap["new_wins_vs_baseline"] == 0, "the graft transferred no capability"
    assert cap["positive_control_new_wins"] == 1
    assert cap["games_won_only_by_positive_control"] == ["r11l"]
    assert "r11l" not in cap["new_win_games_vs_baseline"]
    assert "instrument what the reference does differently" in cap["diagnostic_target"]


def test_harness_acceptance_gates_are_comparative_and_fail_the_measured_result():
    """The old gate (first-win rate >= 0.12) is passed by the BASELINE -- it measures nothing.

    0.12 x 25 = 3 wins required; arm A already wins ~7 of 25 (rate 0.28). A gate the negative
    control clears cannot separate the intervention from doing nothing. The replacements are
    comparative, and the measured result must FAIL them (0 new wins; sign test underpowered).
    """
    m = _harness()
    cap = {
        "available": True,
        "new_wins_vs_baseline": 0,
        "lost_wins_vs_baseline": [],
        "positive_control_new_wins": 1,
    }
    paired = {"B": {"real": {"n_paired_games": 2, "sign_test": {"p_value": 0.5}}}}
    power = m.power_ceiling(["lp85", "vc33"], m.ASSUMED_BASELINE_WIN_GAMES)
    gates = m.acceptance_gates(cap, paired, power)
    assert gates["acceptance_gate_capability"]["passed"] is False
    assert gates["acceptance_gate_efficiency"]["passed"] is False, "p=0.5 at n=2 must not pass"
    assert gates["acceptance_gate_efficiency"]["min_n_required"] == 6
    assert gates["acceptance_gates_all_passed"] is False

    # And they PASS on a result that genuinely earns it.
    good_cap = dict(cap, new_wins_vs_baseline=1)
    good_paired = {"B": {"real": {"n_paired_games": 6, "sign_test": {"p_value": 0.03125}}}}
    good = m.acceptance_gates(good_cap, good_paired, power)
    assert good["acceptance_gate_capability"]["passed"] is True
    assert good["acceptance_gate_efficiency"]["passed"] is True
    assert good["acceptance_gates_all_passed"] is True

    # A new win bought by losing a baseline win does NOT pass the capability gate.
    traded = m.acceptance_gates(
        dict(cap, new_wins_vs_baseline=1, lost_wins_vs_baseline=["lp85"]), good_paired, power
    )
    assert traded["acceptance_gate_capability"]["passed"] is False


def test_harness_regression_guard_is_derived_from_measured_rows_not_a_hardcoded_claim():
    """The guard list must be self-consistent with the baseline it guards.

    The constant was previously commented "measured at budget 2000" with no artifact behind it
    (the nearest real artifact is a different policy with a different 15-game win set), which is
    the fabrication-adjacent shape CLAUDE.md's Adversarial Artifact Verification discipline
    targets. It is now derived from arm A's own rows, and the fallback is labelled ASSUMED.
    """
    m = _harness()
    rows = [
        _hrow("A", "lp85", "real", 1, 1, first=20),
        _hrow("A", "vc33", "real", 1, 0),
        _hrow("A", "cd82", "real", 1, 1, first=1747),
    ]
    guard, prov = m._guard_games_from_rows(rows)
    assert guard == ["cd82", "lp85"], "only games arm A actually won in THIS run"
    assert prov.startswith("derived_from_arm_A")

    empty_guard, empty_prov = m._guard_games_from_rows([])
    assert empty_guard == list(m.ASSUMED_BASELINE_WIN_GAMES)
    assert empty_prov.startswith("ASSUMED_fallback")
    assert not hasattr(m, "BASELINE_WIN_GAMES"), "the unsourced '(measured)' constant must be gone"

    cmp_ = m.compare_to_baseline(
        m.aggregate(rows, ["lp85", "vc33", "cd82"]), ["lp85", "cd82"], rows
    )
    assert cmp_["regression_guard_games"] == ["cd82", "lp85"]
    assert cmp_["regression_guard_provenance"].startswith("derived_from_arm_A")


def test_harness_levels_banked_total_is_flagged_as_not_arm_comparable():
    """Arms A-D early-stop at the first level-up; arm E runs to budget. Do not compare the sums.

    Measured on the same 3 games: A-D = 2, E = 4 -- half that gap is the stopping rule, not
    capability. The flag has to travel with the number or a capstone will compare it.
    """
    m = _harness()
    rows = [
        dict(_hrow("A", "lp85", "real", 1, 1, first=20), levels_capped_by_early_stop=True),
        dict(_hrow("E", "lp85", "real", 1, 2, first=30), levels_capped_by_early_stop=False),
    ]
    agg = m.aggregate(rows, ["lp85"])
    assert agg["A|real"]["levels_capped_by_early_stop"] is True
    assert agg["E|real"]["levels_capped_by_early_stop"] is False
    assert agg["A|real"]["levels_banked_total_cross_arm_comparable"] is False
    assert agg["E|real"]["levels_banked_total_cross_arm_comparable"] is False


def test_harness_scope_and_verdict_distinguish_a_smoke_from_the_full_run():
    """A 3-game smoke at 1/5 the budget must NOT carry a verdict that says 'measured'."""
    m = _harness()
    full = m.run_scope(m.ALL_GAMES, tuple(m.ARMS), m.CONDITIONS, m.DEFAULT_BUDGET)
    smoke = m.run_scope(("lp85", "vc33", "r11l"), tuple(m.ARMS), m.CONDITIONS, 400)
    assert full["full_declared_spec"] is True
    assert smoke["full_declared_spec"] is False and smoke["n_games"] == 3

    cap = {"available": True, "new_wins_vs_baseline": 0, "positive_control_new_wins": 1}
    v_full = m.verdict_for(full, cap, positive_control_ran=True, error_rate=0.0)
    v_smoke = m.verdict_for(smoke, cap, positive_control_ran=True, error_rate=0.0)
    assert v_full.startswith("complete_") and "measured" in v_full
    assert v_smoke.startswith("partial_") and "not_full_spec" in v_smoke
    # Both must carry the capability result in the verdict itself.
    for v in (v_full, v_smoke):
        assert "graft_new_wins_0" in v and "control_new_wins_1" in v
    # No positive control -> uninterpretable, whatever the scope.
    assert "uninterpretable" in m.verdict_for(full, cap, positive_control_ran=False, error_rate=0.0)


def test_harness_spec_deviations_declare_the_tier_advancement_substitution():
    """A null on B/B2/D must not be written up as 'just-explore does not transfer'.

    The reference advances its priority group on UNREACHABILITY FROM THE CURRENT NODE
    (graph_explorer._maybe_advance_group: `while distance == INFINITY`); the graft advances only
    on GLOBAL set-exhaustion, which is STRICTER. The substitution is defensible (Carnot can
    always RESET-replay, so no distance is ever infinite) but it was missing from the artifact's
    spec_deviations, so a flat result would have been attributed to the reference mechanism
    rather than to this stricter variant.
    """
    m = _harness()
    art = m.run(
        games=[],
        arms=["A"],
        conditions=["real"],
        budget=1,
        n_seeds=1,
        artifact_path=Path(m.REPO) / "results" / "_test_5836_deviations_only.json",
        replay_limit=0,
    )
    joined = " ".join(
        f"{d.get('spec', '')} {d.get('actual', '')} {d.get('why', '')} "
        f"{d.get('consequence_for_interpretation', '')}"
        for d in art["spec_deviations"]
    ).lower()
    assert "_maybe_advance_group" in joined or "maybe_advance_group" in joined
    assert "global set-exhaustion" in joined
    assert "does not falsify the reference mechanism" in joined
    assert "top-k" in joined or "top_k" in joined, "the draw deviation must be declared"
    assert "reset" in joined, "arm E's reset-count convention must be declared"
    assert "0.12" in joined, "the retired non-discriminative gate must be recorded"
    (Path(m.REPO) / "results" / "_test_5836_deviations_only.json").unlink(missing_ok=True)


def test_harness_recompute_derived_reanalyses_without_inventing_a_measurement():
    """`--recompute` may only re-derive analysis from measured rows -- never fabricate one.

    The measured rows are the expensive, irreplaceable part; the aggregates/paired stats/gates
    are pure functions of them. This asserts (a) an artifact with no rows is returned untouched
    (no synthesized numbers), (b) the derived sections and the verdict are rebuilt from the rows,
    and (c) the reproducibility checksum -- computed over rows + config -- is UNCHANGED by a
    recompute, since the measurement did not change.
    """
    m = _harness()
    assert m.recompute_derived({"experiment": 5836}) == {"experiment": 5836}
    assert m.recompute_derived({"per_cell_rows": []})["per_cell_rows"] == []

    rows = [
        _hrow("A", "lp85", "real", 1, 1, first=20),
        _hrow("A", "r11l", "real", 1, 0),
        _hrow("B", "lp85", "real", 1, 1, first=5),
        _hrow("B", "r11l", "real", 1, 0),
        _hrow("E", "lp85", "real", 1, 2, first=30),
        _hrow("E", "r11l", "real", 1, 1, first=14),
    ]
    stale = {
        "per_cell_rows": rows,
        "config": {
            "games": ["lp85", "r11l"],
            "arms": ["A", "B", "E"],
            "conditions": ["real"],
            "budget_actions_per_game": 2000,
        },
        "honest_verdict": "complete_stale_claim",
        "new_wins_vs_baseline": 99,
        "duration_s": 123.4,
    }
    before = m._reproducibility_checksum(stale)
    out = m.recompute_derived(dict(stale))

    assert out["new_wins_vs_baseline"] == 0, "recomputed from the rows, not the stale claim"
    assert out["positive_control_new_wins"] == 1
    assert out["honest_verdict"].startswith("partial_"), "1 of 3 conditions -> reduced scope"
    assert out["acceptance_gates_all_passed"] is False
    assert out["derived_sections_recomputed_from_measured_rows"] is True
    assert out["duration_s"] == 123.4, "measured fields must be preserved verbatim"
    assert out["per_cell_rows"] == rows
    assert out["reproducibility_checksum"] == before, (
        "the checksum covers rows+config; a re-analysis of the SAME measurement must not "
        "change it, or a legitimate recompute would look like corpus drift"
    )


def test_harness_preconditions_are_real_observations():
    m = _harness()
    pre = m.check_preconditions()
    by = {p["resource"]: p for p in pre}
    for required in (
        "offline_arcade_environment_files",
        "frontier_discipline_module",
        "live_explorer_flags_wired",
        "llm_proposer_deliberately_absent",
    ):
        assert required in by, required
    assert by["frontier_discipline_module"]["available"] is True
    assert by["live_explorer_flags_wired"]["available"] is True


def test_harness_never_fabricates_arm_e_and_blocks_when_the_reference_is_missing(monkeypatch):
    """Arm E must be recorded as ran:false with a reason, never synthesized."""
    m = _harness()
    monkeypatch.setattr(m, "JE_ROOT", Path("/definitely/not/here"))
    runner, reason = m.load_just_explore_runner()
    assert runner is None
    assert reason.startswith("reference_clone_absent")
    # A cell for arm E with no runner records the absence rather than inventing a number.
    cell = m.run_cell("E", "vc33", budget=1, seed=1, variant=0, reflect=None, je_runner=None)
    assert cell["ran"] is False and "reason" in cell
    assert "levels" not in cell, "an unrun arm must not carry a levels number"


def test_harness_artifact_has_the_required_principle_annotated_fields():
    m = _harness()
    required = (
        "honest_verdict",
        "inference_substrate",
        "verifier_is_oracle",
        "solve_provenance",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
        "preconditions_checked",
    )
    for f in required:
        assert f in m.FIELD_PRINCIPLES, f
        assert len(m.FIELD_PRINCIPLES[f]) > 40, f"{f} principle must explain WHY"
    # The substrate value must be the canonical CLAUDE.md token for a no-LLM live-agent run.
    art = m.run(
        games=[],
        arms=["A"],
        conditions=["real"],
        budget=1,
        n_seeds=1,
        artifact_path=Path(m.REPO) / "results" / "_test_5836_shape_only.json",
        replay_limit=0,
    )
    assert art["inference_substrate"] == "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    assert art["solve_provenance"] == "development_proxy"
    assert art["verifier_is_oracle"] is True, "level-ups come from the env oracle -- disclose it"
    assert art["honest_verdict"].startswith(("complete_", "blocked_"))
    # No games measured -> the positive control never ran -> NOT interpretable, and it says so.
    assert art["ab_interpretable"] is False
    assert "uninterpretable" in art["honest_verdict"]
    assert "offline_reproduced" not in art, (
        "must not emit a solve-claim shape: all 25 public games are already registry-cleared, "
        "so a solve claim would trip the adversarial duplicate check"
    )
    (Path(m.REPO) / "results" / "_test_5836_shape_only.json").unlink(missing_ok=True)


def test_barrier_never_livelocks_the_decision_loop():
    """ADVERSARIAL: the barrier must never leave next_move unable to act OR to terminate.

    The dangerous shape is a gate that makes _frontier return a node while _pop_frontier_batch
    returns nothing for it -- the pending queue stays empty and _serve() raises IndexError -- or
    a state where neither an action nor explored_out is ever produced. This drives the real
    decision loop over a hand-built graph whose ONLY remaining work sits at the top tier, and
    asserts it either emits an action or terminates within a bounded number of turns.
    """
    exp = StepwiseExplorer(tier_exhaustion=True, frontier_gradient=True)
    exp.graph = {
        "ROOT": {"path": [], "untested": [_row(6, tier=4, x=1, y=1)], "value": None},
        "CHILD": {
            "path": [{"action": 1, "data": None}],
            "untested": [_row(6, tier=4, x=2, y=2)],
            "value": None,
        },
    }
    exp.root, exp.cur = "ROOT", "ROOT"
    exp._record_forward_edge("ROOT", {"action": 1, "data": None}, "CHILD")

    acted = 0
    for _ in range(50):
        th = exp._frontier()
        if th is None:
            break
        node = exp.graph[th]
        assert exp._node_has_open_tier(node), "_frontier must never return a fully-deferred node"
        batch = exp._pop_frontier_batch(node)
        assert batch, "a returned frontier node must yield at least one action (else _serve raises)"
        acted += len(batch)
    assert acted == 2, f"both deferred rows must eventually be spent, got {acted}"
    assert exp._frontier() is None, "and then the search must terminate, not spin"
    assert exp._tier_deferrals == 0, "the defensive fail-open path must never have been needed"


def test_every_explorer_arm_pins_all_seven_gated_flags():
    """REQ-ARC-WMTE-5836 / SCENARIO: arm-definition drift cannot silently recontaminate a control.

    WHY THIS EXISTS (2026-07-25). An arm that pins only a SUBSET of the gated flags inherits module
    defaults for the rest, so the moment a flag is flipped that arm's meaning changes underneath
    already-published numbers. It happened: arm A pinned 2 of 7 and arm B2 pinned 3 of 7, so after
    the frontier and HUD flips, arm A stopped being the pre-flip agent and arm B2 -- the HUD A/B's
    own CONTROL -- became the HUD TREATMENT. Any future A/B using them as a control would have been
    measuring a contaminated control. This test makes that class of drift impossible to reintroduce
    quietly: adding a new gated flag, or a new arm, fails here until every arm pins it explicitly.

    Arm E is exempt: it is the just-explore reference shim and constructs no StepwiseExplorer.
    """
    m = _harness()
    seven = set(m.GATED_FLAGS)
    assert len(seven) == 7
    for name, arm in m.ARMS.items():
        if name == "E":
            assert arm["kwargs"] == {}, "the reference shim must take no explorer kwargs"
            continue
        missing = seven - set(arm["kwargs"])
        assert not missing, f"arm {name} inherits defaults for {sorted(missing)}"


def test_pinning_preserved_each_arms_measured_semantics():
    """The pinning change must not REDEFINE any arm whose numbers are already published."""
    m = _harness()
    a, b2 = m.ARMS["A"]["kwargs"], m.ARMS["B2"]["kwargs"]
    # arm A is the pre-flip agent: every lever off
    assert a["tier_exhaustion"] is False and a["tier_uniform_random"] is False
    assert a["frontier_gradient"] is False
    assert a["edge_bar_hud_mask"] is False
    # arm B2 is the frontier configuration that was flipped live, with the HUD levers OFF -- that is
    # what it was when it served as the HUD A/B's control
    assert b2["tier_exhaustion"] is True and b2["tier_uniform_random"] is True
    assert b2["tier_click_vocab_only"] is True and b2["frontier_gradient"] is False
    assert b2["edge_bar_hud_mask"] is False
    # G3 is the flip-candidate HUD arm: frontier on plus all three HUD stages
    g3 = m.ARMS["G3"]["kwargs"]
    assert g3["edge_bar_hud_mask"] is True
    assert g3["hud_mask_collapse_guard"] is True and g3["hud_mask_stage2_confirm"] is True
