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
    target = fd.gradient_frontier_target(
        forward, None, ["SHALLOW_OPEN", "NEAR_OPEN"], "CUR"
    )
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

    The submitted agent's search order must be unchanged until the A/B greenlights a flip, so
    with the flags off _pop_untested is still an unconditional pop(0) even over rows whose
    stamped tiers would otherwise defer them.
    """
    exp = StepwiseExplorer()
    assert exp.tier_exhaustion_enabled is False
    assert exp.tier_uniform_random_enabled is False
    assert exp.frontier_gradient_enabled is False

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
    exp = StepwiseExplorer(tier_exhaustion=True)
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
    exp = StepwiseExplorer(tier_exhaustion=True)
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
    exp = StepwiseExplorer(tier_exhaustion=True, frontier_batch_size="all")
    node = {
        "path": [],
        "untested": [_row(6, tier=4, x=1, y=1), _row(6, tier=0, x=2, y=2), _row(6, tier=0, x=3, y=3)],
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
        "NEAR": {"path": [{"action": 1, "data": None}] * 3, "untested": [_row(6, x=2, y=2)],
                 "value": None},
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


def test_env_flags_toggle_the_mechanisms_without_mutating_module_globals(monkeypatch):
    """The A/B harness flips arms by env var; module globals must stay at their OFF defaults."""
    import carnot.agentic.arc_competition_agent as agent_mod

    monkeypatch.setenv("CARNOT_ARC_FRONTIER_TIER_EXHAUSTION", "1")
    monkeypatch.setenv("CARNOT_ARC_FRONTIER_GRADIENT", "1")
    exp = StepwiseExplorer()
    assert exp.tier_exhaustion_enabled is True
    assert exp.frontier_gradient_enabled is True
    assert agent_mod.SUBMITTED_FRONTIER_TIER_EXHAUSTION_ENABLED is False
    assert agent_mod.SUBMITTED_FRONTIER_DISTANCE_GRADIENT_ENABLED is False
    # an explicit kwarg still wins over the env var
    assert StepwiseExplorer(tier_exhaustion=False).tier_exhaustion_enabled is False
