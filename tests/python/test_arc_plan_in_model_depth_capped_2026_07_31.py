"""REQ-ARC-WMTE-6047-E: `plan_in_model` reports a depth-capped search as depth-capped.

WHY THIS FILE EXISTS. On 2026-07-31 the goal gate
(`arc_llm_reinduction._goal_satisfiability_check`) had its depth axis split out of
`queue_exhausted` (REQ-ARC-WMTE-6047-D) after it mislabelled tn36's measured-reachable win
condition as `degenerate_goal_predicate`. The adversarial review of that fix pointed out that
**the identical conflation still lived in the function the gate guards** -- `plan_in_model` closed
both of its search loops with

    "max_nodes_reached" if nodes >= max_nodes else "queue_exhausted"

while the loop body above did `if len(path) >= max_depth: continue`, discarding a popped node
WITHOUT EXPANDING IT. So the frontier could drain purely because the cap threw work away, and the
diagnostics would call that "queue_exhausted" -- i.e. "I searched the reachable set and there is no
plan."

The proof that this is not hypothetical was already sitting in the sibling fix's own evidence:
`results/arc_goal_gate_depth_20260731/tn36_depth_label.json` records

    plan_at_max_depth_40.diagnostics.termination_reason == "queue_exhausted"

for the tn36 engine + goal + root whose plan the same function FINDS at `max_depth=80`, 61 actions
long. The label was wrong on the real cell, in the real function, on the real artifact.

WHAT IS AND IS NOT CHANGED. `plan_in_model` returns exactly what it returned before on every path:
the same plan, or the same `None`. Only `diagnostics` moved -- a third `termination_reason` value
`depth_capped`, and a `depth_truncated_nodes` count. Nothing in the tree branches on the string
(checked: the only readers are `scripts/arc_plan_in_model_nav_solve.py`, which formats it for
display, and the `ttt_prior_engine_plan_diagnostics` blob the agent stores verbatim), which is why
this is safe to change and also why it was never caught by a behavioural test.

PRECEDENCE. Budget first, then depth, then genuine exhaustion. A search that spent its node budget
is budget-limited whatever else it did -- raising `max_depth` alone would not have helped it,
because it never got to look. Unlike the sibling fix in the gate's own BFS, the two axes here are
NOT mutually exclusive: the best-first loop pops in energy order, so it can drop a deep node early
and then keep spending budget elsewhere. That makes the ordering load-bearing rather than
defensive, and it is pinned below with a search that genuinely does both.

`queue_exhausted` now means what it always claimed: nothing left to search AND nothing thrown away,
so the negative result is real evidence about the engine rather than about the cap.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import _termination_reason, plan_in_model


# --------------------------------------------------------------------------------------
# A deterministic toy world: one cell of a 1xN row is filled per click, left to right.
# The goal is "the whole row is filled", at depth N. This is the SHAPE of tn36 (one cell
# per action against a depth cap), reduced to something that runs in milliseconds.
# --------------------------------------------------------------------------------------


def _chain_engine(grid, action, data):
    """Fill the leftmost unfilled cell. Every action produces a brand-new state."""
    g = np.asarray(grid).copy()
    zeros = np.flatnonzero(g[0] == 0)
    if zeros.size == 0:
        return g
    g[0, int(zeros[0])] = 1
    return g


def _row_complete(grid):
    return bool(np.all(np.asarray(grid)[0] == 1))


def _root(n: int) -> np.ndarray:
    return np.zeros((1, n), dtype=np.int16)


def test_the_helper_orders_budget_over_depth_over_exhaustion():
    """The precedence table, stated directly, so the ordering is falsifiable on its own.

    Asserted as four independent cases rather than one composite, because the interesting
    failure is a SWAP (depth stealing the budget verdict), and a composite assertion that
    happened to pass on three of four would still report green.
    """
    # Budget spent wins even when nodes were also dropped at the cap.
    assert _termination_reason(20000, 20000, 5) == "max_nodes_reached"
    assert _termination_reason(20000, 20000, 0) == "max_nodes_reached"
    # Budget intact + something discarded unexpanded -> the cap is the binding explanation.
    assert _termination_reason(1480, 20000, 1) == "depth_capped"
    # Budget intact + nothing discarded -> the frontier really did empty. Evidence.
    assert _termination_reason(1480, 20000, 0) == "queue_exhausted"


def test_blind_bfs_depth_capped_search_is_not_reported_as_exhausted():
    """The tn36 shape in the goal_energy=None loop: reachable goal, cap too low.

    The load-bearing assertion is NOT just the new string -- it is that `depth_truncated_nodes`
    is non-zero AND the budget is untouched, because that pair is what makes "the frontier
    emptied because I threw work away" distinguishable from "the frontier emptied because there
    was nothing left".
    """
    diag: dict = {}
    plan = plan_in_model(
        _chain_engine,
        _row_complete,
        _root(12),
        max_nodes=20000,
        max_depth=4,
        diagnostics=diag,
    )
    assert plan is None
    assert diag["termination_reason"] == "depth_capped"
    assert diag["depth_truncated_nodes"] >= 1
    # The budget is nowhere near spent -- this is exactly the tn36 arithmetic (1480 << 20000).
    assert diag["nodes_expanded"] < 20000
    assert diag["used_goal_energy_search"] is False


def test_blind_bfs_same_search_with_an_ample_cap_finds_the_plan():
    """The control that makes the previous test mean something: the goal IS reachable.

    Without this, `depth_capped` on an unreachable goal would look identical, and the test above
    would pass for a search that had nothing to find.
    """
    diag: dict = {}
    plan = plan_in_model(
        _chain_engine,
        _row_complete,
        _root(12),
        max_nodes=20000,
        max_depth=40,
        diagnostics=diag,
    )
    assert plan is not None
    assert len(plan) == 12
    assert diag["termination_reason"] == "plan_found"


def test_blind_bfs_a_genuinely_exhausted_frontier_still_says_queue_exhausted():
    """The degenerate verdict must stay EARNABLE, or the fix has just renamed one label to another.

    Here the engine is a no-op, so the single successor state dedups against the root, the frontier
    drains at depth 1, and nothing is ever discarded at the cap. That is a real exhaustive search
    of a real (tiny) reachable set, and the negative result is real evidence.
    """

    def _noop_engine(grid, action, data):
        return np.asarray(grid).copy()

    diag: dict = {}
    plan = plan_in_model(
        _noop_engine,
        _row_complete,
        _root(12),
        max_nodes=20000,
        max_depth=40,
        diagnostics=diag,
    )
    assert plan is None
    assert diag["termination_reason"] == "queue_exhausted"
    assert diag["depth_truncated_nodes"] == 0


def test_blind_bfs_a_spent_budget_still_says_max_nodes_reached():
    """Budget keeps priority: the new counter must not be able to steal its verdict."""
    diag: dict = {}
    plan = plan_in_model(
        _chain_engine,
        _row_complete,
        _root(64),
        max_nodes=3,
        max_depth=2,
        diagnostics=diag,
    )
    assert plan is None
    assert diag["termination_reason"] == "max_nodes_reached"


def test_goal_energy_best_first_loop_gets_the_same_split():
    """The best-first branch is a SEPARATE loop with its own copy of the two-way string.

    Fixing only the blind-BFS loop would leave the live path -- which supplies a goal_energy
    whenever one was induced -- reporting the old mislabel. This is the mutation that a
    single-loop fix would survive.
    """

    def _energy(g):
        return float(np.count_nonzero(np.asarray(g)[0] == 0))

    diag: dict = {}
    plan = plan_in_model(
        _chain_engine,
        _row_complete,
        _root(12),
        max_nodes=20000,
        max_depth=4,
        goal_energy=_energy,
        diagnostics=diag,
    )
    assert plan is None
    assert diag["used_goal_energy_search"] is True
    assert diag["termination_reason"] == "depth_capped"
    assert diag["depth_truncated_nodes"] >= 1


def test_best_first_can_spend_budget_after_a_depth_drop_so_precedence_is_load_bearing():
    """Unlike the gate's plain BFS, here the two axes genuinely co-occur.

    The sibling fix documented that under a plain FIFO BFS truncation begins only at the deepest
    layer, so no budget can be spent after the first drop and the precedence ordering is merely
    defensive. That reasoning does NOT carry over: this loop pops in energy order, so a deep node
    can be dropped while shallower work remains. This test constructs exactly that state -- both
    `depth_truncated_nodes > 0` AND the budget spent -- and pins that BUDGET wins.

    A branching engine is required (the chain engine has one successor per state, so it can never
    have both a deep node and shallow work outstanding).
    """

    def _branch_engine(grid, action, data):
        g = np.asarray(grid).copy()
        # Each action writes a distinct value into a distinct cell, so states fan out.
        idx = int(action) % g.shape[1]
        g[0, idx] = (int(g[0, idx]) + 1 + int(action)) % 7 + 1
        return g

    def _never(grid):
        return False

    def _energy(g):
        # Prefers deep/dense states, so the heap dives before it broadens -- which is what puts a
        # node at the cap while plenty of shallower frontier is still queued.
        return -float(np.count_nonzero(np.asarray(g)[0]))

    diag: dict = {}
    plan = plan_in_model(
        _branch_engine,
        _never,
        _root(8),
        max_nodes=100,
        max_depth=3,
        goal_energy=_energy,
        diagnostics=diag,
    )
    assert plan is None
    # Both conditions hold at once...
    assert diag["depth_truncated_nodes"] > 0
    assert diag["nodes_expanded"] >= 100
    # ...and the budget is the verdict.
    assert diag["termination_reason"] == "max_nodes_reached"


def test_depth_truncated_nodes_is_always_populated_on_both_loops():
    """A counter that is present only sometimes is worse than absent: a reader cannot tell a
    zero from a missing field, and `.get(k, 0)` silently turns "not measured" into "measured
    zero". Every exit that ran a search must write it -- including `plan_found`, which the first
    draft of this fix omitted and this test caught. (`is_level_complete_none` is deliberately NOT
    covered: it returns before any search exists, so there is nothing to have truncated.)"""
    for kwargs in ({}, {"goal_energy": lambda g: 0.0}):
        for max_depth, expect_plan in ((40, True), (4, False)):
            diag: dict = {}
            plan = plan_in_model(
                _chain_engine,
                _row_complete,
                _root(12),
                max_nodes=20000,
                max_depth=max_depth,
                diagnostics=diag,
                **kwargs,
            )
            assert (plan is not None) is expect_plan
            assert "depth_truncated_nodes" in diag


def test_diagnostics_none_is_still_accepted_and_the_return_is_unchanged():
    """The whole change is inside `if diagnostics is not None`. Callers that pass nothing --
    which is the default, and most of the tree -- must be untouched, and the plan returned with
    diagnostics must be identical to the plan returned without."""
    with_diag: dict = {}
    a = plan_in_model(_chain_engine, _row_complete, _root(6), max_depth=40, diagnostics=with_diag)
    b = plan_in_model(_chain_engine, _row_complete, _root(6), max_depth=40)
    assert a is not None and b is not None
    assert len(a) == len(b) == 6
