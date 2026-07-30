"""REQ-ARC-WMTE-6047: a spent search budget is not a degenerate goal predicate.

THE INCIDENT (2026-07-30 review of the 2026-07-29 counter fix).

`_goal_satisfiability_check` is a pre-veto: it exists to reject an induced `is_level_complete`
that the planner could never satisfy. It stops for two reasons that mean OPPOSITE things:

  * the frontier empties            -> the reachable set was searched out; the goal is unreachable
  * `engine_calls >= max_nodes`     -> the BUDGET ran out; NOTHING was learned about the goal

Both returned `counterexample.kind == "degenerate_goal_predicate"`, so they were
indistinguishable to the caller. That was survivable only while the gate's budget was ~11x more
permissive than the planner's -- the gate counted UNIQUE GRIDS, `plan_in_model` counted RAW
ENGINE CALLS -- because the gate essentially never hit its ceiling first.

The 2026-07-29 counter fix made the two units consistent, which is correct, and in doing so made
this conflation LIVE AND SEVERE. Measured on ka59's change-fidelity-1.0000 world model (git
341f776c9): the concept-correct depth-11 win predicate needs ~137,347 engine calls to demonstrate,
and the shipped budget is 20,000. So the gate returns "degenerate" on a PROVABLY CORRECT goal --
and the caller's GOAL-REPAIR then substitutes an exemplar-derived "strictly fuller than root"
proxy and sets `round_goal_satisfiable = True`. The agent proceeds to plan, successfully, toward a
goal that IS NOT WINNING. A compute ceiling silently became a goal rewrite, and the artifact
reports a satisfiable goal predicate throughout.

All 18 occurrences in the historical corpus were genuine frontier exhaustion, so no recorded
result is invalidated by separating the two; every occurrence from the counter fix onward would
have been the budget case.

WHAT THESE TESTS PIN.

1. Budget exhaustion reports `goal_unreached_within_budget` + `termination: budget_exhausted`,
   and a genuinely unreachable goal still reports `degenerate_goal_predicate` +
   `termination: queue_exhausted`. The two must never collapse again.
2. The SAME predicate on the SAME engine flips kind purely on `max_nodes` -- which is the
   ka59 shape, and the thing that makes the bug a compute artifact rather than a goal defect.
3. "Frontier empty" is not trusted when the budget is what stopped the search, because the inner
   loop discards the current grid's remaining candidate expansions on `break` and they are never
   queued -- so an empty deque at that moment does not mean the successors were explored.
4. Depth-limited exhaustion is still reported as `queue_exhausted` (vetoing on
   unreachable-within-depth is sound: the planner it guards is bounded the same way).
"""

import numpy as np

from carnot.agentic import arc_llm_reinduction as reinduction

# A GROWING-BAR world. Action 1 paints the next cell of a bar; the goal is "the bar is TARGET
# cells long". Reaching it takes exactly TARGET expansions, so the depth at which the goal becomes
# true is fixed by the board and the budget can be dialled either side of it. That reproduces the
# ka59 shape -- a CORRECT goal that is simply DEEP -- without a 64x64 board or a real induced
# engine.
#
# A counter in a single cell was the first attempt and it does NOT work: the gate dedups on
# `to_ascii`, whose key is the grid mod 10, so a counter wraps at 10 and the search terminates at
# depth 9 reporting `queue_exhausted`. Depth 11 was unreachable for a reason that had nothing to
# do with the property under test. The bar keeps every state distinct under that key.
_TARGET = 11
_PAINT = 4


def _make_engine():
    """An engine whose only meaningful action extends the bar by one cell."""

    def engine(grid: np.ndarray, action: int, data=None) -> np.ndarray:
        out = np.array(grid, dtype=int, copy=True)
        if action == 1:
            painted = int((out[0] == _PAINT).sum())
            if painted < out.shape[1]:
                out[0, painted] = _PAINT
        return out

    return engine


def _start(width: int = _TARGET + 3) -> np.ndarray:
    return np.zeros((1, width), dtype=int)


def _goal_at(target: int):
    return lambda g: int((np.asarray(g)[0] == _PAINT).sum()) == int(target)


def _kind(result: dict) -> str:
    return str((result.get("counterexample") or {}).get("kind", ""))


def test_budget_exhaustion_is_not_reported_as_a_degenerate_goal() -> None:
    """The ka59 shape: a REACHABLE goal, too deep for the budget, must not be called degenerate."""
    result = reinduction._goal_satisfiability_check(
        engine=_make_engine(),
        goal=_goal_at(_TARGET),
        start_grid=_start(),
        max_nodes=8,
        max_depth=64,
    )
    assert result["satisfiable"] is False
    assert _kind(result) == "goal_unreached_within_budget", (
        "a spent budget must report its own kind; reporting `degenerate_goal_predicate` is what "
        "triggered GOAL-REPAIR on ka59's correct depth-11 predicate"
    )
    assert result["termination"] == "budget_exhausted"
    assert result["engine_calls"] >= 8


def test_genuinely_unreachable_goal_still_reports_degenerate_with_ample_budget() -> None:
    """The split must not blunt the real veto: unreachable-and-searched-out stays degenerate."""
    result = reinduction._goal_satisfiability_check(
        engine=_make_engine(),
        goal=lambda g: False,
        start_grid=_start(),
        max_nodes=10_000,
        max_depth=4,
    )
    assert result["satisfiable"] is False
    assert _kind(result) == "degenerate_goal_predicate"
    assert result["termination"] == "queue_exhausted"


def test_same_predicate_same_engine_flips_kind_on_budget_alone() -> None:
    """A goal's KIND must not depend on how much compute we happened to grant.

    This is the property that makes the ka59 incident diagnosable: identical goal, identical
    engine, identical board -- only `max_nodes` differs, and it decides whether the agent keeps
    its correct predicate or has it rewritten.
    """
    engine, goal, board = _make_engine(), _goal_at(_TARGET), _start()
    starved = reinduction._goal_satisfiability_check(
        engine=engine, goal=goal, start_grid=board, max_nodes=6, max_depth=64
    )
    generous = reinduction._goal_satisfiability_check(
        engine=engine, goal=goal, start_grid=board, max_nodes=10_000, max_depth=64
    )
    assert _kind(starved) == "goal_unreached_within_budget"
    assert starved["satisfiable"] is False
    # With budget, the same predicate is satisfiable at exactly its true depth.
    assert generous["satisfiable"] is True
    assert generous["first_true_depth"] == _TARGET


def test_empty_frontier_is_not_trusted_when_the_budget_is_what_stopped_us() -> None:
    """`frontier_remaining == 0` must not upgrade a budget stop into an exhaustiveness claim.

    The inner loop `break`s on budget and DISCARDS the current grid's remaining candidates without
    queueing them, so the deque can be empty while successors went unexplored. Keying the verdict
    on `q` being non-empty (rather than on the budget alone) would misreport exactly that case.
    A single-successor world with a budget of 1 produces it: one call, the goal is not yet true,
    the successor IS queued -- so we force the harder shape with max_depth=1, where the queued
    node is then dropped by the depth cap and the deque drains.
    """
    result = reinduction._goal_satisfiability_check(
        engine=_make_engine(),
        goal=_goal_at(_TARGET),
        start_grid=_start(),
        max_nodes=1,
        max_depth=1,
    )
    assert result["satisfiable"] is False
    assert result["termination"] == "budget_exhausted", (
        "the budget is what stopped the search, so the verdict must say so regardless of how "
        "many nodes happen to remain queued"
    )
    assert _kind(result) == "goal_unreached_within_budget"


def test_depth_limited_exhaustion_reports_queue_exhausted() -> None:
    """Unreachable WITHIN max_depth is a sound veto and keeps the degenerate kind."""
    result = reinduction._goal_satisfiability_check(
        engine=_make_engine(),
        goal=_goal_at(_TARGET),
        start_grid=_start(),
        max_nodes=10_000,
        max_depth=3,
    )
    assert result["satisfiable"] is False
    assert result["termination"] == "queue_exhausted"
    assert _kind(result) == "degenerate_goal_predicate"
    assert result["engine_calls"] < 10_000


def test_budget_starved_gate_does_not_rewrite_the_goal_end_to_end() -> None:
    """THE ka59 INCIDENT, end to end: a starved gate must not hand the agent a different goal.

    This is the test that would have caught the live bug. It drives the real
    `execute_bounded_llm_reinduction` with a CORRECT-but-deep goal and an exemplar available, so
    GOAL-REPAIR *could* fire. Before the fix the gate reported `degenerate_goal_predicate`, the
    repair substituted the "strictly fuller than root" proxy, and `planned` came back True against
    a NON-WINNING goal -- a false success. After the fix the round proceeds with the ORIGINAL
    predicate untouched.

    The world branches 5 ways per step and the goal sits at depth 7, so breadth-first exploration
    must cover ~19.5k distinct states (5+25+...+15625) before it can reach the goal -- comfortably
    past the shipped default budget of 20,000 ENGINE CALLS. That is the same shape as ka59, where
    the correct depth-11 predicate needs ~137k calls against the same 20k ceiling.
    """
    width = 8
    target_depth = 7

    def engine(grid: np.ndarray, action: int, data=None) -> np.ndarray:
        out = np.array(grid, dtype=int, copy=True)
        if 1 <= int(action) <= 5:
            filled = int((out[0] != 0).sum())
            if filled < out.shape[1]:
                out[0, filled] = int(action)
        return out

    # Reachable ONLY as the all-fives sequence of length 7: correct, specific, and deep.
    def correct_goal(grid: np.ndarray) -> bool:
        row = np.asarray(grid)[0]
        return bool(int((row != 0).sum()) == target_depth and bool((row[:target_depth] == 5).all()))

    root = np.zeros((1, width), dtype=int)

    class _Proposer:
        model = "fixture"

        def induce(self, _game, _transitions, _cell):
            return True, "candidate"

        def refactor(self, _game, _counterexample):
            return True, "refined"

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    transitions = [
        Transition(
            grid=root.copy(),
            action=5,
            data=None,
            next_grid=engine(root, 5, None),
            level_before=1,
            level_after=1,
        )
    ]

    result = execute_bounded_llm_reinduction(
        game="budget_fixture",
        transitions=transitions,
        cell=1,
        root_grid=root,
        proposer=_Proposer(),
        candidate_provider=lambda e, g: [("c", e, g)],
        load_engine=lambda _game: (engine, correct_goal),
        # A planner that cannot find the depth-7 solution either (it only looks one step ahead),
        # so the ONLY thing this test can observe is whether the GOAL was rewritten.
        plan_in_model=lambda e, g, s: None,
        max_rounds=1,
        # An exemplar IS available, so GOAL-REPAIR is armed. That is the point: the fix must be
        # "the gate did not fire", not "the repair had nothing to work with".
        previous_level_complete_grid=np.array([[5, 5, 5, 5, 5, 5, 5, 0]], dtype=int),
    )

    row0 = result.rounds[0]
    assert row0["goal_undecided_within_budget"] is True, (
        "the gate must record that it ran out of budget rather than deciding"
    )
    assert "goal_repaired" not in row0, (
        "GOAL-REPAIR must NOT fire on a spent budget: substituting a looser proxy here is how a "
        "correct predicate got replaced by a non-winning one on ka59"
    )
    # AND THE ROUND IS SKIPPED -- the conservative resolution of "undecided", reverted here on
    # 2026-07-30 review from an earlier version that fell through to the planner.
    #
    # Falling through is the PERMISSIVE answer: a goal that failed the pre-veto reaches the planner
    # anyway, which relaxes `degenerate_goal_predicate` -- a named quality gate that may only be
    # widened with operator authorisation, and was not. Skipping is at least as strict as the
    # pre-split behaviour on the accept axis and strictly stricter on the rewrite axis (the two
    # assertions above), so it cannot be read as widening anything.
    #
    # The cost is real and is the point of reporting it: at the shipped budget this correct-but-deep
    # goal never gets planned against, exactly as ka59's depth-11 predicate does not. That is a
    # COMPUTE result, and `skipped`/`termination` name it as one instead of laundering it into
    # either a goal rewrite or a false solve.
    assert row0.get("skipped", "") == "goal_unreached_within_budget", (
        "an undecided goal must skip the round rather than reach the planner: falling through "
        "relaxes the degenerate-goal veto, which is a quality gate and operator-gated"
    )
    # The goal is never REWRITTEN -- skipping discards the round, it does not substitute a proxy.
    assert result.goal_predicate is not None
    assert getattr(result, "goal_repaired", None) in (None, "", False)
    # The undecided gate IS recorded in the audit trail, distinguishably from a degenerate one.
    kinds = [str(c.get("kind", "")) for c in result.counterexamples]
    assert "goal_unreached_within_budget" in kinds
    assert "degenerate_goal_predicate" not in kinds, (
        "a spent budget must never be recorded as a disproved predicate"
    )
    assert row0["counterexample"]["kind"] == "goal_unreached_within_budget"
    assert row0["counterexample"]["termination"] == "budget_exhausted"


def test_budget_case_carries_an_explicit_unknown_disclaimer() -> None:
    """The counterexample text must say the result is UNKNOWN, not negative.

    `refactor()` and every artifact reader consume this string. "Degenerate" sends the next round
    after the goal predicate; "unknown" must not, because nothing about the predicate was learned.
    """
    result = reinduction._goal_satisfiability_check(
        engine=_make_engine(),
        goal=_goal_at(_TARGET),
        start_grid=_start(),
        max_nodes=4,
        max_depth=64,
    )
    detail = str((result.get("counterexample") or {}).get("detail", "")).lower()
    assert "unknown" in detail
    assert "not evidence" in detail
