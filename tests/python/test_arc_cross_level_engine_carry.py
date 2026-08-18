"""REQ-ARC-XLEVEL-CARRY-1 (2026-08-17, default OFF): cross-level verified engine carry.

On a level-up reinduction, before paying for a fresh LLM induction, the policy may
re-verify the PREVIOUS level's induced engine against the NEW level's own transitions and
plan in it when it clears the same accuracy bar a fresh induction must clear. Four
contracts pinned here, mirroring test_arc_trajectory_transfer_cascade.py:

1. INERTNESS: flag unset (the shipped default) -> no carry key on the attempt row, the
   LLM tier is reached exactly as before, and no retained state is consumed.
2. FIRE: flag on + a carried engine that predicts the new level's transitions + a goal
   that is false at start and reachable -> plan installed, LLM call skipped.
3. VERIFICATION GATE: flag on + a carried engine that CONTRADICTS the new level's
   transitions -> the carry declines with a recorded reason and the LLM tier still runs.
4. RETENTION: a reinduction outcome with an engine populates the carried slots; a failed
   outcome (engine None) must NOT evict a previously carried engine.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_executable_world_model import Transition


def _policy() -> E3AgentPolicy:
    return E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)


def _shift_right_engine(grid, action, data=None):
    """Toy dynamics: action 1 moves the single colored cell one column right."""
    g = np.asarray(grid).copy()
    if int(action) != 1:
        return g
    ys, xs = np.nonzero(g)
    if len(ys) != 1 or xs[0] + 1 >= g.shape[1]:
        return g
    color = g[ys[0], xs[0]]
    g[ys[0], xs[0]] = 0
    g[ys[0], xs[0] + 1] = color
    return g


def _goal_rightmost(grid) -> bool:
    g = np.asarray(grid)
    ys, xs = np.nonzero(g)
    return len(xs) == 1 and int(xs[0]) == g.shape[1] - 1


def _consistent_transitions(n: int = 4) -> list[Transition]:
    """Transitions the shift-right engine predicts exactly (accuracy 1.0)."""
    rows = []
    for i in range(n):
        g = np.zeros((1, 6), dtype=np.int16)
        g[0, i] = 5
        rows.append(
            Transition(
                grid=g,
                action=1,
                data=None,
                next_grid=_shift_right_engine(g, 1),
                level_before=1,
                level_after=1,
            )
        )
    return rows


def _contradicting_transitions(n: int = 4) -> list[Transition]:
    """Transitions where action 1 did NOTHING -> the shift-right engine fails verification."""
    rows = []
    for i in range(n):
        g = np.zeros((1, 6), dtype=np.int16)
        g[0, i] = 5
        rows.append(
            Transition(grid=g, action=1, data=None, next_grid=g, level_before=1, level_after=1)
        )
    return rows


def _llm_stub(called: dict):
    def _fake(**kwargs):
        called["llm"] = True
        return SimpleNamespace(
            model_specs="test",
            planned=False,
            skipped="test_stub",
            plan=[],
            reinduce_attempts=0,
            defects=[],
            goal_defects=[],
        )

    return _fake


def _arm_level_up(policy, transitions) -> None:
    policy.transitions = list(transitions)
    policy._episode_transition_start = 0
    policy._pending_induction_reason = "level_up_reinduction"
    # A start grid the toy goal is FALSE at (cell at column 0 of 6).
    start = np.zeros((1, 6), dtype=np.int16)
    start[0, 0] = 5
    policy.root_grid = start


# --------------------------------------------------------------------------- #
# (1) inertness: shipped default records nothing and changes nothing          #
# --------------------------------------------------------------------------- #
def test_bare_default_is_off_and_byte_identical(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY", raising=False)
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "0")
    assert agent.SUBMITTED_CROSS_LEVEL_ENGINE_CARRY_ENABLED is False
    called = {"llm": False}
    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _llm_stub(called))

    policy = _policy()
    assert policy.cross_level_engine_carry_enabled is False
    # Even with a perfectly carried engine available, the OFF path must not consume it.
    policy._carried_engine = _shift_right_engine
    policy._carried_goal = _goal_rightmost
    _arm_level_up(policy, _consistent_transitions())

    policy._induce_and_plan()

    assert called["llm"] is True
    attempt = policy.induction_attempts[-1]
    assert "cross_level_engine_carry" not in attempt
    assert attempt.get("engine_source") != "cross_level_engine_carry"


# --------------------------------------------------------------------------- #
# (2) fire: verified carry plans and skips the LLM                            #
# --------------------------------------------------------------------------- #
def test_flag_on_verified_carry_short_circuits_before_llm(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY", "1")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "0")

    def _must_not_reach(**kwargs):
        raise AssertionError("verified carry must skip the LLM tier")

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _must_not_reach)

    policy = _policy()
    assert policy.cross_level_engine_carry_enabled is True
    policy._carried_engine = _shift_right_engine
    policy._carried_goal = _goal_rightmost
    policy._carried_engine_meta = {"induced_at_goal_level": 1}
    _arm_level_up(policy, _consistent_transitions())

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    carry = attempt["cross_level_engine_carry"]
    assert carry["fired"] is True
    assert carry["verify_accuracy"] == 1.0
    assert attempt["planned"] is True
    assert attempt["engine_source"] == "cross_level_engine_carry"
    # The plan drives the cell from column 0 to column 5: five shift-right steps.
    assert [int(step["action"]) for step in policy.plan] == [1] * 5


# --------------------------------------------------------------------------- #
# (3) verification gate: a contradicted engine declines, LLM still runs       #
# --------------------------------------------------------------------------- #
def test_flag_on_failed_verification_falls_through_to_llm(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY", "1")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "0")
    called = {"llm": False}
    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _llm_stub(called))

    policy = _policy()
    policy._carried_engine = _shift_right_engine
    policy._carried_goal = _goal_rightmost
    _arm_level_up(policy, _contradicting_transitions())

    policy._induce_and_plan()

    assert called["llm"] is True
    attempt = policy.induction_attempts[-1]
    carry = attempt["cross_level_engine_carry"]
    assert carry["fired"] is False
    assert carry["reason"] == "carried_engine_failed_new_level_verification"
    assert carry["verify_accuracy"] < 1.0
    assert attempt.get("engine_source") != "cross_level_engine_carry"


def test_flag_on_no_carried_engine_reports_reason(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY", "1")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "0")
    called = {"llm": False}
    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _llm_stub(called))

    policy = _policy()
    assert policy._carried_engine is None
    _arm_level_up(policy, _consistent_transitions())

    policy._induce_and_plan()

    assert called["llm"] is True
    assert policy.induction_attempts[-1]["cross_level_engine_carry"] == {
        "fired": False,
        "reason": "no_carried_engine",
    }


def test_flag_on_insufficient_evidence_defers_without_llm(monkeypatch):
    """One post-boundary transition is not verification -- measured: every live boundary
    arrives with exactly one. The stage must DEFER (skip the LLM this attempt and ask the
    caller to re-arm the reinduction), not burn the boundary on a coin-toss verify."""
    monkeypatch.setenv("CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY", "1")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "0")
    called = {"llm": False}
    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _llm_stub(called))

    policy = _policy()
    policy._carried_engine = _shift_right_engine
    policy._carried_goal = _goal_rightmost
    _arm_level_up(policy, _consistent_transitions(n=1))

    policy._induce_and_plan()

    assert called["llm"] is False
    carry = policy.induction_attempts[-1]["cross_level_engine_carry"]
    assert carry["fired"] is False
    assert carry["deferred"] is True
    assert carry["reason"] == "insufficient_new_level_evidence_deferred"
    assert carry["verify_n_transitions"] == 1
    assert policy._carry_defer_reinduction is True
    assert policy.induction_attempts[-1]["skipped"] == "cross_level_carry_deferred_for_evidence"


def test_flag_on_defer_budget_exhausted_falls_through_to_llm(monkeypatch):
    """The defer budget is finite: a RESET-looping explorer could starve the evidence
    window forever, so after the cap the status-quo LLM path must run regardless."""
    monkeypatch.setenv("CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY", "1")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "0")
    called = {"llm": False}
    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _llm_stub(called))

    policy = _policy()
    policy._carried_engine = _shift_right_engine
    policy._carried_goal = _goal_rightmost
    policy._carry_reinduction_defers = agent._CARRY_MAX_REINDUCTION_DEFERS
    _arm_level_up(policy, _consistent_transitions(n=1))

    policy._induce_and_plan()

    assert called["llm"] is True
    carry = policy.induction_attempts[-1]["cross_level_engine_carry"]
    assert carry["fired"] is False
    assert "deferred" not in carry
    assert carry["reason"] == "insufficient_new_level_evidence_defer_budget_exhausted"
    assert policy._carry_defer_reinduction is False


def test_defer_handler_in_next_move_rearms_and_tracks_transition(monkeypatch):
    """The next_move defer handler must un-latch `induced`, keep the reinduction pending,
    hand back the attempt-count tick, and return an explorer move WITH transition tracking
    (without it the evidence window never grows and the defer loop is pointless)."""
    monkeypatch.setenv("CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY", "1")

    policy = _policy()
    tracked = {"called": False}
    policy.explorer = SimpleNamespace(next_move=lambda frames, latest: (1, None))
    monkeypatch.setattr(
        policy,
        "_track_prev_for_transition",
        lambda mv, latest: tracked.__setitem__("called", True),
    )
    monkeypatch.setattr(policy, "_induce_and_plan", lambda: None)
    policy.phase = "induce"
    policy.induced = False
    policy._level_reinduction_pending = True
    policy._carry_defer_reinduction = True  # as the carry stage would have set it
    attempts_before = policy._induction_attempt_count

    mv = policy.next_move([], None)

    assert mv == (1, None)
    assert tracked["called"] is True
    assert policy.induced is False  # un-latched: the reinduction can re-enter
    assert policy._level_reinduction_pending is True  # still pending, not consumed
    assert policy._carry_defer_reinduction is False  # one-shot, consumed
    assert policy._induction_attempt_count == attempts_before  # a defer is not an attempt
    assert policy.phase == "explore"


def test_flag_on_goal_true_at_start_declines(monkeypatch):
    """The bogus-instant-win guard: a carried goal already true at the plan start must
    decline, not install a degenerate zero-step plan."""
    monkeypatch.setenv("CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY", "1")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_TRANSFER", "0")
    called = {"llm": False}
    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _llm_stub(called))

    policy = _policy()
    policy._carried_engine = _shift_right_engine
    policy._carried_goal = _goal_rightmost
    _arm_level_up(policy, _consistent_transitions())
    start = np.zeros((1, 6), dtype=np.int16)
    start[0, 5] = 5  # already at the rightmost column -> goal True
    policy.root_grid = start

    policy._induce_and_plan()

    assert called["llm"] is True
    carry = policy.induction_attempts[-1]["cross_level_engine_carry"]
    assert carry["fired"] is False
    assert carry["reason"] == "carried_goal_already_true_at_start"


# --------------------------------------------------------------------------- #
# (4) retention                                                               #
# --------------------------------------------------------------------------- #
def test_retention_keeps_engine_and_failed_outcome_does_not_evict():
    policy = _policy()
    good = SimpleNamespace(
        engine=_shift_right_engine,
        goal_predicate=_goal_rightmost,
        heldout_accuracy=1.0,
        accepted_by_heldout_verifier=True,
        selected_candidate_name="toy",
    )
    policy._retain_cross_level_engine(good)
    assert policy._carried_engine is _shift_right_engine
    assert policy._carried_goal is _goal_rightmost
    assert policy._carried_engine_meta["selected_candidate_name"] == "toy"

    # A later FAILED induction (no engine) must not evict the carried engine.
    policy._retain_cross_level_engine(SimpleNamespace(engine=None))
    assert policy._carried_engine is _shift_right_engine

    # A partial stub without even an `engine` attribute must be tolerated (existing tests
    # stub the outcome with bare namespaces).
    policy._retain_cross_level_engine(SimpleNamespace())
    assert policy._carried_engine is _shift_right_engine
