"""Tests for the goal-ENERGY wiring (2026-06-23, closes GAP-ARCH-GOAL-NOT-VERIFIED).

induce_goal_energy is the GRADED counterpart of induce_goal_predicate; plan_in_model(goal_energy=...) makes
the in-model planner BEST-FIRST (descend toward the induced goal) instead of blind FIFO BFS -> fewer
nodes-to-win (the action-efficiency win). The energy is induced per-game from the agent's OWN observed
win/non-win states; an ablation control + a silent-failure guard are mandatory in the live wiring.
"""

import numpy as np
import pytest

# Pre-import the heavy chain at COLLECTION time. plan_in_model -> _model_candidates lazily does
# `from carnot.agentic.arc_graph_explore import _components_detailed`, which pulls in torch (~700 MB) on
# first use. If that happens DURING a test, the per-test RSS watchdog (conftest) flags it as a memory leak.
# Importing it here loads torch before the per-test RSS snapshot, so the watchdog sees no growth.
import carnot.agentic.arc_graph_explore  # noqa: F401
from carnot.agentic.arc_agi3_goal_induction import induce_goal_energy
from carnot.agentic.arc_agi3_world_model import objects
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_executable_world_model import Transition, plan_in_model


def _grid(positions):
    g = np.zeros((9, 9), dtype=int)
    for i, (y, x) in enumerate(positions):
        g[y, x] = i + 1
    return g


_POS = [(1, 1), (1, 7), (4, 4), (7, 1), (7, 7), (2, 5)]


def test_induce_goal_energy_is_graded_and_zero_at_win():
    # goal = reduce objects to <= max_win_objs (2); energy is the violation magnitude
    ge = induce_goal_energy(
        [_grid(_POS[:2]), _grid(_POS[:1]), _grid([])], [_grid(_POS), _grid(_POS[:4])]
    )
    assert ge is not None
    assert ge(_grid(_POS[:2])) == 0.0  # win state (2 objects) -> satisfied
    assert ge(_grid(_POS)) > ge(
        _grid(_POS[:4])
    )  # 6 objects farther than 4 objects (graded, monotone)
    assert ge(_grid(_POS)) == 4.0  # 6 objects, ceiling 2 -> violation 4


def test_induce_goal_energy_needs_two_wins():
    assert induce_goal_energy([_grid(_POS[:1])], [_grid(_POS)]) is None  # <2 win examples


def _remove_click_engine(counter, tag):
    def engine(grid, action, data):
        counter[tag] += 1
        g = grid.copy()
        if action == 6 and data is not None:  # click removes the object at (x, y)
            g[int(data["y"]), int(data["x"])] = 0
        return g

    return engine


def test_plan_in_model_goal_energy_reaches_win_in_fewer_nodes():
    start = _grid(_POS)
    K = 2

    def is_win(g):
        return len(objects(g)) <= K

    ge = induce_goal_energy(
        [_grid(_POS[:2]), _grid(_POS[:1]), _grid([])], [_grid(_POS), _grid(_POS[:3])]
    )
    counts = {"bfs": 0, "ge": 0}
    p_bfs = plan_in_model(
        _remove_click_engine(counts, "bfs"), is_win, start, max_nodes=20000, max_depth=10
    )
    p_ge = plan_in_model(
        _remove_click_engine(counts, "ge"),
        is_win,
        start,
        max_nodes=20000,
        max_depth=10,
        goal_energy=ge,
    )
    assert p_bfs is not None and p_ge is not None  # both reach the win
    assert len(p_ge) == len(p_bfs)  # same plan length (4 removals)
    assert counts["ge"] < counts["bfs"]  # goal-energy explores fewer nodes
    assert counts["ge"] * 2 < counts["bfs"]  # materially fewer (observed ~6x)


def test_plan_in_model_backward_compatible_without_goal_energy():
    # goal_energy=None must keep the exact original FIFO BFS behaviour
    start = _grid(_POS)

    def is_win(g):
        return len(objects(g)) <= 2

    counts = {"bfs": 0}
    p = plan_in_model(
        _remove_click_engine(counts, "bfs"), is_win, start, max_nodes=20000, max_depth=10
    )
    assert p is not None and len(p) == 4


# ---------------------------------------------------------------------------
# REQ-ARC-FCP-5699-19: novelty goal-energy fallback for first-contact levels
# (games with no _previous_level_complete_grid exemplar yet -- REQ-ARC-FCP-5699-18
# root-caused that the pre-existing fallback there was a flat, zero-gradient constant).
# ---------------------------------------------------------------------------


def _policy(**kwargs):
    return E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0, **kwargs)


def _always_false(_grid):
    return False


def _transition(before, after, *, action=6, data=None, level_before=0, level_after=0):
    # Real Transition (a @dataclass, NOT a namedtuple -- t[0]/t[3] index access raises
    # TypeError, silently swallowed by a broad except and masking REQ-ARC-FCP-5699-19's
    # original bug where _novelty_observed_stack() used index access against these objects
    # and got zero observed grids in every real live run despite the env var being set).
    # Tests build REAL Transition objects, never raw tuples, so this class of mismatch
    # between test fixtures and production data shape cannot recur silently again.
    return Transition(before, action, data, after, level_before, level_after)


def test_req_arc_fcp_5699_19_default_stays_binary_no_regression(monkeypatch):
    """Neither env var set (the production default) -- energy_source must stay 'binary',
    identical to pre-5699-19 behaviour. This is the regression-safety anchor."""
    monkeypatch.delenv("CARNOT_ARC_GRADED_GOAL_BIAS", raising=False)
    monkeypatch.delenv("CARNOT_ARC_NOVELTY_GOAL_BIAS", raising=False)
    pol = _policy()
    pol.transitions = [_transition(_grid(_POS[:1]), _grid(_POS[:2]))]
    energy = pol._goal_energy_for_plan(_always_false)
    assert energy is not None
    assert energy.energy_source == "binary"
    assert energy(_grid(_POS)) == pol.goal_guidance_lambda


def test_req_arc_fcp_5699_19_novelty_observed_stack_none_with_no_transitions():
    pol = _policy()
    pol.transitions = []
    assert pol._novelty_observed_stack() is None


def test_req_arc_fcp_5699_19_novelty_observed_stack_built_from_transitions():
    pol = _policy()
    before1, after1 = _grid(_POS[:1]), _grid(_POS[:2])
    pol.transitions = [_transition(before1, after1)]
    stack = pol._novelty_observed_stack()
    assert stack is not None
    assert stack.shape == (2, 9, 9)


def test_req_arc_fcp_5699_19_novelty_fires_when_enabled_and_no_exemplar(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_GRADED_GOAL_BIAS", raising=False)
    monkeypatch.setenv("CARNOT_ARC_NOVELTY_GOAL_BIAS", "1")
    pol = _policy()
    observed = _grid(_POS[:1])
    pol.transitions = [_transition(observed, observed)]  # a static (no-op) transition
    assert pol._previous_level_complete_grid is None
    energy = pol._goal_energy_for_plan(_always_false)
    assert energy is not None
    assert energy.energy_source == "novelty"
    # a grid IDENTICAL to something already concretely observed -> zero novelty -> same
    # flat energy as the pre-existing binary fallback (never WORSE than before).
    assert energy(observed) == pytest.approx(pol.goal_guidance_lambda)
    # a grid maximally DIFFERENT from everything observed -> near-zero energy (attractive
    # to the min-heap search) -- the actual differentiator the binary fallback lacked.
    maximally_different = np.ones_like(observed) * 9
    assert energy(maximally_different) < energy(observed)
    assert energy(maximally_different) == pytest.approx(0.0, abs=1e-6)


def test_req_arc_fcp_5699_19_is_done_always_short_circuits_to_zero_under_novelty(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_NOVELTY_GOAL_BIAS", "1")
    pol = _policy()
    observed = _grid(_POS[:1])
    pol.transitions = [_transition(observed, observed)]

    def is_win(_g):
        return True

    energy = pol._goal_energy_for_plan(is_win)
    assert energy(_grid(_POS)) == 0.0  # a literal win always beats novelty, unconditionally


def test_req_arc_fcp_5699_19_graded_exemplar_takes_priority_over_novelty(monkeypatch):
    """When BOTH env vars are set and an exemplar IS available (multi-level case), the
    graded-exemplar branch must win -- novelty is only the FIRST-CONTACT fallback."""
    monkeypatch.setenv("CARNOT_ARC_GRADED_GOAL_BIAS", "1")
    monkeypatch.setenv("CARNOT_ARC_NOVELTY_GOAL_BIAS", "1")
    pol = _policy()
    pol._previous_level_complete_grid = _grid(_POS[:2])
    pol.transitions = [_transition(_grid(_POS[:1]), _grid(_POS[:2]))]
    energy = pol._goal_energy_for_plan(_always_false)
    assert energy.energy_source == "graded_exemplar"


def test_req_arc_fcp_5699_19_novelty_absent_falls_back_to_binary_when_no_transitions(monkeypatch):
    """CARNOT_ARC_NOVELTY_GOAL_BIAS=1 but zero transitions collected -> no observed grids to
    compare against -> gracefully falls back to the flat binary energy, not a crash."""
    monkeypatch.delenv("CARNOT_ARC_GRADED_GOAL_BIAS", raising=False)
    monkeypatch.setenv("CARNOT_ARC_NOVELTY_GOAL_BIAS", "1")
    pol = _policy()
    pol.transitions = []
    energy = pol._goal_energy_for_plan(_always_false)
    assert energy.energy_source == "binary"
    assert energy(_grid(_POS)) == pol.goal_guidance_lambda


def test_req_arc_fcp_5699_19_call_plan_in_model_records_goal_energy_source(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_NOVELTY_GOAL_BIAS", "1")
    pol = _policy()
    observed = _grid(_POS[:1])
    pol.transitions = [_transition(observed, observed)]

    def fake_plan_in_model(engine, is_done, start_grid, **kwargs):
        return None

    diagnostics: dict = {}
    pol._call_plan_in_model(
        fake_plan_in_model,
        engine=None,
        is_done=_always_false,
        start_grid=observed,
        diagnostics=diagnostics,
    )
    assert diagnostics["goal_energy_source"] == "novelty"
