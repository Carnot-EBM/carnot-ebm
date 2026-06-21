"""Parity guard for WHAT SHIPS — prevents the 2026-06-19 "0.08 incident" from recurring.

Incident: the offline eval measured STRONGER opt-in configs (explorer_bf unlocked cn04) while the
SUBMITTED default (make_carnot_agent(Agent)) shipped bare BFS, and nobody caught it because "better" was
opt-in-only and the headline metric was banked-replay levels, not the submitted path. These tests assert
the shipped agent matches the single-source-of-truth SUBMITTED_AGENT_CONFIG, and that the "wired" flags
reflect REALITY — so a silent divergence between what we measure and what we ship fails CI.
"""

import inspect
import re

import pytest

from carnot.agentic import arc_competition_agent as m
from carnot.agentic.arc_competition_agent import (
    E3AgentPolicy, SUBMITTED_AGENT_CONFIG, StepwiseExplorer, make_carnot_agent)


def _imports(module_name: str, src: str) -> bool:
    """True only if `module_name` is actually IMPORTED (a from/import statement), not merely mentioned
    in a comment or docstring -- the TODO prose names these modules without importing them."""
    return bool(re.search(rf"^\s*(from|import)\s+[\w.]*{re.escape(module_name)}\b", src, re.M))


def test_submission_defaults_to_e3_cascade_not_banked_replay():
    # the submission is make_carnot_agent(Agent) with NO cascade arg -> must default to the generic
    # E3AgentPolicy cascade, NEVER the cascade=False banked-replay ("useless on the hidden eval").
    assert inspect.signature(make_carnot_agent).parameters["cascade"].default is True
    assert SUBMITTED_AGENT_CONFIG["cascade"] is True
    assert SUBMITTED_AGENT_CONFIG["policy"] == "E3AgentPolicy"


def test_shipped_explorer_config_matches_single_source_of_truth():
    """REQ-REPORT-4475-LIVE-STACK: shipped explorer config is the declared default."""
    # the live explorer config must equal the declared SUBMITTED_AGENT_CONFIG; any silent change to the
    # E3AgentPolicy/StepwiseExplorer defaults fails here until SUBMITTED_AGENT_CONFIG is consciously updated.
    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    exp = pol.explorer
    assert exp.value_weight == SUBMITTED_AGENT_CONFIG["value_weight"]
    assert exp.target_levels == SUBMITTED_AGENT_CONFIG["target_levels"]
    assert exp.search_mode == SUBMITTED_AGENT_CONFIG["search_mode"]
    assert exp.frontier_batch_size == SUBMITTED_AGENT_CONFIG["frontier_batch_size"]
    assert exp.navigation_cost_tiebreak == SUBMITTED_AGENT_CONFIG["navigation_cost_tiebreak"]
    # value_weight reverted to 0.0 (2026-06-20): the v3 head is loaded + used as a tiebreaker, but
    # weight>0 was a measured regression (per-node v3 eval too slow). Pin it to 0 until .416 shows a
    # weight>0 beats bare-BFS live AND finishes in budget. (NOT `> 0` — that asserted the regression.)
    assert exp.value_weight == 0.0
    assert exp.target_levels == 1


def test_req_arc_wmte_4548_null_a1_a4_levers_are_not_submitted_by_default():
    """REQ-ARC-WMTE-4548: null A1/A4 levers stay off the submitted path."""
    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    exp = pol.explorer

    assert SUBMITTED_AGENT_CONFIG["target_levels"] == 1
    assert exp.target_levels == 1
    assert exp.frame_change_scorer is None
    assert exp.frame_change_prune_threshold is None
    assert exp.action_prior is None
    assert exp.action_prior_prune_quantile is None


def test_wired_flags_reflect_actual_imports():
    """REQ-REPORT-4475-LIVE-STACK: shipped config declares real router/DSL imports."""
    # router_wired / world_model_dsl_wired must match whether the modules are ACTUALLY referenced in the
    # submission module -- catches BOTH "wired the module but left the flag stale" AND "flag claims wired
    # but the import is missing". This is the exact gap that shipped bare BFS at 0.08.
    src = inspect.getsource(m)
    assert SUBMITTED_AGENT_CONFIG["router_wired"] == _imports("arc_strategy_router", src), (
        "router_wired flag disagrees with whether arc_strategy_router is imported in the submission path")
    assert SUBMITTED_AGENT_CONFIG["world_model_dsl_wired"] == _imports("arc_world_model_dsl", src), (
        "world_model_dsl_wired flag disagrees with whether arc_world_model_dsl is imported")


def test_e3_policy_builds_strategy_route_and_world_model_dsl():
    """SCENARIO-REPORT-4475-LIVE-STACK-PARITY: E3 first contact has router + DSL state."""
    pol = E3AgentPolicy("tn36", proposer=None, value_head=lambda _frame: 0.0)
    assert pol.strategy_route["game"] == "tn36"
    assert pol.strategy_route["name"] == "program_editor"
    assert pol.strategy_route["uses_goal_distance_heuristic"] is False
    assert pol.dsl_model.game_id == "tn36"
    assert pol.explore_budget < SUBMITTED_AGENT_CONFIG["graph_explore_budget"]


def test_stepwise_explorer_prefers_forward_shortest_path_over_reset():
    """SCENARIO-REPORT-4475-LIVE-STACK-FORWARD-NAV: forward edges beat RESET replay."""
    exp = StepwiseExplorer()
    exp.root = "A"
    exp.cur = "A"
    exp.start_level = 0
    exp.best_level = 0
    exp.graph = {
        "A": {"path": [], "untested": [], "value": 0.0},
        "B": {
            "path": [{"action": 7, "data": None}],
            "untested": [{"action": 2, "data": {"x": 1, "y": 2}}],
            "value": 0.0,
        },
    }
    exp.adj = {"A": [({"action": 7, "data": None}, "B")]}

    assert exp.next_move([], None) == (7, None)
    assert exp.next_move([], None) == (2, {"x": 1, "y": 2})
    assert exp.awaiting == {"origin": "B", "action": 2, "data": {"x": 1, "y": 2}}
