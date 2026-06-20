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
    E3AgentPolicy, SUBMITTED_AGENT_CONFIG, make_carnot_agent)


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
    # the live explorer config must equal the declared SUBMITTED_AGENT_CONFIG; any silent change to the
    # E3AgentPolicy/StepwiseExplorer defaults fails here until SUBMITTED_AGENT_CONFIG is consciously updated.
    pol = E3AgentPolicy("paritytest", proposer=None)
    exp = pol.explorer
    assert exp.value_weight == SUBMITTED_AGENT_CONFIG["value_weight"]
    assert exp.target_levels == SUBMITTED_AGENT_CONFIG["target_levels"]
    assert exp.search_mode == SUBMITTED_AGENT_CONFIG["search_mode"]


def test_wired_flags_reflect_actual_imports():
    # router_wired / world_model_dsl_wired must match whether the modules are ACTUALLY referenced in the
    # submission module -- catches BOTH "wired the module but left the flag stale" AND "flag claims wired
    # but the import is missing". This is the exact gap that shipped bare BFS at 0.08.
    src = inspect.getsource(m)
    assert SUBMITTED_AGENT_CONFIG["router_wired"] == _imports("arc_strategy_router", src), (
        "router_wired flag disagrees with whether arc_strategy_router is imported in the submission path")
    assert SUBMITTED_AGENT_CONFIG["world_model_dsl_wired"] == _imports("arc_world_model_dsl", src), (
        "world_model_dsl_wired flag disagrees with whether arc_world_model_dsl is imported")
