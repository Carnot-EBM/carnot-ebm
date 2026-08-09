"""REQ-ARC-WMTE-6243 (2026-08-09): conditional think<->no_think arm fallback.

Motivated directly by exp6221 (REQ-ARC-WMTE-6242, Phase 2a's expanded-roster A/B): sp80's
no_think arm failed to produce ANY parseable engine ("local model code unusable after 3 tries")
while think produced a fully-admitted 1.0 engine on the identical transitions; cd82/ls20/sk48
show the mirror failure (think fails outright, no_think at least scores something). The lever:
if the currently-configured arm produces no scoreable engine at all
(`outcome.heldout_accuracy is None`), retry once with the OTHER arm before giving up.

These tests call `E3AgentPolicy._execute_bounded_llm_reinduction_with_arm_fallback` directly
(not through the full `_induce_and_plan` state machine) -- it is a thin, self-contained wrapper
around the module-level `execute_bounded_llm_reinduction`, and testing it directly is more
robust than driving the whole induce-and-plan branch tree to reach it.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy


def _outcome(heldout_accuracy, planned=False, plan=None):
    return SimpleNamespace(
        model_specs="test",
        planned=planned,
        skipped="",
        plan=plan or [],
        heldout_accuracy=heldout_accuracy,
    )


def _policy(**overrides):
    return E3AgentPolicy("lp85", proposer=SimpleNamespace(), target_levels=3, value_head=None)


def test_flag_off_default_never_retries_even_on_total_failure(monkeypatch):
    """Default OFF (SUBMITTED_THINK_ARM_FALLBACK_ENABLED=False, no env override): a total
    induction failure (heldout_accuracy=None) must NOT trigger a second call -- the load-bearing
    byte-identity property for every default-off lever in this file."""
    calls = []

    def _fake(**kwargs):
        calls.append(dict(kwargs))
        return _outcome(heldout_accuracy=None)

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _fake)
    monkeypatch.delenv("CARNOT_ARC_THINK_ARM_FALLBACK", raising=False)

    policy = _policy()
    assert policy.think_arm_fallback_enabled is False
    attempt = {}
    outcome = policy._execute_bounded_llm_reinduction_with_arm_fallback(attempt, game="lp85")

    assert len(calls) == 1
    assert outcome.heldout_accuracy is None
    assert attempt["think_arm_fallback"] == {"enabled": False}


def test_flag_on_primary_succeeds_never_retries(monkeypatch):
    """Flag on, but the primary arm already produced a scoreable engine (even a low-scoring one,
    e.g. 0.0) -- the lever must not fire. Retrying an already-scored engine wastes wall-clock on
    exactly the floor-effect cases exp6221 showed retrying cannot fix (both arms tied at 0.0)."""
    calls = []

    def _fake(**kwargs):
        calls.append(dict(kwargs))
        return _outcome(heldout_accuracy=0.0)

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _fake)
    monkeypatch.setenv("CARNOT_ARC_THINK_ARM_FALLBACK", "1")

    policy = _policy()
    assert policy.think_arm_fallback_enabled is True
    attempt = {}
    outcome = policy._execute_bounded_llm_reinduction_with_arm_fallback(attempt, game="lp85")

    assert len(calls) == 1
    assert outcome.heldout_accuracy == 0.0
    assert attempt["think_arm_fallback"] == {
        "enabled": True,
        "fired": False,
        "reason": "primary_arm_produced_a_scored_engine",
    }


def test_flag_on_total_failure_retries_with_opposite_arm_and_succeeds(monkeypatch):
    """The sp80 shape: primary arm produces nothing at all; the OTHER arm succeeds. Must fire
    exactly once more, toggle CARNOT_ARC_INDUCE_THINK to the opposite of the primary arm DURING
    the retry call, restore the prior env state afterward, and return the fallback's outcome."""
    seen_think_env_during_calls = []

    def _fake(**kwargs):
        seen_think_env_during_calls.append(os.environ.get("CARNOT_ARC_INDUCE_THINK"))
        if len(seen_think_env_during_calls) == 1:
            return _outcome(heldout_accuracy=None)
        return _outcome(heldout_accuracy=1.0, planned=True, plan=[{"action": 1, "data": None}])

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _fake)
    monkeypatch.setenv("CARNOT_ARC_THINK_ARM_FALLBACK", "1")
    # Primary arm = no_think (explicit, so the test does not depend on the live scored default).
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")

    policy = _policy()
    attempt = {}
    outcome = policy._execute_bounded_llm_reinduction_with_arm_fallback(attempt, game="lp85")

    assert seen_think_env_during_calls == ["0", "1"]  # no_think, then think
    assert outcome.heldout_accuracy == 1.0
    assert outcome.plan == [{"action": 1, "data": None}]
    assert attempt["think_arm_fallback"] == {
        "enabled": True,
        "fired": True,
        "primary_arm_think": False,
        "fallback_arm_think": True,
        "primary_produced_engine": False,
        "fallback_produced_engine": True,
    }
    # Env restored to its PRE-CALL value, not left on the fallback arm.
    assert os.environ.get("CARNOT_ARC_INDUCE_THINK") == "0"


def test_flag_on_total_failure_retries_opposite_direction_when_primary_is_think(monkeypatch):
    """Direction is read from the CURRENT arm at call time, not hardcoded -- if the primary arm
    is think (matches the live scored default per ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT), the
    fallback must try no_think, the cd82/ls20/sk48 shape from exp6221."""
    seen = []

    def _fake(**kwargs):
        seen.append(os.environ.get("CARNOT_ARC_INDUCE_THINK"))
        return _outcome(heldout_accuracy=None if len(seen) == 1 else 0.0)

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _fake)
    monkeypatch.setenv("CARNOT_ARC_THINK_ARM_FALLBACK", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")

    policy = _policy()
    attempt = {}
    policy._execute_bounded_llm_reinduction_with_arm_fallback(attempt, game="lp85")

    assert seen == ["1", "0"]  # think, then no_think
    assert attempt["think_arm_fallback"]["primary_arm_think"] is True
    assert attempt["think_arm_fallback"]["fallback_arm_think"] is False
    assert os.environ.get("CARNOT_ARC_INDUCE_THINK") == "1"  # restored


def test_flag_on_both_arms_fail_returns_original_outcome_not_fallback(monkeypatch):
    """Neither arm produces a scoreable engine (the 11/12 exp6221 floor-tie shape, taken to its
    total-failure extreme) -- the lever must not fabricate a difference; it returns the PRIMARY
    outcome, and the diagnostics say the fallback did not help."""
    calls = []

    def _fake(**kwargs):
        calls.append(1)
        return _outcome(heldout_accuracy=None)

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", _fake)
    monkeypatch.setenv("CARNOT_ARC_THINK_ARM_FALLBACK", "1")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")

    policy = _policy()
    attempt = {}
    outcome = policy._execute_bounded_llm_reinduction_with_arm_fallback(attempt, game="lp85")

    assert len(calls) == 2
    assert outcome.heldout_accuracy is None
    assert attempt["think_arm_fallback"]["fallback_produced_engine"] is False


def test_bare_default_matches_submitted_constant(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_THINK_ARM_FALLBACK", raising=False)
    assert agent.SUBMITTED_THINK_ARM_FALLBACK_ENABLED is False
    policy = _policy()
    assert policy.think_arm_fallback_enabled is False


def test_status_diagnostics_dict_reports_the_lever():
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    assert (
        SUBMITTED_AGENT_CONFIG["think_arm_fallback_enabled"]
        is agent.SUBMITTED_THINK_ARM_FALLBACK_ENABLED
    )
    assert SUBMITTED_AGENT_CONFIG["think_arm_fallback_wired"] is True
