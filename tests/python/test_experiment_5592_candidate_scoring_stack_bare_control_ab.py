"""Tests for Exp 5592 candidate-scoring stack vs bare-control A/B.

Spec refs: REQ-ARC-FCP-5592, SCENARIO-ARC-FCP-5592-STACK-VS-BARE-DELTA.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5592_candidate_scoring_stack_bare_control_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_fcp_5592_spec_declares_ab_contract() -> None:
    """REQ-ARC-FCP-5592: OpenSpec declares the stack-vs-bare-control A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5592") :]

    for marker in (
        "REQ-ARC-FCP-5592",
        "SCENARIO-ARC-FCP-5592-STACK-VS-BARE-DELTA",
        "bare_control_config",
        "levels_gained_headroom_present",
        "efficiency_full_stack_total",
    ):
        assert marker in section


def test_req_arc_fcp_5592_bare_control_kwargs_match_submitted_config() -> None:
    """REQ-ARC-FCP-5592: the ablation kwargs match SUBMITTED_AGENT_CONFIG's own bare_control_config
    manifest, so the measured ablation is the same on/off toggle the codebase already documents."""

    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    manifest = SUBMITTED_AGENT_CONFIG["bare_control_config"]
    assert mod.BARE_CONTROL_KWARGS["target_levels"] == manifest["target_levels"]
    assert mod.BARE_CONTROL_KWARGS["value_weight"] == manifest["value_weight"]
    assert mod.BARE_CONTROL_KWARGS["candidate_router"] == manifest["candidate_router"]
    assert (
        mod.BARE_CONTROL_KWARGS["navigation_cost_tiebreak"] == manifest["navigation_cost_tiebreak"]
    )
    # manifest values are already "enabled=False", i.e. the disabled state; the real
    # constructor kwargs (action_effect_expansion_prior, goal_bias, goal_candidate_guidance)
    # carry no "_enabled" suffix but represent the identical disabled state.
    assert manifest["action_effect_expansion_prior_enabled"] is False
    assert mod.BARE_CONTROL_KWARGS["action_effect_expansion_prior"] is False
    assert manifest["goal_energy_enabled"] is False
    assert mod.BARE_CONTROL_KWARGS["goal_bias"] is None
    assert manifest["goal_energy_candidate_guidance_enabled"] is False
    assert mod.BARE_CONTROL_KWARGS["goal_candidate_guidance"] is False


def test_scenario_arc_fcp_5592_blocked_precondition_never_runs_arms(monkeypatch) -> None:
    """A missing resource fails closed without running any game."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": False,
            "offline_arcade_makes_env": False,
            "e3_policy_import": True,
            "bare_control_config_present": True,
            "ok": False,
        },
    )

    def _fail_if_called(roster, *, budget):
        raise AssertionError("run_both_arms must not run when a precondition is missing")

    monkeypatch.setattr(mod, "run_both_arms", _fail_if_called)

    artifact = mod.build_artifact()

    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["full_stack_results"] == {}
    assert artifact["levels_gained_headroom_present"] is False
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_arc_fcp_5592_full_stack_beats_bare_control(monkeypatch) -> None:
    """A synthetic full-stack-wins result is classified as a stack win."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "bare_control_config_present": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_both_arms",
        lambda roster, *, budget: (
            {g: {"levels": (1 if g == roster[0] else 0), "efficiency": 2.0} for g in roster},
            {g: {"levels": 0, "efficiency": 0.0} for g in roster},
            1.0,
        ),
    )

    artifact = mod.build_artifact(roster=("cd82", "sk48"))

    assert artifact["levels_gained_full_stack_total"] == 1
    assert artifact["levels_gained_bare_control_total"] == 0
    assert "candidate_stack_beats_bare_control_0_to_1_levels" in artifact["honest_verdict"]


def test_scenario_arc_fcp_5592_regression_below_bare_control(monkeypatch) -> None:
    """A synthetic bare-control-wins result is flagged as a regression, not hidden."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "bare_control_config_present": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_both_arms",
        lambda roster, *, budget: (
            {g: {"levels": 0, "efficiency": 0.0} for g in roster},
            {g: {"levels": (1 if g == roster[0] else 0), "efficiency": 2.0} for g in roster},
            1.0,
        ),
    )

    artifact = mod.build_artifact(roster=("cd82", "sk48"))

    assert artifact["honest_verdict"] == "complete: candidate_stack_regression_below_bare_control"


def test_scenario_arc_fcp_5592_tied_levels_efficiency_breaks_tie(monkeypatch) -> None:
    """When levels tie, an efficiency delta (not just level count) still yields a real verdict."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "bare_control_config_present": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_both_arms",
        lambda roster, *, budget: (
            {g: {"levels": (1 if g == roster[0] else 0), "efficiency": 2.0} for g in roster},
            {g: {"levels": (1 if g == roster[0] else 0), "efficiency": 1.0} for g in roster},
            1.0,
        ),
    )

    artifact = mod.build_artifact(roster=("cd82", "sk48"))

    assert artifact["honest_verdict"] == (
        "complete: candidate_stack_ties_levels_but_more_efficient_than_bare_control"
    )


def test_scenario_arc_fcp_5592_honest_null_with_headroom(monkeypatch) -> None:
    """A fully-tied result with headroom present is an honest null, not fabricated."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            "offline_arcade_importable": True,
            "offline_arcade_makes_env": True,
            "e3_policy_import": True,
            "bare_control_config_present": True,
            "ok": True,
        },
    )
    monkeypatch.setattr(
        mod,
        "run_both_arms",
        lambda roster, *, budget: (
            {g: {"levels": (1 if g == roster[0] else 0), "efficiency": 1.0} for g in roster},
            {g: {"levels": (1 if g == roster[0] else 0), "efficiency": 1.0} for g in roster},
            1.0,
        ),
    )

    artifact = mod.build_artifact(roster=("cd82", "sk48"))

    assert artifact["honest_verdict"] == (
        "complete: candidate_stack_honest_null_headroom_present_no_delta"
    )


def test_req_arc_fcp_5592_repository_artifact_confirms_real_headroomed_null() -> None:
    """REQ-ARC-FCP-5592: the checked-in real run is an honest, headroom-present null, and
    the per-game rows show the ablation genuinely took effect (not a construction bug that
    silently made both arms identical)."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["levels_gained_headroom_present"] is True
    assert all(delta == 0 for delta in result["per_game_levels_delta"].values())
    assert (
        result["inference_substrate"] == "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in result

    headroom_game = next(
        g for g in result["roster"] if result["full_stack_results"][g].get("levels", 0) > 0
    )
    full_row = result["full_stack_results"][headroom_game]
    bare_row = result["bare_control_results"][headroom_game]
    # the ablation genuinely changed search behavior on the one game with headroom, even
    # though the CAPPED efficiency metric happened to saturate identically for both.
    assert full_row["actions"] != bare_row["actions"]
