from __future__ import annotations

from types import SimpleNamespace

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy, StepwiseExplorer


def test_req_arc_wmte_4533_stepwise_target_levels_do_not_stop_at_l1(monkeypatch):
    """REQ-ARC-WMTE-4533: target_levels > 1 keeps the explorer past the first level-up."""

    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    explorer = StepwiseExplorer(target_levels=2)

    assert explorer.is_done([], SimpleNamespace(levels_completed=0)) is False
    assert explorer.is_done([object()], SimpleNamespace(levels_completed=1)) is False
    assert explorer.is_done([object(), object()], SimpleNamespace(levels_completed=2)) is True


def test_scenario_arc_wmte_4533_level_boundary_resets_induction(monkeypatch):
    """SCENARIO-ARC-WMTE-4533: a level-up starts a fresh goal-induction episode."""

    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=3, value_head=None)
    policy.induced = True
    policy.plan = [{"action": 1, "data": None}]
    policy.pi = 1
    policy.transitions = [object()]

    assert policy._observe_level_boundary(SimpleNamespace(levels_completed=0), frames_seen=1) == []
    events = policy._observe_level_boundary(SimpleNamespace(levels_completed=1), frames_seen=2)

    assert events[-1]["trigger"] == "level_up"
    assert events[-1]["next_goal_level"] == 2
    assert policy.induced is False
    assert policy.plan == []
    assert policy.pi == 0
    assert policy._level_reinduction_pending is True
    assert policy._current_goal_level == 2
    assert policy._episode_transition_start == len(policy.transitions)


def test_scenario_arc_wmte_4533_reinduction_is_not_stall_only(monkeypatch):
    """SCENARIO-ARC-WMTE-4533: a won L1 does not block L2 re-induction."""

    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("lp85", proposer=object(), target_levels=2, value_head=None)
    policy.explorer.start_level = 0
    policy.explorer.best_level = 1
    policy.transitions = [object()]
    policy._episode_transition_start = 0
    policy._level_reinduction_pending = True
    policy.induced = False

    should_induce, reason = policy._should_enter_induction(stalled=False, won=True)

    assert should_induce is True
    assert reason == "level_up_reinduction"


def test_req_arc_wmte_4533_goal_bias_keeps_depth_primary():
    """REQ-ARC-WMTE-4533: a re-induced predicate biases ties without outranking depth."""

    explorer = StepwiseExplorer()
    explorer.cur = "root"
    explorer.set_goal_bias(lambda frame: 1.0 if frame == "goal" else 0.0, label="l2_predicate")
    explorer.graph = {
        "shallow": {
            "path": [{"action": 1, "data": None}],
            "untested": [{"action": 2, "data": None}],
            "value": 0.0,
            "frame": "plain",
        },
        "deep_goal": {
            "path": [{"action": 1, "data": None}, {"action": 2, "data": None}],
            "untested": [{"action": 3, "data": None}],
            "value": 0.0,
            "frame": "goal",
        },
    }
    assert explorer._frontier() == "shallow"

    explorer.graph["shallow"]["path"].append({"action": 4, "data": None})
    assert explorer._frontier() == "deep_goal"


def test_req_arc_wmte_4533_honest_null_artifact_records_delta_note():
    """REQ-ARC-WMTE-4533: baseline==best nulls are explicit, not tautological wins."""

    from carnot import experiment_4533_per_level_goal_reinduction as exp4533

    artifact = exp4533.build_artifact(
        preconditions_checked={
            "offline_arcade_import_smoke": True,
            "spec_has_req_4533": True,
        },
        target_levels_sweep=[
            {
                "target_levels": 1,
                "core_efficiency": exp4533.CORE_EFFICIENCY_BASELINE,
                "deepest_level_by_game": {"lp85": 1, "m0r0": 1, "sp80": 1, "vc33": 1},
                "core_solves_preserved": True,
            },
            {
                "target_levels": 2,
                "core_efficiency": exp4533.CORE_EFFICIENCY_BASELINE,
                "deepest_level_by_game": {"lp85": 1, "m0r0": 1, "sp80": 1, "vc33": 1},
                "core_solves_preserved": True,
            },
        ],
        positive_control={
            "passed": True,
            "predicate_change_registered": True,
            "l1_predicate": "touch_marker",
            "l2_predicate": "clear_new_target",
        },
        offline_reproduction={},
        random_seed=4533,
        duration_s=0.0,
    )

    assert exp4533.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["core_efficiency_best"] == exp4533.CORE_EFFICIENCY_BASELINE
    assert artifact["efficiency_delta"] == 0.0
    assert "null_delta_methodology_note" in artifact
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["false_negative_risk_checked"] is True


def test_req_arc_wmte_4533_success_requires_offline_reproduction():
    """REQ-ARC-WMTE-4533: a deeper live level is not wired without offline reproduction."""

    from carnot import experiment_4533_per_level_goal_reinduction as exp4533

    artifact = exp4533.build_artifact(
        preconditions_checked={
            "offline_arcade_import_smoke": True,
            "spec_has_req_4533": True,
        },
        target_levels_sweep=[
            {
                "target_levels": 2,
                "core_efficiency": exp4533.CORE_EFFICIENCY_BASELINE + 0.25,
                "deepest_level_by_game": {"lp85": 2, "m0r0": 1, "sp80": 1, "vc33": 1},
                "core_solves_preserved": True,
            },
        ],
        positive_control={"passed": True, "predicate_change_registered": True},
        offline_reproduction={"reproduced": False, "reached_level": 2},
        random_seed=4533,
        duration_s=0.0,
    )

    assert exp4533.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["offline_reproduced"] is False
