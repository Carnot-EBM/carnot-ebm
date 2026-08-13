"""Tests for Exp6387 active reward-machine discrimination.

Spec refs: REQ-ARC-ARM-6387,
SCENARIO-ARC-ARM-6387-LEGAL-DISAGREEMENT,
SCENARIO-ARC-ARM-6387-ABSTAIN-AND-BOUNDS,
SCENARIO-ARC-ARM-6387-TWO-SIDED-EVIDENCE,
SCENARIO-ARC-ARM-6387-LIVE-DEFAULT-OFF,
SCENARIO-ARC-ARM-6387-ARTIFACT-NO-SOLVE.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from carnot import experiment_6387_arc_active_reward_machine_discriminator as exp6387
from carnot.agentic.arc_active_reward_machine_frontier import (
    FRAME_CHANGED_NO_LEVEL,
    LEVEL_UP,
    SAME_FRAME_NO_LEVEL,
    RewardMachineFrontier,
    RewardMachineHypothesis,
    RewardMachineTransition,
    TransitionEvidence,
    default_fixture_manifest,
    reward_machine_frontier_from_transitions,
)
from carnot.agentic.arc_two_sided_goal_contract import ACCEPTED, REJECTED, UNVERIFIABLE


REPO = Path(__file__).resolve().parents[2]


def _evidence(name: str, action: int, symbol: str) -> TransitionEvidence:
    return TransitionEvidence(
        source_transition_id=f"source:{name}:{action}:{symbol}",
        source_tick=0,
        source_action=action,
        observed_symbol=symbol,
        visible_frame_hash_before=f"before:{name}",
        visible_frame_hash_after=f"after:{name}:{symbol}",
        source="game_blind_fixture",
    )


def _hypothesis(name: str, predictions: dict[int, str]) -> RewardMachineHypothesis:
    transitions = [
        RewardMachineTransition(
            source_state="q0",
            action=action,
            target_state=f"q_{symbol}",
            predicted_symbol=symbol,
            evidence=(_evidence(name, action, symbol),),
        )
        for action, symbol in predictions.items()
    ]
    return RewardMachineHypothesis(
        hypothesis_id=name,
        states=("q0", "q_same", "q_changed", "q_level_up"),
        start_state="q0",
        current_state="q0",
        transitions=tuple(transitions),
    )


def test_scenario_arc_arm_6387_legal_disagreement_freezes_before_outcome() -> None:
    """SCENARIO-ARC-ARM-6387-LEGAL-DISAGREEMENT."""

    frontier = RewardMachineFrontier(
        hypotheses=[
            _hypothesis("same", {1: SAME_FRAME_NO_LEVEL, 2: SAME_FRAME_NO_LEVEL, 3: LEVEL_UP}),
            _hypothesis(
                "level",
                {1: SAME_FRAME_NO_LEVEL, 2: LEVEL_UP, 3: SAME_FRAME_NO_LEVEL},
            ),
        ],
        capacity=5,
    )

    selection = frontier.choose_legal_disagreement(
        legal_actions=(1, 2),
        candidate_actions=(1, 2, 3),
        tick=1,
        base_policy_action=(1, None),
    )

    assert selection.action == 2
    assert selection.data is None
    assert selection.expected_elimination == 1
    assert selection.frozen_before_outcome is True
    assert selection.frozen_hypothesis_ids == ("level", "same")
    assert frontier.diagnostics()["legal_action_mutation_count"] == 0

    frontier.add_hypothesis(_hypothesis("late", {2: LEVEL_UP}))
    update = frontier.observe_action_result(
        action=2,
        tick=2,
        level_before=0,
        level_after=1,
        frame_before_hash="frame:a",
        frame_after_hash="frame:b",
        source_transition_id="live:treatment:1",
    )

    assert update.action_frozen_before_outcome is True
    assert update.evaluated_hypothesis_ids == ("level", "same")
    assert update.eliminated_hypothesis_ids == ("same",)
    assert update.kept_hypothesis_ids == ("level",)
    assert update.wrong_elimination_count == 0
    assert "late" in frontier.hypotheses
    assert "same" not in frontier.hypotheses


def test_scenario_arc_arm_6387_abstention_capacity_duplicate_contradiction_timeout() -> None:
    """SCENARIO-ARC-ARM-6387-ABSTAIN-AND-BOUNDS."""

    frontier = RewardMachineFrontier(
        hypotheses=[
            _hypothesis("a", {1: SAME_FRAME_NO_LEVEL, 2: FRAME_CHANGED_NO_LEVEL}),
            _hypothesis("b", {1: SAME_FRAME_NO_LEVEL, 2: FRAME_CHANGED_NO_LEVEL}),
        ],
        capacity=2,
        timeout_ticks=1,
    )

    no_split = frontier.choose_legal_disagreement(
        legal_actions=(1, 2),
        candidate_actions=(1, 2),
        tick=1,
        base_policy_action=(1, None),
    )
    assert no_split.action is None
    assert no_split.reason == "no_safe_legal_disagreement"
    assert no_split.fallback_action == (1, None)
    assert frontier.diagnostics()["base_policy_fallback_count"] == 1

    frontier.add_hypothesis(_hypothesis("c", {2: LEVEL_UP}))
    assert sorted(frontier.hypotheses) == ["b", "c"]
    assert frontier.diagnostics()["eviction_count"] == 1

    chosen = frontier.choose_legal_disagreement(
        legal_actions=(2,),
        candidate_actions=(2,),
        tick=2,
    )
    assert chosen.action == 2
    timed_out = frontier.observe_action_result(
        action=2,
        tick=4,
        level_before=0,
        level_after=1,
        frame_before_hash="before",
        frame_after_hash="after",
        source_transition_id="late-transition",
    )
    assert timed_out.state == "timeout"
    assert frontier.diagnostics()["timeout_count"] == 1

    chosen = frontier.choose_legal_disagreement(
        legal_actions=(2,),
        candidate_actions=(2,),
        tick=5,
    )
    assert chosen.action == 2
    contradiction = frontier.observe_action_result(
        action=2,
        tick=6,
        level_before=0,
        level_after=0,
        frame_before_hash="same",
        frame_after_hash="same",
        source_transition_id="contradiction-transition",
    )
    assert contradiction.state == "contradiction"
    assert contradiction.eliminated_hypothesis_ids == ()
    assert contradiction.wrong_elimination_count == 0
    assert sorted(frontier.hypotheses) == ["b", "c"]

    duplicate = frontier.observe_action_result(
        action=2,
        tick=6,
        level_before=0,
        level_after=0,
        frame_before_hash="same",
        frame_after_hash="same",
        source_transition_id="contradiction-transition",
    )
    assert duplicate.state == "duplicate"
    assert frontier.diagnostics()["duplicate_evidence_count"] == 1


def test_scenario_arc_arm_6387_two_sided_contract_receives_transition_evidence() -> None:
    """SCENARIO-ARC-ARM-6387-TWO-SIDED-EVIDENCE."""

    frontier = RewardMachineFrontier(
        hypotheses=[
            _hypothesis("level", {1: SAME_FRAME_NO_LEVEL, 2: LEVEL_UP}),
            _hypothesis("same", {1: SAME_FRAME_NO_LEVEL, 2: SAME_FRAME_NO_LEVEL}),
        ],
        capacity=5,
    )

    contrast = frontier.choose_legal_disagreement(
        legal_actions=(1,),
        candidate_actions=(1,),
        tick=1,
    )
    assert contrast.action is None
    frontier.force_freeze_for_testing(action=1, tick=1)
    contrast_update = frontier.observe_action_result(
        action=1,
        tick=2,
        level_before=0,
        level_after=0,
        frame_before_hash="same",
        frame_after_hash="same",
        source_transition_id="contrast",
    )
    assert contrast_update.two_sided_admission["level"]["state"] == UNVERIFIABLE

    fire = frontier.choose_legal_disagreement(
        legal_actions=(2,),
        candidate_actions=(2,),
        tick=3,
    )
    assert fire.action == 2
    fire_update = frontier.observe_action_result(
        action=2,
        tick=4,
        level_before=0,
        level_after=1,
        frame_before_hash="pre-win",
        frame_after_hash="post-win",
        source_transition_id="fire",
    )

    assert fire_update.two_sided_admission["level"]["state"] == ACCEPTED
    assert fire_update.two_sided_admission["same"]["state"] == REJECTED
    assert fire_update.two_sided_admission["level"]["solve_credit_allowed"] is True
    assert fire_update.arc_solve_claim is False


@pytest.mark.memory_watchdog_skip
def test_scenario_arc_arm_6387_live_entrypoint_default_off_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-ARM-6387-LIVE-DEFAULT-OFF."""

    import carnot.agentic.arc_competition_agent as comp

    monkeypatch.setattr(comp, "load_cross_game_value_head", lambda: None)
    monkeypatch.setattr(comp, "_load_submitted_candidate_router", lambda game_id: None)
    monkeypatch.setattr(comp, "_load_submitted_frame_change_scorer", lambda: None)
    monkeypatch.setattr(comp, "_load_submitted_goal_energy_bias", lambda: None)
    monkeypatch.delenv("CARNOT_ARC_ACTIVE_REWARD_MACHINE", raising=False)

    assert comp.SUBMITTED_AGENT_CONFIG["active_reward_machine_enabled"] is False
    off_policy = comp.E3AgentPolicy(
        "unit",
        value_head=None,
        candidate_router=None,
        frame_change_scorer=None,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_reward_machine=False,
    )
    assert off_policy.active_reward_machine_enabled is False
    assert off_policy.reward_machine_diagnostics["enabled"] is False

    on_policy = comp.E3AgentPolicy(
        "unit",
        value_head=None,
        candidate_router=None,
        frame_change_scorer=None,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_reward_machine=True,
    )
    assert on_policy.active_reward_machine_enabled is True
    assert on_policy.reward_machine_diagnostics["enabled"] is True

    monkeypatch.setenv("CARNOT_ARC_ACTIVE_REWARD_MACHINE", "1")

    class _Base:
        def __init__(self) -> None:
            self.game_id = "unit"

    agent_cls = comp.make_carnot_agent(_Base)
    agent = agent_cls()
    assert agent._policy.active_reward_machine_enabled is True
    assert "active_reward_machine" in inspect.getsource(comp.E3AgentPolicy)
    assert "CARNOT_ARC_ACTIVE_REWARD_MACHINE" in inspect.getsource(comp.E3AgentPolicy)
    assert "E3AgentPolicy(" in inspect.getsource(comp.make_carnot_agent)


def test_scenario_arc_arm_6387_fixture_manifest_and_transition_factory() -> None:
    """REQ-ARC-ARM-6387: fixtures are game-blind and transitions are visible-only."""

    manifest = default_fixture_manifest()

    assert set(manifest["fixtures"]) >= {
        "unique_disagreement",
        "no_disagreement",
        "delayed_evidence",
        "repeated_frames",
        "contradictory_evidence",
        "deadline_timeout",
    }
    assert manifest["forbidden_access_counts"] == {
        "hidden_source_reads": 0,
        "offline_search_calls": 0,
        "adapter_lookup_calls": 0,
        "oracle_result_before_action_reads": 0,
    }

    class _Transition:
        action = 2
        data = None
        level_before = 0
        level_after = 0
        grid = [[0, 0], [0, 0]]
        next_grid = [[0, 1], [0, 0]]

    frontier = reward_machine_frontier_from_transitions([_Transition()], capacity=3)
    assert 1 <= len(frontier.hypotheses) <= 3
    for hypothesis in frontier.hypotheses.values():
        assert hypothesis.evidence_count >= 1
        assert hypothesis.hidden_source_path == ""


def test_req_arc_arm_6387_unknown_disabled_and_action_normalization_branches() -> None:
    """REQ-ARC-ARM-6387: unsupported actions and disabled frontiers abstain."""

    no_pending = RewardMachineFrontier([])
    assert (
        no_pending.observe_action_result(
            action=1,
            tick=1,
            level_before=0,
            level_after=0,
            frame_before_hash="a",
            frame_after_hash="a",
            source_transition_id="unbound",
        ).state
        == "no_pending_probe"
    )

    disabled = RewardMachineFrontier(
        [
            _hypothesis("a", {1: SAME_FRAME_NO_LEVEL}),
            _hypothesis("b", {1: LEVEL_UP}),
        ],
        enabled=False,
    )
    assert disabled.choose_legal_disagreement(legal_actions=(1,), tick=1).action is None

    unknown = RewardMachineFrontier(
        [
            _hypothesis("a", {1: SAME_FRAME_NO_LEVEL}),
            _hypothesis("b", {1: LEVEL_UP}),
        ]
    )
    assert unknown.hypotheses["a"].predict(9) == "unknown"
    assert unknown.choose_legal_disagreement(legal_actions=(9,), tick=1).action is None

    class _Action:
        value = 4

    normalized = RewardMachineFrontier(
        [
            _hypothesis("a", {4: SAME_FRAME_NO_LEVEL}),
            _hypothesis("b", {4: LEVEL_UP}),
        ]
    )
    selection = normalized.choose_legal_disagreement(
        legal_actions=({"action": 1}, (2, None), _Action(), "ACTION4", "bad", 0),
        candidate_actions=({"action_id": 4},),
        tick=1,
    )
    assert selection.action == 4


@pytest.mark.memory_watchdog_skip
def test_scenario_arc_arm_6387_artifact_no_solve_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-ARM-6387-ARTIFACT-NO-SOLVE."""

    output = tmp_path / "experiment_6387_arc_active_reward_machine_discriminator.json"
    monkeypatch.setattr(
        exp6387,
        "_live_entrypoint_receipts",
        lambda: {
            "entrypoint": "make_carnot_agent -> E3AgentPolicy",
            "make_carnot_agent_importable": True,
            "e3_agent_policy_importable": True,
            "active_reward_machine_kwarg_in_e3_policy": True,
            "env_flag_supported": True,
            "make_carnot_agent_constructs_e3_policy": True,
            "submitted_default_off": True,
            "submitted_default_cannot_change_actions": True,
        },
    )
    artifact = exp6387.build_artifact(
        REPO,
        date="20260813",
        output_path=output,
        tests_run=("focused-tests",),
        duration_s=0.25,
    )

    assert output.exists()
    assert set(exp6387.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert "solve_provenance" not in artifact
    assert artifact["status"] == "complete"
    assert artifact["arc_solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["registry_write_count"] == 0
    assert artifact["arc_active_reward_machine_ready_score"] == 1.0
    assert artifact["hidden_source_offline_search_adapter_and_oracle_access_counts"] == {
        "hidden_source_reads": 0,
        "offline_search_calls": 0,
        "adapter_lookup_calls": 0,
        "oracle_result_before_action_reads": 0,
    }
    assert artifact["protected_files_unchanged"]["ops/arc_solve_registry.yaml"] is True
    assert artifact["preconditions_checked"]["exp6386_ready_score"] == 1.0
