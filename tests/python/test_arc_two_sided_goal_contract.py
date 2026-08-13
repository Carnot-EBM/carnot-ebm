"""Tests for REQ-ARC-WMTE-6386 two-sided ARC goal evidence.

These tests freeze exp6258 and prove the new verifier needs both positive
and negative bounded evidence before a goal predicate can terminate search.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot.agentic.arc_two_sided_goal_contract import (
    ACCEPTED,
    REJECTED,
    UNVERIFIABLE,
    GoalEvidenceEvent,
    GoalEvidenceDecision,
    TwoSidedGoalEvidenceContract,
    exp6258_fixture_boundary,
    replay_exp6258_contract,
    sha256_file,
)
from carnot.experiment_6386_arc_two_sided_goal_evidence_contract import (
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
)


REPO = Path(__file__).resolve().parents[2]
EXP6258 = REPO / "results" / "experiment_6258_goal_veto_confusion_matrix.json"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"


def _event(
    event_id: str,
    *,
    tick: int,
    fired: bool,
    before: int,
    after: int,
    reversal_of: str | None = None,
) -> GoalEvidenceEvent:
    return GoalEvidenceEvent(
        event_id=event_id,
        tick=tick,
        predicate_fired=fired,
        level_before=before,
        level_after=after,
        action=0,
        legal_actions=(0, 1, 2),
        visible_frame_hash=f"frame:{event_id}:{tick}:{before}:{after}:{int(fired)}",
        reversal_of=reversal_of,
    )


def test_exp6258_fixture_is_frozen_for_req_arc_wmte_6386() -> None:
    """REQ-ARC-WMTE-6386: all 29 prior predicates stay in the regression set."""

    boundary = exp6258_fixture_boundary(REPO)
    assert boundary["path_sha256"] == (
        "1a2df53becd4bb24e7db5abb9062a6fe0e4a7759f42842cca7e81e22f5457fef"
    )
    assert boundary["n_predicates"] == 29
    assert boundary["counts"] == {
        "true_accept": 5,
        "false_accept": 21,
        "false_reject": 0,
        "true_reject": 3,
    }


def test_two_sided_acceptance_needs_fire_and_contrast() -> None:
    """SCENARIO-ARC-WMTE-6386-TWO-SIDED-ACCEPTANCE."""

    contract = TwoSidedGoalEvidenceContract(max_window_ticks=5)
    decision = contract.evaluate(
        "goal-a",
        [
            _event("win", tick=2, fired=True, before=0, after=1),
            _event("contrast", tick=3, fired=False, before=1, after=1),
        ],
        firing_witness_ids=("win",),
        nonfiring_contrast_ids=("contrast",),
        window_start_tick=0,
        current_tick=4,
        deadline_tick=5,
    )
    assert decision.state == ACCEPTED
    assert decision.termination_allowed is True
    assert decision.solve_credit_allowed is True


def test_missing_win_window_is_unverifiable() -> None:
    """SCENARIO-ARC-WMTE-6386-MISSING-WIN-IS-UNVERIFIABLE."""

    contract = TwoSidedGoalEvidenceContract(max_window_ticks=5)
    decision = contract.evaluate(
        "goal-a",
        [_event("contrast", tick=1, fired=False, before=0, after=0)],
        firing_witness_ids=("win",),
        nonfiring_contrast_ids=("contrast",),
        window_start_tick=0,
        current_tick=2,
        deadline_tick=5,
    )
    assert decision.state == UNVERIFIABLE
    assert decision.termination_allowed is False
    assert decision.solve_credit_allowed is False


def test_constant_false_and_constant_true_are_rejected() -> None:
    """SCENARIO-ARC-WMTE-6386-CONTRADICTIONS-REJECT."""

    contract = TwoSidedGoalEvidenceContract()
    constant_false = contract.evaluate(
        "constant-false",
        [
            _event("win", tick=1, fired=False, before=0, after=1),
            _event("contrast", tick=2, fired=False, before=1, after=1),
        ],
        firing_witness_ids=("win",),
        nonfiring_contrast_ids=("contrast",),
        window_start_tick=0,
        current_tick=2,
        deadline_tick=3,
    )
    constant_true = contract.evaluate(
        "constant-true",
        [
            _event("win", tick=1, fired=True, before=0, after=1),
            _event("contrast", tick=2, fired=True, before=1, after=1),
        ],
        firing_witness_ids=("win",),
        nonfiring_contrast_ids=("contrast",),
        window_start_tick=0,
        current_tick=2,
        deadline_tick=3,
    )
    assert constant_false.state == REJECTED
    assert "missing_firing_witness" in constant_false.reasons
    assert constant_true.state == REJECTED
    assert "contrast_fired" in constant_true.reasons


def test_deadline_duplicate_reversal_and_contradiction_rules() -> None:
    """REQ-ARC-WMTE-6386: windows, duplicates, deadlines, reversals are bounded."""

    contract = TwoSidedGoalEvidenceContract(max_window_ticks=5)
    delayed = contract.evaluate(
        "delayed",
        [
            _event("win", tick=7, fired=True, before=0, after=1),
            _event("contrast", tick=2, fired=False, before=0, after=0),
        ],
        firing_witness_ids=("win",),
        nonfiring_contrast_ids=("contrast",),
        window_start_tick=0,
        current_tick=7,
        deadline_tick=5,
    )
    assert delayed.state == UNVERIFIABLE
    assert "missing_firing_witness" in delayed.reasons

    duplicate = contract.evaluate(
        "duplicate",
        [
            _event("win", tick=1, fired=True, before=0, after=1),
            _event("win", tick=1, fired=True, before=0, after=1),
            _event("contrast", tick=2, fired=False, before=1, after=1),
        ],
        firing_witness_ids=("win",),
        nonfiring_contrast_ids=("contrast",),
        window_start_tick=0,
        current_tick=2,
        deadline_tick=3,
    )
    assert duplicate.state == ACCEPTED
    assert duplicate.duplicate_event_count == 1

    contradictory = contract.evaluate(
        "contradictory",
        [
            _event("win", tick=1, fired=True, before=0, after=1),
            _event("win", tick=1, fired=False, before=0, after=1),
            _event("contrast", tick=2, fired=False, before=1, after=1),
        ],
        firing_witness_ids=("win",),
        nonfiring_contrast_ids=("contrast",),
        window_start_tick=0,
        current_tick=2,
        deadline_tick=3,
    )
    assert contradictory.state == REJECTED
    assert "contradictory_duplicate" in contradictory.reasons

    reversed_decision = contract.evaluate(
        "reversed",
        [
            _event("win", tick=1, fired=True, before=0, after=1),
            _event("contrast", tick=2, fired=False, before=1, after=1),
            _event("undo-win", tick=3, fired=False, before=1, after=1, reversal_of="win"),
        ],
        firing_witness_ids=("win",),
        nonfiring_contrast_ids=("contrast",),
        window_start_tick=0,
        current_tick=3,
        deadline_tick=4,
    )
    assert reversed_decision.state == UNVERIFIABLE
    assert "missing_firing_witness" in reversed_decision.reasons


def test_unverifiable_hypothesis_can_rank_one_legal_probe_but_not_terminate() -> None:
    """REQ-ARC-WMTE-6386: unverified goals can rank probes only."""

    contract = TwoSidedGoalEvidenceContract()
    decision = contract.evaluate(
        "no-win",
        [_event("contrast", tick=1, fired=False, before=0, after=0)],
        firing_witness_ids=("win",),
        nonfiring_contrast_ids=("contrast",),
        window_start_tick=0,
        current_tick=1,
        deadline_tick=3,
    )
    ranked = contract.rank_one_legal_probe(
        decision,
        [{"action": 1}, {"action": 2}, {"action": 3}],
        legal_actions=(1, 2),
        preferred_action=2,
    )
    assert decision.state == UNVERIFIABLE
    assert decision.termination_allowed is False
    assert decision.solve_credit_allowed is False
    assert [row["action"] for row in ranked] == [2, 1, 3]
    assert contract.probe_rank_count == 1


def test_exp6258_replay_new_matrix_has_no_false_accepts() -> None:
    """REQ-ARC-WMTE-6386: prior false accepts are rejected or unverifiable."""

    replay = replay_exp6258_contract(REPO)
    assert replay["old_confusion_matrix"] == {
        "true_accept": 5,
        "false_accept": 21,
        "false_reject": 0,
        "true_reject": 3,
    }
    assert replay["new_confusion_matrix"]["accepted"] == 5
    assert replay["new_confusion_matrix"]["rejected"] == 24
    assert replay["new_confusion_matrix"]["unverifiable"] == 0
    assert replay["new_false_accept_count"] == 0
    assert replay["new_false_reject_count"] == 0
    assert replay["prior_false_accepts_rejected_or_unverifiable"] == 21


@pytest.mark.memory_watchdog_skip
def test_experiment_6386_artifact_schema_and_contract(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6386: the deliverable fields and no-solve boundary are explicit."""

    artifact = build_artifact(
        REPO,
        date="20260813",
        output_path=tmp_path / "out.json",
        live_entrypoint_receipts={
            "entrypoint": "make_carnot_agent -> E3AgentPolicy -> StepwiseExplorer",
            "contract_kwarg_in_e3_policy": True,
            "contract_forwarded_to_explorer": True,
            "env_flag_supported": True,
            "submitted_default_off": True,
        },
    )
    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert "solve_provenance" not in artifact
    assert artifact["arc_solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["arc_two_sided_goal_contract_ready_score"] == 1.0
    assert artifact["registry_precheck_and_hash"]["sha256"] == sha256_file(REGISTRY)
    assert artifact["protected_files_unchanged"]["ops/arc_solve_registry.yaml"] is True


@pytest.mark.memory_watchdog_skip
def test_live_entrypoint_default_off_and_flag_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-6386-LIVE-DEFAULT-OFF."""

    import carnot.agentic.arc_competition_agent as comp

    monkeypatch.setattr(comp, "load_cross_game_value_head", lambda: None)
    monkeypatch.setattr(comp, "_load_submitted_candidate_router", lambda game_id: None)
    monkeypatch.setattr(comp, "_load_submitted_frame_change_scorer", lambda: None)
    monkeypatch.setattr(comp, "_load_submitted_goal_energy_bias", lambda: None)
    monkeypatch.delenv("CARNOT_ARC_TWO_SIDED_GOAL_CONTRACT", raising=False)

    assert comp.SUBMITTED_AGENT_CONFIG["two_sided_goal_contract_enabled"] is False
    off_policy = comp.E3AgentPolicy(
        "unit",
        value_head=None,
        candidate_router=None,
        frame_change_scorer=None,
        goal_bias=None,
        goal_candidate_guidance=False,
        two_sided_goal_contract=False,
    )
    assert off_policy.two_sided_goal_contract is None
    assert off_policy.explorer.two_sided_goal_contract is None

    on_policy = comp.E3AgentPolicy(
        "unit",
        value_head=None,
        candidate_router=None,
        frame_change_scorer=None,
        goal_bias=None,
        goal_candidate_guidance=False,
        two_sided_goal_contract=True,
    )
    assert on_policy.two_sided_goal_contract is not None
    assert on_policy.explorer.two_sided_goal_contract is on_policy.two_sided_goal_contract

    def _planner(_engine, is_done, start_grid, **_kwargs):
        return ["terminated"] if is_done(start_grid) else []

    diagnostics: dict[str, object] = {}
    assert on_policy._call_plan_in_model(
        _planner,
        object(),
        lambda _grid: True,
        object(),
        diagnostics=diagnostics,
    ) == []
    assert diagnostics["two_sided_goal_contract_state"] == UNVERIFIABLE

    on_policy.two_sided_goal_contract.set_decision(
        GoalEvidenceDecision("accepted", ACCEPTED, ("bounded_two_sided_evidence",))
    )
    assert on_policy._call_plan_in_model(
        _planner,
        object(),
        lambda _grid: True,
        object(),
        diagnostics={},
    ) == ["terminated"]

    monkeypatch.setenv("CARNOT_ARC_TWO_SIDED_GOAL_CONTRACT", "1")

    class _Base:
        def __init__(self) -> None:
            self.game_id = "unit"

    agent_cls = comp.make_carnot_agent(_Base)
    agent = agent_cls()
    assert agent._policy.two_sided_goal_contract is not None
