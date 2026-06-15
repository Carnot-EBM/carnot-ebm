"""Tests for Exp 4236 ARC-AGI-3 SC25 L4 incremental progress.

Spec refs: REQ-PHASE4-063, SCENARIO-PHASE4-063.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_4236_arc_incremental_progress as exp
from carnot.experiment_4236_arc_incremental_progress import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_LEVELS_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    SC25_GAME_ID,
    FrontierOutcome,
    TargetSelection,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    select_deeper_level_target,
    validate_hardened_gap4_l4_suffix,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _hardening_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4187_gap4_graded_execution_gate_hardening",
        "vote_aware_guard_blocked_mispromotion": True,
        "gross_recovery_ledger": {"recovered": 4, "lost": 0},
    }


def _prior_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4224_arc_incremental_progress",
        "honest_verdict": "success: incremental_progress_sc25-635fd71a_advanced_to_L3_total17",
        "target_game": SC25_GAME_ID,
        "target_level": 3,
        "total_levels_solved": 17,
        "levels_completed": 3,
        "new_levels_solved_this_task": 1,
        "real_env_confirmed": True,
        "action_plan": [{"action": 4, "kind": "precast_face_right"}],
    }


def _target() -> TargetSelection:
    return TargetSelection(
        game="sc25",
        game_id=SC25_GAME_ID,
        target_level=4,
        prior_level=3,
        baseline_actions=83,
        selection_mode="deeper_sc25_frontier_after_exp4224_L3",
        selection_reason="selected sc25 L4 after Exp 4224 banked sc25 L3 with hardened GAP-4 evidence",
    )


def _outcome(*, advanced: bool) -> FrontierOutcome:
    validation = validate_hardened_gap4_l4_suffix(
        start_level=3,
        final_level=4 if advanced else 3,
        heldout_transition_count=6,
        predicted_level=4,
        gap4_artifact=_hardening_artifact(),
    )
    return FrontierOutcome(
        target_game=SC25_GAME_ID,
        target_level=4,
        prior_level=3,
        final_level_completed=4 if advanced else 3,
        replay_actions_used=44,
        executed_real_env_actions=14 if advanced else 0,
        exploration_actions_used=51,
        real_env_confirmed=advanced,
        verifier_validated=advanced,
        verification_decisions=[validation],
        action_plan=[
            {"action": 6, "kind": "pattern_click", "spell": "fibcey", "x": 30, "y": 50},
            {"action": 6, "kind": "spell_select", "spell": "sieesc_chwjgc", "x": 1, "y": 1},
            {"action": 2, "kind": "move"},
        ]
        if advanced
        else [],
        phase_trace=[
            {"phase": "observe", "levels_completed": 0},
            {"phase": "replay", "source": "sc25_L1_reestablishment"},
            {"phase": "replay", "source": "sc25_L2_banked_suffix"},
            {"phase": "replay", "source": "sc25_L3_banked_suffix"},
            {"phase": "explore", "source": "copied_env_sc25_L4_multispell_transitions"},
            {"phase": "induce", "mechanic": "multi-spell pattern sequence"},
            validation,
            {"phase": "act", "levels_completed": 4 if advanced else 3},
        ],
        induced_mechanic="sc25 L4 multi-spell pattern sequence followed by exit-touch movement",
        failure_reason="" if advanced else "no_verifier_validated_level_up_candidate",
    )


def test_req_phase4_063_spec_declares_exp4236_contract() -> None:
    """REQ-PHASE4-063: OpenSpec declares the Exp 4236 terminal artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-063" in spec
    assert "SCENARIO-PHASE4-063" in spec
    assert "experiment_4236_arc_incremental_progress.json" in spec
    assert "sc25-635fd71a" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_063_selects_sc25_l4_after_exp4224_l3() -> None:
    """REQ-PHASE4-063: target selection prefers the SC25 L4 frontier."""

    survey = {"per_game_surveys": [{"game": "sc25"}]}
    baselines = {"sc25": (SC25_GAME_ID, [36, 6, 32, 83])}

    assert select_deeper_level_target(survey, baselines, _prior_artifact(), _hardening_artifact()) == _target()

    with pytest.raises(ValueError, match="Exp 4224 sc25 L3 success evidence unavailable"):
        select_deeper_level_target(
            survey,
            baselines,
            {**_prior_artifact(), "real_env_confirmed": False},
            _hardening_artifact(),
        )
    with pytest.raises(ValueError, match="hardened GAP-4 verifier evidence unavailable"):
        select_deeper_level_target(survey, baselines, _prior_artifact(), {"gross_recovery_ledger": {"lost": 1}})
    with pytest.raises(ValueError, match="sc25 offline fixture metadata unavailable"):
        select_deeper_level_target(survey, {"sc25": (SC25_GAME_ID, [36, 6, 32])}, _prior_artifact(), _hardening_artifact())


def test_scenario_phase4_063_artifact_schema_accepts_success_and_complete() -> None:
    """SCENARIO-PHASE4-063: only hardened-verified real-env evidence increments levels."""

    success = build_artifact(_outcome(advanced=True), _target(), random_seed=4236, duration_s=0.2)

    assert success["honest_verdict"] == "success: incremental_progress_sc25-635fd71a_advanced_to_L4_total18"
    assert success["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED + 1
    assert success["levels_completed"] == 4
    assert success["real_env_confirmed"] is True
    assert success["acceptance_gate_passed"] is True
    assert success["inference_substrate"] == INFERENCE_SUBSTRATE
    assert success["solve_trace"]["actions"][0]["kind"] == "pattern_click"
    assert artifact_schema_errors(success) == []

    no_solve = build_artifact(_outcome(advanced=False), _target(), random_seed=4236, duration_s=0.2)
    assert no_solve["honest_verdict"].startswith("complete: incremental_progress_no_solve_sc25-635fd71a_L4")
    assert no_solve["total_levels_solved"] == PRIOR_TOTAL_LEVELS_SOLVED
    assert no_solve["new_levels_solved_this_task"] == 0
    assert no_solve["real_env_confirmed"] is False
    assert no_solve["acceptance_gate_passed"] is True

    blocked = blocked_artifact(target_game=SC25_GAME_ID, target_level=4, random_seed=4236, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["acceptance_gate_passed"] is False
    assert artifact_schema_errors(blocked) == []


def test_scenario_phase4_063_validation_and_schema_reject_fabrication() -> None:
    """SCENARIO-PHASE4-063: validation must precede acting and schema rejects inflation."""

    retained = validate_hardened_gap4_l4_suffix(
        start_level=3,
        final_level=4,
        heldout_transition_count=4,
        predicted_level=4,
        gap4_artifact=_hardening_artifact(),
    )
    rejected = validate_hardened_gap4_l4_suffix(
        start_level=3,
        final_level=3,
        heldout_transition_count=4,
        predicted_level=4,
        gap4_artifact=_hardening_artifact(),
    )

    assert retained["retained"] is True
    assert retained["verifier"] == exp.HARDENED_VERIFIER
    assert rejected["retained"] is False
    assert rejected["energy"] == 1.0
    assert any("honest_verdict must be a string" in err for err in artifact_schema_errors({"honest_verdict": 4236}))
    fabricated = build_artifact(_outcome(advanced=True), _target(), random_seed=4236, duration_s=0.0)
    fabricated["real_env_confirmed"] = False
    assert any("real_env_confirmed must be true for success" in err for err in artifact_schema_errors(fabricated))


def test_scenario_phase4_063_runner_writes_real_env_confirmed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-PHASE4-063: runner writes replay-explore-verify-act evidence."""

    (tmp_path / "results").mkdir()
    (tmp_path / "environment_files" / "sc25" / "635fd71a").mkdir(parents=True)
    (tmp_path / "results" / "arc3_win_condition_survey.json").write_text(
        json.dumps({"per_game_surveys": [{"game": "sc25"}]}),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4224_arc_incremental_progress.json").write_text(
        json.dumps(_prior_artifact()),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_4187_gap4_graded_execution_gate_hardening.json").write_text(
        json.dumps(_hardening_artifact()),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_fixture_available", lambda game_id: True)
    monkeypatch.setattr(exp, "load_environment_baselines", lambda root: {"sc25": (SC25_GAME_ID, [36, 6, 32, 83])})
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_run_sc25_l4_frontier", lambda *args: _outcome(advanced=True))

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: incremental_progress_sc25-635fd71a_advanced_to_L4_total18"
    assert artifact["total_levels_solved"] == 18
    assert artifact["levels_completed"] == 4
    assert artifact["real_env_confirmed"] is True
    assert [row["phase"] for row in artifact["phase_trace"] if row["phase"] in {"observe", "induce", "hardened-gap4-verify"}]
    written = json.loads((tmp_path / "results" / "experiment_4236_arc_incremental_progress.json").read_text(encoding="utf-8"))
    assert written["solve_trace"]["actions"][0]["kind"] == "pattern_click"


def test_req_phase4_063_entrypoint_exists() -> None:
    """REQ-PHASE4-063: the required command path has a Python entrypoint."""

    entrypoint = REPO / "results" / "experiment_4236_arc_incremental_progress.py"

    assert entrypoint.exists()
    assert "carnot.experiment_4236_arc_incremental_progress" in entrypoint.read_text(encoding="utf-8")
