import json
import sys
from pathlib import Path

from carnot.agentic.arc_verifier_validated_reinduction import (
    REQUIRED_ARTIFACT_FIELDS,
    RuleValidation,
    actions_saved_vs_openloop,
    artifact_schema_errors,
    choose_verified_candidate,
    executed_consistency_energy,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3992_incremental_levels_verifier_validated as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _r11l_stall() -> dict[str, object]:
    return {
        "game": "r11l-495a7899",
        "ACCURACY_levels_solved": 1,
        "first_fail_level": 2,
        "per_level_actions": [4],
        "baseline_actions_ref": [4],
        "induced_select_place_mechanic": "L1 select/place around target centroid",
        "level_summaries": [
            {"level": 1, "n_pairs": 2, "n_targets": 1},
            {"level": 2, "n_pairs": 5, "n_targets": 2},
        ],
    }


def _lp85_stall() -> dict[str, object]:
    return {
        "game": "lp85-305b61c3",
        "ACCURACY_levels_solved": 1,
        "first_fail_level": 2,
        "per_level_actions": [5],
        "baseline_actions_ref": [17],
        "induced_mechanic": "L1 button permutation",
        "level_summaries": [
            {"level": 1, "n_buttons": 2},
            {"level": 2, "n_buttons": 0},
        ],
    }


def test_req_phase4_021_spec_declares_verifier_validated_reinduction() -> None:
    """REQ-PHASE4-021: OpenSpec declares Exp 3992 and its terminal fields."""
    spec = SPEC_PATH.read_text("utf-8")

    assert "REQ-PHASE4-021" in spec
    assert "SCENARIO-PHASE4-021" in spec
    assert "GAP-4-style executed-consistency verifier" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_021_artifact_schema_requires_bare_scalars_and_prefix() -> None:
    """REQ-PHASE4-021: Exp 3992 artifacts keep verifier fields bare and auditable."""
    artifact = exp._base_artifact(seed=3992, started=0.0, verdict="complete: l2_wall_holds_r11l_test")
    artifact.update(
        {
            "ACCURACY_levels_solved": 1,
            "new_levels_solved_this_task": 0,
            "verifier_validated_the_rule": False,
            "actions_saved_vs_openloop": 2,
            "game_advanced": "r11l-495a7899_to_L1",
            "per_level_actions": [4],
            "baseline_actions_ref": [4],
            "real_env_confirmed": True,
            "duration_s": 1.0,
        }
    )

    assert artifact_schema_errors(artifact) == []
    bad = dict(artifact)
    bad["actions_saved_vs_openloop"] = "2"
    bad["verifier_validated_the_rule"] = 1
    bad["honest_verdict"] = "done"

    errors = artifact_schema_errors(bad)

    assert any("actions_saved_vs_openloop" in err for err in errors)
    assert any("verifier_validated_the_rule" in err for err in errors)
    assert any("honest_verdict" in err for err in errors)

    missing = dict(artifact)
    del missing["per_level_actions"]
    missing["game_advanced"] = 123
    missing["baseline_actions_ref"] = [4, "10"]
    missing["duration_s"] = "slow"
    missing_errors = artifact_schema_errors(missing)

    assert any("missing required field per_level_actions" in err for err in missing_errors)
    assert any("game_advanced must be a bare string" in err for err in missing_errors)
    assert any("baseline_actions_ref must be a list of bare ints" in err for err in missing_errors)
    assert any("duration_s must be a bare number" in err for err in missing_errors)


def test_scenario_phase4_021_selects_verified_candidate_by_executed_consistency() -> None:
    """SCENARIO-PHASE4-021: candidate rules are chosen by held-out executed consistency."""
    rejected = RuleValidation(
        candidate_id="l1-transfer",
        rule_name="L1 centroid transfer",
        demo_fit=1.0,
        heldout_energy=0.75,
        heldout_n=2,
        predicted_levels_after=2,
        validated_levels_after=1,
        planned_l2_actions=10,
    )
    shallow = RuleValidation(
        candidate_id="single-move-mask",
        rule_name="collision-mask single move",
        demo_fit=1.0,
        heldout_energy=0.0,
        heldout_n=1,
        predicted_levels_after=1,
        validated_levels_after=1,
        planned_l2_actions=2,
    )
    chosen = RuleValidation(
        candidate_id="safe-composite-path",
        rule_name="collision-forbidden safe path",
        demo_fit=1.0,
        heldout_energy=0.0,
        heldout_n=2,
        predicted_levels_after=2,
        validated_levels_after=2,
        planned_l2_actions=8,
    )

    assert choose_verified_candidate([rejected, shallow, chosen], current_level=1) == chosen
    assert choose_verified_candidate([rejected], current_level=1) is None


def test_req_phase4_021_executed_consistency_energy_counts_heldout_mismatches() -> None:
    """REQ-PHASE4-021: held-out transition agreement is a real mismatch-sensitive energy."""
    expected = [
        {"piece_after": [1, 2], "composite_after": [3, 4]},
        {"piece_after": [5, 6], "composite_after": [7, 8]},
    ]
    observed = [
        {"piece_after": [1, 2], "composite_after": [3, 4]},
        {"piece_after": [5, 6], "composite_after": [99, 8]},
    ]

    assert executed_consistency_energy(expected, expected) == 0.0
    assert executed_consistency_energy(expected, observed) == 0.25
    assert executed_consistency_energy([], []) is None
    assert actions_saved_vs_openloop(openloop_actions=2, committed_rejected_actions=0) == 2


def test_scenario_phase4_021_blocks_when_offline_arcade_unavailable(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-021: unavailable offline Arcade writes a blocked artifact."""
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_prior_stalls", lambda: {"r11l": _r11l_stall(), "lp85": _lp85_stall()})
    monkeypatch.setattr(exp, "_load_exp3980", lambda: {"l2_attempted_actions": 2})

    def unavailable() -> object:
        raise RuntimeError("offline missing")

    monkeypatch.setattr(exp, "_load_offline_arcade", unavailable)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["real_env_confirmed"] is False
    assert artifact["verifier_validated_the_rule"] is False
    assert artifact_schema_errors(artifact) == []
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text("utf-8"))["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_scenario_phase4_021_success_uses_real_env_validated_candidate(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-021: a validated real-env advance writes the success verdict."""
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_prior_stalls", lambda: {"r11l": _r11l_stall(), "lp85": _lp85_stall()})
    monkeypatch.setattr(exp, "_load_exp3980", lambda: {"l2_attempted_actions": 2})
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_select_game_id", lambda arc, short_game: "r11l-495a7899")

    def fake_execute(arc, game_id, budget, openloop_actions):
        return {
            "ACCURACY_levels_solved": 2,
            "new_levels_solved_this_task": 1,
            "verifier_validated_the_rule": True,
            "actions_saved_vs_openloop": 2,
            "game_advanced": "r11l-495a7899_to_L2",
            "per_level_actions": [4, 8],
            "baseline_actions_ref": [4, 10],
            "real_env_confirmed": True,
            "first_fail_level": 3,
            "selected_candidate": "safe-composite-path",
            "candidate_validations": [{"candidate_id": "safe-composite-path", "heldout_energy": 0.0}],
            "level_summaries": [{"level": 2, "levels_completed_after": 2}],
            "per_level": [{"level": 2, "levels_completed_after": 2, "actions_used": 8}],
            "solve_log": [{"level": 2, "action": "click", "x": 1, "y": 2}],
            "rule_diagnosis": "L2 safe-composite path passed executed consistency",
        }

    monkeypatch.setattr(exp, "_execute_r11l_verifier_validated", fake_execute)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: verifier_validated_reinduction_advanced_r11l_to_L2"
    assert artifact["ACCURACY_levels_solved"] == 2
    assert artifact["new_levels_solved_this_task"] == 1
    assert artifact["verifier_validated_the_rule"] is True
    assert artifact_schema_errors(artifact) == []
