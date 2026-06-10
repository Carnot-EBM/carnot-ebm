import json
import sys
from pathlib import Path

from carnot.agentic.arc_level_reinduction import (
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    choose_reinduction_candidate,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_3980_incremental_levels_reinduction as exp


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _r11l_stall():
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


def _lp85_stall():
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


def test_req_phase4_018_spec_declares_incremental_reinduction():
    """REQ-PHASE4-018: OpenSpec declares Exp 3980 and its bare terminal fields."""
    spec = SPEC_PATH.read_text("utf-8")

    assert "REQ-PHASE4-018" in spec
    assert "SCENARIO-PHASE4-018" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_018_uses_first_fail_diagnostics_to_choose_one_game():
    """REQ-PHASE4-018: the one-game scope is chosen from L2 stall diagnostics."""
    choice = choose_reinduction_candidate({"r11l": _r11l_stall(), "lp85": _lp85_stall()})

    assert choice.short_game == "r11l"
    assert choice.game_id == "r11l-495a7899"
    assert choice.first_fail_level == 2
    assert choice.baseline_actions_ref == [4]
    assert "visible L2 piece-target" in choice.reason


def test_req_phase4_018_artifact_schema_requires_bare_scalars_and_prefix():
    """REQ-PHASE4-018: terminal artifact fields stay auditable and prefix-honest."""
    artifact = exp._base_artifact(seed=3980, started=0.0, verdict="complete: l2_wall_holds_r11l_test")
    artifact.update(
        {
            "ACCURACY_levels_solved": 1,
            "new_levels_solved_this_task": 0,
            "reinduction_found_different_rule": True,
            "game_advanced": "r11l-495a7899_to_L1",
            "per_level_actions": [4],
            "baseline_actions_ref": [4],
            "real_env_confirmed": True,
            "duration_s": 1.0,
        }
    )

    assert artifact_schema_errors(artifact) == []
    bad = dict(artifact)
    bad["new_levels_solved_this_task"] = {"value": 1}
    bad["honest_verdict"] = "done"
    errors = artifact_schema_errors(bad)
    assert any("new_levels_solved_this_task" in err for err in errors)
    assert any("honest_verdict" in err for err in errors)


def test_scenario_phase4_018_blocks_when_offline_arcade_unavailable(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-018: unavailable offline Arcade writes a blocked artifact."""
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_prior_stalls", lambda: {"r11l": _r11l_stall(), "lp85": _lp85_stall()})

    def unavailable():
        raise RuntimeError("offline missing")

    monkeypatch.setattr(exp, "_load_offline_arcade", unavailable)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["real_env_confirmed"] is False
    assert artifact["new_levels_solved_this_task"] == 0
    assert artifact_schema_errors(artifact) == []
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text("utf-8"))["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_scenario_phase4_018_success_uses_real_env_confirmation(monkeypatch, tmp_path):
    """SCENARIO-PHASE4-018: a real-env-confirmed L2 advance writes the success verdict."""
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_prior_stalls", lambda: {"r11l": _r11l_stall(), "lp85": _lp85_stall()})
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(exp, "_select_game_id", lambda arc, short_game: "r11l-495a7899")

    def fake_execute(arc, choice, game_id, budget):
        return {
            "ACCURACY_levels_solved": 2,
            "new_levels_solved_this_task": 1,
            "reinduction_found_different_rule": True,
            "game_advanced": "r11l-495a7899_to_L2",
            "per_level_actions": [4, 6],
            "baseline_actions_ref": [4, 10],
            "real_env_confirmed": True,
            "first_fail_level": 3,
            "per_level": [{"level": 2, "levels_completed_after": 2}],
            "solve_log": [{"level": 2, "action": "click", "x": 1, "y": 2}],
            "rule_diagnosis": "L2 re-induced split-target placement rule",
        }

    monkeypatch.setattr(exp, "_execute_candidate", fake_execute)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: reinduction_advanced_r11l_to_L2"
    assert artifact["ACCURACY_levels_solved"] == 2
    assert artifact["new_levels_solved_this_task"] == 1
    assert artifact["reinduction_found_different_rule"] is True
    assert artifact_schema_errors(artifact) == []
