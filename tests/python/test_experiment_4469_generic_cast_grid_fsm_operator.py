"""Tests for Exp 4469 generic cast-grid phase-FSM operator.

Spec refs: REQ-REPORT-4469, SCENARIO-REPORT-4469.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import yaml

from carnot import experiment_4469_generic_cast_grid_fsm_operator as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _ok_preconditions() -> dict[str, Any]:
    return {
        "sc25_environment_files": True,
        "arc_solver_imports": True,
        "world_model_verifier_imports": True,
        "existing_world_models": 4,
        "gguf_cached": True,
        "igpu_llama_server": False,
        "generator_resource_available": True,
        "baseline_command": mod.BASELINE_COMMAND_TEXT,
        "baseline_exit_code": 0,
        "baseline_pytest_nocov_green": True,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "environment_files" / "sc25" / "fixture").mkdir(parents=True)
    for game in mod.SOLVED_EXAMPLE_GAMES:
        path = root / "results" / "arc_e3" / game / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "def engine(grid, action, data):\n    return grid\n"
            "def is_level_complete(grid):\n    return False\n",
            encoding="utf-8",
        )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "games": [
                    {
                        "game": "sc25",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 5,
                        "mechanic_class": "two_phase_cast_grid_then_tank_exit",
                        "dead_ends": [
                            {
                                "gap_id": mod.SC25_GAP_ID,
                                "status": "open",
                                "failure_mode": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                            }
                        ],
                    }
                ],
                "reproducible_total_levels": 40,
                "reproducible_total_games": 21,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text(
        "<!-- exp4469-gap-sc25-cast-grid:start -->\n"
        "old block\n"
        "<!-- exp4469-gap-sc25-cast-grid:end -->\n",
        encoding="utf-8",
    )


def _clock() -> tuple[dict[str, float], Any, Any]:
    clock = {"t": 10.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    return clock, now, sleep


def _generic_solve_result() -> dict[str, Any]:
    return {
        "solution": list(mod.SC25_GENERIC_L1_EXPECTED),
        "reached_level": 1,
        "operator_result": {
            "operator": "cast_grid_phase_fsm_world_model",
            "grounded": True,
            "recipe_source": "generic_cast_grid_phase_fsm_world_model",
            "target_recipe_withheld": "sc25",
            "counterexample_rounds": 1,
            "solution": list(mod.SC25_GENERIC_L1_EXPECTED),
        },
        "world_model_verification": {
            "world_model_loaded": True,
            "verifier_accuracy": 1.0,
            "transitions_scored": 16,
            "transitions_correct": 16,
            "mismatches": [],
        },
        "plan_and_execute_result": {
            "planned": True,
            "executed": True,
            "level_up": True,
            "plan_len": len(mod.SC25_GENERIC_L1_EXPECTED),
        },
        "counterexample_rounds": 1,
        "solver_source": "generic_cast_grid_phase_fsm_world_model_without_sc25_hand_recipe",
    }


def _generic_reproduce(solution: Sequence[str]) -> dict[str, Any]:
    assert list(solution) == list(mod.SC25_GENERIC_L1_EXPECTED)
    return {"game": "sc25", "claimed_level": 1, "reached_level": 1, "reproduced": True}


def test_req_report_4469_spec_declares_cast_grid_contract() -> None:
    """REQ-REPORT-4469: OpenSpec declares the operator and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4469" in spec
    assert "SCENARIO-REPORT-4469" in spec
    assert "cast_grid_phase_fsm_world_model" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4469_solver_kit_induces_two_phase_cast_grid_fsm() -> None:
    """REQ-REPORT-4469: cast-grid toggle CSP transitions to navigation."""

    result = kit.cast_grid_phase_fsm_world_model(
        game="sc25",
        object_digest=mod.SC25_CAST_GRID_DIGEST,
        few_shot_examples=mod.DEFAULT_CAST_GRID_EXAMPLES,
    )
    cold = kit.cast_grid_phase_fsm_world_model(
        game="sc25",
        object_digest=mod.SC25_CAST_GRID_DIGEST,
        few_shot_examples=(),
    )

    assert result["operator"] == "cast_grid_phase_fsm_world_model"
    assert result["grounded"] is True
    assert result["target_recipe_withheld"] == "sc25"
    assert result["recipe_source"] == "generic_cast_grid_phase_fsm_world_model"
    assert result["predicate_id"] == "toggle_csp_then_navigate_exit"
    assert result["phase_model"]["phases"] == ["config_toggle", "navigate_exit"]
    assert result["solution"] == list(mod.SC25_GENERIC_L1_EXPECTED)
    assert result["solution"][:4] == ["cell0,1", "cell1,0", "cell1,2", "cell2,1"]
    assert result["solution"][4:] == ["move3"] * 12
    assert result["counterexample_rounds"] == 1
    assert result["verifier_is_oracle"] is True
    assert cold["grounded"] is False
    assert cold["residual"] == "missing_cast_grid_phase_fsm_few_shot_examples"

    grid = mod.synthetic_sc25_l1_grid()
    for label in result["solution"]:
        action, data = mod.sc25_label_to_action_data(label)
        grid = result["engine"](grid, action, data)

    assert result["is_level_complete"](grid) is True
    assert int(np.count_nonzero(grid == mod.SC25_CAST_GRID_DIGEST["cast_active_color"])) == 0

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert "cast_grid_phase_fsm_world_model" in operators
    selected = kit.select_primitive_operators(mechanic_class="two_phase_cast_grid_then_tank_exit")
    assert selected[0].operator == "cast_grid_phase_fsm_world_model"


def test_scenario_report_4469_run_reproduces_sc25_generically(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4469: generic sc25 L1 closure is reproduction-gated."""

    _write_fixture_repo(tmp_path)
    _clock_state, now, sleep = _clock()

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        few_shot_examples=mod.DEFAULT_CAST_GRID_EXAMPLES,
        solve_sc25_fn=lambda _examples: _generic_solve_result(),
        reproduce_generic_fn=_generic_reproduce,
        no_regression_fn=lambda _root: True,
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == "success: sc25_generic_cast_grid_fsm_L1_offline_reproduced"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["sc25_resolved_generically"] is True
    assert artifact["sc25_generic_level_reproduced"] == 1
    assert artifact["counterexample_rounds"] == 1
    assert artifact["offline_reproduced"] is True
    assert artifact["no_regression"] is True
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["generic_operator_result"]["target_recipe_withheld"] == "sc25"

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    sc25 = next(row for row in registry["games"] if row["game"] == "sc25")
    assert sc25["latest_exp4469_generic_cast_grid"]["sc25_resolved_generically"] is True
    assert sc25["dead_ends"][0]["status"] == "filled"
    gaps = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "status: filled" in gaps
    assert mod.SC25_GAP_ID in gaps


def test_req_report_4469_measured_no_generalize_is_complete(tmp_path: Path) -> None:
    """REQ-REPORT-4469: a real generic miss reports the residual, not partial."""

    _write_fixture_repo(tmp_path)
    _clock_state, now, sleep = _clock()

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        few_shot_examples=mod.DEFAULT_CAST_GRID_EXAMPLES,
        solve_sc25_fn=lambda _examples: {
            **_generic_solve_result(),
            "solution": [],
            "reached_level": 0,
            "operator_result": {
                "operator": "cast_grid_phase_fsm_world_model",
                "grounded": False,
                "solution": [],
                "counterexample_rounds": 1,
                "residual": "cast_grid_phase_fsm_candidate_did_not_ground",
            },
        },
        reproduce_generic_fn=lambda _solution: pytest.fail("reproduce must not run"),
        no_regression_fn=lambda _root: True,
        write_registry=False,
        write_gaps=False,
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == "complete: sc25_generic_cast_grid_fsm_no_reproduced_level_gap_logged"
    assert artifact["sc25_resolved_generically"] is False
    assert artifact["sc25_generic_level_reproduced"] == 0
    assert artifact["offline_reproduced"] is False
    assert artifact["missing_verifier_gaps"][0]["residual_delta"] == "cast_grid_phase_fsm_candidate_did_not_ground"
    assert "partial:" not in artifact["honest_verdict"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4469_blocked_precondition_and_schema_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4469: blocked resources do not fabricate generic closure."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "sc25_environment_files": False, "ok": False},
        few_shot_examples=mod.DEFAULT_CAST_GRID_EXAMPLES,
        solve_sc25_fn=lambda _examples: calls.append("solve") or {},
        reproduce_generic_fn=lambda _solution: calls.append("reproduce") or {},
        no_regression_fn=lambda _root: calls.append("regression") or True,
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_offline_env_sc25"
    assert artifact["inference_substrate"] == mod.BLOCKED_INFERENCE_SUBSTRATE
    assert artifact["sc25_resolved_generically"] is False
    assert artifact["sc25_generic_level_reproduced"] == 0
    assert artifact["offline_reproduced"] is False
    assert artifact["no_regression"] is False
    assert mod.artifact_schema_errors(artifact) == []

    bad: Mapping[str, Any] = {
        **artifact,
        "honest_verdict": "partial: fake",
        "inference_substrate": None,
        "sc25_resolved_generically": "true",
        "sc25_generic_level_reproduced": "1",
        "counterexample_rounds": "1",
        "offline_reproduced": "true",
        "no_regression": "true",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "random_seed": "4469",
        "reproducibility_checksum": "bad",
        "no_3090_inference": False,
        "submitted_to_leaderboard": True,
        "field_principles": {},
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "inference_substrate must not be None" in errors
    assert "sc25_resolved_generically must be bare bool" in errors
    assert "sc25_generic_level_reproduced must be bare int" in errors
    assert "counterexample_rounds must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "no_regression must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "no_3090_inference must be true" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4469" in errors

    short_cached = {**artifact, "inference_substrate": mod.INFERENCE_SUBSTRATE, "duration_s": 0.1}
    short_live = {**artifact, "inference_substrate": mod.LIVE_LLM_SUBSTRATE, "duration_s": 1.0}
    assert "cached verifier substrate requires duration_s >= 1.0" in mod.artifact_schema_errors(short_cached)
    assert "live_llm_inference requires duration_s >= 60.0" in mod.artifact_schema_errors(short_live)
