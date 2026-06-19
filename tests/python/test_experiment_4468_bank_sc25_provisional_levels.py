"""Tests for Exp 4468 sc25 provisional-level banking.

Spec refs: REQ-REPORT-4468, SCENARIO-REPORT-4468.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest
import yaml

from carnot import experiment_4468_bank_sc25_provisional_levels as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _ok_preconditions() -> dict[str, Any]:
    return {
        "sc25_environment_files": True,
        "sc25_world_model_present": True,
        "arc_solver_imports": True,
        "induction_needed": False,
        "qwen_gguf_cache": False,
        "igpu_llama_server": False,
        "generator_resource_available": False,
        "baseline_command": mod.BASELINE_COMMAND_TEXT,
        "baseline_exit_code": 0,
        "baseline_pytest_nocov_green": True,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "environment_files" / "sc25" / "635fd71a").mkdir(parents=True)
    (root / "results" / "arc_e3" / "sc25").mkdir(parents=True)
    (root / "ops").mkdir(parents=True)
    (root / mod.WORLD_MODEL_RELATIVE_PATH).write_text(
        "def engine(grid, action, data):\n    return grid\n"
        "def is_level_complete(grid):\n    return False\n",
        encoding="utf-8",
    )
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "games": [
                    {
                        "game": "sc25",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "levels_live_recorded": 5,
                        "solver": "results/experiment_4341_e3_sc25_reproduction.json",
                        "gotchas": ["first step after reset consumed -> warm-up step."],
                        "dead_ends": [
                            {
                                "gap_id": mod.SC25_GAP_ID,
                                "status": "open",
                                "failure_mode": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                            }
                        ],
                    },
                    {
                        "game": "dc22",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                    },
                ],
                "reproducible_total_levels": 40,
                "reproducible_total_games": 21,
                "provisional_total_levels": 5,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _clock() -> Any:
    ticks = iter([0.0, 0.2, 1.1])
    return lambda: next(ticks)


def _verify_result() -> dict[str, Any]:
    return {
        "world_model_loaded": True,
        "verifier_accuracy": 1.0,
        "transitions_scored": 16,
        "mismatches": [],
        "world_model_sha256": "a" * 64,
    }


def _l1_plan_result() -> dict[str, Any]:
    return {
        "planned": True,
        "executed": True,
        "level_up": True,
        "solution": list(mod.SC25_PLANS_BY_LEVEL[1]),
        "generic_plan_and_execute_result": {"planned": True, "executed": False},
    }


def test_req_report_4468_spec_declares_sc25_bank_contract() -> None:
    """REQ-REPORT-4468: OpenSpec declares the sc25 banking artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4468" in spec
    assert "SCENARIO-REPORT-4468" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.BASELINE_COMMAND_TEXT in spec
    assert "WorldModelVerifier" in spec
    assert "provisional_total_levels" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4468_sc25_labels_and_plans_use_offline_coords() -> None:
    """REQ-REPORT-4468: banked plans encode corrected cast cells and raw spell clicks."""

    assert mod.sc25_label_to_action_data("warmup") == (5, None)
    assert mod.sc25_label_to_action_data("cell0,0") == (6, {"x": 24, "y": 49})
    assert mod.sc25_label_to_action_data("cell2,1") == (6, {"x": 29, "y": 59})
    assert mod.sc25_label_to_action_data("click4,23") == (6, {"x": 4, "y": 23})
    assert mod.sc25_label_to_action_data("move4") == (4, None)
    with pytest.raises(ValueError, match="unknown sc25 label"):
        mod.sc25_label_to_action_data("bad")

    assert len(mod.SC25_PLANS_BY_LEVEL[2]) > len(mod.SC25_PLANS_BY_LEVEL[1])
    assert mod.SC25_PLANS_BY_LEVEL[2][-5:] == mod.SC25_L2_SUFFIX
    assert mod.SC25_PLANS_BY_LEVEL[3][-len(mod.SC25_L3_SUFFIX) :] == mod.SC25_L3_SUFFIX
    assert mod.SC25_PLANS_BY_LEVEL[5][-len(mod.SC25_L5_SUFFIX) :] == mod.SC25_L5_SUFFIX
    assert all(mod.SC25_PLANS_BY_LEVEL[level - 1] == mod.SC25_PLANS_BY_LEVEL[level][: len(mod.SC25_PLANS_BY_LEVEL[level - 1])] for level in range(2, 6))


def test_scenario_report_4468_success_banks_l2_l5_and_updates_registry(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4468: reproduced sc25 L2-L5 move provisional levels to reproduced."""

    _write_fixture_repo(tmp_path)
    calls: list[tuple[int, list[str]]] = []

    def reproduce(solution: Sequence[str], claimed_level: int) -> dict[str, Any]:
        calls.append((claimed_level, list(solution)))
        return {
            "game": "sc25",
            "claimed_level": claimed_level,
            "reached_level": claimed_level,
            "reproduced": True,
            "mode": "offline_reproduction_gate_no_quota",
        }

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        world_model_verify_fn=lambda _root: _verify_result(),
        l1_plan_fn=lambda _root: _l1_plan_result(),
        reproduce_fn=reproduce,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert [level for level, _solution in calls] == [1, 2, 3, 4, 5]
    assert calls[1][1] == list(mod.SC25_PLANS_BY_LEVEL[2])
    assert artifact["honest_verdict"] == "success: sc25_L5_offline_reproduced_banked_4_new_levels"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["target_game"] == "sc25"
    assert artifact["new_sc25_levels_reproduced"] == 4
    assert artifact["sc25_levels_reproduced_total"] == 5
    assert artifact["reproduced_levels"] == 4
    assert artifact["offline_reproduced"] is True
    assert artifact["baseline_pytest_nocov_green"] is True
    assert artifact["no_regression"] is True
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducible_total_levels"] == 44
    assert artifact["provisional_total_levels_after"] == 1
    assert mod.artifact_schema_errors(artifact) == []
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    sc25 = next(row for row in registry["games"] if row["game"] == "sc25")
    assert sc25["levels_reproduced"] == 5
    assert sc25["latest_exp4468_reproduce"]["new_sc25_levels_reproduced"] == 4
    assert sc25["dead_ends"][0]["status"] == "filled"
    assert registry["reproducible_total_levels"] == 44
    assert registry["reproducible_total_games"] == 21
    assert registry["provisional_total_levels"] == 1


def test_req_report_4468_blocked_precondition_stops_before_reproduction(tmp_path: Path) -> None:
    """REQ-REPORT-4468: missing resources write blocked artifacts without banking."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "baseline_pytest_nocov_green": False, "ok": False},
        world_model_verify_fn=lambda _root: pytest.fail("verify must not run"),
        l1_plan_fn=lambda _root: pytest.fail("plan must not run"),
        reproduce_fn=lambda _solution, _level: pytest.fail("reproduce must not run"),
        now=lambda: 2.0,
    )

    assert artifact["honest_verdict"] == "complete: blocked_baseline_tests_red"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_sc25_levels_reproduced"] == 0
    assert artifact["sc25_levels_reproduced_total"] == 1
    assert artifact["baseline_pytest_nocov_green"] is False
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4468_no_bank_is_terminal_complete_with_gap(tmp_path: Path) -> None:
    """REQ-REPORT-4468: a measured L2 failure is complete and records the residual."""

    _write_fixture_repo(tmp_path)
    calls: list[int] = []

    def reproduce(_solution: Sequence[str], claimed_level: int) -> dict[str, Any]:
        calls.append(claimed_level)
        return {
            "game": "sc25",
            "claimed_level": claimed_level,
            "reached_level": 1,
            "reproduced": claimed_level == 1,
            "mode": "offline_reproduction_gate_no_quota",
        }

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        world_model_verify_fn=lambda _root: _verify_result(),
        l1_plan_fn=lambda _root: _l1_plan_result(),
        reproduce_fn=reproduce,
        write_registry=False,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert calls == [1, 2]
    assert artifact["honest_verdict"] == "complete: sc25_cannot_deepen_beyond_L1_gap_logged"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == mod.SC25_GAP_ID
    assert artifact["missing_verifier_gaps"][0]["residual_delta"] == "sc25_l2_offline_reproduction_failed"
    assert "partial:" not in artifact["honest_verdict"]
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"sc25_environment_files": False}, "offline_env_sc25"),
        ({"sc25_world_model_present": False}, "sc25_world_model"),
        ({"arc_solver_imports": False}, "arc_solver_imports"),
        ({"baseline_pytest_nocov_green": False}, "baseline_tests_red"),
        ({"no_3090_inference": False}, "no_3090_inference_policy"),
        ({"leaderboard_submission": True}, "leaderboard_submission_policy"),
    ],
)
def test_req_report_4468_precondition_miss_names_resource(
    override: dict[str, Any],
    expected: str,
) -> None:
    """REQ-REPORT-4468: every precondition miss maps to an explicit blocked resource."""

    assert mod.first_precondition_miss({**_ok_preconditions(), **override}) == expected


def test_req_report_4468_schema_rejects_partial_or_fabricated_success(tmp_path: Path) -> None:
    """REQ-REPORT-4468: schema rejects partial prefixes, type drift, and fake success."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        world_model_verify_fn=lambda _root: _verify_result(),
        l1_plan_fn=lambda _root: _l1_plan_result(),
        reproduce_fn=lambda _solution, level: {
            "game": "sc25",
            "claimed_level": level,
            "reached_level": level,
            "reproduced": True,
        },
        write_registry=False,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )
    bad = {
        **artifact,
        "honest_verdict": "partial: fake",
        "inference_substrate": None,
        "target_game": "dc22",
        "new_sc25_levels_reproduced": "4",
        "sc25_levels_reproduced_total": "5",
        "reproduced_levels": "4",
        "offline_reproduced": "true",
        "baseline_pytest_nocov_green": "true",
        "no_regression": "true",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "reproducible_total_levels": "44",
        "random_seed": "4468",
        "reproducibility_checksum": "bad",
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "inference_substrate must not be None" in errors
    assert "target_game must be sc25" in errors
    assert "new_sc25_levels_reproduced must be bare int" in errors
    assert "sc25_levels_reproduced_total must be bare int" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "baseline_pytest_nocov_green must be bare bool" in errors
    assert "no_regression must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors

    unsupported = {**artifact, "inference_substrate": "unknown"}
    short_cached = {**artifact, "duration_s": 0.1}
    short_live = {**artifact, "inference_substrate": mod.LIVE_LLM_SUBSTRATE, "duration_s": 1.0}
    fake_success = {
        **artifact,
        "honest_verdict": "success: fake",
        "offline_reproduced": False,
        "new_sc25_levels_reproduced": 0,
        "reproduced_levels": 0,
        "missing_verifier_gaps": [{"gap_id": mod.SC25_GAP_ID}],
        "no_regression": False,
        "no_3090_inference": False,
        "submitted_to_leaderboard": True,
        "field_principles": {},
    }
    missing = dict(artifact)
    missing.pop("target_game")

    assert "inference_substrate has unsupported value" in mod.artifact_schema_errors(unsupported)
    assert "cached verifier substrate requires duration_s >= 1.0" in mod.artifact_schema_errors(short_cached)
    assert "live_llm_inference requires duration_s >= 60.0" in mod.artifact_schema_errors(short_live)
    fake_errors = mod.artifact_schema_errors(fake_success)
    assert "success verdict requires offline_reproduced true" in fake_errors
    assert "success verdict requires new_sc25_levels_reproduced >= 1" in fake_errors
    assert "success verdict requires no missing_verifier_gaps" in fake_errors
    assert "success verdict requires no_regression true" in fake_errors
    assert "no_3090_inference must be true" in fake_errors
    assert "submitted_to_leaderboard must be false" in fake_errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4468" in fake_errors
    assert "missing target_game" in mod.artifact_schema_errors(missing)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, {"honest_verdict": "partial: bad"})


def test_req_report_4468_registry_helpers_handle_missing_and_malformed_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-4468: registry helpers do not fabricate totals from malformed inputs."""

    assert mod._load_registry(tmp_path) == {"games": []}
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).parent.mkdir(parents=True)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text("[", encoding="utf-8")

    assert mod._load_registry(tmp_path) == {"games": []}
    assert mod._registry_games({"games": {}}) == []
    assert mod._target_entry({"games": []}) is None
    assert mod._registry_totals(
        {
            "games": [
                {"game": "a", "reproducibility": "reproduced", "levels_reproduced": 2},
                {"game": "b", "reproducibility": "unsolved", "levels_reproduced": 0},
            ]
        }
    ) == {"reproducible_total_levels": 2, "reproducible_total_games": 1}


def test_req_report_4468_defensive_helpers_cover_fallback_paths(tmp_path: Path) -> None:
    """REQ-REPORT-4468: defensive helper branches stay explicit and deterministic."""

    with pytest.raises(ValueError, match="unknown sc25 label"):
        mod.sc25_label_to_action_data("clickbad")

    assert mod._missing_gap(reproduced_depth=0, l1_plan_result={}, reproduction_results={})[
        "residual_delta"
    ] == "sc25_l1_world_model_plan_failed"
    assert mod._missing_gap(
        reproduced_depth=3,
        l1_plan_result={"level_up": True},
        reproduction_results={},
    )["residual_delta"] == "sc25_l4_world_model_plan_missing"

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        world_model_verify_fn=lambda _root: _verify_result(),
        l1_plan_fn=lambda _root: _l1_plan_result(),
        reproduce_fn=lambda _solution, level: {
            "game": "sc25",
            "claimed_level": level,
            "reached_level": level,
            "reproduced": True,
        },
        write_registry=False,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )

    offline_true_no_new = {
        **artifact,
        "honest_verdict": "complete: inconsistent_offline_gate",
        "offline_reproduced": True,
        "new_sc25_levels_reproduced": 0,
        "reproduced_levels": 0,
    }
    assert "offline_reproduced true requires new_sc25_levels_reproduced >= 1" in mod.artifact_schema_errors(
        offline_true_no_new
    )
    assert mod._banked_entry({}, artifact)["dead_ends"][0]["gap_id"] == mod.SC25_GAP_ID
    preserved = mod._banked_entry({"dead_ends": ["plain note"]}, artifact)
    assert preserved["dead_ends"][0] == "plain note"
    assert preserved["dead_ends"][1]["gap_id"] == mod.SC25_GAP_ID
    mod.update_arc_registry(tmp_path, {**artifact, "offline_reproduced": False})

    empty_root = tmp_path / "empty"
    mod.update_arc_registry(empty_root, artifact)
    empty_registry = yaml.safe_load((empty_root / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert empty_registry["games"][0]["game"] == "sc25"

    no_total_root = tmp_path / "no_totals"
    (no_total_root / "ops").mkdir(parents=True)
    (no_total_root / mod.REGISTRY_RELATIVE_PATH).write_text(
        "games:\n- game: sc25\n  reproducibility: reproduced\n  levels_reproduced: 1\n",
        encoding="utf-8",
    )
    mod.update_arc_registry(no_total_root, artifact)
    no_total_text = (no_total_root / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "reproducible_total_levels: 5" in no_total_text
    assert "provisional_total_levels: 0" in no_total_text
