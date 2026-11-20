"""Tests for Exp 4445 generic object-motion world-model operator.

Spec refs: REQ-REPORT-4445, SCENARIO-REPORT-4445.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from carnot import experiment_4445_generic_object_motion_world_model_operator as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_fixture_repo(root: Path, *, examples: int = 4, env: bool = True) -> None:
    if env:
        for game in mod.TARGET_GAMES:
            (root / "environment_files" / game / "fixture").mkdir(parents=True)
    for game in mod.SOLVED_EXAMPLE_GAMES[:examples]:
        path = root / "results" / "arc_e3" / game / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"def engine(grid, action, data):\n    return grid\n\nGAME = {game!r}\n",
            encoding="utf-8",
        )


def _ok_preconditions() -> dict[str, Any]:
    return {
        "ar25_env_present": True,
        "ka59_env_present": True,
        "existing_world_models": 4,
        "focused_baseline_selected_green": True,
        "focused_baseline_exact_command_green": False,
        "focused_baseline_exact_command_blocker": "repo_addopts_package_wide_coverage_on_focused_k_slice",
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _reproduce_ar25(solution: Sequence[str]) -> dict[str, Any]:
    assert list(solution) == ["3"] * 5 + ["2"] * 10
    return {"game": "ar25", "claimed_level": 1, "reached_level": 1, "reproduced": True}


def _reproduce_ka59(solution: Sequence[str]) -> dict[str, Any]:
    assert list(solution) == ["4", "4", "4", "3", "2", "3", "3", "3", "C:1", "1", "4"]
    return {"game": "ka59", "claimed_level": 1, "reached_level": 1, "reproduced": True}


def test_req_report_4445_spec_declares_object_motion_contract() -> None:
    """REQ-REPORT-4445: OpenSpec declares the operator and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4445" in spec
    assert "SCENARIO-REPORT-4445" in spec
    assert "object_motion_world_model" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4445_solver_kit_operator_reflects_ar25_slots() -> None:
    """REQ-REPORT-4445: reflect/translate plans synthesize without ar25's recipe."""

    result = kit.object_motion_world_model(
        game="ar25",
        object_digest=mod.AR25_OBJECT_MOTION_DIGEST,
        few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
    )
    cold = kit.object_motion_world_model(
        game="ar25",
        object_digest=mod.AR25_OBJECT_MOTION_DIGEST,
        few_shot_examples=(),
    )

    assert result["operator"] == "object_motion_world_model"
    assert result["grounded"] is True
    assert result["target_recipe_withheld"] == "ar25"
    assert result["transition_families"] == ["translate", "reflect"]
    assert result["solution"] == ["3"] * 5 + ["2"] * 10
    assert result["verifier"]["grounded_transition_count"] >= 2
    assert cold["grounded"] is False
    assert cold["residual"] == "missing_object_motion_few_shot_examples"

    case = mod.build_active_data_cases()["ar25"][0]
    observed = result["engine"](np.asarray(case["before"]), case["action"], case["data"])
    assert np.array_equal(observed, np.asarray(case["expected"]))

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert "object_motion_world_model" in operators
    selected = kit.select_primitive_operators(mechanic_class="reflection world_model object motion")
    assert selected[0].operator == "object_motion_world_model"


def test_req_report_4445_solver_kit_operator_pushes_ka59_slots() -> None:
    """SCENARIO-REPORT-4445: push/select plans synthesize without ka59's recipe."""

    result = kit.object_motion_world_model(
        game="ka59",
        object_digest=mod.KA59_OBJECT_MOTION_DIGEST,
        few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
    )

    assert result["grounded"] is True
    assert result["transition_families"] == ["translate", "push"]
    assert result["solution"] == ["4", "4", "4", "3", "2", "3", "3", "3", "C:1", "1", "4"]
    assert result["object_slots"]["pushed_block"]["role"] == "dynamic_selected_after_push"
    assert result["grounded_win_condition"]["fires_on_win"] is True

    click_case = next(case for case in mod.build_active_data_cases()["ka59"] if case["action"] == 6)
    observed = result["engine"](np.asarray(click_case["before"]), click_case["action"], click_case["data"])
    assert np.array_equal(observed, np.asarray(click_case["expected"]))


def test_req_report_4445_accuracy_arms_measure_conditioning_lift() -> None:
    """REQ-REPORT-4445: with-examples accuracy is compared to a cold arm."""

    cases = mod.build_active_data_cases()
    with_examples = mod.evaluate_object_motion_models(cases, mod.DEFAULT_OBJECT_MOTION_EXAMPLES)
    cold = mod.evaluate_object_motion_models(cases, ())

    assert with_examples["accuracy"] == 1.0
    assert cold["accuracy"] < with_examples["accuracy"]
    assert with_examples["per_game"]["ar25"]["correct"] == with_examples["per_game"]["ar25"]["total"]
    assert cold["per_game"]["ka59"]["correct"] < cold["per_game"]["ka59"]["total"]


def test_scenario_report_4445_run_closes_both_residuals_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4445: generic plans are reproduction-gated and terminal."""

    _write_fixture_repo(tmp_path)
    clock = {"t": 30.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
        reproduce_fns={"ar25": _reproduce_ar25, "ka59": _reproduce_ka59},
        no_regression_fn=lambda _root: True,
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == "success: ar25_ka59_object_motion_generic_L1_offline_reproduced"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["residuals_closed_generically"] == ["ar25", "ka59"]
    assert artifact["world_model_accuracy_with_examples"] == 1.0
    assert artifact["world_model_accuracy_cold"] < artifact["world_model_accuracy_with_examples"]
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["no_regression"] is True
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is True
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["per_game"]["ar25"]["operator_result"]["target_recipe_withheld"] == "ar25"


def test_req_report_4445_no_help_result_is_complete_with_gap(tmp_path: Path) -> None:
    """REQ-REPORT-4445: measured no-help results are complete negative findings."""

    _write_fixture_repo(tmp_path)
    cases = mod.build_active_data_cases()
    cold = mod.evaluate_object_motion_models(cases, mod.DEFAULT_OBJECT_MOTION_EXAMPLES)
    with_examples = mod.evaluate_object_motion_models(cases, mod.DEFAULT_OBJECT_MOTION_EXAMPLES)
    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions=_ok_preconditions(),
        few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
        active_data_cases=cases,
        cold_metrics=cold,
        with_examples_metrics=with_examples,
        operator_results={
            game: kit.object_motion_world_model(
                game=game,
                object_digest=mod.OBJECT_MOTION_DIGESTS[game],
                few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
            )
            for game in mod.TARGET_GAMES
        },
        reproduction_results={
            "ar25": {"game": "ar25", "reproduced": False, "reached_level": 0},
            "ka59": {"game": "ka59", "reproduced": False, "reached_level": 0},
        },
        no_regression=True,
        started_at=1.0,
        ended_at=2.2,
    )

    assert artifact["honest_verdict"] == "complete: object_motion_examples_no_reproduced_level_gap_logged"
    assert artifact["offline_reproduced"] is False
    assert artifact["residuals_closed_generically"] == []
    assert artifact["missing_verifier_gaps"] == mod.DEFAULT_OBJECT_MOTION_GAPS
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4445_blocked_precondition_and_schema_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4445: blocked resources do not fabricate metrics or solves."""

    _write_fixture_repo(tmp_path, env=False)
    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "ar25_env_present": False, "ok": False},
        few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
        reproduce_fns={
            "ar25": lambda solution: calls.append("ar25") or {},
            "ka59": lambda solution: calls.append("ka59") or {},
        },
        no_regression_fn=lambda _root: calls.append("regression") or True,
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_offline_env_ar25"
    assert artifact["world_model_accuracy_with_examples"] is None
    assert artifact["world_model_accuracy_cold"] is None
    assert artifact["residuals_closed_generically"] == []
    assert artifact["offline_reproduced"] is False
    assert artifact["no_regression"] is False
    assert mod.artifact_schema_errors(artifact) == []

    bad: Mapping[str, Any] = {
        **artifact,
        "honest_verdict": "partial: retry",
        "inference_substrate": None,
        "residuals_closed_generically": "ar25",
        "world_model_accuracy_with_examples": 0.5,
        "world_model_accuracy_cold": None,
        "reproduced_levels": "0",
        "offline_reproduced": "false",
        "no_regression": "true",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "random_seed": "4445",
        "reproducibility_checksum": "bad",
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "missing inference_substrate" in errors
    assert "residuals_closed_generically must be list" in errors
    assert "blocked artifacts must not fabricate accuracy metrics" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "no_regression must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors


def test_req_report_4445_utility_branches_are_deterministic(tmp_path: Path) -> None:
    """REQ-REPORT-4445: helpers expose deterministic misses and schema branches."""

    _write_fixture_repo(tmp_path)
    examples = mod.gather_world_model_examples(tmp_path)
    assert [row["game"] for row in examples] == list(mod.SOLVED_EXAMPLE_GAMES)
    assert examples[0]["sha256"]
    assert mod.gather_world_model_examples(tmp_path, example_games=("missing",)) == []
    assert mod._as_int("bad") == 0

    for override, expected in (
        ({"ka59_env_present": False}, "offline_env_ka59"),
        ({"existing_world_models": 1}, "few_shot_world_models"),
        ({"focused_baseline_selected_green": False}, "pre_refactor_focused_pytest"),
        ({"no_3090_inference": False}, "no_3090_inference_policy"),
        ({"leaderboard_submission": True}, "leaderboard_submission_policy"),
    ):
        assert mod.first_precondition_miss({**_ok_preconditions(), **override}) == expected

    grid = np.zeros((3, 3), dtype=int)
    assert np.array_equal(mod._cold_engine(grid, 6, {"x": "bad", "y": 1}), grid)

    class _Sprite:
        x = 10
        y = 20
        width = 6
        height = 8

    class _Level:
        grid_size = (60, 60)

        def get_sprites_by_tag(self, _tag: str) -> list[_Sprite]:
            return [_Sprite(), _Sprite()]

    class _Game:
        current_level = _Level()

    class _Env:
        _game = _Game()

    click = mod._click_data_for_label(_Env(), "C:1", mod.KA59_OBJECT_MOTION_DIGEST)
    assert click == {"x": 15, "y": 26}

    base_cases = mod.build_active_data_cases()
    cold = mod.evaluate_object_motion_models(base_cases, ())
    with_examples = mod.evaluate_object_motion_models(base_cases, mod.DEFAULT_OBJECT_MOTION_EXAMPLES)
    ar25_operator = kit.object_motion_world_model(
        game="ar25",
        object_digest=mod.AR25_OBJECT_MOTION_DIGEST,
        few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
    )
    ka59_operator = kit.object_motion_world_model(
        game="ka59",
        object_digest=mod.KA59_OBJECT_MOTION_DIGEST,
        few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
    )

    one_closed = mod.build_artifact(
        root=tmp_path,
        preconditions=_ok_preconditions(),
        few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
        active_data_cases=base_cases,
        cold_metrics=with_examples,
        with_examples_metrics=with_examples,
        operator_results={"ar25": ar25_operator, "ka59": ka59_operator},
        reproduction_results={
            "ar25": {"game": "ar25", "reproduced": True, "reached_level": 1},
            "ka59": {"game": "ka59", "reproduced": False, "reached_level": 0},
        },
        no_regression=True,
        started_at=1.0,
        ended_at=2.1,
    )
    assert one_closed["honest_verdict"] == "success: object_motion_generic_residual_closed_ar25"
    assert [gap["game"] for gap in one_closed["missing_verifier_gaps"]] == ["ka59"]

    accuracy_only = mod.build_artifact(
        root=tmp_path,
        preconditions=_ok_preconditions(),
        few_shot_examples=mod.DEFAULT_OBJECT_MOTION_EXAMPLES,
        active_data_cases=base_cases,
        cold_metrics=cold,
        with_examples_metrics=with_examples,
        operator_results={"ar25": ar25_operator, "ka59": ka59_operator},
        reproduction_results={
            "ar25": {"game": "ar25", "reproduced": False, "reached_level": 0},
            "ka59": {"game": "ka59", "reproduced": False, "reached_level": 0},
        },
        no_regression=True,
        started_at=1.0,
        ended_at=2.1,
    )
    assert accuracy_only["honest_verdict"] == "success: object_motion_examples_improved_world_model_accuracy"
    assert accuracy_only["missing_verifier_gaps"] == []

    calls: list[str] = []
    clock = {"t": 1.0}
    unsupported = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        few_shot_examples=({"game": "xx", "rule_id": "none", "predicate": "unsupported"},),
        reproduce_fns={
            "ar25": lambda solution: calls.append("ar25") or {},
            "ka59": lambda solution: calls.append("ka59") or {},
        },
        no_regression_fn=lambda _root: True,
        now=lambda: clock["t"],
        sleep_fn=lambda seconds: clock.__setitem__("t", clock["t"] + seconds),
    )
    assert calls == []
    assert unsupported["offline_reproduced"] is False

    malformed = {
        **accuracy_only,
        "inference_substrate": mod.LIVE_LLM_SUBSTRATE,
        "duration_s": 1.0,
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "world_model_accuracy_with_examples": 0.0,
        "world_model_accuracy_cold": 0.0,
        "missing_verifier_gaps": [],
        "model_specs": {"no_3090_inference": False, "leaderboard_submission": True},
        "field_principles": {**mod.FIELD_PRINCIPLES, "random_seed": {"principle": "wrong"}},
    }
    malformed_errors = mod.artifact_schema_errors(malformed)
    assert "live_llm_inference requires duration_s >= 60.0" in malformed_errors
    assert "offline_reproduced true requires reproduced_levels >= 1" in malformed_errors
    assert "missing_verifier_gaps must list residuals when neither gate passes" in malformed_errors
    assert "model_specs.no_3090_inference must be true" in malformed_errors
    assert "model_specs.leaderboard_submission must be false" in malformed_errors
    assert "field_principles.random_seed must match REQ-REPORT-4445" in malformed_errors

    short_cached = {**accuracy_only, "duration_s": 0.0}
    assert "cached verifier substrate requires duration_s >= 1.0" in mod.artifact_schema_errors(short_cached)


def test_req_report_4445_write_artifact_rejects_invalid_schema(tmp_path: Path) -> None:
    """REQ-REPORT-4445: invalid terminal artifacts are not written."""

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, {"honest_verdict": "partial: invalid"})
