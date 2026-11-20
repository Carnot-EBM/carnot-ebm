"""Tests for Exp 4470 generic sb26 color-match slot verifier.

Spec refs: REQ-REPORT-4470, SCENARIO-REPORT-4470.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import yaml

from carnot import experiment_4470_color_match_slot_operator_solve_sb26 as mod
from carnot.agentic import arc_solve_learning as learning
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _ok_preconditions() -> dict[str, Any]:
    return {
        "sb26_environment_files": True,
        "arc_solver_kit_importable": True,
        "arc_solve_learning_importable": True,
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


def _recommendation() -> dict[str, Any]:
    return {
        "target_game": "sb26",
        "target_features": {"action_type": "click", "win_kw": ["click", "color"]},
        "strategy": {"routed_mechanic": "color_match_slot_sequence", "solver": "generic_color_match"},
        "retrieved_primitives": [
            {
                "name": "color_match_slot_sequence_verifier",
                "operator": "color_match_slot_sequence_verifier",
                "mechanic_class": "color_match_slot_sequence",
                "score": 11.5,
                "matched_cues": ["mechanic:color_match_slot_sequence", "slot", "undo"],
            }
        ],
        "selected_generic_operators": [{"operator": "color_match_slot_sequence_verifier"}],
        "recommended": [
            {
                "game": "s5i5",
                "similarity": 4.0,
                "solver": "generic config verifier",
                "win_condition": "ordered color slot sequence residual routes from config predicates",
                "action_model": "ACTION6 clicks, ACTION7 undo, ACTION5 validate",
            }
        ],
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "environment_files" / "sb26" / "7fbdac44").mkdir(parents=True)
    (root / "ops").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "games": [
                    {
                        "game": "sb26",
                        "reproducibility": "unsolved",
                        "levels_reproduced": 0,
                        "dead_ends": [
                            {
                                "gap_id": mod.SB26_GAP_ID,
                                "status": "open",
                                "residual_delta": "missing_color_match_slot_sequence_verifier",
                            }
                        ],
                    }
                ],
                "reproducible_total_levels": 44,
                "reproducible_total_games": 21,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text(
        "<!-- exp4458-gap-sb26-color-match-slot-sequence:start -->\n"
        "old sb26 gap\n"
        "<!-- exp4458-gap-sb26-color-match-slot-sequence:end -->\n",
        encoding="utf-8",
    )


def _clock() -> tuple[dict[str, float], Any, Any]:
    clock = {"t": 50.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    return clock, now, sleep


def test_req_report_4470_spec_declares_color_match_contract() -> None:
    """REQ-REPORT-4470: OpenSpec declares the operator, gate, and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4470" in spec
    assert "SCENARIO-REPORT-4470" in spec
    assert "color_match_slot_sequence_verifier" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.BASELINE_COMMAND_TEXT in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4470_solver_kit_grounds_ordered_color_slot_sequence() -> None:
    """REQ-REPORT-4470: the generic operator grounds item->slot matching with undo evidence."""

    result = kit.color_match_slot_sequence_verifier(
        game="sb26",
        object_digest=mod.SB26_L1_COLOR_MATCH_DIGEST,
        few_shot_examples=mod.DEFAULT_COLOR_MATCH_EXAMPLES,
    )
    no_examples = kit.color_match_slot_sequence_verifier(
        game="sb26",
        object_digest=mod.SB26_L1_COLOR_MATCH_DIGEST,
        few_shot_examples=(),
    )
    missing_item = kit.color_match_slot_sequence_verifier(
        game="sb26",
        object_digest={
            **mod.SB26_L1_COLOR_MATCH_DIGEST,
            "items": [dict(row) for row in mod.SB26_L1_COLOR_MATCH_DIGEST["items"] if row["color"] != 15],
        },
        few_shot_examples=mod.DEFAULT_COLOR_MATCH_EXAMPLES,
    )

    assert result["operator"] == "color_match_slot_sequence_verifier"
    assert result["grounded"] is True
    assert result["predicate_id"] == "color_match_slot_sequence"
    assert result["target_recipe_withheld"] == "sb26"
    assert result["solution"] == list(mod.SB26_L1_EXPECTED_LABELS)
    assert result["solution"][-1] == "validate"
    assert result["counterexample_rounds"] >= 1
    assert result["counterexamples"][0]["rejected_candidate"] == "unordered_color_bag_match"
    assert result["undo_recovery_solution"] == ["undo"]
    assert result["verifier"]["wrong_order_rejected"] is True
    assert result["verifier"]["undo_aware"] is True
    assert result["grounded_win_condition"]["fires_on_win"] is True
    assert result["grounded_win_condition"]["rejects_nonwins"] is True
    assert result["verifier_is_oracle"] is True
    assert no_examples["grounded"] is False
    assert no_examples["residual"] == "missing_color_match_slot_sequence_few_shot_examples"
    assert missing_item["grounded"] is False
    assert missing_item["residual"] == "missing_matching_item_for_slot"

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert "color_match_slot_sequence_verifier" in operators
    selected = kit.select_primitive_operators(mechanic_class="color_match_slot_sequence", game="sb26")
    assert selected[0].operator == "color_match_slot_sequence_verifier"


def test_req_report_4470_recommend_approach_routes_sb26_to_new_operator() -> None:
    """REQ-REPORT-4470: recommend_approach(sb26) surfaces the new generic operator first."""

    recommendation = learning.recommend_approach("sb26")

    assert recommendation["selected_generic_operators"][0]["operator"] == "color_match_slot_sequence_verifier"
    assert recommendation["retrieved_primitives"][0]["operator"] == "color_match_slot_sequence_verifier"


def test_scenario_report_4470_run_reproduces_sb26_and_closes_gap(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4470: sb26 L1 is banked only through the offline reproduction gate."""

    _write_fixture_repo(tmp_path)
    _clock_state, now, sleep = _clock()
    calls: list[list[str]] = []

    def reproduce(solution: Sequence[str]) -> dict[str, Any]:
        calls.append([str(label) for label in solution])
        return {"game": "sb26", "claimed_level": 1, "reached_level": 1, "reproduced": True}

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        recommendation_fn=lambda _game: _recommendation(),
        few_shot_examples=mod.DEFAULT_COLOR_MATCH_EXAMPLES,
        reproduce_fn=reproduce,
        now=now,
        sleep_fn=sleep,
    )

    assert calls == [list(mod.SB26_L1_EXPECTED_LABELS)]
    assert artifact["honest_verdict"] == "success: sb26_color_match_slot_sequence_L1_offline_reproduced"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["target_game"] == "sb26"
    assert artifact["color_match_operator_built"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["offline_reproduced"] is True
    assert artifact["counterexample_rounds"] >= 1
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducible_total_levels"] == 45
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["operator_result"]["predicate_id"] == "color_match_slot_sequence"

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    sb26 = next(row for row in registry["games"] if row["game"] == "sb26")
    assert sb26["reproducibility"] == "reproduced"
    assert sb26["levels_reproduced"] == 1
    assert sb26["mechanic_class"] == "color_match_slot_sequence"
    assert sb26["latest_exp4470_color_match"]["offline_reproduced"] is True
    assert sb26["dead_ends"][0]["status"] == "filled"
    assert registry["reproducible_total_levels"] == 45
    assert registry["reproducible_total_games"] == 22
    gaps = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "status: filled" in gaps
    assert "closed_by_color_match_slot_sequence_verifier" in gaps


def test_req_report_4470_no_bank_logs_refined_residual(tmp_path: Path) -> None:
    """REQ-REPORT-4470: a genuine no-bank run is terminal complete with a refined residual."""

    _write_fixture_repo(tmp_path)
    _clock_state, now, sleep = _clock()

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        recommendation_fn=lambda _game: _recommendation(),
        few_shot_examples=mod.DEFAULT_COLOR_MATCH_EXAMPLES,
        solve_sb26_fn=lambda _examples: {
            "solution": [],
            "operator_result": {
                "operator": "color_match_slot_sequence_verifier",
                "grounded": False,
                "solution": [],
                "counterexample_rounds": 2,
                "residual": "offline_env_rejected_color_match_candidate_after_undo_refinement",
                "verifier_is_oracle": True,
            },
            "counterexample_rounds": 2,
        },
        reproduce_fn=lambda _solution: pytest.fail("ungrounded candidate must not reproduce"),
        write_registry=False,
        write_gaps=False,
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == "complete: sb26_color_match_slot_sequence_no_reproduced_level_gap_logged"
    assert artifact["color_match_operator_built"] is True
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["counterexample_rounds"] == 2
    assert artifact["missing_verifier_gaps"][0]["residual_delta"] == (
        "offline_env_rejected_color_match_candidate_after_undo_refinement"
    )
    assert "partial:" not in artifact["honest_verdict"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4470_blocked_precondition_and_schema_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4470: blocked resources and malformed artifacts cannot fabricate success."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "sb26_environment_files": False, "ok": False},
        recommendation_fn=lambda game: calls.append(game) or _recommendation(),
        reproduce_fn=lambda _solution: pytest.fail("reproduce must not run"),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_offline_env_sb26"
    assert artifact["inference_substrate"] == mod.BLOCKED_INFERENCE_SUBSTRATE
    assert artifact["target_game"] == "sb26"
    assert artifact["color_match_operator_built"] is False
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []

    bad: Mapping[str, Any] = {
        **artifact,
        "honest_verdict": "partial: fake",
        "inference_substrate": None,
        "target_game": "",
        "color_match_operator_built": "true",
        "reproduced_levels": "1",
        "offline_reproduced": "true",
        "counterexample_rounds": "1",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "reproducible_total_levels": "45",
        "random_seed": "4470",
        "reproducibility_checksum": "bad",
        "no_3090_inference": False,
        "submitted_to_leaderboard": True,
        "field_principles": {},
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "inference_substrate must not be None" in errors
    assert "target_game must be sb26" in errors
    assert "color_match_operator_built must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "counterexample_rounds must be bare int" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "no_3090_inference must be true" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4470" in errors

    short_cached = {**artifact, "inference_substrate": mod.INFERENCE_SUBSTRATE, "duration_s": 0.1}
    short_live = {**artifact, "inference_substrate": mod.LIVE_LLM_SUBSTRATE, "duration_s": 1.0}
    fabricated = {
        **artifact,
        "honest_verdict": "success: sb26_color_match_slot_sequence_L1_offline_reproduced",
        "offline_reproduced": False,
        "reproduced_levels": 0,
    }
    assert "cached verifier substrate requires duration_s >= 1.0" in mod.artifact_schema_errors(short_cached)
    assert "live_llm_inference requires duration_s >= 60.0" in mod.artifact_schema_errors(short_live)
    assert "success verdict requires offline_reproduced true" in mod.artifact_schema_errors(fabricated)
    assert "success verdict requires reproduced_levels >= 1" in mod.artifact_schema_errors(fabricated)
