"""Tests for Exp 4471 rotated generic first-contact attempt.

Spec refs: REQ-REPORT-4471, SCENARIO-REPORT-4471.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import yaml

from carnot import experiment_4471_first_contact_rotated_new_game as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _ok_preconditions() -> dict[str, Any]:
    return {
        "target_env_present": True,
        "arc_solver_kit_importable": True,
        "arc_solve_learning_importable": True,
        "qwen_gguf_cached": True,
        "igpu_llama_server_available": False,
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
        "target_game": "re86",
        "target_features": {"action_type": "keyboard", "win_kw": ["pattern", "template"]},
        "strategy": {"routed_mechanic": "graph_explore", "solver": "arc_graph_explore.graph_explore_solve_v2"},
        "retrieved_primitives": [
            {
                "name": "graph_astar_action_cost",
                "operator": "graph_astar_action_cost",
                "mechanic_class": "graph_explore_navigation",
                "score": 9.0,
                "matched_cues": ["mechanic:graph_explore", "graph", "keyboard"],
            },
            {
                "name": "mechanic_graph_explore_navigation",
                "operator": "graph_astar_action_cost",
                "mechanic_class": "graph_explore_navigation",
                "score": 7.0,
                "matched_cues": ["graph explore"],
            },
        ],
        "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        "recommended": [
            {
                "game": "tu93",
                "similarity": 6.0,
                "solver": "graph navigation recipe",
                "win_condition": "keyboard navigation to a target",
                "action_model": "keyboard ACTION1-4",
            },
            {
                "game": "tr87",
                "similarity": 6.0,
                "solver": "glyph rewrite recipe",
                "win_condition": "editable sequence must match pattern rewrite",
                "action_model": "keyboard ACTION1-4",
            },
        ],
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "environment_files" / "re86" / "8af5384d").mkdir(parents=True)
    (root / "ops").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "games": [
                    {
                        "game": "tu93",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 5,
                        "mechanic_class": "graph_explore",
                        "solver": "graph recipe fixture",
                    }
                ],
                "reproducible_total_levels": 45,
                "reproducible_total_games": 22,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text("# Verifier Gaps\n", encoding="utf-8")


def _clock() -> tuple[dict[str, float], Any, Any]:
    clock = {"t": 10.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    return clock, now, sleep


def test_req_report_4471_spec_declares_rotated_first_contact_contract() -> None:
    """REQ-REPORT-4471: OpenSpec declares routing, gate, and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4471" in spec
    assert "SCENARIO-REPORT-4471" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.BASELINE_COMMAND_TEXT in spec
    assert "{bp35, lf52, re86}" in spec
    assert "missing_pattern_match_sprite_resize_verifier" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4471_selects_re86_and_graph_operator_from_routing() -> None:
    """REQ-REPORT-4471: re86 is the rotated target and graph primitive is selected."""

    selection = mod.select_rotated_target(
        {
            "bp35": {"target_game": "bp35", "retrieved_primitives": [{"operator": "graph_astar_action_cost", "score": 9.0}]},
            "lf52": {"target_game": "lf52", "retrieved_primitives": [{"operator": "graph_astar_action_cost", "score": 9.0}]},
            "re86": _recommendation(),
        }
    )
    operator = mod.select_generic_operator(_recommendation())

    assert selection["target_game"] == "re86"
    assert selection["reason"] == "pattern_match_sprite_resize_is_best_rotated_gap_for_existing_graph_route"
    assert operator == {
        "operator": "graph_astar_action_cost",
        "routed_to": "tu93",
        "reason": "retrieved_graph_primitive_matches_re86_keyboard_pattern_route",
        "source": "retrieved_primitives",
    }
    assert mod.target_digest_for("re86")["rule_family"] == "sprite_overlay_pattern_match"


def test_scenario_report_4471_re86_no_level_is_terminal_and_logs_residual(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4471: no generic re86 bank logs the sprite verifier residual."""

    _write_fixture_repo(tmp_path)
    _clock_state, now, sleep = _clock()

    def solve(
        *,
        target_game: str,
        selected_operator: Mapping[str, str],
        few_shot_examples: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del few_shot_examples
        return mod.target_digest_for(target_game), {
            "operator": selected_operator["operator"],
            "game": target_game,
            "grounded": False,
            "solution": [],
            "predicate_id": "",
            "counterexample_rounds": 2,
            "residual": "missing_pattern_match_sprite_resize_verifier",
            "graph_search": {"expansions": 6000, "states": 2500, "reached_level": 0},
            "verifier_is_oracle": True,
        }

    artifact = mod.run(
        root=tmp_path,
        target_game="re86",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        ground_operator_fn=solve,
        reproduce_fn=lambda _solution: pytest.fail("ungrounded re86 must not reproduce"),
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == "complete: generic_first_contact_re86_routed_no_new_level"
    assert artifact["target_game"] == "re86"
    assert artifact["routed_to"] == "tu93"
    assert artifact["retrieved_primitives"][0]["operator"] == "graph_astar_action_cost"
    assert artifact["selected_operator"]["operator"] == "graph_astar_action_cost"
    assert artifact["operator_result"]["grounded"] is False
    assert artifact["operator_result"]["graph_search"]["expansions"] == 6000
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["reproducible_total_levels"] == 45
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == mod.RE86_GAP_ID
    assert artifact["missing_verifier_gaps"][0]["residual_delta"] == "missing_pattern_match_sprite_resize_verifier"
    assert "partial:" not in artifact["honest_verdict"]
    assert artifact["verifier_is_oracle"] is True
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["target_game"] == "re86"
    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    re86 = next(row for row in registry["games"] if row["game"] == "re86")
    assert re86["reproducibility"] == "unsolved"
    assert re86["levels_reproduced"] == 0
    assert re86["dead_ends"][0]["gap_id"] == mod.RE86_GAP_ID
    assert registry["reproducible_total_levels"] == 45
    gaps = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "exp4471-gap-re86-pattern-match-sprite-resize:start" in gaps
    assert "movement: still_open" in gaps


def test_scenario_report_4471_success_banks_one_level(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4471: reproduced re86 L1 banks only through reproduce()."""

    _write_fixture_repo(tmp_path)
    _clock_state, now, sleep = _clock()
    calls: list[list[str]] = []

    def solve(
        *,
        target_game: str,
        selected_operator: Mapping[str, str],
        few_shot_examples: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del few_shot_examples
        return mod.target_digest_for(target_game), {
            "operator": selected_operator["operator"],
            "game": target_game,
            "grounded": True,
            "predicate_id": "sprite_overlay_pattern_match",
            "solution": ['{"action": 4}', '{"action": 5}'],
            "counterexample_rounds": 3,
            "verifier_is_oracle": True,
        }

    def reproduce(solution: Sequence[str]) -> dict[str, Any]:
        calls.append([str(label) for label in solution])
        return {"game": "re86", "claimed_level": 1, "reached_level": 1, "reproduced": True}

    artifact = mod.run(
        root=tmp_path,
        target_game="re86",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        ground_operator_fn=solve,
        reproduce_fn=reproduce,
        now=now,
        sleep_fn=sleep,
    )

    assert calls == [['{"action": 4}', '{"action": 5}']]
    assert artifact["honest_verdict"] == "success: generic_first_contact_re86_L1_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["reproducible_total_levels"] == 46
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    re86 = next(row for row in registry["games"] if row["game"] == "re86")
    assert re86["reproducibility"] == "reproduced"
    assert re86["levels_reproduced"] == 1
    assert registry["reproducible_total_levels"] == 46
    assert registry["reproducible_total_games"] == 23
    assert "movement: filled" in (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")


def test_req_report_4471_blocked_precondition_stops_before_routing(tmp_path: Path) -> None:
    """REQ-REPORT-4471: blocked resources write terminal artifacts without routing."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        target_game="re86",
        preconditions_checked={**_ok_preconditions(), "generator_resource_available": False, "ok": False},
        recommend_fn=lambda game: calls.append(game) or _recommendation(),
        reproduce_fn=lambda _solution: pytest.fail("reproduce must not run"),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_qwen_generator_resource"
    assert artifact["inference_substrate"] == mod.BLOCKED_INFERENCE_SUBSTRATE
    assert artifact["target_game"] == "re86"
    assert artifact["routed_to"] == ""
    assert artifact["retrieved_primitives"] == []
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"target_env_present": False}, "offline_env_re86"),
        ({"arc_solver_kit_importable": False}, "arc_solver_kit"),
        ({"arc_solve_learning_importable": False}, "arc_solve_learning"),
        ({"generator_resource_available": False}, "qwen_generator_resource"),
        ({"baseline_pytest_nocov_green": False}, "baseline_pytest_nocov"),
        ({"no_3090_inference": False}, "no_3090_inference_policy"),
        ({"leaderboard_submission": True}, "leaderboard_submission_policy"),
    ],
)
def test_req_report_4471_precondition_miss_names_resource(
    override: dict[str, Any],
    expected: str,
) -> None:
    """REQ-REPORT-4471: blocked resources are explicit and deterministic."""

    assert mod.first_precondition_miss({**_ok_preconditions(), **override}, "re86") == expected


def test_req_report_4471_schema_rejects_fabrication_and_type_drift(tmp_path: Path) -> None:
    """REQ-REPORT-4471: schema rejects fake success, partial prefixes, and missing evidence."""

    _write_fixture_repo(tmp_path)
    _clock_state, now, sleep = _clock()
    artifact = mod.run(
        root=tmp_path,
        target_game="re86",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        ground_operator_fn=lambda **_: (
            mod.target_digest_for("re86"),
            {
                "operator": "graph_astar_action_cost",
                "game": "re86",
                "grounded": False,
                "solution": [],
                "counterexample_rounds": 1,
                "residual": "missing_pattern_match_sprite_resize_verifier",
                "verifier_is_oracle": True,
            },
        ),
        reproduce_fn=lambda _solution: pytest.fail("ungrounded re86 must not reproduce"),
        write_registry=False,
        write_gaps=False,
        now=now,
        sleep_fn=sleep,
    )
    bad: Mapping[str, Any] = {
        **artifact,
        "honest_verdict": "partial: fake",
        "inference_substrate": None,
        "duration_s": 0.5,
        "target_game": "",
        "routed_to": "",
        "retrieved_primitives": [],
        "reproduced_levels": "0",
        "offline_reproduced": "false",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "reproducible_total_levels": "45",
        "random_seed": "4471",
        "reproducibility_checksum": "bad",
        "no_3090_inference": False,
        "submitted_to_leaderboard": True,
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": {"principle": "wrong"}},
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "inference_substrate must not be None" in errors
    assert "target_game must be non-empty string" in errors
    assert "routed_to must be non-empty string for attempted runs" in errors
    assert "retrieved_primitives must be non-empty list for attempted runs" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "no_3090_inference must be true" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4471" in errors

    fabricated = {
        **artifact,
        "honest_verdict": "success: generic_first_contact_re86_L1_offline_reproduced",
        "offline_reproduced": False,
        "reproduced_levels": 0,
    }
    assert "success verdict requires offline_reproduced true" in mod.artifact_schema_errors(fabricated)
    assert "success verdict requires reproduced_levels >= 1" in mod.artifact_schema_errors(fabricated)
    assert "offline_reproduced true requires reproduced_levels >= 1" in mod.artifact_schema_errors(
        {**artifact, "offline_reproduced": True, "reproduced_levels": 0}
    )
    assert "inference_substrate has unsupported value" in mod.artifact_schema_errors(
        {**artifact, "inference_substrate": "unsupported"}
    )
    assert "cached verifier substrate requires duration_s >= 1.0" in mod.artifact_schema_errors(
        {**artifact, "duration_s": 0.1}
    )
    assert "live_llm_inference requires duration_s >= 60.0" in mod.artifact_schema_errors(
        {**artifact, "inference_substrate": mod.LIVE_LLM_SUBSTRATE, "duration_s": 1.0}
    )

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, {"honest_verdict": "partial: invalid"})


def test_req_report_4471_helper_branches_are_deterministic(tmp_path: Path) -> None:
    """REQ-REPORT-4471: helper fallbacks and label encoding stay deterministic."""

    _write_fixture_repo(tmp_path)
    sleep_calls: list[float] = []
    ticks = iter([0.0, 1.05])
    assert mod._sleep_until_verifier_floor(
        started_at=0.0,
        now=lambda: next(ticks),
        sleep_fn=sleep_calls.append,
    ) == 1.05
    assert sleep_calls == [mod.VERIFIER_SCORING_DURATION_TARGET_S]

    assert mod._closest_recipe({"recommended": []}) == {}
    assert mod._retrieved_primitives({"retrieved_primitives": "bad"}) == []
    assert mod.select_generic_operator({"recommended": [{"game": "tr87", "solver": "glyph rewrite"}]}) == {
        "operator": "glyph_rewrite_rule_verifier",
        "routed_to": "tr87",
        "reason": "routed_recipe_contains_glyph_or_pattern_rewrite",
        "source": "closest_recipe",
    }
    assert mod.select_generic_operator({"recommended": [{"game": "ka59", "solver": "object motion world model"}]}) == {
        "operator": "object_motion_world_model",
        "routed_to": "ka59",
        "reason": "routed_recipe_contains_object_motion",
        "source": "closest_recipe",
    }
    assert mod.select_generic_operator({"selected_generic_operators": [{"operator": "object_centric_digest"}]}) == {
        "operator": "object_centric_digest",
        "routed_to": "",
        "reason": "fallback_to_router_selected_operator",
        "source": "selected_generic_operators",
    }
    assert mod.select_generic_operator({})["operator"] == "object_centric_digest"
    assert mod.target_digest_for("unknown")["rule_family"] == "unknown_first_contact"
    assert mod._trajectory_to_labels([{"action": 4}, {"action": 6, "data": {"x": 3, "y": 7}}]) == [
        '{"action":4}',
        '{"action":6,"data":{"x":3,"y":7}}',
    ]
    assert mod._load_registry(tmp_path / "missing") == {"games": []}
    bad_registry_root = tmp_path / "bad-registry"
    (bad_registry_root / "ops").mkdir(parents=True)
    (bad_registry_root / mod.REGISTRY_RELATIVE_PATH).write_text("games: [\n", encoding="utf-8")
    assert mod._load_registry(bad_registry_root) == {"games": []}
    assert mod._registry_games({"games": "bad"}) == []
    assert mod._registry_totals({"games": [{"game": "x", "levels_reproduced": 2}]}) == {
        "reproducible_total_levels": 2,
        "reproducible_total_games": 1,
    }
    assert mod.select_rotated_target({"lf52": {"target_game": "lf52"}}) == {
        "target_game": "lf52",
        "reason": "first_available_rotated_target",
    }
    assert mod.select_rotated_target({}) == {
        "target_game": "re86",
        "reason": "default_rotated_target",
    }


def test_req_report_4471_defensive_ledger_branches(tmp_path: Path) -> None:
    """REQ-REPORT-4471: registry and gap rewrites are idempotent across existing rows."""

    _write_fixture_repo(tmp_path)
    _clock_state, now, sleep = _clock()

    artifact = mod.run(
        root=tmp_path,
        target_game="re86",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        ground_operator_fn=lambda **_: (
            mod.target_digest_for("re86"),
            {
                "operator": "graph_astar_action_cost",
                "game": "re86",
                "grounded": False,
                "solution": [],
                "counterexample_rounds": 1,
                "residual": "missing_pattern_match_sprite_resize_verifier",
                "verifier_is_oracle": True,
            },
        ),
        reproduce_fn=lambda _solution: pytest.fail("ungrounded re86 must not reproduce"),
        now=now,
        sleep_fn=sleep,
    )
    assert "complete no-new-level verdict requires missing_verifier_gaps" in mod.artifact_schema_errors(
        {**artifact, "missing_verifier_gaps": []}
    )

    # Existing target and gap rows are replaced, not duplicated.
    before = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert any(row["game"] == "re86" for row in before["games"])
    mod.update_arc_registry(tmp_path, artifact)
    after = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert sum(1 for row in after["games"] if row["game"] == "re86") == 1
    re86 = next(row for row in after["games"] if row["game"] == "re86")
    assert sum(1 for row in re86["dead_ends"] if row["gap_id"] == mod.RE86_GAP_ID) == 1
    mod.update_verifier_gaps(tmp_path, artifact)
    gap_text = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert gap_text.count("exp4471-gap-re86-pattern-match-sprite-resize:start") == 1

    success_artifact = {
        **artifact,
        "honest_verdict": "success: generic_first_contact_re86_L1_offline_reproduced",
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "missing_verifier_gaps": [],
    }
    banked = mod._banked_entry({"dead_ends": [{"gap_id": mod.RE86_GAP_ID, "status": "open"}]}, success_artifact)
    assert banked["dead_ends"][0]["status"] == "filled"

    no_newline_root = tmp_path / "no-newline-registry"
    (no_newline_root / "ops").mkdir(parents=True)
    (no_newline_root / mod.REGISTRY_RELATIVE_PATH).write_text(
        "games: []reproducible_total_levels: 0\n",
        encoding="utf-8",
    )
    mod._write_registry(
        no_newline_root,
        {
            "games": [{"game": "re86", "levels_reproduced": 0}],
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
        },
        target_game="re86",
    )
    assert "- game: re86" in (no_newline_root / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")

    no_totals_root = tmp_path / "no-totals-registry"
    (no_totals_root / "ops").mkdir(parents=True)
    (no_totals_root / mod.REGISTRY_RELATIVE_PATH).write_text("games:\n", encoding="utf-8")
    mod._write_registry(
        no_totals_root,
        {
            "games": [{"game": "re86", "levels_reproduced": 0}],
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
        },
        target_game="re86",
    )
    assert "reproducible_total_games: 0" in (
        no_totals_root / mod.REGISTRY_RELATIVE_PATH
    ).read_text(encoding="utf-8")

    empty_root = tmp_path / "empty-registry"
    mod._write_registry(
        empty_root,
        {
            "games": [{"game": "re86", "levels_reproduced": 0}],
            "reproducible_total_levels": 0,
            "reproducible_total_games": 0,
        },
        target_game="re86",
    )
    assert (empty_root / mod.REGISTRY_RELATIVE_PATH).exists()

    no_newline_gap_root = tmp_path / "no-newline-gap"
    (no_newline_gap_root / "ops").mkdir(parents=True)
    (no_newline_gap_root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text("header", encoding="utf-8")
    mod.update_verifier_gaps(no_newline_gap_root, artifact)
    assert "header\n" in (no_newline_gap_root / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
