"""Tests for Exp 4446 generic first-contact routed bank.

Spec refs: REQ-REPORT-4446, SCENARIO-REPORT-4446.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4446_drive_generic_first_contact_bank as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _ok_preconditions() -> dict[str, Any]:
    return {
        "target_env_present": True,
        "arc_solver_kit_importable": True,
        "arc_solve_learning_importable": True,
        "generator_resource_available": True,
        "qwen_gguf_cached": True,
        "igpu_llama_server_available": False,
        "focused_exp4423_pytest_green": True,
        "focused_exp4423_exact_command_green": False,
        "focused_exp4423_exact_command_blocker": "repo_addopts_package_wide_coverage",
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _recommendation() -> dict[str, Any]:
    return {
        "target_game": "vc33",
        "recommended": [
            {
                "game": "s5i5",
                "similarity": 6.0,
                "solver": "python/carnot/experiment_4421_config_rule_solve_unseen.py",
                "win_condition": "L1 marker-coverage config: controlled marker sprites occupy targets",
                "action_model": "ACTION6 click-only; h_extend/v_extend marker config controls",
            }
        ],
        "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        "strategy": {"routed_mechanic": "graph_explore"},
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "environment_files" / "vc33" / "5430563c").mkdir(parents=True)
    (root / "ops").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "games": [
                    {
                        "game": "s5i5",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "solver": "config-rule fixture",
                    },
                    {
                        "game": "vc33",
                        "reproducibility": "unsolved",
                        "levels_reproduced": 0,
                        "solver": "scripts/arc_loop_solve.py --game vc33",
                        "dead_ends": [
                            {
                                "gap_id": mod.VC33_GAP_ID,
                                "status": "open",
                                "failure_mode": "needs_per_game_RE",
                            }
                        ],
                    },
                ],
                "reproducible_total_levels": 38,
                "reproducible_total_games": 19,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text("# Verifier Gaps\n", encoding="utf-8")


def _clock() -> Any:
    ticks = iter([0.0, 1.1, 1.1])
    return lambda: next(ticks)


def test_req_report_4446_spec_declares_routed_bank_contract() -> None:
    """REQ-REPORT-4446: OpenSpec declares the driver and required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4446" in spec
    assert "SCENARIO-REPORT-4446" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "arc_solve_learning.recommend_approach(target_game)" in spec
    assert "config_rule_verifier" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4446_selects_config_operator_from_s5i5_route() -> None:
    """REQ-REPORT-4446: routed config recipes select the generic config-rule operator."""

    selection = mod.select_generic_operator(_recommendation())

    assert selection["operator"] == "config_rule_verifier"
    assert selection["routed_to"] == "s5i5"
    assert selection["reason"] == "routed_recipe_contains_config_rule_markers"


def test_scenario_report_4446_success_banks_vc33_and_updates_ledgers(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4446: reproduced vc33 L1 is banked through the offline gate."""

    _write_fixture_repo(tmp_path)
    calls: list[list[str]] = []

    def reproduce(solution: list[str]) -> dict[str, Any]:
        calls.append(list(solution))
        return {
            "game": "vc33",
            "reached_level": 1,
            "claimed_level": 1,
            "reproduced": True,
            "mode": "offline_reproduction_gate_no_quota",
        }

    artifact = mod.run(
        root=tmp_path,
        target_game="vc33",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        reproduce_fn=reproduce,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert calls == [["lower_click", "lower_click", "lower_click"]]
    assert artifact["honest_verdict"] == "success: generic_first_contact_vc33_L1_offline_reproduced"
    assert artifact["target_game"] == "vc33"
    assert artifact["routed_to"] == "s5i5"
    assert artifact["selected_operator"]["operator"] == "config_rule_verifier"
    assert artifact["operator_result"]["grounded"] is True
    assert artifact["operator_result"]["predicate_id"] == "marker_coverage"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is True
    assert artifact["duration_s"] >= 1.0
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))[
        "reproduced_levels"
    ] == 1

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    vc33 = next(row for row in registry["games"] if row["game"] == "vc33")
    assert vc33["reproducibility"] == "reproduced"
    assert vc33["levels_reproduced"] == 1
    assert vc33["dead_ends"][0]["status"] == "filled"
    assert registry["reproducible_total_levels"] == 39
    assert registry["reproducible_total_games"] == 20
    gaps = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "exp4446-gap-4423-vc33-unselectable-first-contact:start" in gaps
    assert "status: filled" in gaps


def test_scenario_report_4446_no_level_is_terminal_complete_with_gap(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4446: grounded no-bank attempts log the residual as complete."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        target_game="vc33",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        reproduce_fn=lambda _solution: {
            "game": "vc33",
            "reached_level": 0,
            "claimed_level": 1,
            "reproduced": False,
            "mode": "offline_reproduction_gate_no_quota",
        },
        write_registry=False,
        write_gaps=True,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "complete: generic_first_contact_vc33_routed_no_new_level"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == mod.VC33_GAP_ID
    assert artifact["missing_verifier_gaps"][0]["residual_delta"] == "support_clearance_replay_failed"
    assert "partial:" not in artifact["honest_verdict"]
    assert mod.artifact_schema_errors(artifact) == []
    assert "movement: still_open" in (
        tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH
    ).read_text(encoding="utf-8")


def test_req_report_4446_blocked_precondition_stops_before_routing(tmp_path: Path) -> None:
    """REQ-REPORT-4446: missing resources write blocked artifacts without routing."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        target_game="vc33",
        preconditions_checked={**_ok_preconditions(), "generator_resource_available": False},
        recommend_fn=lambda game: calls.append(game) or _recommendation(),
        reproduce_fn=lambda _solution: pytest.fail("reproduce must not run"),
        now=lambda: 2.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_qwen_generator_resource"
    assert artifact["target_game"] == "vc33"
    assert artifact["routed_to"] == ""
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"target_env_present": False}, "offline_env_vc33"),
        ({"arc_solver_kit_importable": False}, "arc_solver_kit"),
        ({"arc_solve_learning_importable": False}, "arc_solve_learning"),
        ({"focused_exp4423_pytest_green": False}, "focused_exp4423_pytest"),
        ({"no_3090_inference": False}, "no_3090_inference_policy"),
        ({"leaderboard_submission": True}, "leaderboard_submission_policy"),
    ],
)
def test_req_report_4446_precondition_miss_names_resource(
    override: dict[str, Any],
    expected: str,
) -> None:
    """REQ-REPORT-4446: precondition misses map to explicit blocked resources."""

    assert mod.first_precondition_miss({**_ok_preconditions(), **override}) == expected


def test_req_report_4446_schema_rejects_fabricated_or_malformed_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-4446: schema rejects partial prefixes, type drift, and fake success."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        target_game="vc33",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        reproduce_fn=lambda _solution: {
            "game": "vc33",
            "reached_level": 1,
            "claimed_level": 1,
            "reproduced": True,
        },
        write_registry=False,
        write_gaps=False,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )
    bad = {
        **artifact,
        "honest_verdict": "partial: retry_later",
        "inference_substrate": None,
        "duration_s": 0.5,
        "target_game": 4446,
        "routed_to": "",
        "reproduced_levels": "1",
        "offline_reproduced": "true",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "random_seed": "4446",
        "reproducibility_checksum": "z" * 64,
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
    assert "reproduced_levels must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "no_3090_inference must be true" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4446" in errors

    fabricated = {
        **artifact,
        "offline_reproduced": False,
        "reproduced_levels": 0,
    }
    assert "success verdict requires offline_reproduced true" in mod.artifact_schema_errors(fabricated)

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, {"honest_verdict": "partial: invalid"})


def test_req_report_4446_helper_branches_and_defensive_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-4446: helper branches stay deterministic and ledger edits are idempotent."""

    sleep_calls: list[float] = []
    ticks = iter([0.0, 1.05])
    assert mod._sleep_until_verifier_floor(
        started_at=0.0,
        now=lambda: next(ticks),
        sleep_fn=sleep_calls.append,
    ) == 1.05
    assert sleep_calls == [mod.VERIFIER_SCORING_DURATION_TARGET_S]

    assert mod._closest_recipe({"recommended": []}) == {}
    assert mod.select_generic_operator(
        {"recommended": [{"game": "ar25", "solver": "object motion reflect"}]}
    )["operator"] == "object_motion_world_model"
    assert mod.select_generic_operator(
        {
            "recommended": [{"game": "plain", "solver": "plain"}],
            "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        }
    ) == {
        "operator": "graph_astar_action_cost",
        "routed_to": "plain",
        "reason": "fallback_to_router_selected_operator",
    }
    assert mod.select_generic_operator({})["operator"] == "object_centric_digest"

    digest, ungrounded = mod._ground_operator(
        target_game="vc33",
        selected_operator={"operator": "object_centric_digest"},
        few_shot_examples=[],
    )
    assert digest["abstract_rule_family"] == "support_clearance_as_marker_coverage"
    assert ungrounded["residual"] == "selected_operator_not_config_rule_verifier"
    assert mod._missing_gap(
        target_game="vc33",
        routed_to="plain",
        operator_result=ungrounded,
        reproduction_result={"reproduced": False},
    )["residual_delta"] == "selected_operator_not_config_rule_verifier"
    assert mod._missing_gap(
        target_game="vc33",
        routed_to="plain",
        operator_result={"operator": "config_rule_verifier", "grounded": True},
        reproduction_result={"reproduced": True},
    )["residual_delta"] == "none"

    real_import = __import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "carnot":
            raise ImportError("blocked for fallback coverage")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    assert mod.extract_few_shot_examples(tmp_path) == []
    monkeypatch.setattr("builtins.__import__", real_import)

    _write_fixture_repo(tmp_path)
    ungrounded_artifact = mod.run(
        root=tmp_path,
        target_game="vc33",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: {
            "recommended": [{"game": "plain", "solver": "plain"}],
            "selected_generic_operators": [{"operator": "object_centric_digest"}],
        },
        reproduce_fn=lambda _solution: pytest.fail("ungrounded operator must not reproduce"),
        write_registry=False,
        write_gaps=False,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )
    assert ungrounded_artifact["missing_verifier_gaps"][0]["residual_delta"] == (
        "selected_operator_not_config_rule_verifier"
    )

    schema_artifact = {
        **ungrounded_artifact,
        "inference_substrate": "unsupported",
        "duration_s": 0.5,
    }
    errors = mod.artifact_schema_errors(schema_artifact)
    assert "inference_substrate has unsupported value" in errors
    cached_duration_errors = mod.artifact_schema_errors(
        {**ungrounded_artifact, "inference_substrate": mod.INFERENCE_SUBSTRATE, "duration_s": 0.5}
    )
    assert "cached verifier substrate requires duration_s >= 1.0" in cached_duration_errors
    live_errors = mod.artifact_schema_errors(
        {**ungrounded_artifact, "inference_substrate": mod.LIVE_LLM_SUBSTRATE, "duration_s": 1.0}
    )
    assert "live_llm_inference requires duration_s >= 60.0" in live_errors
    offline_bad = {
        **ungrounded_artifact,
        "honest_verdict": "complete: bad_offline_gate",
        "offline_reproduced": True,
        "reproduced_levels": 0,
    }
    assert "offline_reproduced true requires reproduced_levels >= 1" in mod.artifact_schema_errors(
        offline_bad
    )

    assert mod._load_registry(tmp_path / "missing-root") == {"games": []}
    bad_registry_root = tmp_path / "bad-registry"
    (bad_registry_root / "ops").mkdir(parents=True)
    (bad_registry_root / mod.REGISTRY_RELATIVE_PATH).write_text("games: [\n", encoding="utf-8")
    assert mod._load_registry(bad_registry_root) == {"games": []}
    assert mod._registry_games({"games": "bad"}) == []
    assert mod._registry_totals({"games": [{"game": "x", "levels_reproduced": 2}]}) == {
        "reproducible_total_levels": 2,
        "reproducible_total_games": 1,
    }
    assert mod._target_entry({"games": [{"game": "x"}]}, "vc33") is None

    appended = mod._banked_entry({}, {**ungrounded_artifact, "offline_reproduced": True, "reproduced_levels": 1})
    assert appended["dead_ends"][0]["gap_id"] == mod.VC33_GAP_ID
    appended_after_other = mod._banked_entry(
        {"dead_ends": [{"gap_id": "OTHER"}]},
        {**ungrounded_artifact, "offline_reproduced": True, "reproduced_levels": 1},
    )
    assert [row["gap_id"] for row in appended_after_other["dead_ends"]] == ["OTHER", mod.VC33_GAP_ID]

    registry_append_root = tmp_path / "registry-append"
    (registry_append_root / "ops").mkdir(parents=True)
    (registry_append_root / mod.REGISTRY_RELATIVE_PATH).write_text("games: []\n", encoding="utf-8")
    mod.update_arc_registry(
        registry_append_root,
        {**ungrounded_artifact, "offline_reproduced": True, "reproduced_levels": 1},
    )
    loaded_append = yaml.safe_load(
        (registry_append_root / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert loaded_append["games"][0]["game"] == "vc33"

    registry_no_totals = tmp_path / "registry-no-totals"
    (registry_no_totals / "ops").mkdir(parents=True)
    (registry_no_totals / mod.REGISTRY_RELATIVE_PATH).write_text(
        "games:\n- game: vc33\n  reproducibility: unsolved\n  levels_reproduced: 0\n",
        encoding="utf-8",
    )
    mod.update_arc_registry(
        registry_no_totals,
        {**ungrounded_artifact, "offline_reproduced": True, "reproduced_levels": 1},
    )
    assert "reproducible_total_levels: 1" in (
        registry_no_totals / mod.REGISTRY_RELATIVE_PATH
    ).read_text(encoding="utf-8")

    registry_no_file = tmp_path / "registry-no-file"
    mod._write_registry(
        registry_no_file,
        {"games": [{"game": "vc33"}], "reproducible_total_levels": 0, "reproducible_total_games": 0},
        target_game="vc33",
    )
    assert (registry_no_file / mod.REGISTRY_RELATIVE_PATH).exists()

    before = (tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    mod.update_arc_registry(tmp_path, {**ungrounded_artifact, "offline_reproduced": False})
    assert (tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8") == before

    gap_root = tmp_path / "gap-root"
    (gap_root / "ops").mkdir(parents=True)
    (gap_root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text("header", encoding="utf-8")
    mod.update_verifier_gaps(gap_root, ungrounded_artifact)
    assert "movement: still_open" in (gap_root / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    mod.update_verifier_gaps(
        gap_root,
        {**ungrounded_artifact, "offline_reproduced": True, "reproduced_levels": 1},
    )
    assert "movement: filled" in (gap_root / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
