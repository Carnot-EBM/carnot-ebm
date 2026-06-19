"""Tests for Exp 4458 generic first-contact new-game attempt.

Spec refs: REQ-REPORT-4458, SCENARIO-REPORT-4458.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import yaml

from carnot import experiment_4458_first_contact_new_game as mod


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
        "target_game": "sb26",
        "target_features": {"action_type": "click", "win_kw": ["click"]},
        "strategy": {"routed_mechanic": "config_rule", "solver": "generic_config_rule"},
        "retrieved_primitives": [
            {
                "name": "config_rule_verifier",
                "operator": "config_rule_verifier",
                "mechanic_class": "config_rule",
                "score": 8.5,
                "matched_cues": ["color", "slot", "sequence"],
            },
            {
                "name": "glyph_rewrite_rule_verifier",
                "operator": "glyph_rewrite_rule_verifier",
                "mechanic_class": "config_substitution",
                "score": 2.0,
                "matched_cues": ["sequence"],
            },
        ],
        "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        "recommended": [
            {
                "game": "ft09",
                "similarity": 4.0,
                "solver": "python/carnot/experiment_4444_generic_config_rule_verifier_operator.py",
                "win_condition": "local constraint color-cycle config rule",
                "action_model": "ACTION6 click color-cycle cells; frame level gate",
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
                        "game": "ft09",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "solver": "config-rule fixture",
                        "mechanic_class": "local_constraint_color_cycle",
                    }
                ],
                "reproducible_total_levels": 38,
                "reproducible_total_games": 20,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text("# Verifier Gaps\n", encoding="utf-8")


def _clock() -> Any:
    ticks = iter([0.0, 1.1, 1.1])
    return lambda: next(ticks)


def test_req_report_4458_spec_declares_new_game_contract() -> None:
    """REQ-REPORT-4458: OpenSpec declares routing, LILO retrieval, and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4458" in spec
    assert "SCENARIO-REPORT-4458" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "arc_solve_learning.recommend_approach(target_game)" in spec
    assert "retrieve_primitives(digest)" in spec
    assert "sb26" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4458_selects_best_fitting_retrieved_operator() -> None:
    """REQ-REPORT-4458: LILO primitives take precedence when selecting the generic operator."""

    selection = mod.select_generic_operator(_recommendation())

    assert selection == {
        "operator": "config_rule_verifier",
        "routed_to": "ft09",
        "reason": "retrieved_config_rule_primitive_matches_slot_sequence",
        "source": "retrieved_primitives",
    }


def test_scenario_report_4458_sb26_no_level_is_terminal_and_logs_residual(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4458: sb26 routes through primitives and logs missing slot sequencing."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        target_game="sb26",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        reproduce_fn=lambda _solution: pytest.fail("ungrounded sb26 must not reproduce"),
        write_registry=True,
        write_gaps=True,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "complete: generic_first_contact_sb26_routed_no_new_level"
    assert artifact["target_game"] == "sb26"
    assert artifact["routed_to"] == "ft09"
    assert artifact["retrieved_primitives"][0]["operator"] == "config_rule_verifier"
    assert artifact["selected_operator"]["operator"] == "config_rule_verifier"
    assert artifact["target_digest"]["rule_family"] == "color_match_slot_sequence"
    assert artifact["operator_result"]["grounded"] is False
    assert artifact["operator_result"]["residual"] == "missing_config_rule_verifier_grounding"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["reproducible_total_levels"] == 38
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == mod.SB26_GAP_ID
    assert artifact["missing_verifier_gaps"][0]["residual_delta"] == "missing_color_match_slot_sequence_verifier"
    assert "partial:" not in artifact["honest_verdict"]
    assert artifact["verifier_is_oracle"] is True
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))[
        "target_game"
    ] == "sb26"

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    sb26 = next(row for row in registry["games"] if row["game"] == "sb26")
    assert sb26["reproducibility"] == "unsolved"
    assert sb26["levels_reproduced"] == 0
    assert sb26["dead_ends"][0]["gap_id"] == mod.SB26_GAP_ID
    assert registry["reproducible_total_levels"] == 38
    gaps = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "exp4458-gap-sb26-color-match-slot-sequence:start" in gaps
    assert "movement: still_open" in gaps


def test_scenario_report_4458_success_banks_one_reproduced_level(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4458: reproduced L1 banks only through the offline gate."""

    _write_fixture_repo(tmp_path)
    calls: list[list[str]] = []

    def ground(
        *,
        target_game: str,
        selected_operator: Mapping[str, str],
        few_shot_examples: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del few_shot_examples
        digest = mod.sb26_color_match_sequence_digest()
        return digest, {
            "operator": selected_operator["operator"],
            "game": target_game,
            "grounded": True,
            "predicate_id": "color_match_slot_sequence",
            "solution": ["click:36,59", "click:23,30", "validate"],
            "counterexample_rounds": 2,
            "verifier_is_oracle": True,
        }

    def reproduce(solution: Sequence[str]) -> dict[str, Any]:
        calls.append(list(solution))
        return {
            "game": "sb26",
            "claimed_level": 1,
            "reached_level": 1,
            "reproduced": True,
            "mode": "offline_reproduction_gate_no_quota",
        }

    artifact = mod.run(
        root=tmp_path,
        target_game="sb26",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        ground_operator_fn=ground,
        reproduce_fn=reproduce,
        write_registry=True,
        write_gaps=True,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert calls == [["click:36,59", "click:23,30", "validate"]]
    assert artifact["honest_verdict"] == "success: generic_first_contact_sb26_L1_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["reproducible_total_levels"] == 39
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    sb26 = next(row for row in registry["games"] if row["game"] == "sb26")
    assert sb26["reproducibility"] == "reproduced"
    assert sb26["levels_reproduced"] == 1
    assert registry["reproducible_total_levels"] == 39
    assert registry["reproducible_total_games"] == 21
    assert "movement: filled" in (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )


def test_req_report_4458_blocked_precondition_stops_before_routing(tmp_path: Path) -> None:
    """REQ-REPORT-4458: missing resources produce terminal blocked artifacts."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        target_game="sb26",
        preconditions_checked={**_ok_preconditions(), "generator_resource_available": False},
        recommend_fn=lambda game: calls.append(game) or _recommendation(),
        reproduce_fn=lambda _solution: pytest.fail("reproduce must not run"),
        now=lambda: 2.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_qwen_generator_resource"
    assert artifact["inference_substrate"] == mod.BLOCKED_INFERENCE_SUBSTRATE
    assert artifact["target_game"] == "sb26"
    assert artifact["routed_to"] == ""
    assert artifact["retrieved_primitives"] == []
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"target_env_present": False}, "offline_env_sb26"),
        ({"arc_solver_kit_importable": False}, "arc_solver_kit"),
        ({"arc_solve_learning_importable": False}, "arc_solve_learning"),
        ({"generator_resource_available": False}, "qwen_generator_resource"),
        ({"focused_exp4423_pytest_green": False}, "focused_exp4423_pytest"),
        ({"no_3090_inference": False}, "no_3090_inference_policy"),
        ({"leaderboard_submission": True}, "leaderboard_submission_policy"),
    ],
)
def test_req_report_4458_precondition_miss_names_resource(
    override: dict[str, Any],
    expected: str,
) -> None:
    """REQ-REPORT-4458: blocked resources are explicit and deterministic."""

    assert mod.first_precondition_miss({**_ok_preconditions(), **override}, "sb26") == expected


def test_req_report_4458_schema_rejects_fabrication_and_type_drift(tmp_path: Path) -> None:
    """REQ-REPORT-4458: schema rejects fake success, partial prefixes, and missing evidence."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        target_game="sb26",
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        reproduce_fn=lambda _solution: pytest.fail("ungrounded sb26 must not reproduce"),
        write_registry=False,
        write_gaps=False,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )
    bad: dict[str, Any] = {
        **artifact,
        "honest_verdict": "partial: retry_later",
        "inference_substrate": None,
        "duration_s": 0.5,
        "target_game": 4458,
        "routed_to": "",
        "retrieved_primitives": [],
        "reproduced_levels": "0",
        "offline_reproduced": "false",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "reproducible_total_levels": "38",
        "random_seed": "4458",
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
    assert "field_principles.honest_verdict must match REQ-REPORT-4458" in errors

    fabricated = {
        **artifact,
        "honest_verdict": "success: generic_first_contact_sb26_L1_offline_reproduced",
        "offline_reproduced": False,
        "reproduced_levels": 0,
    }
    assert "success verdict requires offline_reproduced true" in mod.artifact_schema_errors(
        fabricated
    )
    assert "success verdict requires reproduced_levels >= 1" in mod.artifact_schema_errors(
        fabricated
    )

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, {"honest_verdict": "partial: invalid"})
