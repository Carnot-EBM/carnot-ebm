"""Tests for Exp 4444 generic config-rule verifier operator.

Spec refs: REQ-REPORT-4444, SCENARIO-REPORT-4444.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot import experiment_4444_generic_config_rule_verifier_operator as mod
from carnot.agentic import arc_solver_kit as kit


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _examples() -> list[dict[str, str]]:
    return [
        {
            "game": "s5i5",
            "rule_id": "marker_coverage",
            "predicate": "controlled marker sprites cover target marker coordinates",
        },
        {
            "game": "ft09",
            "rule_id": "local_color_cycle_constraint",
            "predicate": "neighbor equality/inequality constraints are satisfied after color cycling",
        },
        {
            "game": "g50t",
            "rule_id": "target_offset_toggle",
            "predicate": "player reaches target offset and commits the toggle",
        },
    ]


def _ft09_digest() -> dict[str, Any]:
    return {
        "game": "ft09",
        "rule_family": "local_constraint_color_cycle",
        "constraints": [
            {
                "grid": [22, 22],
                "center_color": 8,
                "pattern": [[0, 2, 2], [0, 8, 0], [0, 2, 2]],
            }
        ],
        "cells": [
            {"grid": [18, 18], "color": 9, "kind": "Hkx"},
            {"grid": [22, 18], "color": 9, "kind": "Hkx"},
            {"grid": [26, 18], "color": 9, "kind": "Hkx"},
            {"grid": [18, 22], "color": 9, "kind": "Hkx"},
            {"grid": [26, 22], "color": 9, "kind": "Hkx"},
            {"grid": [18, 26], "color": 9, "kind": "Hkx"},
            {"grid": [22, 26], "color": 9, "kind": "Hkx"},
            {"grid": [26, 26], "color": 9, "kind": "Hkx"},
        ],
        "color_cycle": [9, 8],
        "neighbor_step": 4,
        "click_scale": 2,
    }


def _dc22_groundable_digest() -> dict[str, Any]:
    return {
        "game": "dc22",
        "rule_family": "marker_coverage",
        "controlled_markers": [[0, 0]],
        "target_markers": [[2, 0]],
        "step": 1,
        "horizontal_label": "toggle_right",
        "vertical_label": "toggle_down",
    }


def _ok_preconditions() -> dict[str, Any]:
    return {
        "ft09_env_present": True,
        "dc22_env_present": True,
        "arc_solver_kit_importable": True,
        "generator_resource_available": True,
        "qwen_gguf_cached": True,
        "igpu_llama_server_available": True,
        "focused_baseline_selected_green": True,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "environment_files/ft09/fixture").mkdir(parents=True, exist_ok=True)
    (root / "environment_files/dc22/fixture").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        "\n".join(
            [
                "schema_version: 1",
                "updated: '2026-06-19'",
                "general_gotchas:",
                "- id: primitive_config_rule_grounding",
                "  operator: config_rule_grounding",
                "games:",
                "- game: s5i5",
                "  reproducibility: reproduced",
                "  levels_reproduced: 1",
                "  win_condition: marker coverage",
                "- game: ft09",
                "  reproducibility: reproduced",
                "  levels_reproduced: 1",
                "  win_condition: local color cycle",
                "- game: dc22",
                "  reproducibility: unsolved",
                "  levels_reproduced: 0",
                "reproducible_total_levels: 37",
                "reproducible_total_games: 18",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _ft09_reproduce(solution: Sequence[str]) -> dict[str, Any]:
    assert list(solution) == ["click:36,36", "click:36,44", "click:52,44", "click:36,52"]
    return {"game": "ft09", "claimed_level": 1, "reached_level": 1, "reproduced": True}


def _dc22_reproduce(solution: Sequence[str]) -> dict[str, Any]:
    assert list(solution) == ["toggle_right", "toggle_right"]
    return {"game": "dc22", "claimed_level": 1, "reached_level": 0, "reproduced": False}


def test_req_report_4444_spec_declares_generic_config_rule_verifier_contract() -> None:
    """REQ-REPORT-4444: OpenSpec declares the .411 verifier operator and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4444" in spec
    assert "SCENARIO-REPORT-4444" in spec
    assert "config_rule_verifier" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4444_solver_kit_operator_preserves_legacy_marker_grounding() -> None:
    """REQ-REPORT-4444: config_rule_verifier composes the legacy marker helper."""

    result = kit.config_rule_verifier(
        game="s5i5",
        object_digest={
            "rule_family": "marker_coverage",
            "controlled_markers": [[9, 33], [30, 9]],
            "target_markers": [[9, 51], [51, 9]],
            "step": 3,
            "horizontal_label": "h_extend",
            "vertical_label": "v_extend",
        },
        few_shot_examples=_examples(),
    )

    assert result["operator"] == "config_rule_verifier"
    assert result["legacy_operator"] == "config_rule_grounding"
    assert result["grounded"] is True
    assert result["predicate_id"] == "marker_coverage"
    assert result["solution"] == ["h_extend"] * 7 + ["v_extend"] * 6

    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert "config_rule_grounding" in operators
    assert "config_rule_verifier" in operators
    selected = kit.select_primitive_operators(mechanic_class="local_constraint_color_cycle")
    assert [row.operator for row in selected][:2] == ["config_rule_verifier", "config_rule_grounding"]


def test_req_report_4444_solver_kit_operator_grounds_local_constraint_color_cycle() -> None:
    """SCENARIO-REPORT-4444: ft09-style local constraints ground without ft09's recipe."""

    result = kit.config_rule_verifier(
        game="ft09",
        object_digest=_ft09_digest(),
        few_shot_examples=_examples(),
    )

    assert result["grounded"] is True
    assert result["predicate_id"] == "local_constraint_color_cycle"
    assert result["recipe_source"] == "generic_config_rule_verifier"
    assert result["target_recipe_withheld"] == "ft09"
    assert result["solution"] == ["click:36,36", "click:36,44", "click:52,44", "click:36,52"]
    assert result["verifier"]["start_violation_count"] == 4
    assert result["verifier"]["final_violation_count"] == 0
    assert result["grounded_win_condition"]["fires_on_win"] is True
    assert result["grounded_win_condition"]["rejects_nonwins"] is True


def test_req_report_4444_solver_kit_operator_rejects_ungrounded_candidates() -> None:
    """REQ-REPORT-4444: ungrounded config candidates are rejected instead of solved."""

    result = kit.config_rule_verifier(
        game="dc22",
        object_digest={"game": "dc22", "components": {"toggles": []}},
        few_shot_examples=_examples(),
    )

    assert result["grounded"] is False
    assert result["solution"] == []
    assert result["residual"] == "missing_config_rule_verifier_grounding"
    assert result["verifier_is_oracle"] is True


def test_scenario_report_4444_run_closes_ft09_and_logs_grounded_dc22_gap(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4444: ft09 gates offline and dc22 grounded-no-level is terminal."""

    _write_fixture_repo(tmp_path)
    clock = {"t": 100.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        few_shot_examples=_examples(),
        ft09_digest=_ft09_digest(),
        dc22_digest=_dc22_groundable_digest(),
        reproduce_ft09_fn=_ft09_reproduce,
        reproduce_dc22_fn=_dc22_reproduce,
        no_regression_fn=lambda _root: True,
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == "complete: ft09_generic_resolved_dc22_grounded_no_level_gap_logged"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["ft09_resolved_generically"] is True
    assert artifact["dc22_state"] == "grounded_no_level"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["no_regression"] is True
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["ft09_operator_result"]["target_recipe_withheld"] == "ft09"


def test_req_report_4444_blocked_precondition_and_schema_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4444: blocked resources do not fabricate solver or regression claims."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "dc22_env_present": False, "ok": False},
        few_shot_examples=_examples(),
        ft09_digest=_ft09_digest(),
        dc22_digest=_dc22_groundable_digest(),
        reproduce_ft09_fn=lambda _solution: calls.append("ft09") or {},
        reproduce_dc22_fn=lambda _solution: calls.append("dc22") or {},
        no_regression_fn=lambda _root: calls.append("regression") or True,
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_offline_env_dc22"
    assert artifact["ft09_resolved_generically"] is False
    assert artifact["dc22_state"] == "not_grounded"
    assert artifact["offline_reproduced"] is False
    assert artifact["no_regression"] is False
    assert mod.artifact_schema_errors(artifact) == []

    bad: Mapping[str, Any] = {
        **artifact,
        "honest_verdict": "partial: retry",
        "inference_substrate": None,
        "ft09_resolved_generically": "true",
        "dc22_state": "maybe",
        "reproduced_levels": "1",
        "offline_reproduced": "true",
        "no_regression": "true",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "random_seed": "4444",
        "reproducibility_checksum": "bad",
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "missing inference_substrate" in errors
    assert "ft09_resolved_generically must be bare bool" in errors
    assert "dc22_state must be solved / grounded_no_level / not_grounded" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "no_regression must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
