"""Tests for Exp 4479 re86 sprite-overlay verifier solve.

Spec refs: REQ-REPORT-4479, SCENARIO-REPORT-4479.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4479_solve_re86 as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _ok_preconditions() -> dict[str, Any]:
    return {
        "arc_solver_kit_importable": True,
        "offline_arcade_reachable": True,
        "target_env_present": True,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "ops").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    (root / "environment_files" / "re86" / "8af5384d").mkdir(parents=True)
    (root / mod.ARC_REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "games": [
                    {
                        "game": "sb26",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                    },
                    {
                        "game": "re86",
                        "reproducibility": "unsolved",
                        "levels_reproduced": 0,
                        "dead_ends": [{"gap_id": mod.RE86_GAP_ID, "status": "open"}],
                    },
                ],
                "reproducible_total_levels": 45,
                "reproducible_total_games": 22,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text(
        "<!-- exp4471-gap-re86-pattern-match-sprite-resize:start -->\n"
        "### GAP-4471-RE86-MISSING-PATTERN-MATCH-SPRITE-RESIZE-VERIFIER: fixture\n"
        "- status: open\n"
        "<!-- exp4471-gap-re86-pattern-match-sprite-resize:end -->\n",
        encoding="utf-8",
    )
    (root / mod.VERIFIER_REGISTRY_RELATIVE_PATH).write_text("roles: []\n", encoding="utf-8")


def _cross(size: int, color: int, *, center_zero: bool = False) -> list[list[int]]:
    center = size // 2
    pixels = [[-1 for _ in range(size)] for _ in range(size)]
    for index in range(size):
        pixels[center][index] = color
        pixels[index][center] = color
    if center_zero:
        pixels[center][center] = 0
    return pixels


def test_req_report_4479_spec_declares_re86_sprite_overlay_contract() -> None:
    """REQ-REPORT-4479: OpenSpec declares verifier, gate, and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4479" in spec
    assert "SCENARIO-REPORT-4479" in spec
    assert "sprite_overlay_resize_verifier" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4479_solver_kit_operator_matches_overlay_sources_and_cycles() -> None:
    """REQ-REPORT-4479: verifier derives executable movement/cycle labels."""

    digest = {
        "rule_family": "sprite_overlay_pattern_match",
        "movement_step": 3,
        "target_match_ignore_colors": [-1, 4],
        "actions": {
            "up": "U",
            "down": "D",
            "left": "L",
            "right": "R",
            "cycle": "C",
        },
        "active_source_index": 1,
        "sources": [
            {"id": "eleven", "x": 6, "y": 6, "pixels": _cross(5, 11)},
            {"id": "nine", "x": 0, "y": 9, "pixels": _cross(5, 9, center_zero=True)},
        ],
        "required_pixels": [
            {"x": 2, "y": 2, "color": 11},
            {"x": 8, "y": 3, "color": 9},
        ],
    }

    result = kit.sprite_overlay_resize_verifier("fixture", digest, [])

    assert result["operator"] == "sprite_overlay_resize_verifier"
    assert result["grounded"] is True
    assert result["predicate_id"] == "sprite_overlay_pattern_match_resize"
    assert result["coverage"]["missing_required_pixels"] == []
    assert result["solution"] == ["U", "U", "R", "R", "C", "U", "U", "L", "L"]
    operators = {row.operator for row in kit.primitive_operator_registry()}
    assert "sprite_overlay_resize_verifier" in operators
    selected = kit.select_primitive_operators(mechanic_class="pattern_match_sprite_resize", game="re86")
    assert selected[0].operator == "sprite_overlay_resize_verifier"


def test_req_report_4479_solver_kit_operator_uses_explicit_resize_variant() -> None:
    """REQ-REPORT-4479: resize variants are explicit evidence, not fabricated."""

    digest = {
        "rule_family": "sprite_overlay_pattern_match",
        "movement_step": 1,
        "actions": {"up": "U", "down": "D", "left": "L", "right": "R"},
        "active_source_index": 0,
        "sources": [
            {
                "id": "growable",
                "x": 0,
                "y": 0,
                "pixels": _cross(3, 6, center_zero=True),
                "variants": [
                    {
                        "id": "grown_cross",
                        "pixels": _cross(5, 6, center_zero=True),
                        "pre_labels": ["GROW"],
                    }
                ],
            }
        ],
        "required_pixels": [
            {"x": 0, "y": 2, "color": 6},
            {"x": 4, "y": 2, "color": 6},
        ],
    }

    result = kit.sprite_overlay_resize_verifier("fixture", digest, [])

    assert result["grounded"] is True
    assert result["solution"] == ["GROW"]
    assert result["placements"][0]["variant_id"] == "grown_cross"
    assert result["placements"][0]["resize_variant_used"] is True


def test_scenario_report_4479_default_re86_digest_reproduces_l1() -> None:
    """SCENARIO-REPORT-4479: real re86 labels pass arc_solver_kit.reproduce()."""

    digest = mod.build_re86_object_digest()
    result = kit.sprite_overlay_resize_verifier("re86", digest, [])
    gate = mod.reproduce_re86_solution(result["solution"])

    assert result["grounded"] is True
    assert len(result["solution"]) == 20
    assert result["coverage"]["covered_required_pixels"] == 8
    assert gate["reproduced"] is True
    assert gate["reached_level"] >= 1


def test_scenario_report_4479_run_banks_re86_and_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4479: reproduced re86 L1 banks exactly one level."""

    _write_fixture_repo(tmp_path)
    clock = {"t": 100.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        object_digest_fn=mod.build_re86_object_digest,
        reproduce_fn=mod.reproduce_re86_solution,
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == "success: re86_L1_sprite_overlay_resize_offline_reproduced"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["target_game"] == "re86"
    assert artifact["sprite_overlay_verifier_built"] is True
    assert artifact["registered_verifier_operator"] == "sprite_overlay_resize_verifier"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["reproducible_total_levels"] == 46
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproduction_result"]["reproduced"] is True
    registry = yaml.safe_load((tmp_path / mod.ARC_REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    re86 = next(row for row in registry["games"] if row["game"] == "re86")
    assert re86["reproducibility"] == "reproduced"
    assert re86["levels_reproduced"] == 1
    assert re86["latest_exp4479_solve_re86"]["artifact"] == mod.RESULT_RELATIVE_PATH
    assert registry["reproducible_total_levels"] == 46
    assert "movement: filled" in (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    verifier_registry = (tmp_path / mod.VERIFIER_REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "sprite_overlay_resize_verifier" in verifier_registry


def test_req_report_4479_no_advance_emits_missing_verifier_gap(tmp_path: Path) -> None:
    """REQ-REPORT-4479: ungrounded/no-bank runs emit gaps instead of success."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        object_digest_fn=lambda: {
            "rule_family": "sprite_overlay_pattern_match",
            "movement_step": 3,
            "actions": {"up": "U"},
            "sources": [{"id": "bad", "x": 0, "y": 0, "pixels": [[9]]}],
            "required_pixels": [{"x": 100, "y": 100, "color": 11}],
        },
        reproduce_fn=lambda _solution: pytest.fail("ungrounded candidate must not reproduce"),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "complete: re86_sprite_overlay_resize_no_new_level_gap_logged"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == mod.RE86_GAP_ID
    assert artifact["missing_verifier_gaps"][0]["residual_delta"] == "sprite_overlay_required_pixels_uncovered"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4479_blocked_precondition_and_schema_guards(tmp_path: Path) -> None:
    """REQ-REPORT-4479: blocked resources stop before solving and schema rejects fabrication."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "offline_arcade_reachable": False, "ok": False},
        object_digest_fn=lambda: calls.append("digest") or {},
        reproduce_fn=lambda _solution: pytest.fail("blocked run must not reproduce"),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_offline_arcade"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []

    bad = {
        **artifact,
        "honest_verdict": "partial: fake",
        "inference_substrate": None,
        "target_game": "",
        "sprite_overlay_verifier_built": "true",
        "registered_verifier_operator": "",
        "offline_reproduced": "false",
        "reproduced_levels": "0",
        "reproducible_total_levels": "46",
        "preconditions_checked": [],
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "solution_labels": {},
        "reproduction_result": [],
        "random_seed": "4479",
        "reproducibility_checksum": "bad",
        "submitted_to_leaderboard": True,
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": {"principle": "wrong"}},
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must not be None" in errors
    assert "target_game must be re86" in errors
    assert "sprite_overlay_verifier_built must be bare bool" in errors
    assert "registered_verifier_operator must be non-empty string" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "preconditions_checked must be dict" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "solution_labels must be list" in errors
    assert "reproduction_result must be dict" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4479" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, bad)
