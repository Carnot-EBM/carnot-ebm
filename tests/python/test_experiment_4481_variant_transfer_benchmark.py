"""Tests for Exp 4481 reflection-variant transfer benchmark.

Spec refs: REQ-REPORT-4481, SCENARIO-REPORT-4481.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4481_variant_transfer_benchmark as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_fixture_repo(root: Path) -> None:
    for game in ("ar25", "bp35", "cd82"):
        (root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reproducible_total_levels": 3,
                "games": [
                    {"game": "ar25", "reproducibility": "reproduced", "levels_reproduced": 1},
                    {"game": "bp35", "reproducibility": "unsolved", "levels_reproduced": 2},
                    {"game": "cd82", "reproducibility": "unsolved", "levels_reproduced": 0},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _ok_preconditions() -> dict[str, Any]:
    return {
        "registry_parseable": True,
        "arc_variant_generator_import": True,
        "arc_solver_kit_import": True,
        "offline_env_files_present": True,
        "offline_env_games": ["ar25", "bp35", "cd82"],
        "solved_games": ["ar25", "bp35"],
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _fake_reflection_runner(
    game: str, spec: Mapping[str, Any], _budget: int
) -> dict[str, Any]:
    solved = game == "bp35" or spec["variant_signature"] == "ar25~reflect01"
    reached = 1 if solved else 0
    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": spec["variant"],
        "kind": spec["kind"],
        "reflect": spec["reflect"],
        "attempted": True,
        "solved": solved,
        "reached_level": reached,
        "actions": 4 if solved else 0,
        "reproduction_gate": {
            "game": game,
            "reached_level": reached,
            "claimed_level": reached,
            "reproduced": solved,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "blocked_reason": "",
    }


def test_req_report_4481_spec_declares_reflection_transfer_contract() -> None:
    """REQ-REPORT-4481: OpenSpec declares the reflection benchmark artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4481" in spec
    assert "SCENARIO-REPORT-4481" in spec
    assert "reflection variants" in spec
    assert "Color-permutation variants SHALL NOT be part" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_4481_tallies_reflection_transfer_per_solved_game(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4481: solved games get reflection variants and per-game rates."""

    _write_fixture_repo(tmp_path)
    clock = {"t": 5.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        variant_runner=_fake_reflection_runner,
        reflection_variants=(1, 2),
        budget=9,
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == (
        "success: reflection_variant_transfer_3_of_4_rate_0.7500_games_2"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["solved_games"] == ["ar25", "bp35"]
    assert artifact["variant_plan"]["reflection_variants"] == [1, 2]
    assert artifact["variant_plan"]["color_variants"] == []
    assert artifact["variants_attempted"] == 4
    assert artifact["variants_solved"] == 3
    assert artifact["transfer_solve_rate"] == pytest.approx(0.75)
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["reproducible_total_levels"] == 3
    assert artifact["preconditions_checked"]["solved_games"] == ["ar25", "bp35"]
    assert artifact["verifier_is_oracle"] is True
    assert artifact["no_3090_inference"] is True
    assert artifact["leaderboard_submission"] is False

    by_game = {row["game"]: row for row in artifact["per_game"]}
    assert by_game["ar25"] == {
        "game": "ar25",
        "source_levels_reproduced": 1,
        "variants_attempted": 2,
        "variants_solved": 1,
        "transfer_solve_rate": 0.5,
    }
    assert by_game["bp35"]["transfer_solve_rate"] == 1.0
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["variants_solved"] == 3
    assert len(written["reproducibility_checksum"]) == 64


def test_req_report_4481_preconditions_block_before_runner(tmp_path: Path) -> None:
    """REQ-REPORT-4481: missing resources write terminal blocked artifacts without runner calls."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "arc_variant_generator_import": False, "ok": False},
        variant_runner=lambda game, _spec, _budget: calls.append(game) or {},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_arc_variant_generator"
    assert artifact["variants_attempted"] == 0
    assert artifact["variants_solved"] == 0
    assert artifact["transfer_solve_rate"] == 0.0
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4481_schema_rejects_fabricated_or_malformed_results(tmp_path: Path) -> None:
    """REQ-REPORT-4481: schema catches bad prefixes, wrapped metrics, and false gates."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        variant_runner=_fake_reflection_runner,
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    bad = {
        **artifact,
        "honest_verdict": "partial: fake",
        "inference_substrate": None,
        "offline_reproduced": "true",
        "reproduced_levels": "3",
        "preconditions_checked": [],
        "solved_games": "ar25",
        "per_game": [
            "bad",
            {
                "game": "ar25",
                "source_levels_reproduced": "1",
                "variants_attempted": "1",
                "variants_solved": "1",
                "transfer_solve_rate": "1.0",
            },
        ],
        "variant_plan": [],
        "variant_attempts": [
            {
                "game": "ar25",
                "attempted": True,
                "solved": True,
                "reproduction_gate": {"reproduced": False, "reached_level": 0},
            }
        ],
        "variants_attempted": "2",
        "variants_solved": "1",
        "transfer_solve_rate": {"principle": "wrapped"},
        "reproducible_total_levels": "3",
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": {"principle": "wrong"}},
        "verifier_is_oracle": False,
        "random_seed": "4481",
        "reproducibility_checksum": "bad",
        "no_3090_inference": False,
        "leaderboard_submission": True,
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must not be None" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "preconditions_checked must be dict" in errors
    assert "solved_games must be list" in errors
    assert "per_game[0] must be dict" in errors
    assert "per_game[1].source_levels_reproduced must be bare int" in errors
    assert "per_game[1].variants_attempted must be bare int" in errors
    assert "per_game[1].variants_solved must be bare int" in errors
    assert "per_game[1].transfer_solve_rate must be bare float" in errors
    assert "variant_plan must be dict" in errors
    assert "solved variant_attempts must have reproduced gate evidence" in errors
    assert "variants_attempted must be bare int" in errors
    assert "variants_solved must be bare int" in errors
    assert "transfer_solve_rate must be bare float" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4481" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "no_3090_inference must be true" in errors
    assert "leaderboard_submission must be false" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, bad)


def test_req_report_4481_registry_and_rate_helpers_are_deterministic(tmp_path: Path) -> None:
    """REQ-REPORT-4481: registry parsing, planning, and rates are deterministic."""

    _write_fixture_repo(tmp_path)

    assert mod.first_precondition_miss({**_ok_preconditions(), "registry_parseable": False}) == (
        "registry_parse"
    )
    assert mod.first_precondition_miss(
        {**_ok_preconditions(), "offline_env_files_present": False}
    ) == "offline_env_files"
    assert mod.first_precondition_miss(
        {**_ok_preconditions(), "arc_solver_kit_import": False}
    ) == "arc_solver_kit"
    assert mod.first_precondition_miss(
        {**_ok_preconditions(), "no_3090_inference": False}
    ) == "no_3090_inference_policy"
    assert mod.first_precondition_miss(
        {**_ok_preconditions(), "leaderboard_submission": True}
    ) == "leaderboard_submission_policy"
    assert mod._transfer_rate(0, 0) == 0.0
    assert mod._transfer_rate(1, 4) == 0.25

    registry = mod.load_registry(tmp_path)
    assert mod.solved_game_rows(registry, tmp_path) == [
        {"game": "ar25", "levels_reproduced": 1},
        {"game": "bp35", "levels_reproduced": 2},
    ]
    assert mod.reproducible_total_levels(registry) == 3
    assert [row["variant_signature"] for row in mod.reflection_variant_specs(["bp35", "ar25"], (2, 1))] == [
        "ar25~reflect01",
        "ar25~reflect02",
        "bp35~reflect01",
        "bp35~reflect02",
    ]
    assert mod._attempt_summary_by_game([{"game": "", "attempted": True, "solved": True}]) == {}

    malformed = {
        **mod.run(
            root=tmp_path,
            preconditions_checked=_ok_preconditions(),
            variant_runner=_fake_reflection_runner,
            now=lambda: 1.0,
            sleep_fn=lambda _seconds: None,
        ),
        "variant_attempts": "bad",
        "offline_reproduced": False,
        "variants_solved": 1,
        "reproduced_levels": 1,
    }
    malformed_errors = mod.artifact_schema_errors(malformed)
    assert "variant_attempts must be list" in malformed_errors
    assert "offline_reproduced false cannot accompany counted solves" in malformed_errors
