"""Tests for Exp 4583 exploration-diversity floor transfer.

Spec refs: REQ-CAPSTONE-4583, SCENARIO-CAPSTONE-4583,
SCENARIO-CAPSTONE-4583-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4583_diversity_floor_transfer as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _preconditions(games: tuple[str, ...]) -> dict[str, Any]:
    return {
        "ok": True,
        "offline_arcade": True,
        "arc_variant_generator_importable": True,
        "offline_env_public_games": list(games),
        "leaderboard_submission": False,
    }


def _runner_factory(
    solved_by_mode: Mapping[str, set[str]],
    actions_by_mode: Mapping[str, Mapping[str, int]] | None = None,
    reproduced_by_mode: Mapping[str, set[str]] | None = None,
):
    actions_by_mode = actions_by_mode or {}
    reproduced_by_mode = reproduced_by_mode or solved_by_mode

    def _runner(mode: str):
        def run(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
            signature = str(spec["variant_signature"])
            solved = signature in solved_by_mode.get(mode, set())
            reproduced = signature in reproduced_by_mode.get(mode, set())
            reached = 1 if solved else 0
            gate_level = reached if reproduced else 0
            actions = int(actions_by_mode.get(mode, {}).get(signature, 8 if solved else 19))
            return {
                "game": game,
                "variant_signature": signature,
                "variant": int(spec["variant"]),
                "kind": spec["kind"],
                "reflect": spec.get("reflect"),
                "attempted": True,
                "solved": solved,
                "winner_generated": solved,
                "reached_level": reached,
                "actions": actions,
                "actions_to_first_levelup": actions if solved else None,
                "solution_labels": ["ACTION1"] if solved else [],
                "reproduction_gate": {
                    "game": game,
                    "claimed_level": reached,
                    "reached_level": gate_level,
                    "reproduced": reproduced,
                },
                "blocked_reason": "",
                "diversity_mode": mode,
            }

        return run

    return _runner


def test_req_capstone_4583_spec_declares_diversity_floor_contract() -> None:
    """REQ-CAPSTONE-4583: OpenSpec declares the diversity-floor artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4583" in spec
    assert "SCENARIO-CAPSTONE-4583" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_capstone_4583_temporary_diversity_forces_and_restores_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4583: diversity arms force CARNOT_ARC_EXPLORE_DIVERSITY deterministically."""

    monkeypatch.setenv("CARNOT_ARC_EXPLORE_DIVERSITY", "operator")
    with mod._temporary_diversity(True):
        assert os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] == "1"
    assert os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] == "operator"

    monkeypatch.delenv("CARNOT_ARC_EXPLORE_DIVERSITY", raising=False)
    with mod._temporary_diversity(False):
        assert os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] == "0"
    assert "CARNOT_ARC_EXPLORE_DIVERSITY" not in os.environ


def test_req_capstone_4583_make_variant_runner_sets_on_off_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4583: the live runner toggles only the shipped diversity flag."""

    seen: list[str | None] = []

    def fake_default_runner(
        game: str, spec: Mapping[str, Any], budget: int
    ) -> dict[str, Any]:
        seen.append(os.environ.get("CARNOT_ARC_EXPLORE_DIVERSITY"))
        return {
            "game": game,
            "variant_signature": spec["variant_signature"],
            "attempted": True,
            "solved": False,
            "actions": budget,
        }

    monkeypatch.setattr(mod.exp4550, "default_variant_runner", fake_default_runner)
    spec = {"variant": 1, "kind": "color", "variant_signature": "g1~color01"}

    assert mod.make_variant_runner("diversity_on")("g1", spec, 7)["diversity_enabled"] is True
    assert mod.make_variant_runner("diversity_off")("g1", spec, 7)["diversity_enabled"] is False
    assert seen == ["1", "0"]


def test_scenario_capstone_4583_success_reports_firstwin_lift_and_actions(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4583: diversity-on beats matched diversity-off variants."""

    games = ("g1", "g2", "g3", "g4")
    off = {"g1~color01"}
    on = {"g1~color01", "g2~color01", "g3~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {"diversity_off": off, "diversity_on": on},
            actions_by_mode={
                "diversity_off": {"g1~color01": 10},
                "diversity_on": {
                    "g1~color01": 9,
                    "g2~color01": 11,
                    "g3~color01": 13,
                },
            },
        ),
        update_registry=False,
    )

    assert artifact["honest_verdict"] == "success: diversity_floor_transfer_firstwin_up_2"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["firstwin_count_diversity_off"] == 1
    assert artifact["firstwin_count_diversity_on"] == 3
    assert artifact["firstwin_delta"] == 2
    assert artifact["solve_rate_without_diversity"] == pytest.approx(0.25)
    assert artifact["solve_rate_with_diversity"] == pytest.approx(0.75)
    assert artifact["median_actions_to_first_levelup_without_diversity"] == 10.0
    assert artifact["median_actions_to_first_levelup_with_diversity"] == 11.0
    assert artifact["actions_delta"] == -1.0
    assert artifact["diversity_off_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["newly_reached_wins"] == ["g2~color01", "g3~color01"]
    assert artifact["chosen_submitted_config"] == "keep_diversity_floor_on"
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4583_honest_null_annotates_zero_delta(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4583-FIELD-PRINCIPLES: zero firstwin delta is annotated."""

    games = ("g1", "g2")
    solved = {"g1~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {"diversity_off": solved, "diversity_on": solved},
            actions_by_mode={
                "diversity_off": {"g1~color01": 8},
                "diversity_on": {"g1~color01": 8},
            },
        ),
        update_registry=False,
    )

    assert artifact["honest_verdict"] == (
        "complete: diversity_floor_no_transfer_honest_null_gap_sharpened"
    )
    assert artifact["firstwin_delta"] == 0
    assert artifact["actions_delta"] == 0.0
    assert "honest no-transfer null" in artifact["null_delta_methodology_note"]
    assert artifact["diversity_off_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["chosen_submitted_config"] == "leave_diversity_floor_default_off"
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4583_control_failure_keeps_false_negative_risk_open(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4583: diversity-on regression is not a valid null."""

    games = ("g1", "g2")
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {
                "diversity_off": {"g1~color01", "g2~color01"},
                "diversity_on": {"g1~color01"},
            }
        ),
        update_registry=False,
    )

    assert artifact["honest_verdict"] == (
        "complete: diversity_floor_regression_control_failed_false_negative_risk_open"
    )
    assert artifact["firstwin_delta"] == -1
    assert artifact["diversity_off_control_passed"] is False
    assert artifact["false_negative_risk_checked"] is False
    assert artifact["offline_reproduced"] is True
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4583_unreproduced_new_win_blocks_success(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4583: newly reached wins need offline reproduce evidence to count."""

    games = ("g1", "g2")
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {
                "diversity_off": {"g1~color01"},
                "diversity_on": {"g1~color01", "g2~color01"},
            },
            reproduced_by_mode={
                "diversity_off": {"g1~color01"},
                "diversity_on": {"g1~color01"},
            },
        ),
        update_registry=False,
    )

    assert artifact["firstwin_delta"] == 1
    assert artifact["offline_reproduced"] is False
    assert artifact["honest_verdict"] == "complete: diversity_floor_new_win_unreproduced_no_bank"
    assert "g2~color01" in artifact["unreproduced_new_wins"]
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4583_precondition_misses_and_blocked_artifact(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4583: missing resources produce terminal blocked artifacts."""

    assert mod._first_precondition_miss({"offline_arcade": False}) == "offline_arcade"
    assert (
        mod._first_precondition_miss(
            {"offline_arcade": True, "arc_variant_generator_importable": False}
        )
        == "arc_variant_generator_import"
    )
    assert (
        mod._first_precondition_miss(
            {
                "offline_arcade": True,
                "arc_variant_generator_importable": True,
                "leaderboard_submission": True,
            }
        )
        == "leaderboard_submission"
    )

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=("g1",),
        variant_ids=(1,),
        preconditions_checked={"offline_arcade": False, "arc_variant_generator_importable": True},
    )

    assert artifact["honest_verdict"] == "complete: blocked_offline_arcade"
    assert artifact["false_negative_risk_checked"] is False
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4583_registry_bank_rewrites_only_when_level_is_new() -> None:
    """REQ-CAPSTONE-4583: reproduced newly reached wins can bump the registry total."""

    registry_text = "\n".join(
        [
            "schema_version: 1",
            "games:",
            "- game: g1",
            "  reproducibility: reproduced",
            "  levels_reproduced: 1",
            "reproducible_total_levels: 1",
            "",
        ]
    )
    wins = [
        {
            "game": "g1",
            "variant_signature": "g1~color01",
            "reached_level": 2,
            "reproduction_gate": {"reproduced": True, "reached_level": 2, "claimed_level": 2},
            "solution_labels": ["ACTION1"],
        }
    ]

    updated, update = mod.apply_registry_banks(registry_text, wins)

    assert update["updated"] is True
    assert update["banked_levels"] == 1
    assert update["new_total_declared"] == 2
    assert "levels_reproduced: 2" in updated
    assert "reproducible_total_levels: 2" in updated

    unchanged, no_update = mod.apply_registry_banks(updated, wins)
    assert no_update["updated"] is False
    assert no_update["banked_levels"] == 0
    assert unchanged == updated


def test_req_capstone_4583_registry_defensive_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4583: registry banking handles missing rows and disabled evidence."""

    assert mod._as_int("not-int", 7) == 7
    assert mod._gate_reproduced(None) is False
    assert mod._registry_game_levels({"games": [{"game": "other"}]}, "g1") == 0
    assert mod._game_block_bounds("schema_version: 1\n", "g1") is None

    unreproduced = [
        {
            "game": "g1",
            "variant_signature": "g1~color01",
            "reached_level": 1,
            "reproduction_gate": {"reproduced": False, "reached_level": 0, "claimed_level": 1},
        }
    ]
    same_text, no_bank = mod.apply_registry_banks("schema_version: 1\n", unreproduced)
    assert no_bank["updated"] is False
    assert same_text == "schema_version: 1\n"

    missing_game_text = "schema_version: 1\nreproducible_total_levels: 0\n"
    inserted, insert_update = mod.apply_registry_banks(
        missing_game_text,
        [
            {
                "game": "g2",
                "variant_signature": "g2~color01",
                "reached_level": 2,
                "reproduced": True,
            }
        ],
    )
    assert insert_update["banked_levels"] == 2
    assert "- game: g2" in inserted
    assert "reproducible_total_levels: 2" in inserted

    no_total, no_total_update = mod.apply_registry_banks(
        "schema_version: 1\n",
        [{"game": "g3", "variant_signature": "g3~color01", "reached_level": 1, "reproduced": True}],
    )
    assert no_total_update["new_total_declared"] == 1
    assert no_total.endswith("reproducible_total_levels: 1\n")

    no_level_block = "\n".join(
        [
            "games:",
            "- game: g4",
            "  reproducibility: reproduced",
            "reproducible_total_levels: 0",
            "",
        ]
    )
    inserted_level, _update = mod.apply_registry_banks(
        no_level_block,
        [{"game": "g4", "variant_signature": "g4~color01", "reached_level": 1, "reproduced": True}],
    )
    assert "  levels_reproduced: 1" in inserted_level
    assert mod._replace_game_level("- game: g5", "g5", 3) == "- game: g5\n  levels_reproduced: 3"

    record = {
        "game": "g6",
        "variant_signature": "g6~color01",
        "reached_level": 1,
        "reproduced": True,
    }
    assert mod._registry_update(tmp_path, [record], update_registry=True)["reason"] == "registry_missing"
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 0\n", encoding="utf-8")
    update = mod._registry_update(tmp_path, [record], update_registry=True)
    assert update["updated"] is True
    assert "reproducible_total_levels: 1" in registry_path.read_text(encoding="utf-8")


def test_scenario_capstone_4583_zero_delta_without_control_notes_open_risk(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4583: zero delta without an off arm is not a closed null."""

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=(),
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(()),
        variant_runner_factory=_runner_factory(
            {"diversity_off": set(), "diversity_on": set()}
        ),
        update_registry=False,
    )

    assert artifact["firstwin_delta"] == 0
    assert artifact["diversity_off_control_passed"] is False
    assert "false-negative risk remains open" in artifact["null_delta_methodology_note"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4583_validate_artifact_reports_schema_errors(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4583-FIELD-PRINCIPLES: validation rejects bad field shapes."""

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=("g1",),
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(("g1",)),
        variant_runner_factory=_runner_factory(
            {"diversity_off": set(), "diversity_on": set()}
        ),
        update_registry=False,
    )
    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "pending",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "firstwin_count_diversity_on": "0",
            "firstwin_count_diversity_off": "0",
            "firstwin_delta": "0",
            "median_actions_to_first_levelup_with_diversity": "0",
            "actions_delta": "0",
            "solve_rate_with_diversity": "0",
            "diversity_off_control_passed": "false",
            "false_negative_risk_checked": "false",
            "offline_reproduced": "false",
            "null_delta_methodology_note": "",
            "field_principles": {},
        }
    )

    errors = mod.validate_artifact(bad)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "firstwin_count_diversity_on must be a bare int" in errors
    assert "median_actions_to_first_levelup_with_diversity must be float or null" in errors
    assert "diversity_off_control_passed must be a bare bool" in errors
    assert any(error.startswith("missing field principle") for error in errors)

    note_errors = mod.validate_artifact(dict(artifact, null_delta_methodology_note=""))
    assert "null_delta_methodology_note required for zero firstwin_delta" in note_errors
    principle_errors = mod.validate_artifact(dict(artifact, field_principles=[]))
    assert "field_principles missing" in principle_errors


def test_scenario_capstone_4583_write_and_run_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-4583: write_artifact and run produce the JSON artifact."""

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=("g1",),
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(("g1",)),
        variant_runner_factory=_runner_factory(
            {"diversity_off": set(), "diversity_on": set()}
        ),
        update_registry=False,
    )
    path = mod.write_artifact(tmp_path, artifact=artifact)
    assert path.exists()
    assert mod._read_json(path)["reproducibility_checksum"] == artifact[
        "reproducibility_checksum"
    ]

    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, artifact=dict(artifact, honest_verdict="pending"))

    calls: list[tuple[str, Any]] = []

    def fake_build(root: Path | str, **kwargs: Any) -> dict[str, Any]:
        calls.append(("build", root, kwargs))
        return artifact

    def fake_write(root: Path | str, *, artifact: Mapping[str, Any] | None = None) -> Path:
        calls.append(("write", artifact))
        return tmp_path / "written.json"

    monkeypatch.setattr(mod, "build_artifact", fake_build)
    monkeypatch.setattr(mod, "write_artifact", fake_write)
    assert mod.run(tmp_path, write=True, update_registry=False) == artifact
    assert calls[0][0] == "build"
    assert calls[1][0] == "write"
    calls.clear()
    assert mod.run(tmp_path, write=False, update_registry=False) == artifact
    assert calls == [("build", tmp_path, {"update_registry": False})]
