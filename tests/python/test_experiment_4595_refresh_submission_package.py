"""Tests for Exp 4595 operator-resubmit package refresh.

Spec refs: REQ-CAPSTONE-4595, SCENARIO-CAPSTONE-4595,
SCENARIO-CAPSTONE-4595-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from carnot import experiment_4595_refresh_submission_package as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _solution(action_count: int) -> list[JsonDict]:
    return [{"action": 2} for _ in range(action_count)]


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "ops").mkdir()
    (root / "results" / "arc3_live_banked_trajectories").mkdir(parents=True)
    registry = {
        "schema_version": 1,
        "reproducible_total_levels": 60,
        "games": [
            {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 51},
            {"game": "ar25", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "ft09", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "sc25", "reproducibility": "reproduced", "levels_reproduced": 5},
            {"game": "capped", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "no_path", "reproducibility": "reproduced", "levels_reproduced": 3},
        ],
    }
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(yaml.safe_dump(registry), encoding="utf-8")
    _write_json(
        root / mod.PREVIOUS_PACKAGE_RELATIVE_PATH,
        {
            "experiment": "experiment_4580_submission_package_live_gap_close",
            "package_manifest": [
                {
                    "game": "alpha",
                    "levels": 51,
                    "offline_reproduced_level": 51,
                    "trajectory_path": "results/arc3_live_banked_trajectories/alpha.json",
                    "action_count": 1,
                    "env_matched": True,
                    "source": "fixture_alpha",
                },
                {
                    "game": "ar25",
                    "levels": 1,
                    "offline_reproduced_level": 1,
                    "trajectory_path": "results/arc3_live_banked_trajectories/ar25.json",
                    "action_count": 15,
                    "env_matched": True,
                    "source": "old_ar25",
                },
                {
                    "game": "ft09",
                    "levels": 1,
                    "offline_reproduced_level": 1,
                    "trajectory_path": "results/arc3_live_banked_trajectories/ft09.json",
                    "action_count": 4,
                    "env_matched": True,
                    "source": "old_ft09",
                },
                {
                    "game": "sc25",
                    "levels": 5,
                    "offline_reproduced_level": 5,
                    "trajectory_path": "results/arc3_live_banked_trajectories/sc25.json",
                    "action_count": 102,
                    "adaptive_solver": "sc25_dynamic_cast_grid_origin_step",
                    "env_matched": True,
                    "source": "old_sc25",
                },
                {
                    "game": "capped",
                    "levels": 5,
                    "offline_reproduced_level": 5,
                    "trajectory_path": "results/arc3_live_banked_trajectories/capped.json",
                    "action_count": 3,
                    "env_matched": True,
                    "source": "over_claim_fixture",
                },
                {
                    "game": "no_path",
                    "levels": 3,
                    "offline_reproduced_level": 3,
                    "trajectory_path": "",
                    "action_count": 0,
                    "env_matched": True,
                    "source": "missing_path_fixture",
                },
            ],
        },
    )
    _write_json(root / mod.PREVIOUS_RESULT_RELATIVE_PATH, {"live_submittable_level_count": 53})
    for game, count in {"alpha": 1, "ar25": 26, "ft09": 4, "sc25": 102, "capped": 3}.items():
        _write_json(
            root / "results" / "arc3_live_banked_trajectories" / f"{game}.json",
            {
                "schema": "carnot.arc3.flat_trajectory_bank.v1",
                "game": game,
                "action_count": count,
                "source": f"old_{game}",
                "solution": _solution(count),
            },
        )
    _write_json(
        root / "results" / "arc_loop_solve_ft09.json",
        {
            "game": "ft09",
            "offline_reproduced": True,
            "reached_level": 2,
            "reproduction_gate": {"game": "ft09", "claimed_level": 2, "reached_level": 2, "reproduced": True},
            "solution": [{"action": 6, "data": {"x": x, "y": 16}} for x in range(11)],
        },
    )
    _write_json(
        root / "results" / "arc_loop_solve_ar25.json",
        {
            "game": "ar25",
            "offline_reproduced": True,
            "reached_level": 2,
            "reproduction_gate": {"game": "ar25", "claimed_level": 2, "reached_level": 2, "reproduced": True},
            "solution": _solution(26),
        },
    )
    _write_json(
        root / "results" / "experiment_4593_levelup_selfplay.json",
        {
            "honest_verdict": "success: ft09_L2_offline_reproduced",
            "target_game": "ft09",
            "offline_reproduced": True,
            "reproduction_gate": {"game": "ft09", "claimed_level": 2, "reached_level": 2, "reproduced": True},
        },
    )
    _write_json(
        root / "results" / "experiment_4592_generation_completeness_wiring.json",
        {"honest_verdict": "success: generation_completeness_winner_generated_2of25_above_1of25"},
    )
    return root


def test_req_capstone_4595_spec_declares_refresh_contract() -> None:
    """REQ-CAPSTONE-4595: OpenSpec declares the operator-resubmit refresh contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4595",
        "SCENARIO-CAPSTONE-4595",
        "SCENARIO-CAPSTONE-4595-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4595_refresh_folds_l2_banks_without_overclaim(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4595: current L2 banks are folded in only to registry depth."""

    root = _fixture_root(tmp_path)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    by_game = {row["game"]: row for row in artifact["per_game_submittable"]}

    assert artifact["live_submittable_count_prev"] == 53
    assert artifact["live_submittable_level_count"] == 62
    assert artifact["count_delta"] == 9
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["honest_verdict"] == "success: package_refreshed_live_submittable_62_above_33"
    assert artifact["offline_reproduced"] is True
    assert set(artifact["levels_folded_in"]) == {"ar25", "ft09"}

    assert by_game["ar25"]["submittable_level"] == 2
    assert by_game["ar25"]["claimed_level"] == 2
    assert by_game["ar25"]["trajectory_action_count"] == 26
    assert by_game["ft09"]["submittable_level"] == 2
    assert by_game["ft09"]["trajectory_action_count"] == 11
    assert by_game["ft09"]["drift_robust"] is True
    assert by_game["capped"]["submittable_level"] == 2
    assert by_game["capped"]["claim_capped"] is True
    assert by_game["no_path"]["submittable_level"] == 0
    assert by_game["no_path"]["exclusion_reason"] == "missing_trajectory_or_adaptive_resolver"

    refreshed_ft09 = json.loads((root / "results/arc3_live_banked_trajectories/ft09.json").read_text())
    assert refreshed_ft09["action_count"] == 11
    assert refreshed_ft09["source"] == "results/arc_loop_solve_ft09.json"


def test_scenario_capstone_4595_writes_driver_loadable_package(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4595: the package manifest is the operator driver deliverable."""

    root = _fixture_root(tmp_path)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    package = json.loads((root / artifact["refreshed_package_path"]).read_text(encoding="utf-8"))
    manifest_by_game = {row["game"]: row for row in package["package_manifest"]}

    assert package["source_result_path"] == mod.RESULT_RELATIVE_PATH
    assert package["claimed_total_levels"] == artifact["live_submittable_level_count"]
    assert package["operator_only"] is True
    assert package["submitted_to_leaderboard"] is False
    assert manifest_by_game["ft09"]["levels"] == 2
    assert manifest_by_game["ft09"]["action_count"] == 11
    assert manifest_by_game["ft09"]["adaptive_solver"] == "env_adaptive_resolve_operator:ft09"
    assert manifest_by_game["capped"]["levels"] == 2
    assert "no_path" not in manifest_by_game


def test_req_capstone_4595_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4595: missing resources stop before package claims are fabricated."""

    root = _fixture_root(tmp_path)
    artifact = mod.run(
        root,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )

    assert artifact["honest_verdict"] == "blocked_offline_arcade"
    assert artifact["live_submittable_level_count"] == 0
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["ok"] is False
    assert (root / mod.RESULT_RELATIVE_PATH).exists()


def test_req_capstone_4595_schema_catches_mismatches() -> None:
    """REQ-CAPSTONE-4595: schema rejects false readiness and over-claimed rows."""

    bad = {
        field: None for field in mod.REQUIRED_ARTIFACT_FIELDS
    }
    bad.update(
        {
            "honest_verdict": "success: package_refreshed_live_submittable_10_above_33",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "verifier_is_oracle": False,
            "live_submittable_level_count": 10,
            "live_submittable_count_prev": 53,
            "count_delta": 0,
            "levels_folded_in": [],
            "refreshed_package_path": mod.PACKAGE_RELATIVE_PATH,
            "per_game_submittable": [
                {
                    "game": "bad",
                    "claimed_level": 3,
                    "registry_reproduced_level": 2,
                    "offline_reproduced_level": 2,
                    "submittable_level": 3,
                    "has_replayable_trajectory": False,
                    "has_env_adaptive_resolver": False,
                }
            ],
            "ready_for_operator_submit": True,
            "offline_reproduced": True,
            "random_seed": mod.RANDOM_SEED,
            "reproducibility_checksum": "0" * 64,
            "preconditions_checked": {"ok": True},
            "null_delta_methodology_note": "",
        }
    )

    errors = mod.artifact_schema_errors(bad)

    assert "count_delta must equal live_submittable_level_count - live_submittable_count_prev" in errors
    assert "ready_for_operator_submit requires count strictly above 33" in errors
    assert "null_delta_methodology_note required when count_delta is zero" in errors
    assert "bad submittable exceeds offline reproduction" in errors
    assert "bad counted without trajectory or adaptive resolver" in errors


def test_req_capstone_4595_pure_helper_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4595: helper branches handle missing and defensive inputs."""

    root = _fixture_root(tmp_path)
    assert mod._as_int(True, 7) == 7
    assert mod._as_int("5") == 5
    assert mod._as_int("x", 4) == 4
    assert mod._package_rows({"package_manifest": "bad"}) == {}
    assert mod._registry_levels({"games": "bad"}) == {}
    assert mod._registry_levels({"games": ["bad"]}) == {}
    assert mod._load_json(root / "missing.json") == {}
    assert mod._load_yaml(root / "missing.yaml") == {}
    assert mod._actions_from_payload({}) == []
    assert mod._actions_from_payload({"solution": []}) == []
    assert mod._actions_from_payload({"solve_trace": {"actions": [{"action": 1}]}}) == [{"action": 1}]
    assert mod._trajectory_actions(root, "") == []
    assert mod._trajectory_actions(root, "results/arc3_live_banked_trajectories/alpha.json")
    assert mod._loop_artifact_for("ft09") == "results/arc_loop_solve_ft09.json"
    assert mod._loop_artifact_for("unknown") == ""
    assert mod._is_reproduced_loop({"reproduction_gate": {"reproduced": True, "reached_level": 2}}, 2)
    assert not mod._is_reproduced_loop({"reproduction_gate": {"reproduced": True, "reached_level": 1}}, 2)
    assert mod.first_precondition_miss({"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}}) is None
    assert mod.first_precondition_miss({"offline_arcade": {"ok": True}, "registry_loadable": {"ok": False}}) == "registry"
    assert mod._terminal_prefixed("complete: ok")
    assert not mod._terminal_prefixed("working")
    assert mod._honest_verdict(53, 0) == "complete: package_refreshed_unchanged_depth."

    package_payload = mod.build_package_payload(
        [
            {
                "game": "zero",
                "submittable_level": 0,
                "offline_reproduced_level": 0,
                "registry_reproduced_level": 0,
            }
        ],
        result_path=mod.RESULT_RELATIVE_PATH,
    )
    assert package_payload["package_manifest"] == []

    bad = {
        "inference_substrate": "bad",
        "verifier_is_oracle": True,
        "live_submittable_level_count": "x",
        "live_submittable_count_prev": "x",
        "count_delta": "x",
        "levels_folded_in": "x",
        "refreshed_package_path": "",
        "per_game_submittable": ["bad"],
        "ready_for_operator_submit": "x",
        "offline_reproduced": "x",
        "random_seed": "x",
        "reproducibility_checksum": "bad",
        "preconditions_checked": [],
    }
    errors = mod.artifact_schema_errors(bad)
    assert "missing honest_verdict" in errors
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate must equal the declared offline packaging substrate" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "live_submittable_level_count must be bare int" in errors
    assert "levels_folded_in must be list" in errors
    assert "ready_for_operator_submit must be bare bool" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "preconditions_checked must be mapping" in errors
    assert "reproducibility_checksum must be sha256 hex" in errors
    assert "per_game_submittable rows must be mappings" in errors

    try:
        mod.write_artifact(tmp_path, bad)
    except ValueError as exc:
        assert "missing honest_verdict" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("write_artifact should reject invalid artifacts")
