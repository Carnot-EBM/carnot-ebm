"""Tests for Exp 4643 operator-resubmit package refresh.

Spec refs: REQ-CAPSTONE-4643, SCENARIO-CAPSTONE-4643,
SCENARIO-CAPSTONE-4643-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml
import pytest

from carnot import experiment_4643_refresh_submission_package as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _solution(count: int, action: int = 1) -> list[JsonDict]:
    return [{"action": action} for _ in range(count)]


def _fixture_root(tmp_path: Path, *, a3_banked: bool = True) -> Path:
    root = tmp_path
    (root / "ops").mkdir()
    (root / "results" / "arc3_live_banked_trajectories").mkdir(parents=True)
    registry = {
        "schema_version": 1,
        "reproducible_total_levels": 57 if a3_banked else 56,
        "games": [
            {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 48},
            {"game": "lp85", "reproducibility": "reproduced", "levels_reproduced": 5},
            {
                "game": "ft09",
                "reproducibility": "reproduced",
                "levels_reproduced": 3 if a3_banked else 2,
            },
            {"game": "drift", "reproducibility": "reproduced", "levels_reproduced": 1},
        ],
        "general_gotchas": [
            {
                "id": "primitive_env_adaptive_resolve_operator",
                "latest_exp4584_transfer": {"transfer_games": ["drift"]},
            }
        ],
    }
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(yaml.safe_dump(registry), encoding="utf-8")
    _write_json(
        root / mod.BASELINE_PACKAGE_RELATIVE_PATH,
        {
            "experiment": "experiment_4631_submission_package_operator_resubmit",
            "package_manifest": [
                {
                    "game": "alpha",
                    "levels": 48,
                    "offline_reproduced_level": 48,
                    "trajectory_path": "results/arc3_live_banked_trajectories/alpha.json",
                    "action_count": 2,
                    "env_matched": True,
                    "source": "baseline_alpha",
                },
                {
                    "game": "lp85",
                    "levels": 5,
                    "offline_reproduced_level": 5,
                    "trajectory_path": "results/arc3_live_banked_trajectories/lp85.json",
                    "action_count": 5,
                    "env_matched": True,
                    "source": "baseline_lp85",
                },
                {
                    "game": "ft09",
                    "levels": 2,
                    "offline_reproduced_level": 2,
                    "trajectory_path": "results/arc3_live_banked_trajectories/ft09.json",
                    "action_count": 11,
                    "env_matched": True,
                    "source": "results/arc_loop_solve_ft09.json",
                    "adaptive_solver": "env_adaptive_resolve_operator:ft09",
                },
                {
                    "game": "drift",
                    "levels": 1,
                    "offline_reproduced_level": 1,
                    "trajectory_path": "",
                    "action_count": 0,
                    "env_matched": False,
                    "source": "drifted_coordinates",
                },
            ],
        },
    )
    _write_json(root / mod.BASELINE_RESULT_RELATIVE_PATH, {"live_submittable_level_count": 56})
    for game, count in {"alpha": 2, "lp85": 5, "ft09": 11}.items():
        _write_json(
            root / "results" / "arc3_live_banked_trajectories" / f"{game}.json",
            {
                "schema": "carnot.arc3.flat_trajectory_bank.v1",
                "game": game,
                "action_count": count,
                "source": f"baseline_{game}",
                "solution": _solution(count),
            },
        )
    if a3_banked:
        _write_json(
            root / "results" / "arc_loop_solve_ft09.json",
            {
                "game": "ft09",
                "offline_reproduced": True,
                "reached_level": 3,
                "reproduced_levels": 3,
                "reproduction_gate": {
                    "game": "ft09",
                    "claimed_level": 3,
                    "reached_level": 3,
                    "reproduced": True,
                },
                "solution_labels": ['{"action":6,"data":{"x":1,"y":2}}'] * 25,
            },
        )
        a3_payload: JsonDict = {
            "honest_verdict": "success: ft09_L3_offline_reproduced",
            "target_game": "ft09",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reached_level": 3,
            "standing_loop_result_path": "results/arc_loop_solve_ft09.json",
            "reproduction_gate": {
                "game": "ft09",
                "claimed_level": 3,
                "reached_level": 3,
                "reproduced": True,
            },
            "solution_labels": ['{"action":6,"data":{"x":1,"y":2}}'] * 25,
        }
    else:
        a3_payload = {
            "honest_verdict": "complete: ft09_delta_identified_no_bank",
            "target_game": "ft09",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "reached_level": 2,
            "standing_loop_result_path": "results/arc_loop_solve_ft09.json",
            "reproduction_gate": {
                "game": "ft09",
                "claimed_level": 3,
                "reached_level": 2,
                "reproduced": False,
            },
            "solution_labels": ['{"action":6,"data":{"x":1,"y":2}}'] * 11,
        }
    _write_json(root / mod.A3_LEVELUP_RELATIVE_PATH, a3_payload)
    reproduced_variant = {
        "game": "lp85",
        "variant_signature": "lp85~color01",
        "first_win": True,
        "solved": True,
        "reached_level": 1,
        "reproduction_gate": {
            "game": "lp85",
            "claimed_level": 1,
            "reached_level": 1,
            "reproduced": True,
        },
        "solution_labels": ['{"action":6,"data":{"x":1,"y":2}}'],
    }
    _write_json(
        root / mod.A1_VARIANT_RELATIVE_PATH,
        {
            "honest_verdict": "complete: goal_energy_generation_no_live_lift_honest_null",
            "baseline_measurement": {"variant_attempts": [reproduced_variant]},
            "goal_energy_measurement": {"variant_attempts": [reproduced_variant]},
            "uniform_measurement": {"variant_attempts": [reproduced_variant]},
        },
    )
    _write_json(
        root / mod.A2_VARIANT_RELATIVE_PATH,
        {
            "honest_verdict": "complete: action_effect_expansion_prior_no_deeper_solve",
            "expansion_measurement": {"attempts": [reproduced_variant]},
        },
    )
    return root


def test_req_capstone_4643_spec_declares_refresh_contract() -> None:
    """REQ-CAPSTONE-4643: OpenSpec declares the refreshed package contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4643",
        "SCENARIO-CAPSTONE-4643",
        "SCENARIO-CAPSTONE-4643-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4643_folds_ft09_l3_without_overclaim(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4643: A3 folds only to registry-backed replayable depth."""

    root = _fixture_root(tmp_path, a3_banked=True)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    by_game = {row["game"]: row for row in artifact["per_game_submittable"]}
    package = json.loads((root / artifact["refreshed_package_path"]).read_text(encoding="utf-8"))
    manifest_by_game = {row["game"]: row for row in package["package_manifest"]}

    assert artifact["live_submittable_count_prev"] == 56
    assert artifact["live_submittable_level_count"] == 57
    assert artifact["count_delta"] == 1
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["honest_verdict"] == "success: package_refreshed_live_submittable_57_above_33"
    assert artifact["levels_folded_in"] == ["ft09"]
    assert artifact["submitted_to_leaderboard"] is False

    assert by_game["ft09"]["submittable_level"] == 3
    assert by_game["ft09"]["trajectory_action_count"] == 25
    assert by_game["ft09"]["source"] == "results/arc_loop_solve_ft09.json"
    assert by_game["ft09"]["has_env_adaptive_resolver"] is True
    assert by_game["ft09"]["adaptive_solver"] == "env_adaptive_resolve_operator:ft09"
    assert by_game["drift"]["submittable_level"] == 1
    assert by_game["drift"]["has_env_adaptive_resolver"] is True

    assert package["source_result_path"] == mod.RESULT_RELATIVE_PATH
    assert package["claimed_total_levels"] == 57
    assert package["operator_only"] is True
    assert package["submitted_to_leaderboard"] is False
    assert (
        manifest_by_game["ft09"]["env_match_basis"]
        == "offline_reproduction_gated_package_refresh_4643"
    )
    assert manifest_by_game["ft09"]["levels"] == 3

    a1_audit = [
        row
        for row in artifact["upstream_fold_audit"]
        if row["game"] == "lp85" and row["source_artifact"] == mod.A1_VARIANT_RELATIVE_PATH
    ]
    a2_audit = [
        row
        for row in artifact["upstream_fold_audit"]
        if row["game"] == "lp85" and row["source_artifact"] == mod.A2_VARIANT_RELATIVE_PATH
    ]
    assert len(a1_audit) == 1
    assert len(a2_audit) == 1
    assert a1_audit[0]["reason"] == "already_submittable"
    assert a2_audit[0]["reason"] == "already_submittable"
    assert a1_audit[0]["folded"] is False
    assert a2_audit[0]["folded"] is False


def test_scenario_capstone_4643_unchanged_depth_is_annotated(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4643: no new bank yields an honest null delta."""

    root = _fixture_root(tmp_path, a3_banked=False)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)

    assert artifact["live_submittable_level_count"] == 56
    assert artifact["count_delta"] == 0
    assert artifact["levels_folded_in"] == []
    assert artifact["honest_verdict"] == "complete: package_refreshed_unchanged_depth."
    assert ".427 A4" in artifact["null_delta_methodology_note"]
    assert artifact["ready_for_operator_submit"] is True


def test_req_capstone_4643_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4643: missing resources stop before claims are fabricated."""

    root = _fixture_root(tmp_path)
    artifact = mod.run(
        root,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )

    assert artifact["honest_verdict"] == "blocked_offline_arcade"
    assert artifact["live_submittable_level_count"] == 0
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert (root / mod.RESULT_RELATIVE_PATH).exists()


def test_req_capstone_4643_schema_and_checksum_helpers(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4643: schema rejects false readiness and checksum drift."""

    root = _fixture_root(tmp_path)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    assert mod.artifact_schema_errors(artifact) == []
    assert mod.compute_reproducibility_checksum(artifact) == artifact["reproducibility_checksum"]

    bad = dict(artifact)
    bad["count_delta"] = 0
    bad["reproducibility_checksum"] = "bad"
    bad["per_game_submittable"] = [
        "not_a_row",
        {
            "game": "bad",
            "submittable_level": 2,
            "offline_reproduced_level": 1,
            "registry_reproduced_level": 1,
            "has_replayable_trajectory": False,
            "has_env_adaptive_resolver": False,
        }
    ]

    errors = mod.artifact_schema_errors(bad)

    assert "reproducibility_checksum must match the artifact payload" in errors
    assert (
        "count_delta must equal live_submittable_level_count - live_submittable_count_prev"
        in errors
    )
    assert "per_game_submittable rows must be mappings" in errors
    assert "bad submittable exceeds offline reproduction" in errors
    assert "bad counted without trajectory or adaptive resolver" in errors

    low_ready = dict(artifact)
    low_ready["live_submittable_level_count"] = 33
    low_ready["live_submittable_count_prev"] = 33
    low_ready["count_delta"] = 0
    low_ready["ready_for_operator_submit"] = True
    low_ready["null_delta_methodology_note"] = "declared null"
    low_ready["reproducibility_checksum"] = mod.compute_reproducibility_checksum(low_ready)
    assert (
        "ready_for_operator_submit requires count strictly above 33"
        in mod.artifact_schema_errors(low_ready)
    )

    empty_errors = mod.artifact_schema_errors({})
    assert "missing honest_verdict" in empty_errors
    assert "honest_verdict must be terminal-prefixed" in empty_errors


def test_req_capstone_4643_defensive_helpers_are_scoped(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4643: defensive filters skip malformed rows without claims."""

    variants = list(
        mod._iter_reproduced_variants(
            {
                "measurement": {
                    "attempts": [
                        "bad",
                        {"game": "lp85", "reproduction_gate": {"reproduced": False}},
                        {
                            "game": "lp85",
                            "variant_signature": "lp85~ok",
                            "reached_level": 1,
                            "reproduction_gate": {"reproduced": True, "reached_level": 1},
                        },
                        {
                            "game": "lp85",
                            "variant_signature": "lp85~ok",
                            "reached_level": 1,
                            "reproduction_gate": {"reproduced": True, "reached_level": 1},
                        },
                    ]
                }
            },
            measurement_keys=("measurement",),
        )
    )
    package = mod.build_package_payload(
        [
            {"game": "skip", "submittable_level": 0},
            {
                "game": "keep",
                "submittable_level": 1,
                "offline_reproduced_level": 1,
                "registry_reproduced_level": 1,
                "trajectory_path": "results/keep.json",
                "trajectory_action_count": 1,
            },
        ],
        result_path=mod.RESULT_RELATIVE_PATH,
    )

    assert len(variants) == 1
    assert variants[0][0] == "lp85"
    assert [row["game"] for row in package["package_manifest"]] == ["keep"]

    with pytest.raises(ValueError, match="missing honest_verdict"):
        mod.write_artifact(tmp_path, {})
