"""Tests for Exp 4655 operator-resubmit package refresh.

Spec refs: REQ-CAPSTONE-4655, SCENARIO-CAPSTONE-4655,
SCENARIO-CAPSTONE-4655-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4655_refresh_submission_package as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _solution(count: int, action: int = 6) -> list[JsonDict]:
    return [{"action": action, "data": {"x": index, "y": index + 1}} for index in range(count)]


def _labels(count: int) -> list[str]:
    return [json.dumps(action, sort_keys=True) for action in _solution(count)]


def _fixture_root(tmp_path: Path, *, a3_banked: bool = True) -> Path:
    root = tmp_path
    (root / "ops").mkdir()
    (root / "results" / "arc3_live_banked_trajectories").mkdir(parents=True)
    registry = {
        "schema_version": 1,
        "reproducible_total_levels": 58 if a3_banked else 57,
        "games": [
            {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 54},
            {
                "game": "vc33",
                "reproducibility": "reproduced",
                "levels_reproduced": 2 if a3_banked else 1,
            },
            {"game": "drift", "reproducibility": "reproduced", "levels_reproduced": 2},
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
            "experiment": "experiment_4643_submission_package_operator_resubmit",
            "package_manifest": [
                {
                    "game": "alpha",
                    "levels": 54,
                    "offline_reproduced_level": 54,
                    "trajectory_path": "results/arc3_live_banked_trajectories/alpha.json",
                    "action_count": 4,
                    "env_matched": True,
                    "source": "baseline_alpha",
                },
                {
                    "game": "vc33",
                    "levels": 1,
                    "offline_reproduced_level": 1,
                    "trajectory_path": "results/arc3_live_banked_trajectories/vc33.json",
                    "action_count": 3,
                    "env_matched": True,
                    "source": "baseline_vc33",
                },
                {
                    "game": "drift",
                    "levels": 2,
                    "offline_reproduced_level": 2,
                    "trajectory_path": "",
                    "action_count": 0,
                    "env_matched": False,
                    "source": "drifted_coordinates",
                },
            ],
        },
    )
    _write_json(root / mod.BASELINE_RESULT_RELATIVE_PATH, {"live_submittable_level_count": 57})
    for game, count in {"alpha": 4, "vc33": 3}.items():
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
    loop_payload = {
        "game": "vc33",
        "offline_reproduced": a3_banked,
        "reached_level": 2 if a3_banked else 1,
        "reproduced_levels": 2 if a3_banked else 1,
        "reproduction_gate": {
            "game": "vc33",
            "claimed_level": 2,
            "reached_level": 2 if a3_banked else 1,
            "reproduced": a3_banked,
        },
        "solution_labels": _labels(10),
    }
    _write_json(root / "results" / "arc_loop_solve_vc33.json", loop_payload)
    _write_json(
        root / mod.A3_LEVELUP_RELATIVE_PATH,
        {
            "honest_verdict": (
                "success: vc33_L2_offline_reproduced"
                if a3_banked
                else "complete: vc33_delta_identified_no_bank"
            ),
            "target_game": "vc33",
            "offline_reproduced": a3_banked,
            "reproduced_levels": 1 if a3_banked else 0,
            "reached_level": 2 if a3_banked else 1,
            "standing_loop_result_path": "results/arc_loop_solve_vc33.json",
            "reproduction_gate": loop_payload["reproduction_gate"],
            "solution_labels": loop_payload["solution_labels"],
        },
    )
    reproduced_variant = {
        "game": "alpha",
        "variant_signature": "alpha~value",
        "first_win": True,
        "solved": True,
        "reached_level": 1,
        "reproduction_gate": {
            "game": "alpha",
            "claimed_level": 1,
            "reached_level": 1,
            "reproduced": True,
        },
        "solution_labels": _labels(1),
    }
    _write_json(
        root / mod.A1_VARIANT_RELATIVE_PATH,
        {
            "honest_verdict": "complete: value_routing_cost_fixed_no_live_lift",
            "value_routed_measurement": {"variant_attempts": [reproduced_variant]},
            "live_baseline_value_weight_zero": {
                "measurement": {"variant_attempts": [reproduced_variant]}
            },
        },
    )
    _write_json(
        root / mod.A2_VARIANT_RELATIVE_PATH,
        {
            "honest_verdict": "complete: energy_fitness_qd_no_winner_generated",
            "qd_measurement": {
                "attempts": [
                    {
                        "game": "omega",
                        "variant_signature": "omega~qd",
                        "reached_level": 0,
                        "reproduction_gate": {"reproduced": False, "reached_level": 0},
                    }
                ]
            },
        },
    )
    return root


def test_req_capstone_4655_spec_declares_refresh_contract() -> None:
    """REQ-CAPSTONE-4655: OpenSpec declares the refreshed package contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4655",
        "SCENARIO-CAPSTONE-4655",
        "SCENARIO-CAPSTONE-4655-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4655_folds_vc33_l2_without_overclaim(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4655: A3 folds only to registry-backed replayable depth."""

    root = _fixture_root(tmp_path, a3_banked=True)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    by_game = {row["game"]: row for row in artifact["per_game_submittable"]}
    package = json.loads((root / artifact["refreshed_package_path"]).read_text(encoding="utf-8"))
    manifest_by_game = {row["game"]: row for row in package["package_manifest"]}

    assert artifact["live_submittable_count_prev"] == 57
    assert artifact["live_submittable_level_count"] == 58
    assert artifact["count_delta"] == 1
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["honest_verdict"] == "success: package_refreshed_live_submittable_58_above_33"
    assert artifact["levels_folded_in"] == ["vc33"]
    assert artifact["submitted_to_leaderboard"] is False

    assert by_game["vc33"]["submittable_level"] == 2
    assert by_game["vc33"]["trajectory_action_count"] == 10
    assert by_game["vc33"]["source"] == "results/arc_loop_solve_vc33.json"
    assert by_game["drift"]["has_env_adaptive_resolver"] is True
    assert by_game["drift"]["adaptive_solver"] == "env_adaptive_resolve_operator:drift"

    assert package["source_result_path"] == mod.RESULT_RELATIVE_PATH
    assert package["claimed_total_levels"] == 58
    assert package["operator_only"] is True
    assert package["submitted_to_leaderboard"] is False
    assert (
        manifest_by_game["vc33"]["env_match_basis"]
        == "offline_reproduction_gated_package_refresh_4655"
    )
    assert manifest_by_game["vc33"]["levels"] == 2

    a1_audit = [
        row
        for row in artifact["upstream_fold_audit"]
        if row["game"] == "alpha" and row["source_artifact"] == mod.A1_VARIANT_RELATIVE_PATH
    ]
    assert len(a1_audit) == 1
    assert a1_audit[0]["reason"] == "already_submittable"
    assert a1_audit[0]["folded"] is False


def test_scenario_capstone_4655_unchanged_depth_is_annotated(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4655: no new bank yields an honest null delta."""

    root = _fixture_root(tmp_path, a3_banked=False)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)

    assert artifact["live_submittable_level_count"] == 57
    assert artifact["count_delta"] == 0
    assert artifact["levels_folded_in"] == []
    assert artifact["honest_verdict"] == "complete: package_refreshed_unchanged_depth."
    assert ".428 A4" in artifact["null_delta_methodology_note"]
    assert artifact["ready_for_operator_submit"] is True


def test_req_capstone_4655_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4655: missing resources stop before claims are fabricated."""

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


def test_req_capstone_4655_schema_helpers_reject_bad_rows(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4655: schema helpers reject overclaims and checksum drift."""

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
        },
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

    with pytest.raises(ValueError, match="missing honest_verdict"):
        mod.write_artifact(tmp_path, {})
