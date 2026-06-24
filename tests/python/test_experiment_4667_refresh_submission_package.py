"""Tests for Exp 4667 operator-resubmit package refresh.

Spec refs: REQ-CAPSTONE-4667, SCENARIO-CAPSTONE-4667,
SCENARIO-CAPSTONE-4667-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4667_refresh_submission_package as mod


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
        "reproducible_total_levels": 59 if a3_banked else 58,
        "games": [
            {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 53},
            {"game": "vc33", "reproducibility": "reproduced", "levels_reproduced": 2},
            {
                "game": "dc22",
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
            "experiment": "experiment_4655_submission_package_operator_resubmit",
            "package_manifest": [
                {
                    "game": "alpha",
                    "levels": 53,
                    "offline_reproduced_level": 53,
                    "trajectory_path": "results/arc3_live_banked_trajectories/alpha.json",
                    "action_count": 4,
                    "env_matched": True,
                    "source": "baseline_alpha",
                },
                {
                    "game": "vc33",
                    "levels": 2,
                    "offline_reproduced_level": 2,
                    "trajectory_path": "results/arc3_live_banked_trajectories/vc33.json",
                    "action_count": 4,
                    "env_matched": True,
                    "source": "baseline_vc33",
                },
                {
                    "game": "dc22",
                    "levels": 1,
                    "offline_reproduced_level": 1,
                    "trajectory_path": "results/arc3_live_banked_trajectories/dc22.json",
                    "action_count": 3,
                    "env_matched": True,
                    "source": "baseline_dc22",
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
    _write_json(root / mod.BASELINE_RESULT_RELATIVE_PATH, {"live_submittable_level_count": 58})
    for game, count in {"alpha": 4, "vc33": 4, "dc22": 3}.items():
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
        "game": "dc22",
        "offline_reproduced": a3_banked,
        "reached_level": 2 if a3_banked else 1,
        "reproduced_levels": 2 if a3_banked else 1,
        "reproduction_gate": {
            "game": "dc22",
            "claimed_level": 2,
            "reached_level": 2 if a3_banked else 1,
            "reproduced": a3_banked,
        },
        "solution": _solution(10),
        "solution_labels": _labels(10),
    }
    _write_json(root / "results" / "arc_loop_solve_dc22.json", loop_payload)
    _write_json(
        root / mod.A3_LEVELUP_RELATIVE_PATH,
        {
            "honest_verdict": (
                "success: dc22_L2_offline_reproduced"
                if a3_banked
                else "complete: dc22_delta_identified_no_bank"
            ),
            "target_game": "dc22",
            "offline_reproduced": a3_banked,
            "reproduced_levels": 1 if a3_banked else 0,
            "reached_level": 2 if a3_banked else 1,
            "standing_loop_result_path": "results/arc_loop_solve_dc22.json",
            "reproduction_gate": loop_payload["reproduction_gate"],
            "solution_labels": loop_payload["solution_labels"],
        },
    )
    reproduced_alpha = {
        "game": "alpha",
        "variant_signature": "alpha~l2-goal",
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
            "honest_verdict": "complete: l2_goal_induction_no_deepening",
            "per_game": {"alpha": reproduced_alpha},
        },
    )
    _write_json(
        root / mod.A2_VARIANT_RELATIVE_PATH,
        {
            "honest_verdict": "complete: dagger_distribution_shift_no_new_bank",
            "baseline_measurement": {
                "variant_attempts": [
                    reproduced_alpha,
                    {"reproduction_gate": {"reproduced": True, "reached_level": 1}},
                ]
            },
            "value_routed_measurement": {"variant_attempts": [reproduced_alpha]},
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


def test_req_capstone_4667_spec_declares_refresh_contract() -> None:
    """REQ-CAPSTONE-4667: OpenSpec declares the refreshed package contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4667",
        "SCENARIO-CAPSTONE-4667",
        "SCENARIO-CAPSTONE-4667-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4667_folds_dc22_l2_without_overclaim(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4667: A3 folds only to registry-backed replayable depth."""

    root = _fixture_root(tmp_path, a3_banked=True)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    by_game = {row["game"]: row for row in artifact["per_game_submittable"]}
    package = json.loads((root / artifact["refreshed_package_path"]).read_text(encoding="utf-8"))
    manifest_by_game = {row["game"]: row for row in package["package_manifest"]}

    assert artifact["live_submittable_count_prev"] == 58
    assert artifact["live_submittable_level_count"] == 59
    assert artifact["count_delta"] == 1
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["honest_verdict"] == "success: package_refreshed_live_submittable_59_above_33"
    assert artifact["levels_folded_in"] == ["dc22"]
    assert artifact["submitted_to_leaderboard"] is False

    assert by_game["dc22"]["submittable_level"] == 2
    assert by_game["dc22"]["trajectory_action_count"] == 10
    assert by_game["dc22"]["source"] == "results/arc_loop_solve_dc22.json"
    assert by_game["drift"]["has_env_adaptive_resolver"] is True
    assert by_game["drift"]["adaptive_solver"] == "env_adaptive_resolve_operator:drift"

    assert package["source_result_path"] == mod.RESULT_RELATIVE_PATH
    assert package["claimed_total_levels"] == 59
    assert package["operator_only"] is True
    assert package["submitted_to_leaderboard"] is False
    assert (
        manifest_by_game["dc22"]["env_match_basis"]
        == "offline_reproduction_gated_package_refresh_4667"
    )
    assert manifest_by_game["dc22"]["levels"] == 2

    a1_audit = [
        row
        for row in artifact["upstream_fold_audit"]
        if row["game"] == "alpha" and row["source_artifact"] == mod.A1_VARIANT_RELATIVE_PATH
    ]
    a2_audit = [
        row
        for row in artifact["upstream_fold_audit"]
        if row["game"] == "alpha" and row["source_artifact"] == mod.A2_VARIANT_RELATIVE_PATH
    ]
    assert len(a1_audit) == 1
    assert len(a2_audit) == 1
    assert a1_audit[0]["reason"] == "already_submittable"
    assert a2_audit[0]["reason"] == "already_submittable"
    assert a1_audit[0]["folded"] is False
    assert a2_audit[0]["folded"] is False


def test_scenario_capstone_4667_unchanged_depth_is_annotated(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4667: no new bank yields an honest null delta."""

    root = _fixture_root(tmp_path, a3_banked=False)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)

    assert artifact["live_submittable_level_count"] == 58
    assert artifact["count_delta"] == 0
    assert artifact["levels_folded_in"] == []
    assert artifact["honest_verdict"] == "complete: package_refreshed_unchanged_depth."
    assert ".429 A4" in artifact["null_delta_methodology_note"]
    assert artifact["ready_for_operator_submit"] is True


def test_req_capstone_4667_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4667: missing resources stop before claims are fabricated."""

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


def test_req_capstone_4667_schema_helpers_reject_bad_rows(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4667: schema helpers reject overclaims and checksum drift."""

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

    package = mod.build_package_payload(
        [{"game": "zero", "submittable_level": 0}], result_path=mod.RESULT_RELATIVE_PATH
    )
    assert package["package_manifest"] == []

    with pytest.raises(ValueError, match="missing honest_verdict"):
        mod.write_artifact(tmp_path, {})
