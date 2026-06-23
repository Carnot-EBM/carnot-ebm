"""Tests for Exp 4619 operator-resubmit package refresh.

Spec refs: REQ-CAPSTONE-4619, SCENARIO-CAPSTONE-4619,
SCENARIO-CAPSTONE-4619-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from carnot import experiment_4619_refresh_submission_package as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _solution(count: int) -> list[JsonDict]:
    return [{"action": 1} for _ in range(count)]


def _fixture_root(tmp_path: Path, *, a3_banked: bool = True) -> Path:
    root = tmp_path
    (root / "ops").mkdir()
    (root / "results" / "arc3_live_banked_trajectories").mkdir(parents=True)
    registry = {
        "schema_version": 1,
        "reproducible_total_levels": 56 if a3_banked else 55,
        "games": [
            {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 48},
            {"game": "lp85", "reproducibility": "reproduced", "levels_reproduced": 5},
            {
                "game": "sk48",
                "reproducibility": "reproduced",
                "levels_reproduced": 2 if a3_banked else 1,
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
            "experiment": "experiment_4607_submission_package_operator_resubmit",
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
                    "game": "sk48",
                    "levels": 1,
                    "offline_reproduced_level": 1,
                    "trajectory_path": "results/arc3_live_banked_trajectories/sk48.json",
                    "action_count": 1,
                    "env_matched": True,
                    "source": "baseline_sk48",
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
    _write_json(root / mod.BASELINE_RESULT_RELATIVE_PATH, {"live_submittable_level_count": 55})
    for game, count in {"alpha": 2, "lp85": 5, "sk48": 1}.items():
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
            root / "results" / "arc_loop_solve_sk48.json",
            {
                "game": "sk48",
                "offline_reproduced": True,
                "reached_level": 2,
                "reproduction_gate": {
                    "game": "sk48",
                    "claimed_level": 2,
                    "reached_level": 2,
                    "reproduced": True,
                },
                "solution": _solution(2),
            },
        )
        a3_payload: JsonDict = {
            "honest_verdict": "success: sk48_L2_offline_reproduced",
            "target_game": "sk48",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reached_level": 2,
            "standing_loop_result_path": "results/arc_loop_solve_sk48.json",
            "reproduction_gate": {
                "game": "sk48",
                "claimed_level": 2,
                "reached_level": 2,
                "reproduced": True,
            },
            "solution_labels": ['{"action":1}', '{"action":1}'],
        }
    else:
        a3_payload = {
            "honest_verdict": "complete: sk48_delta_identified_no_bank",
            "target_game": "sk48",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "reached_level": 1,
            "standing_loop_result_path": "results/arc_loop_solve_sk48.json",
            "reproduction_gate": {
                "game": "sk48",
                "claimed_level": 2,
                "reached_level": 1,
                "reproduced": False,
            },
            "solution_labels": ['{"action":1}'],
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
        root / mod.A2_VARIANT_RELATIVE_PATH,
        {
            "honest_verdict": "complete: spatial_value_head_graduated_no_live_value_honest_null",
            "graduated_measurement": {
                "variant_attempts": [
                    reproduced_variant,
                    "malformed_attempt",
                    {
                        "game": "alpha",
                        "variant_signature": "alpha~color01",
                        "first_win": False,
                        "solved": False,
                        "reproduction_gate": {"reproduced": False, "reached_level": 0},
                    },
                ]
            },
            "linear_measurement": {"variant_attempts": [reproduced_variant]},
            "bare_measurement": {"variant_attempts": [reproduced_variant]},
        },
    )
    return root


def test_req_capstone_4619_spec_declares_refresh_contract() -> None:
    """REQ-CAPSTONE-4619: OpenSpec declares the refreshed package contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4619",
        "SCENARIO-CAPSTONE-4619",
        "SCENARIO-CAPSTONE-4619-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4619_folds_a3_and_audits_a2_without_overclaim(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4619: A3/A2 candidates fold only to registry-backed depth."""

    root = _fixture_root(tmp_path, a3_banked=True)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    by_game = {row["game"]: row for row in artifact["per_game_submittable"]}
    package = json.loads((root / artifact["refreshed_package_path"]).read_text(encoding="utf-8"))
    manifest_by_game = {row["game"]: row for row in package["package_manifest"]}

    assert artifact["live_submittable_count_prev"] == 55
    assert artifact["live_submittable_level_count"] == 56
    assert artifact["count_delta"] == 1
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["honest_verdict"] == "success: package_refreshed_live_submittable_56_above_33"
    assert artifact["levels_folded_in"] == ["sk48"]

    assert by_game["sk48"]["submittable_level"] == 2
    assert by_game["sk48"]["trajectory_action_count"] == 2
    assert by_game["sk48"]["source"] == "results/arc_loop_solve_sk48.json"
    assert by_game["drift"]["submittable_level"] == 1
    assert by_game["drift"]["has_env_adaptive_resolver"] is True
    assert by_game["drift"]["adaptive_solver"] == "env_adaptive_resolve_operator:drift"

    assert package["source_result_path"] == mod.RESULT_RELATIVE_PATH
    assert package["claimed_total_levels"] == 56
    assert package["operator_only"] is True
    assert package["submitted_to_leaderboard"] is False
    assert (
        manifest_by_game["sk48"]["env_match_basis"]
        == "offline_reproduction_gated_package_refresh_4619"
    )
    assert manifest_by_game["sk48"]["levels"] == 2
    assert manifest_by_game["drift"]["adaptive_solver"] == "env_adaptive_resolve_operator:drift"

    a2_audit = [
        row
        for row in artifact["upstream_fold_audit"]
        if row["game"] == "lp85" and row["source_artifact"] == mod.A2_VARIANT_RELATIVE_PATH
    ]
    assert len(a2_audit) == 1
    assert a2_audit[0]["reason"] == "already_submittable"
    assert a2_audit[0]["folded"] is False


def test_scenario_capstone_4619_unchanged_depth_is_annotated(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4619: no new bank yields an honest null delta."""

    root = _fixture_root(tmp_path, a3_banked=False)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)

    assert artifact["live_submittable_level_count"] == 55
    assert artifact["count_delta"] == 0
    assert artifact["levels_folded_in"] == []
    assert artifact["honest_verdict"] == "complete: package_refreshed_unchanged_depth."
    assert ".425 A4" in artifact["null_delta_methodology_note"]
    assert artifact["ready_for_operator_submit"] is True


def test_req_capstone_4619_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4619: missing resources stop before package claims are fabricated."""

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


def test_req_capstone_4619_schema_and_checksum_helpers(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4619: schema rejects false readiness and checksum drift."""

    root = _fixture_root(tmp_path)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    assert mod.artifact_schema_errors(artifact) == []
    assert mod.compute_reproducibility_checksum(artifact) == artifact["reproducibility_checksum"]

    bad = dict(artifact)
    bad["count_delta"] = 0
    bad["reproducibility_checksum"] = "bad"
    bad["per_game_submittable"] = [
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
    assert "bad submittable exceeds offline reproduction" in errors
    assert "bad counted without trajectory or adaptive resolver" in errors

    package = mod.build_package_payload(
        [{"game": "zero", "submittable_level": 0}], result_path=mod.RESULT_RELATIVE_PATH
    )
    assert package["package_manifest"] == []
    try:
        mod.write_artifact(tmp_path, bad)
    except ValueError as exc:
        assert "reproducibility_checksum must match the artifact payload" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("write_artifact should reject invalid artifacts")
