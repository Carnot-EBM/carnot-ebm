"""Tests for Exp 4607 operator-resubmit package refresh.

Spec refs: REQ-CAPSTONE-4607, SCENARIO-CAPSTONE-4607,
SCENARIO-CAPSTONE-4607-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from carnot import experiment_4607_refresh_submission_package as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _solution(action_count: int, action: int = 2) -> list[JsonDict]:
    return [{"action": action} for _ in range(action_count)]


def _fixture_root(tmp_path: Path, *, a3_banked: bool = True) -> Path:
    root = tmp_path
    (root / "ops").mkdir()
    (root / "results" / "arc3_live_banked_trajectories").mkdir(parents=True)
    registry = {
        "schema_version": 1,
        "reproducible_total_levels": 56 if a3_banked else 55,
        "games": [
            {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 52},
            {
                "game": "dc22",
                "reproducibility": "reproduced",
                "levels_reproduced": 2 if a3_banked else 1,
            },
            {"game": "drift", "reproducibility": "reproduced", "levels_reproduced": 1},
            {"game": "capped", "reproducibility": "reproduced", "levels_reproduced": 1},
            {"game": "no_path", "reproducibility": "reproduced", "levels_reproduced": 1},
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
            "experiment": "experiment_4595_submission_package_operator_resubmit",
            "package_manifest": [
                {
                    "game": "alpha",
                    "levels": 52,
                    "offline_reproduced_level": 52,
                    "trajectory_path": "results/arc3_live_banked_trajectories/alpha.json",
                    "action_count": 1,
                    "env_matched": True,
                    "source": "baseline_alpha",
                },
                {
                    "game": "dc22",
                    "levels": 1,
                    "offline_reproduced_level": 1,
                    "trajectory_path": "results/arc3_live_banked_trajectories/dc22.json",
                    "action_count": 20,
                    "env_matched": True,
                    "source": "baseline_dc22",
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
                {
                    "game": "capped",
                    "levels": 3,
                    "offline_reproduced_level": 3,
                    "trajectory_path": "results/arc3_live_banked_trajectories/capped.json",
                    "action_count": 3,
                    "env_matched": True,
                    "source": "over_claim_fixture",
                },
                {
                    "game": "no_path",
                    "levels": 1,
                    "offline_reproduced_level": 1,
                    "trajectory_path": "",
                    "action_count": 0,
                    "env_matched": True,
                    "source": "missing_path_fixture",
                },
            ],
        },
    )
    _write_json(root / mod.BASELINE_RESULT_RELATIVE_PATH, {"live_submittable_level_count": 55})
    for game, count in {"alpha": 1, "dc22": 20, "capped": 3}.items():
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
            root / "results" / "arc_loop_solve_dc22.json",
            {
                "game": "dc22",
                "offline_reproduced": True,
                "reached_level": 2,
                "reproduction_gate": {
                    "game": "dc22",
                    "claimed_level": 2,
                    "reached_level": 2,
                    "reproduced": True,
                },
                "solution_labels": ['{"action":1}', '{"action":2}', '{"action":3}'],
            },
        )
        a3_payload: JsonDict = {
            "honest_verdict": "success: dc22_L2_offline_reproduced",
            "target_game": "dc22",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reached_level": 2,
            "standing_loop_result_path": "results/arc_loop_solve_dc22.json",
            "reproduction_gate": {
                "game": "dc22",
                "claimed_level": 2,
                "reached_level": 2,
                "reproduced": True,
            },
            "solution_labels": ['{"action":1}', '{"action":2}', '{"action":3}'],
        }
    else:
        a3_payload = {
            "honest_verdict": "complete: dc22_delta_identified_no_bank",
            "target_game": "dc22",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "reached_level": 2,
            "standing_loop_result_path": "results/arc_loop_solve_dc22.json",
            "reproduction_gate": {
                "game": "dc22",
                "claimed_level": 2,
                "reached_level": 1,
                "reproduced": False,
            },
            "solution_labels": ['{"action":1}'],
        }
    _write_json(root / mod.A3_LEVELUP_RELATIVE_PATH, a3_payload)
    _write_json(
        root / mod.A1_WORLD_MODEL_RELATIVE_PATH,
        {
            "honest_verdict": "success: world_model_trust_energy_pass_rate_up_6_first_win_up",
            "measurements": [
                {"game": "alpha", "new_first_win": True, "offline_reproduced": True},
            ],
        },
    )
    _write_json(
        root / mod.A2_LIVE_INTEGRATION_RELATIVE_PATH,
        {
            "honest_verdict": "complete: live_integration_no_value_honest_null_gap_sharpened",
            "integrated_measurement": {
                "variant_attempts": [
                    {
                        "game": "alpha",
                        "reproduction_gate": {
                            "claimed_level": 1,
                            "game": "alpha",
                            "reached_level": 1,
                            "reproduced": True,
                        },
                        "solution_labels": ['{"action":6,"data":{"x":1,"y":2}}'],
                    }
                ]
            },
        },
    )
    return root


def test_req_capstone_4607_spec_declares_refresh_contract() -> None:
    """REQ-CAPSTONE-4607: OpenSpec declares the refreshed package contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4607",
        "SCENARIO-CAPSTONE-4607",
        "SCENARIO-CAPSTONE-4607-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4607_folds_a3_bank_without_overclaim(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4607: A3 banks fold in only to current registry depth."""

    root = _fixture_root(tmp_path, a3_banked=True)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    by_game = {row["game"]: row for row in artifact["per_game_submittable"]}

    assert artifact["live_submittable_count_prev"] == 55
    assert artifact["live_submittable_level_count"] == 56
    assert artifact["count_delta"] == 1
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["honest_verdict"] == "success: package_refreshed_live_submittable_56_above_33"
    assert artifact["levels_folded_in"] == ["dc22"]

    assert by_game["dc22"]["submittable_level"] == 2
    assert by_game["dc22"]["trajectory_action_count"] == 3
    assert by_game["dc22"]["source"] == "results/arc_loop_solve_dc22.json"
    assert by_game["drift"]["submittable_level"] == 1
    assert by_game["drift"]["has_env_adaptive_resolver"] is True
    assert by_game["drift"]["adaptive_solver"] == "env_adaptive_resolve_operator:drift"
    assert by_game["capped"]["submittable_level"] == 1
    assert by_game["capped"]["claim_capped"] is True
    assert by_game["no_path"]["submittable_level"] == 0
    assert by_game["no_path"]["exclusion_reason"] == "missing_trajectory_or_adaptive_resolver"

    refreshed_dc22 = json.loads(
        (root / "results/arc3_live_banked_trajectories/dc22.json").read_text()
    )
    assert refreshed_dc22["action_count"] == 3
    assert refreshed_dc22["source"] == "results/arc_loop_solve_dc22.json"
    dc22_audit = next(row for row in artifact["upstream_fold_audit"] if row["game"] == "dc22")
    assert dc22_audit["folded"] is True


def test_scenario_capstone_4607_unchanged_depth_is_annotated(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4607: no current registry bank yields an honest null delta."""

    root = _fixture_root(tmp_path, a3_banked=False)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)

    assert artifact["live_submittable_level_count"] == 55
    assert artifact["count_delta"] == 0
    assert artifact["levels_folded_in"] == []
    assert artifact["honest_verdict"] == "complete: package_refreshed_unchanged_depth."
    assert artifact["null_delta_methodology_note"]
    dc22_audit = next(row for row in artifact["upstream_fold_audit"] if row["game"] == "dc22")
    assert dc22_audit["folded"] is False
    assert dc22_audit["reason"] == "upstream_not_offline_reproduced"


def test_scenario_capstone_4607_writes_driver_loadable_package(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4607: the package manifest is the operator deliverable."""

    root = _fixture_root(tmp_path, a3_banked=True)
    artifact = mod.run(root, offline_arcade_checker=lambda: True)
    package = json.loads((root / artifact["refreshed_package_path"]).read_text(encoding="utf-8"))
    manifest_by_game = {row["game"]: row for row in package["package_manifest"]}

    assert package["source_result_path"] == mod.RESULT_RELATIVE_PATH
    assert package["claimed_total_levels"] == artifact["live_submittable_level_count"]
    assert package["operator_only"] is True
    assert package["submitted_to_leaderboard"] is False
    assert manifest_by_game["dc22"]["levels"] == 2
    assert manifest_by_game["dc22"]["action_count"] == 3
    assert manifest_by_game["drift"]["adaptive_solver"] == "env_adaptive_resolve_operator:drift"
    assert manifest_by_game["capped"]["levels"] == 1
    assert "no_path" not in manifest_by_game


def test_req_capstone_4607_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4607: missing resources stop before package claims are fabricated."""

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


def test_req_capstone_4607_schema_catches_mismatches() -> None:
    """REQ-CAPSTONE-4607: schema rejects false readiness and over-claimed rows."""

    bad = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    bad.update(
        {
            "honest_verdict": "success: package_refreshed_live_submittable_10_above_33",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "verifier_is_oracle": False,
            "live_submittable_level_count": 10,
            "live_submittable_count_prev": 55,
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

    assert (
        "count_delta must equal live_submittable_level_count - live_submittable_count_prev"
        in errors
    )
    assert "ready_for_operator_submit requires count strictly above 33" in errors
    assert "null_delta_methodology_note required when count_delta is zero" in errors
    assert "bad submittable exceeds offline reproduction" in errors
    assert "bad counted without trajectory or adaptive resolver" in errors


def test_req_capstone_4607_pure_helper_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4607: helper branches handle defensive inputs."""

    root = _fixture_root(tmp_path)
    assert mod._solution_label_actions(['{"action":1}', {"action": 2}, "bad", "[]"]) == [
        {"action": 1},
        {"action": 2},
    ]
    assert mod._candidate_actions({"solution": [{"action": 3}]}) == [{"action": 3}]
    assert mod._candidate_actions({"solution_labels": ['{"action":4}']}) == [{"action": 4}]
    assert mod._candidate_actions({}) == []
    assert mod._candidate_level({"reproduction_gate": {"reached_level": 2}}, fallback=1) == 2
    assert mod._candidate_level({"reached_level": 3}, fallback=1) == 3
    assert mod._candidate_level({}, fallback=4) == 4
    assert mod._is_candidate_reproduced({"offline_reproduced": True}, 1)
    assert not mod._is_candidate_reproduced(
        {"reproduction_gate": {"reproduced": True, "reached_level": 1}}, 2
    )
    assert (
        mod.first_precondition_miss(
            {"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}}
        )
        is None
    )
    assert (
        mod.first_precondition_miss(
            {"offline_arcade": {"ok": True}, "registry_loadable": {"ok": False}}
        )
        == "registry"
    )
    assert mod._honest_verdict(55, 0) == "complete: package_refreshed_unchanged_depth."
    assert mod._previous_live_count({}) == 55
    assert (
        mod._env_adaptive_resolvers({"general_gotchas": []})["ft09"]
        == "env_adaptive_resolve_operator:ft09"
    )

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


def test_req_capstone_4607_defensive_audit_and_candidate_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4607: defensive upstream parsing keeps claims offline-gated."""

    root = _fixture_root(tmp_path)
    assert mod._env_adaptive_resolvers({})["sc25"] == "sc25_dynamic_cast_grid_origin_step"
    assert (
        mod._env_adaptive_resolvers({"general_gotchas": [{}, {"id": "unrelated"}]})["ft09"]
        == "env_adaptive_resolve_operator:ft09"
    )

    _write_json(root / mod.A3_LEVELUP_RELATIVE_PATH, {})
    _write_json(
        root / mod.A1_WORLD_MODEL_RELATIVE_PATH, {"measurements": ["bad", {"new_first_win": True}]}
    )
    _write_json(
        root / mod.A2_LIVE_INTEGRATION_RELATIVE_PATH,
        {"integrated_measurement": {"variant_attempts": ["bad", {}]}},
    )
    assert mod.collect_upstream_candidates(root) == {}

    payload, source = mod._candidate_payload(
        root,
        {
            "source_payload_path": "results/missing_payload.json",
            "source_artifact": "source.json",
            "payload": {"offline_reproduced": True},
        },
    )
    assert payload == {"offline_reproduced": True}
    assert source == "source.json"

    actions, path, source, level, refreshes = mod._refreshed_actions(
        root,
        game="branch",
        registry_level=2,
        previous_level=1,
        previous_row={},
        candidates=[
            {
                "target_level": 3,
                "offline_reproduced": True,
                "source_artifact": "bad_gate.json",
                "payload": {"reproduction_gate": {"reproduced": False, "reached_level": 1}},
            },
            {
                "target_level": 2,
                "offline_reproduced": True,
                "source_artifact": "no_actions.json",
                "payload": {"offline_reproduced": True, "reached_level": 2},
            },
        ],
    )
    assert actions == []
    assert path == ""
    assert source == ""
    assert level == 0
    assert refreshes == []

    audit = mod._candidate_audit(
        candidates_by_game={
            "missing": [{"offline_reproduced": True, "target_level": 1, "source_artifact": "a"}],
            "above": [{"offline_reproduced": True, "target_level": 3, "source_artifact": "b"}],
            "stuck": [{"offline_reproduced": True, "target_level": 2, "source_artifact": "c"}],
            "covered": [{"offline_reproduced": True, "target_level": 2, "source_artifact": "d"}],
        },
        registry_levels={"above": 1, "stuck": 2, "covered": 2},
        previous_by_game={"stuck": {"levels": 1}, "covered": {"levels": 1}},
        rows_by_game={"stuck": {"submittable_level": 1}, "covered": {"submittable_level": 2}},
        folded=[],
    )
    reasons = {row["game"]: row["reason"] for row in audit}
    assert reasons == {
        "above": "above_current_registry_depth",
        "covered": "covered_without_delta",
        "missing": "not_in_current_registry",
        "stuck": "no_new_replayable_depth",
    }
