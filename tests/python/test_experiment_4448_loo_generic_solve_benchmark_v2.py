"""Tests for Exp 4448 ARC LOO generic-solver v2 re-measurement.

Spec refs: REQ-REPORT-4448, SCENARIO-REPORT-4448.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4448_loo_generic_solve_benchmark_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
V1_GAMES = ("tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09")


def _ok_preconditions() -> dict[str, Any]:
    return {
        "offline_env_files_present": True,
        "offline_env_games": list(V1_GAMES),
        "arc_solver_kit_import": True,
        "arc_solve_learning_import": True,
        "qwen_gguf_cached": True,
        "igpu_llama_server_available": False,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _v1_artifact() -> dict[str, Any]:
    rows = [
        ("tr87", False, "tu93", "missing_glyph_rewrite_rule_verifier_without_tr87_adapter"),
        ("tu93", True, "tr87", "none"),
        ("lp85", True, "r11l", "none"),
        ("sc25", False, "cd82", "missing_cast_grid_spell_shrink_tank_exit_verifier"),
        ("ka59", False, "sp80", "missing_push_block_world_model_and_dynamic_selection"),
        ("ar25", False, "sp80", "missing_reflection_world_model_and_object_motion_plan"),
        ("ft09", False, "s5i5", "missing_local_constraint_color_cycle_verifier"),
    ]
    return {
        "experiment": "experiment_4432_loo_generic_solve_benchmark",
        "honest_verdict": "complete: generic_loo_solve_count_2_of_7_gate_passed",
        "generic_loo_solve_count": 2,
        "heldout_games": list(V1_GAMES),
        "per_game": [
            {
                "game": game,
                "solved_without_own_recipe": solved,
                "routed_to": routed_to,
                "residual_delta": residual,
            }
            for game, solved, routed_to, residual in rows
        ],
        "attempts": [
            {
                "game": game,
                "solved_without_own_recipe": solved,
                "reproduction_gate": {
                    "game": game,
                    "reproduced": solved,
                    "reached_level": 1 if solved else 0,
                },
                "offline_reproduced": solved,
            }
            for game, solved, _routed_to, _residual in rows
        ],
        "reproducibility_checksum": "1" * 64,
    }


def _config_artifact(*, closed: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4444_generic_config_rule_verifier_operator",
        "honest_verdict": "complete: ft09_generic_resolved_dc22_not_grounded_gap_logged",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "duration_s": 1.05,
        "ft09_resolved_generically": closed,
        "offline_reproduced": closed,
        "ft09_reproduction_result": {
            "game": "ft09",
            "reproduced": closed,
            "reached_level": 1 if closed else 0,
        },
    }


def _object_artifact(*, closed: tuple[str, ...] = ("ar25", "ka59")) -> dict[str, Any]:
    return {
        "experiment": "experiment_4445_generic_object_motion_world_model_operator",
        "honest_verdict": "success: ar25_ka59_object_motion_generic_L1_offline_reproduced",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "duration_s": 1.05,
        "residuals_closed_generically": list(closed),
        "offline_reproduced": bool(closed),
        "per_game": {
            game: {
                "operator_result": {
                    "game": game,
                    "grounded": game in closed,
                    "operator": "object_motion_world_model",
                    "target_recipe_withheld": game,
                },
                "reproduction_result": {
                    "game": game,
                    "reproduced": game in closed,
                    "reached_level": 1 if game in closed else 0,
                },
            }
            for game in ("ar25", "ka59")
        },
    }


def _library_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4447_lilo_documented_primitive_library",
        "honest_verdict": "success: documented_primitive_library_retrieval_gate_passed",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "library_coverage": 1.0,
        "retrieval_precision_at_1": 1.0,
        "constant_leak_violations": [],
        "per_game": [
            {
                "game": "tr87",
                "identified": True,
                "top_operator": "config_rule_grounding",
                "top_primitive": "config_rule_grounding",
            },
            {
                "game": "sc25",
                "identified": True,
                "top_operator": "active_data_collection",
                "top_primitive": "active_data_collection",
            },
            {
                "game": "ft09",
                "identified": True,
                "top_operator": "config_rule_verifier",
                "top_primitive": "config_rule_verifier",
            },
            {
                "game": "ar25",
                "identified": True,
                "top_operator": "object_motion_world_model",
                "top_primitive": "object_motion_world_model",
            },
            {
                "game": "ka59",
                "identified": True,
                "top_operator": "object_motion_world_model",
                "top_primitive": "object_motion_world_model",
            },
        ],
    }


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixture_repo(root: Path) -> str:
    for game in V1_GAMES:
        (root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version": 1,
        "games": [
            {"game": game, "reproducibility": "reproduced", "levels_reproduced": 1}
            for game in V1_GAMES
        ],
    }
    registry_text = yaml.safe_dump(registry, sort_keys=False)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(registry_text, encoding="utf-8")
    _write_json(root, mod.V1_RELATIVE_PATH, _v1_artifact())
    _write_json(root, mod.CONFIG_RULE_RELATIVE_PATH, _config_artifact())
    _write_json(root, mod.OBJECT_MOTION_RELATIVE_PATH, _object_artifact())
    _write_json(root, mod.DOCUMENTED_LIBRARY_RELATIVE_PATH, _library_artifact())
    return registry_text


def test_req_report_4448_spec_declares_v2_remeasurement_contract() -> None:
    """REQ-REPORT-4448: OpenSpec declares the v2 LOO benchmark and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4448" in spec
    assert "SCENARIO-REPORT-4448" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "generic_loo_solve_count_v2" in spec
    assert "generic_loo_solve_count_v1_baseline" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4448_counts_only_reproduction_gated_v1_and_411_closures(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4448: .411 operators close ft09/ar25/ka59 over the same v1 K."""

    registry_before = _write_fixture_repo(tmp_path)
    clock = {"t": 10.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        now=now,
        sleep_fn=sleep,
    )

    assert (tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8") == registry_before
    assert artifact["honest_verdict"] == "success: generic_loo_solve_count_v2_5_of_7_beats_v1_2"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["generic_loo_solve_count_v2"] == 5
    assert artifact["generic_loo_solve_count_v1_baseline"] == 2
    assert artifact["loo_gate_passed"] is True
    assert artifact["heldout_games"] == list(V1_GAMES)
    assert artifact["offline_reproduced"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["no_3090_inference"] is True
    assert artifact["leaderboard_submission"] is False

    by_game = {row["game"]: row for row in artifact["per_game"]}
    assert by_game["tu93"]["closed_by_operator"] == "v1_generic_loop_reproduction_gate"
    assert by_game["lp85"]["closed_by_operator"] == "v1_generic_loop_reproduction_gate"
    assert by_game["ft09"] == {
        "game": "ft09",
        "solved_without_own_recipe": True,
        "closed_by_operator": "config_rule_verifier",
        "residual_delta": "none",
    }
    assert by_game["ar25"]["closed_by_operator"] == "object_motion_world_model"
    assert by_game["ka59"]["closed_by_operator"] == "object_motion_world_model"
    assert by_game["tr87"]["solved_without_own_recipe"] is False
    assert by_game["sc25"]["solved_without_own_recipe"] is False
    assert {gap["game"] for gap in artifact["missing_verifier_gaps"]} == {"tr87", "sc25"}
    assert {row["game"] for row in artifact["closed_residuals_by_new_operator"]} == {
        "ft09",
        "ar25",
        "ka59",
    }
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["generic_loo_solve_count_v2"] == 5
    assert len(written["reproducibility_checksum"]) == 64


def test_req_report_4448_documented_retrieval_alone_does_not_count_as_solve(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4448: documented-library matches are guidance unless reproduction-gated."""

    _write_fixture_repo(tmp_path)
    _write_json(tmp_path, mod.CONFIG_RULE_RELATIVE_PATH, _config_artifact(closed=False))
    _write_json(tmp_path, mod.OBJECT_MOTION_RELATIVE_PATH, _object_artifact(closed=()))

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "complete: generic_loo_solve_count_v2_2_of_7_flat_vs_v1_2"
    assert artifact["generic_loo_solve_count_v2"] == 2
    retrieval_only = {gap["game"]: gap for gap in artifact["missing_verifier_gaps"]}
    assert retrieval_only["ft09"]["retrieved_operator"] == "config_rule_verifier"
    assert retrieval_only["ar25"]["retrieved_operator"] == "object_motion_world_model"
    assert retrieval_only["ka59"]["retrieved_operator"] == "object_motion_world_model"
    assert all(gap["residual_delta"] != "none" for gap in artifact["missing_verifier_gaps"])
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4448_model_cache_blocks_only_induction_targets(tmp_path: Path) -> None:
    """REQ-REPORT-4448: missing model cache marks induction rows and continues other folds."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "qwen_gguf_cached": False},
        llm_induction_games={"ft09"},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    ft09 = next(row for row in artifact["per_game"] if row["game"] == "ft09")
    assert ft09 == {
        "game": "ft09",
        "solved_without_own_recipe": False,
        "closed_by_operator": "none",
        "residual_delta": "blocked_model_not_cached",
    }
    assert artifact["generic_loo_solve_count_v2"] == 4
    assert {gap["game"] for gap in artifact["missing_verifier_gaps"]} == {"tr87", "sc25", "ft09"}
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4448_precondition_block_writes_honest_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-4448: missing offline resources stop before measuring."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "offline_env_files_present": False, "ok": False},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "complete: blocked_offline_env_files"
    assert artifact["generic_loo_solve_count_v2"] == 0
    assert artifact["per_game"] == []
    assert artifact["offline_reproduced"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4448_schema_rejects_malformed_or_fabricated_results(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4448: schema catches non-bare counts, ungated claims, and bad prefixes."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    bad: dict[str, Any] = {
        **artifact,
        "honest_verdict": "partial: invalid",
        "inference_substrate": None,
        "generic_loo_solve_count_v2": "5",
        "generic_loo_solve_count_v1_baseline": 3,
        "per_game": [{"game": "x", "solved_without_own_recipe": True, "residual_delta": "none"}],
        "offline_reproduced": "true",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "random_seed": "4448",
        "reproducibility_checksum": "bad",
        "duration_s": 0.0,
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": {"principle": "wrong"}},
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with complete:/success:/passed:/shipped:" in errors
    assert "inference_substrate must not be None" in errors
    assert "generic_loo_solve_count_v2 must be bare int" in errors
    assert "generic_loo_solve_count_v1_baseline must be bare int = 2" in errors
    assert "per_game[0] missing closed_by_operator" in errors
    assert "generic_loo_solve_count_v2 must match solved per_game rows" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "cached verifier substrate requires duration_s >= 1.0" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4448" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, bad)


def test_req_report_4448_defensive_helpers_cover_missing_and_lower_paths(tmp_path: Path) -> None:
    """REQ-REPORT-4448: helper branches remain deterministic for blocked/lower measurements."""

    assert mod.load_json(tmp_path, "missing.json") is None
    (tmp_path / "bad.json").write_text("[]\n", encoding="utf-8")
    assert mod.load_json(tmp_path, "bad.json") is None
    assert mod._file_sha256(tmp_path, "missing.json") is None
    assert mod._environment_games(tmp_path) == set()
    (tmp_path / "environment_files" / "aa" / "fixture").mkdir(parents=True)
    assert mod._environment_games(tmp_path) == {"aa"}
    assert mod.checksum_is_hex("0" * 64) is True
    assert mod.checksum_is_hex("z" * 64) is False
    assert mod.gate_reproduced({"reproduced": True, "reached_level": "bad"}) is False
    assert mod._heldout_games({"heldout_games": ["too_few"]}) == []
    fallback_rows = {"per_game": [{"game": game} for game in V1_GAMES]}
    assert mod._heldout_games(fallback_rows) == list(V1_GAMES)
    assert mod._heldout_games({"per_game": [{"game": "too_few"}]}) == []
    assert mod._rows_by_game("bad") == {}
    assert mod._v1_reproduced_games({"attempts": "bad"}) == set()
    assert mod._v1_reproduced_games({"per_game": [], "attempts": ["bad"]}) == set()
    assert mod._library_by_game(None) == {}
    assert mod._config_rule_closes_ft09(None) is False
    assert mod._object_motion_closes_game(None, "ar25") is False
    assert mod._object_motion_closes_game({"residuals_closed_generically": "bad"}, "ar25") is False
    assert (
        mod._object_motion_closes_game(
            {"residuals_closed_generically": ["ar25"], "per_game": "bad"},
            "ar25",
        )
        is False
    )
    assert (
        mod._object_motion_closes_game(
            {"residuals_closed_generically": ["ar25"], "per_game": {"ar25": "bad"}},
            "ar25",
        )
        is False
    )
    assert mod.first_precondition_miss({"offline_env_files_present": True}) == "arc_solver_kit_import"
    assert (
        mod.first_precondition_miss(
            {"offline_env_files_present": True, "arc_solver_kit_import": True}
        )
        == "arc_solve_learning_import"
    )
    assert (
        mod.first_precondition_miss(
            {
                "offline_env_files_present": True,
                "arc_solver_kit_import": True,
                "arc_solve_learning_import": True,
                "no_3090_inference": False,
            }
        )
        == "no_3090_inference_policy"
    )
    assert (
        mod.first_precondition_miss(
            {
                "offline_env_files_present": True,
                "arc_solver_kit_import": True,
                "arc_solve_learning_import": True,
                "no_3090_inference": True,
                "leaderboard_submission": True,
            }
        )
        == "leaderboard_submission_policy"
    )

    _write_fixture_repo(tmp_path)
    v1 = _v1_artifact()
    for attempt in v1["attempts"]:
        attempt["reproduction_gate"] = {"reproduced": False, "reached_level": 0}
        attempt["offline_reproduced"] = False
    v1["per_game"][1]["solved_without_own_recipe"] = False
    v1["per_game"][1]["residual_delta"] = "regressed"
    v1["per_game"][2]["solved_without_own_recipe"] = False
    v1["per_game"][2]["residual_delta"] = "regressed"
    _write_json(tmp_path, mod.V1_RELATIVE_PATH, v1)
    _write_json(tmp_path, mod.CONFIG_RULE_RELATIVE_PATH, _config_artifact(closed=False))
    _write_json(tmp_path, mod.OBJECT_MOTION_RELATIVE_PATH, _object_artifact(closed=()))

    lower = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert lower["honest_verdict"] == "complete: generic_loo_solve_count_v2_0_of_7_lower_than_v1_2"
    assert lower["generic_loo_solve_count_v2"] == 0
    assert len(lower["missing_verifier_gaps"]) == 7

    source_block_root = tmp_path / "source_block"
    for game in V1_GAMES:
        (source_block_root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    missing_source = mod.run(
        root=source_block_root,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert missing_source["honest_verdict"] == "complete: blocked_source_artifacts"

    short_v1_root = tmp_path / "short_v1"
    _write_fixture_repo(short_v1_root)
    _write_json(short_v1_root, mod.V1_RELATIVE_PATH, {"heldout_games": ["too_few"], "per_game": []})
    short_v1 = mod.run(
        root=short_v1_root,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert short_v1["honest_verdict"] == "complete: blocked_v1_heldout_target_count"

    assert "missing honest_verdict" in mod.artifact_schema_errors({})
    malformed_rows = {
        **lower,
        "generic_loo_solve_count_v2": 1,
        "per_game": [
            "bad",
            {
                "game": "x",
                "solved_without_own_recipe": "true",
                "closed_by_operator": 7,
                "residual_delta": "none",
            },
            {
                "game": "y",
                "solved_without_own_recipe": True,
                "closed_by_operator": "config_rule_verifier",
                "residual_delta": "still_open",
            },
        ],
        "offline_reproduced": False,
        "honest_verdict": "success: fabricated",
        "no_3090_inference": False,
        "leaderboard_submission": True,
    }
    malformed_errors = mod.artifact_schema_errors(malformed_rows)
    assert "per_game[0] must be dict" in malformed_errors
    assert "per_game[1].solved_without_own_recipe must be bare bool" in malformed_errors
    assert "per_game[1].closed_by_operator must be string" in malformed_errors
    assert "per_game[2] solved row requires residual_delta none" in malformed_errors
    assert "offline_reproduced false cannot accompany counted solves" in malformed_errors
    assert "success verdict requires generic_loo_solve_count_v2 > 2" in malformed_errors
    assert "no_3090_inference must be true" in malformed_errors
    assert "leaderboard_submission must be false" in malformed_errors

    assert "per_game must be list" in mod.artifact_schema_errors({**lower, "per_game": "bad"})
