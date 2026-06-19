"""Tests for Exp 4459 ARC LOO generic-solver v3 re-measurement.

Spec refs: REQ-REPORT-4459, SCENARIO-REPORT-4459.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4448_loo_generic_solve_benchmark_v2 as v2
from carnot import experiment_4459_loo_generic_solve_benchmark_v3 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
V2_GAMES = ("tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09")


def _ok_preconditions() -> dict[str, Any]:
    return {
        "offline_env_files_present": True,
        "offline_env_games": list(V2_GAMES),
        "arc_solver_kit_import": True,
        "arc_solve_learning_import": True,
        "qwen_gguf_cached": True,
        "igpu_llama_server_available": False,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _v2_artifact() -> dict[str, Any]:
    rows = [
        ("tr87", False, "none", "missing_glyph_rewrite_rule_verifier_without_tr87_adapter"),
        ("tu93", True, "v1_generic_loop_reproduction_gate", "none"),
        ("lp85", True, "v1_generic_loop_reproduction_gate", "none"),
        ("sc25", False, "none", "missing_cast_grid_spell_shrink_tank_exit_verifier"),
        ("ka59", True, "object_motion_world_model", "none"),
        ("ar25", True, "object_motion_world_model", "none"),
        ("ft09", True, "config_rule_verifier", "none"),
    ]
    return {
        "experiment": "experiment_4448_loo_generic_solve_benchmark_v2",
        "honest_verdict": "success: generic_loo_solve_count_v2_5_of_7_beats_v1_2",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "generic_loo_solve_count_v2": 5,
        "heldout_games": list(V2_GAMES),
        "per_game": [
            {
                "game": game,
                "solved_without_own_recipe": solved,
                "closed_by_operator": operator,
                "residual_delta": residual,
            }
            for game, solved, operator, residual in rows
        ],
        "reproduction_evidence": [
            {
                "game": game,
                "operator": operator,
                "source": v2.RESULT_RELATIVE_PATH,
            }
            for game, solved, operator, _residual in rows
            if solved
        ],
        "missing_verifier_gaps": [
            {
                "game": "tr87",
                "residual_delta": "missing_glyph_rewrite_rule_verifier_without_tr87_adapter",
                "retrieved_operator": "config_rule_grounding",
                "v1_routed_to": "tu93",
            },
            {
                "game": "sc25",
                "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                "retrieved_operator": "active_data_collection",
                "v1_routed_to": "cd82",
            },
        ],
        "offline_reproduced": True,
        "verifier_is_oracle": True,
        "random_seed": 4448,
        "reproducibility_checksum": "1" * 64,
    }


def _glyph_artifact(*, closed: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4456_generic_glyph_rewrite_operator",
        "honest_verdict": "success: tr87_generic_glyph_rewrite_L1_offline_reproduced",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "tr87_resolved_generically": closed,
        "offline_reproduced": closed,
        "generic_operator_result": {
            "game": "tr87",
            "operator": "glyph_rewrite_rule_verifier",
            "target_recipe_withheld": "tr87",
            "grounded": closed,
        },
        "generic_reproduction_result": {
            "game": "tr87",
            "reproduced": closed,
            "reached_level": 1 if closed else 0,
        },
        "missing_verifier_gaps": [] if closed else [{"game": "tr87"}],
        "verifier_is_oracle": True,
    }


def _cast_artifact(*, closed: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4457_cast_grid_phase_fsm_world_model",
        "honest_verdict": "success: sc25_cast_grid_phase_fsm_L1_offline_reproduced",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "sc25_resolved_generically": closed,
        "offline_reproduced": closed,
        "generic_operator_result": {
            "game": "sc25",
            "operator": "cast_grid_phase_fsm_world_model",
            "target_recipe_withheld": "sc25",
            "grounded": closed,
        },
        "generic_reproduction_result": {
            "game": "sc25",
            "reproduced": closed,
            "reached_level": 1 if closed else 0,
        },
        "missing_verifier_gaps": [] if closed else [{"game": "sc25"}],
        "verifier_is_oracle": True,
    }


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixture_repo(root: Path, *, cast_artifact: bool = True) -> str:
    for game in V2_GAMES:
        (root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version": 1,
        "games": [
            {"game": game, "reproducibility": "reproduced", "levels_reproduced": 1}
            for game in V2_GAMES
        ],
    }
    registry_text = yaml.safe_dump(registry, sort_keys=False)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(registry_text, encoding="utf-8")
    _write_json(root, mod.V2_RELATIVE_PATH, _v2_artifact())
    _write_json(root, mod.GLYPH_REWRITE_RELATIVE_PATH, _glyph_artifact())
    if cast_artifact:
        _write_json(root, "results/experiment_4457_cast_grid_phase_fsm_world_model.json", _cast_artifact())
    return registry_text


def test_req_report_4459_spec_declares_v3_remeasurement_contract() -> None:
    """REQ-REPORT-4459: OpenSpec declares the v3 LOO benchmark and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4459" in spec
    assert "SCENARIO-REPORT-4459" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "generic_loo_solve_count_v3" in spec
    assert "generic_loo_solve_count_v2_baseline" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4459_counts_reproduction_gated_412_closures(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4459: .412 operators can close tr87/sc25 over the same v2 K."""

    registry_before = _write_fixture_repo(tmp_path, cast_artifact=True)
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
    assert artifact["honest_verdict"] == "success: generic_loo_solve_count_v3_7_of_7_beats_v2_5"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["generic_loo_solve_count_v3"] == 7
    assert artifact["generic_loo_solve_count_v2_baseline"] == 5
    assert artifact["loo_gate_passed"] is True
    assert artifact["heldout_games"] == list(V2_GAMES)
    assert artifact["offline_reproduced"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["no_3090_inference"] is True
    assert artifact["leaderboard_submission"] is False

    by_game = {row["game"]: row for row in artifact["per_game"]}
    assert by_game["tr87"] == {
        "game": "tr87",
        "solved_without_own_recipe": True,
        "closed_by_operator": "glyph_rewrite_rule_verifier",
        "residual_delta": "none",
    }
    assert by_game["sc25"]["closed_by_operator"] == "cast_grid_phase_fsm_world_model"
    assert by_game["ka59"]["closed_by_operator"] == "object_motion_world_model"
    assert {row["game"] for row in artifact["closed_residuals_by_412_operator"]} == {"tr87", "sc25"}
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["generic_loo_solve_count_v3"] == 7
    assert len(written["reproducibility_checksum"]) == 64


def test_req_report_4459_missing_cast_artifact_keeps_sc25_residual(tmp_path: Path) -> None:
    """REQ-REPORT-4459: absent .412 sc25 evidence is logged, not counted."""

    _write_fixture_repo(tmp_path, cast_artifact=False)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "success: generic_loo_solve_count_v3_6_of_7_beats_v2_5"
    assert artifact["generic_loo_solve_count_v3"] == 6
    by_game = {row["game"]: row for row in artifact["per_game"]}
    assert by_game["tr87"]["closed_by_operator"] == "glyph_rewrite_rule_verifier"
    assert by_game["sc25"] == {
        "game": "sc25",
        "solved_without_own_recipe": False,
        "closed_by_operator": "none",
        "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
    }
    assert {gap["game"] for gap in artifact["missing_verifier_gaps"]} == {"sc25"}
    assert artifact["missing_verifier_gaps"][0]["attempt_mode"] == "v3_412_operator_remeasurement"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4459_model_cache_blocks_only_induction_targets(tmp_path: Path) -> None:
    """REQ-REPORT-4459: missing model cache marks induction rows and continues other folds."""

    _write_fixture_repo(tmp_path, cast_artifact=False)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "qwen_gguf_cached": False},
        llm_induction_games={"tr87"},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    tr87 = next(row for row in artifact["per_game"] if row["game"] == "tr87")
    assert tr87 == {
        "game": "tr87",
        "solved_without_own_recipe": False,
        "closed_by_operator": "none",
        "residual_delta": "blocked_model_not_cached",
    }
    assert artifact["honest_verdict"] == "complete: generic_loo_solve_count_v3_5_of_7_flat_vs_v2_5"
    assert artifact["generic_loo_solve_count_v3"] == 5
    assert {gap["game"] for gap in artifact["missing_verifier_gaps"]} == {"tr87", "sc25"}
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4459_precondition_and_source_blocks_write_honest_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-4459: missing offline resources or v2 artifact block before measuring."""

    _write_fixture_repo(tmp_path)

    blocked = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "offline_env_files_present": False, "ok": False},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert blocked["honest_verdict"] == "complete: blocked_offline_env_files"
    assert blocked["generic_loo_solve_count_v3"] == 0
    assert blocked["per_game"] == []
    assert blocked["offline_reproduced"] is False
    assert mod.artifact_schema_errors(blocked) == []

    source_block_root = tmp_path / "source_block"
    for game in V2_GAMES:
        (source_block_root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    missing_source = mod.run(
        root=source_block_root,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert missing_source["honest_verdict"] == "complete: blocked_v2_source_artifact"
    assert mod.artifact_schema_errors(missing_source) == []


def test_req_report_4459_schema_rejects_malformed_or_fabricated_results(tmp_path: Path) -> None:
    """REQ-REPORT-4459: schema catches non-bare counts, ungated claims, and bad prefixes."""

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
        "generic_loo_solve_count_v3": "6",
        "generic_loo_solve_count_v2_baseline": 4,
        "per_game": [{"game": "x", "solved_without_own_recipe": True, "residual_delta": "none"}],
        "offline_reproduced": "true",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "random_seed": "4459",
        "reproducibility_checksum": "bad",
        "duration_s": 0.0,
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": {"principle": "wrong"}},
        "no_3090_inference": False,
        "leaderboard_submission": True,
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with complete:/success:/passed:/shipped:" in errors
    assert "inference_substrate must not be None" in errors
    assert "generic_loo_solve_count_v3 must be bare int" in errors
    assert "generic_loo_solve_count_v2_baseline must be bare int = 5" in errors
    assert "per_game[0] missing closed_by_operator" in errors
    assert "generic_loo_solve_count_v3 must match solved per_game rows" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "cached verifier substrate requires duration_s >= 1.0" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4459" in errors
    assert "no_3090_inference must be true" in errors
    assert "leaderboard_submission must be false" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, bad)


def test_req_report_4459_defensive_helpers_and_blockers_are_deterministic(tmp_path: Path) -> None:
    """REQ-REPORT-4459: fallback helpers and blocked paths remain deterministic."""

    assert mod._heldout_games({"per_game": [{"game": game} for game in V2_GAMES]}) == list(V2_GAMES)
    assert mod._heldout_games({"heldout_games": ["too_few"], "per_game": []}) == []
    assert mod._v2_solved_games({"offline_reproduced": False, "per_game": []}) == set()
    assert mod._operator_result_matches(
        {
            "generic_operator_result": "bad",
            "generic_solve_result": {
                "operator_result": {
                    "game": "tr87",
                    "operator": "glyph_rewrite_rule_verifier",
                    "target_recipe_withheld": "tr87",
                    "grounded": True,
                }
            },
        },
        game="tr87",
        operator="glyph_rewrite_rule_verifier",
    )
    assert (
        mod._operator_result_matches(
            {"generic_operator_result": {"game": "tr87", "operator": "wrong"}},
            game="tr87",
            operator="glyph_rewrite_rule_verifier",
        )
        is False
    )
    assert mod._glyph_rewrite_closes_tr87(None) is False
    assert mod._cast_grid_closes_sc25(None) is False
    assert mod._retrieved_operator({"missing_verifier_gaps": "bad"}, "tr87") == ""
    assert mod._retrieved_operator({"missing_verifier_gaps": [{"game": "sc25"}]}, "tr87") == ""
    assert mod._verdict(4, 7) == "complete: generic_loo_solve_count_v3_4_of_7_lower_than_v2_5"
    assert mod._residual_for_open("tr87", {}, None) == "missing_glyph_rewrite_rule_verifier_without_tr87_adapter"
    assert mod._residual_for_open("sc25", {}, None) == "missing_cast_grid_spell_shrink_tank_exit_verifier"
    assert mod._residual_for_open("xx", {}, None) == "missing_reproduction_gate_evidence"
    assert mod._missing_heldout_env_games({"offline_env_games": "bad"}, ["tr87"]) == ["tr87"]

    _write_fixture_repo(tmp_path)
    short_v2_root = tmp_path / "short_v2"
    _write_fixture_repo(short_v2_root)
    _write_json(short_v2_root, mod.V2_RELATIVE_PATH, {"heldout_games": ["too_few"], "per_game": []})
    short_v2 = mod.run(
        root=short_v2_root,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert short_v2["honest_verdict"] == "complete: blocked_v2_heldout_target_count"

    missing_env = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "offline_env_games": ["tr87"]},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert missing_env["honest_verdict"] == "complete: blocked_offline_env_files_tu93"

    assert "missing honest_verdict" in mod.artifact_schema_errors({})
    malformed_rows = {
        **short_v2,
        "honest_verdict": "success: fabricated",
        "generic_loo_solve_count_v3": 1,
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
                "closed_by_operator": "none",
                "residual_delta": "still_open",
            },
        ],
        "offline_reproduced": False,
    }
    malformed_errors = mod.artifact_schema_errors(malformed_rows)
    assert "per_game[0] must be dict" in malformed_errors
    assert "per_game[1].solved_without_own_recipe must be bare bool" in malformed_errors
    assert "per_game[1].closed_by_operator must be string" in malformed_errors
    assert "per_game[2] solved row requires closed_by_operator" in malformed_errors
    assert "per_game[2] solved row requires residual_delta none" in malformed_errors
    assert "offline_reproduced false cannot accompany counted solves" in malformed_errors
    assert "success verdict requires generic_loo_solve_count_v3 > 5" in malformed_errors
    assert "per_game must be list" in mod.artifact_schema_errors({**short_v2, "per_game": "bad"})
