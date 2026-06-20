"""Tests for Exp 4472 manufactured-variant generic-transfer benchmark v4.

Spec refs: REQ-REPORT-4472, SCENARIO-REPORT-4472.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4472_variant_generic_transfer_benchmark_v4 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
PUBLIC_GAMES = (
    "ar25",
    "bp35",
    "cd82",
    "cn04",
    "dc22",
    "ft09",
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "ls20",
    "m0r0",
    "r11l",
    "re86",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "sp80",
    "su15",
    "tn36",
    "tr87",
    "tu93",
    "vc33",
    "wa30",
)
V3_GAMES = ("tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09")


def _ok_preconditions() -> dict[str, Any]:
    return {
        "offline_env_files_present": True,
        "offline_env_games": list(PUBLIC_GAMES),
        "arc_variant_generator_import": True,
        "arc_solver_kit_import": True,
        "arc_solve_learning_import": True,
        "baseline_smoke_command": mod.BASELINE_SMOKE_COMMAND_TEXT,
        "baseline_smoke_green": True,
        "qwen_gguf_cached": False,
        "igpu_llama_server_available": False,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _v3_artifact() -> dict[str, Any]:
    rows = [
        ("tr87", True, "glyph_rewrite_rule_verifier", "none"),
        ("tu93", True, "v1_generic_loop_reproduction_gate", "none"),
        ("lp85", True, "v1_generic_loop_reproduction_gate", "none"),
        ("sc25", False, "none", "missing_cast_grid_spell_shrink_tank_exit_verifier"),
        ("ka59", True, "object_motion_world_model", "none"),
        ("ar25", True, "object_motion_world_model", "none"),
        ("ft09", True, "config_rule_verifier", "none"),
    ]
    return {
        "experiment": "experiment_4459_loo_generic_solve_benchmark_v3",
        "honest_verdict": "success: generic_loo_solve_count_v3_6_of_7_beats_v2_5",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "generic_loo_solve_count_v3": 6,
        "heldout_games": list(V3_GAMES),
        "per_game": [
            {
                "game": game,
                "solved_without_own_recipe": solved,
                "closed_by_operator": operator,
                "residual_delta": residual,
            }
            for game, solved, operator, residual in rows
        ],
        "missing_verifier_gaps": [
            {
                "game": "sc25",
                "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                "retrieved_operator": "active_data_collection",
                "attempt_mode": "v3_412_operator_remeasurement",
            }
        ],
        "offline_reproduced": True,
        "verifier_is_oracle": True,
        "random_seed": 4459,
        "reproducibility_checksum": "1" * 64,
    }


def _cast_artifact(*, closed: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4469_generic_cast_grid_fsm_operator",
        "honest_verdict": "success: sc25_generic_cast_grid_fsm_L1_offline_reproduced",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "target_game": "sc25",
        "sc25_resolved_generically": closed,
        "sc25_generic_level_reproduced": 1 if closed else 0,
        "offline_reproduced": closed,
        "generic_operator_result": {
            "game": "sc25",
            "operator": "cast_grid_phase_fsm_world_model",
            "target_recipe_withheld": "sc25",
            "grounded": closed,
        },
        "reproduction_result": {
            "game": "sc25",
            "reached_level": 1 if closed else 0,
            "claimed_level": 1,
            "reproduced": closed,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "missing_verifier_gaps": [] if closed else [{"game": "sc25"}],
        "verifier_is_oracle": True,
    }


def _write_json(root: Path, rel_path: str, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixture_repo(root: Path, *, cast_closed: bool = True) -> str:
    for game in PUBLIC_GAMES:
        (root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version": 1,
        "games": [
            {"game": game, "reproducibility": "reproduced", "levels_reproduced": 1}
            for game in V3_GAMES
        ],
    }
    registry_text = yaml.safe_dump(registry, sort_keys=False)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(registry_text, encoding="utf-8")
    _write_json(root, mod.V3_RELATIVE_PATH, _v3_artifact())
    _write_json(root, mod.CAST_GRID_RELATIVE_PATH, _cast_artifact(closed=cast_closed))
    return registry_text


def _fake_variant_runner(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
    solved = game in {"lp85", "sc25", "tr87"}
    reached = 1 if solved else 0
    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": spec["variant"],
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": solved,
        "reached_level": reached,
        "actions": 3 if solved else 0,
        "reproduction_gate": {
            "game": game,
            "reached_level": reached,
            "claimed_level": reached,
            "reproduced": solved,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "blocked_reason": "",
    }


def test_req_report_4472_spec_declares_variant_transfer_contract() -> None:
    """REQ-REPORT-4472: OpenSpec declares the v4 benchmark and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4472" in spec
    assert "SCENARIO-REPORT-4472" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "generic_transfer_rate_over_variants" in spec
    assert "generic_loo_solve_count_v4" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4472_measures_variants_and_closes_sc25_loo(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4472: variants are tallied and exp4469 closes the v3 sc25 residual."""

    registry_before = _write_fixture_repo(tmp_path, cast_closed=True)
    clock = {"t": 20.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        variant_runner=_fake_variant_runner,
        color_variants=(1,),
        reflection_variants=(),
        budget=7,
        now=now,
        sleep_fn=sleep,
    )

    assert (tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8") == registry_before
    assert artifact["honest_verdict"] == (
        "success: generic_transfer_variants_3_of_25_rate_0.1200_loo_v4_7_beats_v3_6"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["variants_attempted"] == 25
    assert artifact["variants_solved"] == 3
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(3 / 25)
    assert artifact["generic_loo_solve_count_v4"] == 7
    assert artifact["generic_loo_solve_count_v3_baseline"] == 6
    assert artifact["offline_reproduced"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["no_3090_inference"] is True
    assert artifact["leaderboard_submission"] is False
    assert artifact["variant_plan"]["color_variants"] == [1]
    assert artifact["variant_plan"]["reflection_variants"] == []
    assert len(artifact["variant_attempts"]) == 25

    by_game = {row["game"]: row for row in artifact["per_game"]}
    assert by_game["sc25"] == {
        "game": "sc25",
        "variant_transfer_rate": 1.0,
        "loo_solved_without_own_recipe": True,
        "closed_by_operator": "cast_grid_phase_fsm_world_model",
        "residual_delta": "none",
    }
    assert by_game["lp85"]["variant_transfer_rate"] == 1.0
    assert by_game["bp35"]["variant_transfer_rate"] == 0.0
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["variants_attempted"] == 25
    assert len(written["reproducibility_checksum"]) == 64


def test_req_report_4472_absent_cast_evidence_keeps_v4_flat(tmp_path: Path) -> None:
    """REQ-REPORT-4472: missing .413 cast-grid evidence leaves sc25 as a residual."""

    _write_fixture_repo(tmp_path, cast_closed=False)

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        variant_runner=_fake_variant_runner,
        color_variants=(1,),
        reflection_variants=(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == (
        "complete: generic_transfer_variants_3_of_25_rate_0.1200_loo_v4_6_flat_vs_v3_6"
    )
    assert artifact["generic_loo_solve_count_v4"] == 6
    sc25 = next(row for row in artifact["per_game"] if row["game"] == "sc25")
    assert sc25["loo_solved_without_own_recipe"] is False
    assert sc25["closed_by_operator"] == "none"
    assert sc25["residual_delta"] == "missing_cast_grid_spell_shrink_tank_exit_verifier"
    assert artifact["missing_verifier_gaps"][0]["game"] == "sc25"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4472_preconditions_block_without_measurement(tmp_path: Path) -> None:
    """REQ-REPORT-4472: missing resources write terminal blocked artifacts."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "baseline_smoke_green": False, "ok": False},
        variant_runner=_fake_variant_runner,
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert artifact["honest_verdict"] == "complete: blocked_baseline_smoke"
    assert artifact["variants_attempted"] == 0
    assert artifact["generic_transfer_rate_over_variants"] == 0.0
    assert artifact["generic_loo_solve_count_v4"] == 0
    assert artifact["offline_reproduced"] is False
    assert mod.artifact_schema_errors(artifact) == []

    source_block = tmp_path / "source_block"
    for game in PUBLIC_GAMES:
        (source_block / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    missing_source = mod.run(
        root=source_block,
        preconditions_checked=_ok_preconditions(),
        variant_runner=_fake_variant_runner,
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert missing_source["honest_verdict"] == "complete: blocked_v3_source_artifact"
    assert mod.artifact_schema_errors(missing_source) == []


def test_req_report_4472_schema_rejects_malformed_or_fabricated_results(tmp_path: Path) -> None:
    """REQ-REPORT-4472: schema catches bad prefixes, non-bare fields, and ungated claims."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        variant_runner=_fake_variant_runner,
        color_variants=(1,),
        reflection_variants=(),
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    bad = {
        **artifact,
        "honest_verdict": "partial: invalid",
        "inference_substrate": None,
        "generic_transfer_rate_over_variants": {"principle": "wrapped"},
        "variants_attempted": "25",
        "variants_solved": "3",
        "generic_loo_solve_count_v4": "7",
        "generic_loo_solve_count_v3_baseline": 5,
        "per_game": [{"game": "x", "variant_transfer_rate": "1.0"}],
        "offline_reproduced": "true",
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "random_seed": "4472",
        "reproducibility_checksum": "bad",
        "duration_s": 0.0,
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": {"principle": "wrong"}},
        "variant_attempts": [
            {
                "game": "x",
                "attempted": True,
                "solved": True,
                "reproduction_gate": {"reproduced": False, "reached_level": 0},
            }
        ],
        "no_3090_inference": False,
        "leaderboard_submission": True,
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with complete:/success:/passed:/shipped:" in errors
    assert "inference_substrate must not be None" in errors
    assert "generic_transfer_rate_over_variants must be bare float" in errors
    assert "variants_attempted must be bare int" in errors
    assert "variants_solved must be bare int" in errors
    assert "generic_loo_solve_count_v4 must be bare int" in errors
    assert "generic_loo_solve_count_v3_baseline must be bare int = 6" in errors
    assert "per_game[0] missing loo_solved_without_own_recipe" in errors
    assert "variants_solved must match solved variant_attempts" in errors
    assert (
        "generic_transfer_rate_over_variants must equal variants_solved/variants_attempted"
        in errors
    )
    assert "solved variant_attempts must have reproduced gate evidence" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "cached verifier substrate requires duration_s >= 1.0" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4472" in errors
    assert "no_3090_inference must be true" in errors
    assert "leaderboard_submission must be false" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, bad)


def test_req_report_4472_defensive_helpers_are_deterministic(tmp_path: Path) -> None:
    """REQ-REPORT-4472: helper fallbacks and variant plans stay deterministic."""

    assert mod._environment_games(tmp_path / "missing") == set()
    (tmp_path / "env_probe" / "environment_files" / "zz99").mkdir(parents=True)
    assert mod._environment_games(tmp_path / "env_probe") == {"zz99"}
    for key, reason in (
        ("offline_env_files_present", "offline_env_files"),
        ("arc_variant_generator_import", "arc_variant_generator_import"),
        ("arc_solver_kit_import", "arc_solver_kit_import"),
        ("arc_solve_learning_import", "arc_solve_learning_import"),
        ("baseline_smoke_green", "baseline_smoke"),
    ):
        checks = _ok_preconditions()
        checks[key] = False
        assert mod.first_precondition_miss(checks) == reason
    assert mod.first_precondition_miss({**_ok_preconditions(), "no_3090_inference": False}) == (
        "no_3090_inference_policy"
    )
    assert mod.first_precondition_miss({**_ok_preconditions(), "leaderboard_submission": True}) == (
        "leaderboard_submission_policy"
    )

    plan = mod.manufactured_variant_specs(
        ["b", "a"], color_variants=(2, 1), reflection_variants=(3,)
    )
    assert [row["variant_signature"] for row in plan[:3]] == [
        "a~color01",
        "a~color02",
        "a~reflect03",
    ]
    assert mod._variant_transfer_rate(0, 0) == 0.0
    assert mod._variant_transfer_rate(1, 4) == 0.25
    assert mod._public_games({"offline_env_games": "bad"}) == []
    assert mod._public_games({"offline_env_games": list(PUBLIC_GAMES)}) == list(PUBLIC_GAMES)
    assert mod._verdict(0, 25, 0.0, 5) == (
        "complete: generic_transfer_variants_0_of_25_rate_0.0000_loo_v4_5_lower_than_v3_6"
    )
    assert mod._cast_grid_closes_sc25(None) is False
    assert mod._cast_grid_closes_sc25(_cast_artifact(closed=True)) is True
    assert mod._cast_grid_closes_sc25(_cast_artifact(closed=False)) is False
    assert mod._operator_result_matches(
        {
            "generic_operator_result": "bad",
            "generic_solve_result": {
                "operator_result": {
                    "game": "sc25",
                    "operator": "cast_grid_phase_fsm_world_model",
                    "target_recipe_withheld": "sc25",
                    "grounded": True,
                }
            },
        },
        game="sc25",
        operator="cast_grid_phase_fsm_world_model",
    )
    assert (
        mod._operator_result_matches(
            {"generic_operator_result": {"game": "sc25", "operator": "wrong"}},
            game="sc25",
            operator="cast_grid_phase_fsm_world_model",
        )
        is False
    )
    assert mod._heldout_games({"per_game": [{"game": "x"}]}) == ["x"]
    assert mod._heldout_games({}) == []
    assert mod._variant_summary_by_game([{"game": "", "attempted": True, "solved": True}]) == {}

    _write_fixture_repo(tmp_path)
    short_games = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "offline_env_games": ["lp85"]},
        variant_runner=_fake_variant_runner,
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert short_games["honest_verdict"] == "complete: blocked_public_game_count"

    assert "missing honest_verdict" in mod.artifact_schema_errors({})
    malformed = {
        **short_games,
        "honest_verdict": "complete: malformed_measurement",
        "variants_attempted": 1,
        "variants_solved": 1,
        "generic_transfer_rate_over_variants": 1.0,
        "per_game": [
            "bad",
            {
                "game": "x",
                "variant_transfer_rate": 1.0,
                "loo_solved_without_own_recipe": "true",
                "closed_by_operator": 7,
                "residual_delta": "none",
            },
            {
                "game": "y",
                "variant_transfer_rate": 0.0,
                "loo_solved_without_own_recipe": True,
                "closed_by_operator": "none",
                "residual_delta": "still_open",
            },
        ],
        "variant_attempts": ["bad"],
        "offline_reproduced": False,
    }
    malformed_errors = mod.artifact_schema_errors(malformed)
    assert "per_game[0] must be dict" in malformed_errors
    assert "per_game[1].loo_solved_without_own_recipe must be bare bool" in malformed_errors
    assert "per_game[1].closed_by_operator must be string" in malformed_errors
    assert "per_game[2] loo solved row requires closed_by_operator" in malformed_errors
    assert "per_game[2] loo solved row requires residual_delta none" in malformed_errors
    assert "generic_loo_solve_count_v4 must match solved per_game LOO rows" in malformed_errors
    assert "variants_attempted must be >= 25 for completed measurement" in malformed_errors
    assert "offline_reproduced false cannot accompany counted solves" in malformed_errors

    not_list_errors = mod.artifact_schema_errors({**short_games, "variant_attempts": "bad"})
    assert "variant_attempts must be list" in not_list_errors

    success_flat = {
        **short_games,
        "honest_verdict": "success: fabricated",
        "duration_s": 1.1,
        "generic_transfer_rate_over_variants": 0.0,
        "variants_attempted": 25,
        "variant_attempts": [],
        "per_game": [
            {
                "game": f"g{i}",
                "variant_transfer_rate": 0.0,
                "loo_solved_without_own_recipe": i < mod.V3_BASELINE,
                "closed_by_operator": "operator" if i < mod.V3_BASELINE else "none",
                "residual_delta": "none" if i < mod.V3_BASELINE else "not_in_plain_loo_benchmark",
            }
            for i in range(25)
        ],
        "generic_loo_solve_count_v4": mod.V3_BASELINE,
        "offline_reproduced": True,
    }
    assert "success verdict requires generic_loo_solve_count_v4 > 6" in mod.artifact_schema_errors(
        success_flat
    )
