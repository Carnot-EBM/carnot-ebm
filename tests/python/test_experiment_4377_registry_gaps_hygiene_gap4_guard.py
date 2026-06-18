"""Tests for Exp 4377 registry/gaps hygiene, GAP-4 guard, and stamp durability.

Spec refs: REQ-VERIFY-4377, SCENARIO-VERIFY-4377.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4377_registry_gaps_hygiene_gap4_guard as wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4377 as exp4377


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
RESULTS_WRAPPER_PATH = Path("results/experiment_4377_registry_gaps_hygiene_gap4_guard.py")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4377.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "eval": {"metric": "pass_at_2"},
                "registry_roles": [],
            },
            {
                "verifier_id": exp4377.ACTION_COST_VERIFIER_ID,
                "domain": "arc_agi3_interactive",
                "version": 1,
                "kind": "search_heuristic",
                "eval": {"eval_exp_4366": exp4377.EXP4366_PATH},
            },
            {
                "verifier_id": exp4377.FOVER_VERIFIER_ID,
                "domain": "math_reasoning",
                "version": 4,
                "kind": "ensemble",
                "eval": {"metric": "fover_dual_condition_auroc"},
            },
        ]
    }


def _minimal_arc_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": "2026-06-17",
        "games": [
            {"game": "lp85", "reproducibility": "reproduced", "levels_reproduced": 4},
            {"game": "tn36", "reproducibility": "reproduced", "levels_reproduced": 7},
            {"game": "ft09", "reproducibility": "reproduced", "levels_reproduced": 1},
        ],
        "reproducible_total_levels": 33,
        "reproducible_total_games": 17,
        "provisional_total_levels": 5,
    }


def _minimal_arc_registry_text() -> str:
    return (
        '# fixture comment\n'
        'schema_version: 1\n'
        'updated: "2026-06-17"\n'
        'games:\n'
        '  - game: lp85\n'
        '    reproducibility: reproduced\n'
        '    levels_reproduced: 4   # old\n'
        '    solver: "old solver"\n'
        '    reproduce: "old reproduce"\n'
        '\n'
        '  - game: sc25\n'
        '    reproducibility: reproduced\n'
        '    levels_reproduced: 1\n'
        '#   reproduced levels = r11l 1 + lp85 4 + ft09 1 = 33 across 17 games\n'
        'reproducible_total_levels: 33   # old total\n'
        'reproducible_total_games: 17\n'
        'provisional_total_levels: 5\n'
    )


def _minimal_gaps_text() -> str:
    return (
        "# Verifier Gaps\n\n"
        "<!-- exp4366-gap-e3-world-model-rule-sc25-l2-4361:start -->\n"
        "### GAP-E3-WORLD-MODEL-RULE-SC25-L2-4361: Exp 4366 .403 verifier gap update\n"
        "- status: open\n"
        "- evidence: old sc25 residual.\n"
        "- failure mode: sc25 old\n"
        "- missing discriminator: sc25 old\n"
        "- candidate design: old design\n"
        "- priority: high\n"
        "<!-- exp4366-gap-e3-world-model-rule-sc25-l2-4361:end -->\n\n"
        "<!-- exp4366-gap-e3-world-model-rule-tn36-l8-4361:start -->\n"
        "### GAP-E3-WORLD-MODEL-RULE-TN36-L8-4361: Exp 4366 .403 verifier gap update\n"
        "- status: open\n"
        "- evidence: old tn36 residual.\n"
        "- failure mode: tn36 old\n"
        "- missing discriminator: tn36 old\n"
        "- candidate design: old design\n"
        "- priority: high\n"
        "<!-- exp4366-gap-e3-world-model-rule-tn36-l8-4361:end -->\n\n"
        "<!-- exp4366-gap-e3-world-model-rule-lp85-l5-4361:start -->\n"
        "### GAP-E3-WORLD-MODEL-RULE-LP85-L5-4361: Exp 4366 .403 verifier gap update\n"
        "- status: open\n"
        "- evidence: old lp85 residual.\n"
        "- failure mode: lp85 old\n"
        "- missing discriminator: lp85 old\n"
        "- candidate design: old design\n"
        "- priority: high\n"
        "<!-- exp4366-gap-e3-world-model-rule-lp85-l5-4361:end -->\n\n"
        "<!-- exp4366-gap-e3-world-model-rule-ar25-l2-4362:start -->\n"
        "### GAP-E3-WORLD-MODEL-RULE-AR25-L2-4362: Exp 4366 .403 verifier gap update\n"
        "- status: open\n"
        "- evidence: old ar25 residual.\n"
        "- failure mode: ar25 old\n"
        "- missing discriminator: ar25 old\n"
        "- candidate design: old design\n"
        "- priority: high\n"
        "<!-- exp4366-gap-e3-world-model-rule-ar25-l2-4362:end -->\n\n"
        "<!-- exp4366-gap-e3-world-model-rule-ka59-l2-4362:start -->\n"
        "### GAP-E3-WORLD-MODEL-RULE-KA59-L2-4362: Exp 4366 .403 verifier gap update\n"
        "- status: open\n"
        "- evidence: old ka59 residual.\n"
        "- failure mode: ka59 old\n"
        "- missing discriminator: ka59 old\n"
        "- candidate design: old design\n"
        "- priority: high\n"
        "<!-- exp4366-gap-e3-world-model-rule-ka59-l2-4362:end -->\n"
    )


def _minimal_v404_payloads() -> dict[str, dict[str, Any]]:
    return {
        exp4377.EXP4370_PATH: {
            "honest_verdict": "complete: clean_powered_null_linear_not_beaten",
            "acceptance_gate_passed": True,
            "llm_heuristic_beats_linear": False,
            "static_leakage_clean": True,
            "reproduction_gated": True,
            "n_held_out_levels": 9,
            "held_out_actions_by_heuristic": {
                "bfs_baseline": 646,
                "linear": 646,
                "llm_generated": 646,
            },
            "missing_verifier_gaps": [
                {
                    "gap_id": "GAP-4370",
                    "failure_mode": "no clean LLM-generated heuristic reduced actions",
                    "missing_discriminator": "shorter-plan feature",
                    "candidate_design": "richer transition features",
                    "priority": "medium",
                }
            ],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:b9678da39c175a78c7079f67905b265e890dc814d1947e5cf3ba93ec1974f769",
        },
        exp4377.EXP4371_PATH: {
            "experiment": 4371,
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "gate_check_summary": "exp4370.llm_heuristic_beats_linear failed",
            "gates_evaluated": [{"passed": False}],
        },
        exp4377.EXP4372_PATH: {
            "honest_verdict": "success_e3_deeper_lp85_reproduced",
            "new_levels_reproduced": 1,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
            "per_target_scorecard": [
                {
                    "game": "lp85",
                    "prior_best_level": 4,
                    "new_reproduced_level": 5,
                    "offline_reproduced": True,
                    "residual_win_mechanic_gap_class": "none",
                    "verifier_accuracy": 1.0,
                    "world_model_path": "python/carnot/agentic/arc_game_adapters.py",
                },
                {
                    "game": "tn36",
                    "prior_best_level": 7,
                    "new_reproduced_level": 7,
                    "offline_reproduced": False,
                    "residual_win_mechanic_gap_class": "tn36_l8_program_editor_object_control_gap",
                    "verifier_accuracy": 0.875,
                    "world_model_path": "scripts/arc3_tn36_offline_solver.py",
                },
                {
                    "game": "sc25",
                    "prior_best_level": 1,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_win_mechanic_gap_class": "sc25_l2_spell_delta_gap",
                    "verifier_accuracy": 1.0,
                    "world_model_path": "results/arc_e3/sc25/world_model.py",
                },
            ],
            "reproducibility_checksum": "41b38325905de2fb0d6eb707701bc2635bb4e9a0a0d1a914e566f06f3c3b17bf",
        },
        exp4377.EXP4373_PATH: {
            "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 33,
            "verifier_is_oracle": True,
            "per_game_scorecard": [
                {
                    "game": "ar25",
                    "prior_best_level": 1,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_gap_class": "ar25_l2_action7_undo_stack_hidden_rule_gap",
                    "verifier_accuracy": 0.958333,
                    "mechanic_checks_passed": False,
                    "plan_action_count": 15,
                    "world_model_path": "results/arc_e3/ar25/world_model.py",
                    "world_model_sha256": "395b5d9bee22d00991e24ba08919993eadf245d401b9d92d34b17303e315202c",
                },
                {
                    "game": "ka59",
                    "prior_best_level": 1,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_gap_class": "ka59_l2_hidden_step_counter_hud_register_gap",
                    "verifier_accuracy": 0.15625,
                    "mechanic_checks_passed": False,
                    "plan_action_count": 11,
                    "world_model_path": "results/arc_e3/ka59/world_model.py",
                    "world_model_sha256": "e20305436149d9bf0490980036a6b0e20c26503f0d3fd09573552b8789ff76bf",
                },
                {
                    "game": "ft09",
                    "prior_best_level": 1,
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "residual_gap_class": "ft09_l2_residual_world_model_mismatch_gap",
                    "verifier_accuracy": 0.15625,
                    "mechanic_checks_passed": False,
                    "plan_action_count": 4,
                    "world_model_path": "results/arc_e3/ft09/world_model.py",
                    "world_model_sha256": "8e4563289295741b6728075a411d0aa4ae7165140d2f3e5e5e885312dcf650ee",
                },
            ],
            "reproducibility_checksum": "4c775367e4b09eaf060085f9df2e1617bb79af96c87fcd0996ef15bce2b79310",
        },
        exp4377.EXP4374_PATH: {
            "honest_verdict": "retired_in_generation_conversion_unmeasurable",
            "acceptance_gate": True,
            "s3_guided_beats_control": False,
            "controls_differentiated": False,
            "codila_control_differentiates": False,
            "scorer_requalified_leak_clean": False,
            "s3_minus_best_of_n_delta": 0.0,
            "s3_gain_ci95": [0.0, 0.0],
            "benchmark_n": 0,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "038463fa613c1cb3bbde0212d915b5b55fdff754325e36de55d377de27f20c17",
        },
        exp4377.EXP4375_PATH: {
            "honest_verdict": "complete: detector_beats_chance_zero_selection_headroom_fover",
            "detector_beats_chance": True,
            "detector_auroc": 0.918304,
            "detector_auroc_ci95": [0.909296, 0.926923],
            "n_candidates": 8829,
            "selection_headroom": {
                "headroom": 0.0,
                "oracle_at_k": 0.812097,
                "vote_at_1": 0.812097,
            },
            "per_verifier_auroc": {
                "fr11_session_memory": 0.871382,
                "tier0r_curry_howard": 0.901686,
            },
            "missing_verifier_gaps": [],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:fc6a9ba3db67c9f2c2d7c9290bef13af467850f9aba93e0c7b1ad659498c59ac",
        },
    }


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "adversarial_verify.py").write_text("# fixture\n", encoding="utf-8")
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        _minimal_arc_registry_text(),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text(
        _minimal_gaps_text(),
        encoding="utf-8",
    )
    for rel_path, payload in _minimal_v404_payloads().items():
        _write_json(tmp_path / rel_path, payload)
    _write_json(
        tmp_path / exp4377.CAPSTONE_V403_PATH,
        {
            "honest_verdict": "complete: v403_fixture",
            "verifier_is_oracle": False,
            "verifier_is_oracle_honored": True,
        },
    )


def _guard() -> dict[str, Any]:
    return {
        "regression_guard_passed": True,
        "replayed_arc1_rule_exec": {
            "n": 31,
            "vote_pass2": 0.4516,
            "gated_pass2": 0.5806,
            "headroom_recovered": 4,
            "vote_wins_lost": 0,
        },
    }


def _stamp_report() -> dict[str, Any]:
    return {
        "capstone_stamp_fix_durable": True,
        "capstone_path": exp4377.CAPSTONE_V403_PATH,
        "capstone_verifier_is_oracle": False,
        "capstone_aggregation_propagates_oracle_stamp": True,
        "capstone_aggregation_uses_available_helper": True,
        "circular_moat_overclaim_fired": False,
        "flag_count": 0,
        "returncode": 0,
        "flags": [],
    }


def test_req_verify_4377_spec_declared() -> None:
    """REQ-VERIFY-4377: OpenSpec declares the .404 hygiene guard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4377",
        "SCENARIO-VERIFY-4377",
        "python/carnot/experiment_4377_registry_gaps_hygiene_gap4_guard.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4377.EXP4377_ARTIFACT_PATH,
        "blocked_<file>_unreadable",
        "capstone_stamp_fix_durable",
        "CIRCULAR_MOAT_OVERCLAIM",
        "reproducible_total_levels=34",
        "GAP-4370",
        "detector_auroc=0.918304",
    ):
        assert marker in spec
    for field in exp4377.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp4377.FIELD_PRINCIPLES.values():
        assert principle in spec
    assert wrapper.main is exp4377.main


def test_scenario_verify_4377_ledgers_record_v404_truth(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4377: .404 outcomes reconcile ledgers without pruning."""

    _write_minimal_repo(tmp_path)
    outcomes = exp4377.load_v404_outcomes(tmp_path)
    gap_entries = exp4377.build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = exp4377.ensure_ledgers_record_v404(
        _minimal_registry(),
        _minimal_gaps_text(),
        _minimal_arc_registry(),
        _guard(),
        outcomes,
        gap_entries,
    )

    assert exp4377.registry_contains_v404(registry) is True
    assert exp4377.arc_registry_contains_v404(arc_registry) is True
    assert exp4377.gaps_contain_v404(gaps_text, gap_entries) is True
    assert summary["registries_reconciled"] is True
    assert exp4377.GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361 in summary["filled_gap_ids"]
    assert "status: filled (exp4372_lp85_l5_world_model)" in gaps_text
    assert "GAP-E3-WORLD-MODEL-RULE-FT09-L2-4373" in gaps_text
    assert "GAP-4370" in gaps_text

    gap4 = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4377.GAP4_VERIFIER_ID)
    assert gap4["eval"]["exp4377_arc_reproducible_total_levels"] == 34
    assert gap4["eval"]["exp4377_diffusiongemma_status"] == (
        "retired_in_generation_conversion_unmeasurable"
    )
    action = next(
        row for row in registry["verifiers"] if row["verifier_id"] == exp4377.ACTION_COST_VERIFIER_ID
    )
    assert action["eval"]["exp4377_llm_heuristic_beats_linear"] is False
    assert action["eval"]["exp4377_held_out_actions_by_heuristic"]["linear"] == 646
    fover = next(row for row in registry["verifiers"] if row["verifier_id"] == exp4377.FOVER_VERIFIER_ID)
    assert fover["eval"]["exp4377_detector_beats_chance"] is True
    assert fover["eval"]["exp4377_detector_auroc"] == 0.918304

    lp85 = exp4377._find_game(arc_registry, "lp85")
    assert lp85 is not None
    assert lp85["levels_reproduced"] == 5
    assert arc_registry["reproducible_total_levels"] == 34


def test_req_verify_4377_run_hygiene_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-4377: terminal artifact exposes bare guard/stamp fields."""

    _write_minimal_repo(tmp_path)
    artifact = exp4377.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    exp4377.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gap4_regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_durable"] is True
    assert artifact["registries_reconciled"] is True
    assert artifact["v404_outcomes"]["llm_generated_action_cost"]["llm_heuristic_beats_linear"] is False
    assert artifact["v404_outcomes"]["verifier_as_detector"]["detector_beats_chance"] is True

    written = json.loads(
        (tmp_path / exp4377.EXP4377_ARTIFACT_PATH).read_text(encoding="utf-8")
    )
    assert written["honest_verdict"] == artifact["honest_verdict"]
    for field in exp4377.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4377.validate_artifact(malformed)
    with pytest.raises(ValueError, match="gap4_regression_guard_passed"):
        exp4377.validate_artifact({**artifact, "gap4_regression_guard_passed": "true"})
    with pytest.raises(ValueError, match="capstone_stamp_fix_durable"):
        exp4377.validate_artifact({**artifact, "capstone_stamp_fix_durable": 1})
    with pytest.raises(ValueError, match="registries_reconciled"):
        exp4377.validate_artifact({**artifact, "registries_reconciled": None})

    fallback_repo = tmp_path / "fallback"
    _write_minimal_repo(fallback_repo)
    (fallback_repo / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(_minimal_arc_registry(), sort_keys=False),
        encoding="utf-8",
    )
    fallback_artifact = exp4377.run_hygiene(
        fallback_repo,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )
    assert fallback_artifact["registries_reconciled"] is True


def test_req_verify_4377_blocks_unreadable_ledgers(tmp_path: Path) -> None:
    """REQ-VERIFY-4377: unreadable registries fail closed before mutation."""

    _write_minimal_repo(tmp_path)
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text("[]\n", encoding="utf-8")

    artifact = exp4377.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    assert artifact["honest_verdict"] == "blocked_arc_solve_registry_unreadable"
    assert artifact["gap4_regression_guard_passed"] is False
    assert artifact["capstone_stamp_fix_durable"] is False
    assert artifact["registries_reconciled"] is False


def test_req_verify_4377_defensive_readers_and_stamp_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4377: malformed inputs are recorded and stamp durability is audited."""

    assert exp4377._load_optional_json(tmp_path, "missing.json")[0] is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert "JSONDecodeError" in exp4377._load_optional_json(tmp_path, "bad.json")[1]
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp4377._load_optional_json(tmp_path, "list.json")[1] == (
        "top-level JSON is not an object"
    )

    assert exp4377._bool(None, "x") is None
    assert exp4377._bool({"x": 1}, "x") is None
    assert exp4377._int(None, "x") == 0
    assert exp4377._int({"x": True}, "x") == 0
    assert exp4377._float(None, "x") is None
    assert exp4377._float({"x": True}, "x") is None
    assert exp4377._str(None, "x") == ""
    assert exp4377._list(None, "x") == []
    assert exp4377._scorecard_map(["bad", {"game": ""}], "residual") == {}
    assert exp4377._read_llm_heuristic(None, "missing")["available"] is False
    assert exp4377._read_skeptic(None, "missing")["available"] is False
    assert exp4377._read_deeper(None, "missing")["targets"] == {}
    assert exp4377._read_blocked_mechanics(None, "missing")["games"] == {}
    assert exp4377._read_diffusiongemma(None, "missing")["available"] is False
    assert exp4377._read_detector(None, "missing")["available"] is False

    missing_repo = tmp_path / "missing_repo"
    preflight = exp4377.check_preconditions(missing_repo)
    assert preflight["ok"] is False
    assert preflight["blocked_file"] == "verifier_registry"
    no_gaps_repo = tmp_path / "no_gaps_repo"
    (no_gaps_repo / "ops").mkdir(parents=True)
    (no_gaps_repo / "ops" / "verifier_registry.yaml").write_text("{}\n", encoding="utf-8")
    (no_gaps_repo / "ops" / "arc_solve_registry.yaml").write_text("{}\n", encoding="utf-8")
    no_gaps_preflight = exp4377.check_preconditions(no_gaps_repo)
    assert no_gaps_preflight["blocked_file"] == "verifier_gaps"

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(
        exp4377.exp4366,
        "run_gap4_regression_guard",
        lambda root: {"regression_guard_passed": True, "root": str(root)},
    )
    assert exp4377.run_gap4_regression_guard(tmp_path)["regression_guard_passed"] is True

    outcomes = exp4377.load_v404_outcomes(tmp_path)
    with_bad_gap = json.loads(json.dumps(outcomes))
    with_bad_gap["llm_generated_action_cost"]["missing_verifier_gaps"].insert(0, "bad")
    assert "GAP-4370" in [gap["gap_id"] for gap in exp4377.build_gap_entries(with_bad_gap)]

    arc_text = (
        '# header comment\nupdated: "2026-06-17"\n\n'
        "games:\n"
        "  - game: lp85\n"
        "    reproducibility: reproduced\n"
        "    levels_reproduced: 4   # old\n"
        '    solver: "old solver"\n'
        '    reproduce: "old reproduce"\n'
        "\n"
        "  - game: sc25\n"
        "    reproducibility: reproduced\n"
        "#   reproduced levels = r11l 1 + lp85 4 + ft09 1 = 33 across 17 games\n"
        "reproducible_total_levels: 33   # old total\n"
    )
    patched_arc_text = exp4377._patch_arc_registry_text(arc_text, outcomes)
    assert patched_arc_text.startswith("# header comment")
    assert 'updated: "2026-06-18"' in patched_arc_text
    assert "levels_reproduced: 5" in patched_arc_text
    assert "lp85 5 +" in patched_arc_text
    assert "reproducible_total_levels: 34" in patched_arc_text
    no_lp85 = json.loads(json.dumps(outcomes))
    no_lp85["arc_e3"]["deeper_high_headroom"]["targets"]["lp85"]["offline_reproduced"] = False
    assert "levels_reproduced: 5" not in exp4377._patch_arc_registry_text(arc_text, no_lp85)
    existing_gap_text = "### GAP-4370: already present\n"
    assert exp4377._replace_or_append_gap(
        existing_gap_text,
        "exp4377-gap-4370",
        {
            "gap_id": "GAP-4370",
            "status": "open",
            "evidence": "fixture",
            "failure_mode": "fixture",
            "missing_discriminator": "fixture",
            "candidate_design": "fixture",
            "priority": "medium",
        },
    ) == existing_gap_text

    skipped = json.loads(json.dumps(outcomes))
    skipped["arc_e3"]["deeper_high_headroom"]["targets"]["tn36"]["offline_reproduced"] = True
    skipped["arc_e3"]["blocked_mechanics"]["games"].pop("ka59")
    assert exp4377.GAP_E3_WORLD_MODEL_RULE_TN36_L8_4361 not in [
        gap["gap_id"] for gap in exp4377.build_gap_entries(skipped)
    ]
    registry: dict[str, Any] = {"verifiers": []}
    gaps = exp4377.build_gap_entries(outcomes)
    exp4377._ensure_gap4_eval(registry, _guard(), outcomes, gaps)
    assert exp4377.GAP4_VERIFIER_ID in [row["verifier_id"] for row in registry["verifiers"]]
    exp4377._ensure_v404_role({"verifiers": []}, outcomes, gaps)
    exp4377._ensure_action_cost_verifier({"verifiers": []}, outcomes)
    exp4377._ensure_fover_detector({"verifiers": []}, outcomes)
    arc: dict[str, Any] = {"games": []}
    assert exp4377._find_game(arc, "missing") is None
    assert exp4377._ensure_game(arc, "new") == {"game": "new"}
    assert exp4377._flags_from_report({}) == []
    assert exp4377._flags_from_report({"reports": [{}]}) == []
    assert exp4377._capstone_aggregation_propagates_oracle_stamp() is True
    assert exp4377._capstone_aggregation_uses_available_helper() is True

    missing_capstone = tmp_path / "missing_capstone"
    _write_minimal_repo(missing_capstone)
    (missing_capstone / exp4377.CAPSTONE_V403_PATH).unlink()
    missing_stamp = exp4377.verify_capstone_stamp_fix_durable(missing_capstone)
    assert missing_stamp["capstone_stamp_fix_durable"] is False
    assert "error" in missing_stamp

    completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=0,
        stdout=json.dumps({"reports": [{"flags": [], "flag_count": 0}], "flagged_count": 0}),
        stderr="",
    )
    monkeypatch.setattr(exp4377.subprocess, "run", lambda *args, **kwargs: completed)
    stamp = exp4377.verify_capstone_stamp_fix_durable(tmp_path)
    assert stamp["capstone_stamp_fix_durable"] is True

    bad_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout=json.dumps(
            {
                "reports": [
                    {
                        "flags": [
                            {
                                "kind": "CIRCULAR_MOAT_OVERCLAIM",
                                "severity": "critical",
                                "detail": "fixture",
                            }
                        ],
                        "flag_count": 1,
                    }
                ],
                "flagged_count": 1,
            }
        ),
        stderr="",
    )
    monkeypatch.setattr(exp4377.subprocess, "run", lambda *args, **kwargs: bad_completed)
    bad_stamp = exp4377.verify_capstone_stamp_fix_durable(tmp_path)
    assert bad_stamp["capstone_stamp_fix_durable"] is False
    assert bad_stamp["circular_moat_overclaim_fired"] is True

    invalid_completed = subprocess.CompletedProcess(
        args=["python", "scripts/adversarial_verify.py"],
        returncode=1,
        stdout="not json",
        stderr="broken",
    )
    monkeypatch.setattr(exp4377.subprocess, "run", lambda *args, **kwargs: invalid_completed)
    invalid_stamp = exp4377.verify_capstone_stamp_fix_durable(tmp_path)
    assert invalid_stamp["capstone_stamp_fix_durable"] is False
    assert invalid_stamp["flags"] == []


def test_req_verify_4377_artifact_schema_rejects_malformed_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4377: schema validation rejects malformed terminal artifacts."""

    _write_minimal_repo(tmp_path)
    artifact = exp4377.run_hygiene(
        tmp_path,
        gap4_guard_runner=lambda _: _guard(),
        capstone_stamp_runner=lambda _: _stamp_report(),
    )

    for field in (
        "preconditions_checked",
        "v404_outcomes",
        "registry_reconciliation",
        "gap4_regression_guard",
        "capstone_stamp_fix",
    ):
        malformed = dict(artifact)
        malformed[field] = []
        with pytest.raises(ValueError, match=field):
            exp4377.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4377.validate_artifact({**artifact, "honest_verdict": "not_terminal"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4377.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="random_seed"):
        exp4377.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="field_principles"):
        exp4377.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4377.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4377"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4377.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_verify_4377_results_entrypoint_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4377: results entrypoint calls the package runner."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4377, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp4377, "run_gap4_regression_guard", lambda _root: _guard())
    monkeypatch.setattr(exp4377, "verify_capstone_stamp_fix_durable", lambda _root: _stamp_report())
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads(
        (tmp_path / exp4377.EXP4377_ARTIFACT_PATH).read_text(encoding="utf-8")
    )
    assert payload["gap4_regression_guard_passed"] is True
    assert payload["capstone_stamp_fix_durable"] is True
    assert payload["registries_reconciled"] is True
