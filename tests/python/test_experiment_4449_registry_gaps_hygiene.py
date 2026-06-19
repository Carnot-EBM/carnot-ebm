"""Tests for Exp 4449 registry and gaps hygiene.

Spec refs: REQ-REPORT-4449, SCENARIO-REPORT-4449.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

from carnot import experiment_4449_registry_gaps_hygiene as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_games() -> list[dict[str, Any]]:
    return [
        {"game": "r11l", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "ls20", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "wa30", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "s5i5", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "lp85", "reproducibility": "reproduced", "levels_reproduced": 5},
        {"game": "sc25", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "cd82", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "sp80", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "su15", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "tu93", "reproducibility": "reproduced", "levels_reproduced": 5},
        {"game": "tn36", "reproducibility": "reproduced", "levels_reproduced": 7},
        {"game": "cn04", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "m0r0", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "sk48", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "ar25", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "ka59", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "ft09", "reproducibility": "reproduced", "levels_reproduced": 1},
        {"game": "tr87", "reproducibility": "reproduced", "levels_reproduced": 6},
        {"game": "g50t", "reproducibility": "unsolved", "levels_reproduced": 0},
    ]


def _fixture_arc_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": "2026-06-19",
        "general_gotchas": [{"id": "offline_is_a_simulator", "note": "offline sim"}],
        "games": _fixture_games(),
        "reproducible_total_levels": 37,
        "reproducible_total_games": 18,
    }


def _fixture_verifier_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": mod.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "eval": {
                    "exp4438_regression_guard_passed": True,
                    "exp4438_arc1_rule_exec_vote_pass2": 0.4516,
                    "exp4438_arc1_rule_exec_gated_pass2": 0.5806,
                },
                "registry_roles": [{"role_id": "old", "status": "kept"}],
            }
        ]
    }


def _fixture_gaps_text() -> str:
    blocks = [
        (
            "exp4427-gap-4423-g50t-unselectable-first-contact",
            "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT",
            "open",
        ),
        (
            "exp4438-gap-4432-loo-ka59-missing-push-block-world-model-and-dynamic-selection",
            "GAP-4432-LOO-KA59-MISSING-PUSH-BLOCK-WORLD-MODEL-AND-DYNAMIC-SELECTION",
            "open",
        ),
        (
            "exp4438-gap-4432-loo-ar25-missing-reflection-world-model-and-object-motion-plan",
            "GAP-4432-LOO-AR25-MISSING-REFLECTION-WORLD-MODEL-AND-OBJECT-MOTION-PLAN",
            "open",
        ),
        (
            "exp4438-gap-4432-loo-ft09-missing-local-constraint-color-cycle-verifier",
            "GAP-4432-LOO-FT09-MISSING-LOCAL-CONSTRAINT-COLOR-CYCLE-VERIFIER",
            "open",
        ),
        (
            "exp4438-gap-4432-loo-tr87-missing-glyph-rewrite-rule-verifier-without-tr87-adapter",
            "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER",
            "open",
        ),
        (
            "exp4438-gap-4432-loo-sc25-missing-cast-grid-spell-shrink-tank-exit-verifier",
            "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
            "open",
        ),
        (
            "exp4438-gap-4423-dc22-unselectable-first-contact",
            "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
            "open",
        ),
    ]
    text = "# Verifier Gaps\n"
    for marker, gap_id, status in blocks:
        text += (
            f"\n<!-- {marker}:start -->\n"
            f"### {gap_id}: stale fixture row\n"
            f"- status: {status}\n"
            "- evidence: stale\n"
            "- failure mode: stale\n"
            "- missing discriminator: stale\n"
            "- candidate design: stale\n"
            "- priority: high\n"
            f"<!-- {marker}:end -->\n"
        )
    return text


def _write_fixture_repo(
    root: Path,
    *,
    omit_4447: bool = False,
    flagged_4445: bool = False,
) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / mod.ARC_REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_fixture_arc_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_fixture_verifier_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (root / mod.GAPS_RELATIVE_PATH).write_text(_fixture_gaps_text(), encoding="utf-8")

    _write_json(
        root / mod.EXP4443_PATH,
        {
            "experiment": "experiment_4443_bank_g50t_example_conditioned_win",
            "honest_verdict": "success: example_conditioned_g50t_L1_banked_with_correct_substrate",
            "target_game": "g50t",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reproducible_total_levels": 38,
            "reproducible_total_games": 19,
            "registry_update": {"gap_filled": True},
            "verifier_is_oracle": True,
            "reproducibility_checksum": "3" * 64,
        },
    )
    _write_json(
        root / mod.EXP4444_PATH,
        {
            "experiment": "experiment_4444_generic_config_rule_verifier_operator",
            "honest_verdict": "complete: ft09_generic_resolved_dc22_not_grounded_gap_logged",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "ft09_resolved_generically": True,
            "dc22_state": "not_grounded",
            "missing_verifier_gaps": [
                {
                    "gap_id": "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                    "game": "dc22",
                    "status": "open",
                    "residual_delta": "missing_config_rule_verifier_grounding",
                }
            ],
            "no_regression": True,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "4" * 64,
        },
    )
    _write_json(
        root / mod.EXP4445_PATH,
        {
            "experiment": "experiment_4445_generic_object_motion_world_model_operator",
            "honest_verdict": "success: ar25_ka59_object_motion_generic_L1_offline_reproduced",
            "flagged_adversarial": flagged_4445,
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "residuals_closed_generically": ["ar25", "ka59"],
            "world_model_accuracy_with_examples": 1.0,
            "world_model_accuracy_cold": 0.25,
            "missing_verifier_gaps": [],
            "no_regression": True,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "5" * 64,
        },
    )
    _write_json(
        root / mod.EXP4446_PATH,
        {
            "experiment": "experiment_4446_drive_generic_first_contact_bank",
            "honest_verdict": "success: generic_first_contact_vc33_L1_offline_reproduced",
            "target_game": "vc33",
            "routed_to": "s5i5",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "selected_operator": {"operator": "config_rule_verifier"},
            "missing_verifier_gaps": [],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "6" * 64,
        },
    )
    if not omit_4447:
        _write_json(
            root / mod.EXP4447_PATH,
            {
                "experiment": "experiment_4447_lilo_documented_primitive_library",
                "honest_verdict": "success: documented_primitive_library_retrieval_gate_passed",
                "library_coverage": 1.0,
                "retrieval_precision_at_1": 1.0,
                "constant_leak_violations": [],
                "primitives_documented": [
                    {
                        "name": "object_motion_world_model",
                        "mechanic_class": "object_motion_world_model",
                        "derived_from_games": ["ar25", "ka59"],
                    }
                ],
                "no_regression": True,
                "verifier_is_oracle": True,
                "reproducibility_checksum": "7" * 64,
            },
        )
    _write_json(
        root / mod.EXP4448_PATH,
        {
            "experiment": "experiment_4448_loo_generic_solve_benchmark_v2",
            "honest_verdict": "success: generic_loo_solve_count_v2_5_of_7_beats_v1_2",
            "generic_loo_solve_count_v1_baseline": 2,
            "generic_loo_solve_count_v2": 5,
            "loo_gate_passed": True,
            "offline_reproduced": True,
            "closed_residuals_by_new_operator": [
                {
                    "game": "ka59",
                    "closed_by_operator": "object_motion_world_model",
                    "v1_residual_delta": "missing_push_block_world_model_and_dynamic_selection",
                },
                {
                    "game": "ar25",
                    "closed_by_operator": "object_motion_world_model",
                    "v1_residual_delta": "missing_reflection_world_model_and_object_motion_plan",
                },
                {
                    "game": "ft09",
                    "closed_by_operator": "config_rule_verifier",
                    "v1_residual_delta": "missing_local_constraint_color_cycle_verifier",
                },
            ],
            "missing_verifier_gaps": [
                {
                    "game": "tr87",
                    "residual_delta": "missing_glyph_rewrite_rule_verifier_without_tr87_adapter",
                    "retrieved_operator": "config_rule_grounding",
                },
                {
                    "game": "sc25",
                    "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                    "retrieved_operator": "active_data_collection",
                },
            ],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "8" * 64,
        },
    )


def _guard_ok(_root: Path) -> dict[str, Any]:
    return {
        "regression_guard_passed": True,
        "arc_oracle_distinct_verifier_beats_vote": True,
        "current": {
            "vote_pass2": 0.4516,
            "gated_pass2": 0.5806,
            "headroom_recovered": 4,
            "vote_wins_lost": 0,
        },
    }


def _stamp_ok(_root: Path) -> dict[str, Any]:
    return {
        "capstone_stamp_fix_durable": True,
        "capstone_verifier_is_oracle_honored": True,
        "circular_moat_overclaim_fired": False,
    }


def test_req_report_4449_spec_declares_registry_hygiene_contract() -> None:
    """REQ-REPORT-4449: OpenSpec names the .411 hygiene contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-4449",
        "SCENARIO-REPORT-4449",
        mod.RESULT_RELATIVE_PATH,
        "regression_guard_passed",
        "reproducible_total_levels",
        "scripts.capstone_aggregate_available",
        "aggregation_from_upstream_artifacts -- reads upstream artifacts; 100us floor",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4449_reconciles_411_ledgers_and_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4449: .411 closures, totals, guard, and stamp are reconciled."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        tmp_path,
        gap4_guard_runner=_guard_ok,
        capstone_stamp_runner=_stamp_ok,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "complete: registry_gaps_hygiene_4449_guard_passed"
    assert type(artifact["regression_guard_passed"]) is bool
    assert artifact["regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_durable"] is True
    assert artifact["reproducible_total_levels"] == 39
    assert artifact["reproducible_total_games"] == 20
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == 0.0001
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["no_production_verifier_edits"] is True
    assert artifact["field_principles"]["honest_verdict"]["principle"] == "terminal-prefixed"
    assert (
        artifact["field_principles"]["regression_guard_passed"]["principle"]
        == "BARE bool (gated-fields-must-be-bare): the GAP-4 execution result did not regress"
    )
    assert (
        artifact["field_principles"]["inference_substrate"]["principle"]
        == "aggregation_from_upstream_artifacts -- reads upstream artifacts; 100us floor"
    )
    assert artifact["availability_report"]["axes"]["g50t_bank"]["verdict"] is True
    assert artifact["availability_report"]["axes"]["loo_v2"]["verdict"] == 5
    assert artifact["registry_reconciliation"]["filled_gap_ids"] == [
        "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT",
        "GAP-4432-LOO-FT09-MISSING-LOCAL-CONSTRAINT-COLOR-CYCLE-VERIFIER",
        "GAP-4432-LOO-AR25-MISSING-REFLECTION-WORLD-MODEL-AND-OBJECT-MOTION-PLAN",
        "GAP-4432-LOO-KA59-MISSING-PUSH-BLOCK-WORLD-MODEL-AND-DYNAMIC-SELECTION",
        "GAP-4423-VC33-UNSELECTABLE-FIRST-CONTACT",
    ]

    arc_registry = yaml.safe_load((tmp_path / mod.ARC_REGISTRY_RELATIVE_PATH).read_text())
    assert arc_registry["reproducible_total_levels"] == 39
    assert arc_registry["reproducible_total_games"] == 20
    assert next(row for row in arc_registry["games"] if row["game"] == "g50t")[
        "levels_reproduced"
    ] == 1
    assert next(row for row in arc_registry["games"] if row["game"] == "vc33")[
        "levels_reproduced"
    ] == 1
    assert arc_registry["latest_hygiene_4449"]["artifact"] == mod.RESULT_RELATIVE_PATH

    gaps = (tmp_path / mod.GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT: Exp 4449 .411 registry gap hygiene" in gaps
    assert "- status: filled (exp4443_bank_g50t_example_conditioned_win)" in gaps
    assert "- status: open\n- evidence: results/experiment_4444_generic_config_rule_verifier_operator.json; dc22_state=not_grounded" in gaps
    assert "GAP-4423-VC33-UNSELECTABLE-FIRST-CONTACT: Exp 4449 .411 registry gap hygiene" in gaps
    assert "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER" in gaps
    assert "results/experiment_4448_loo_generic_solve_benchmark_v2.json; game=sc25" in gaps

    verifier_registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text())
    verifier = verifier_registry["verifiers"][0]
    assert verifier["eval"]["eval_exp_4449"] == mod.RESULT_RELATIVE_PATH
    assert verifier["eval"]["exp4449_regression_guard_passed"] is True
    assert verifier["eval"]["exp4449_reproducible_total_levels"] == 39
    assert verifier["registry_roles"][-1]["role_id"] == mod.V411_ROLE_ID

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert mod.artifact_schema_errors(artifact) == []
    assert importlib.import_module("scripts.capstone_aggregate_available").AxisSpec is not None


def test_req_report_4449_aggregate_available_reports_gaps_without_erasing_axes(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4449: missing/flagged inputs report gaps while other axes compute."""

    _write_fixture_repo(tmp_path, omit_4447=True, flagged_4445=True)

    artifact = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)

    assert artifact["availability_report"]["axes"]["primitive_library"]["missing_artifacts"] == [
        {"axis": "primitive_library", "artifact_key": "4447_primitive_library", "experiment_id": 4447}
    ]
    assert artifact["availability_report"]["axes"]["object_motion"]["flagged_artifacts"] == [
        {
            "axis": "object_motion",
            "artifact_key": "4445_object_motion",
            "experiment_id": 4445,
            "reason": "flagged_adversarial",
        }
    ]
    assert artifact["availability_report"]["axes"]["g50t_bank"]["verdict"] is True
    assert artifact["registry_reconciliation"]["registries_reconciled"] is True
    assert "GAP-4432-LOO-AR25-MISSING-REFLECTION-WORLD-MODEL-AND-OBJECT-MOTION-PLAN" not in artifact[
        "registry_reconciliation"
    ]["filled_gap_ids"]
    assert mod.EXP4445_PATH in artifact["excluded_artifacts"]


def test_req_report_4449_schema_and_guard_helpers_reject_bad_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-4449: bare fields, guard logic, and precondition helpers are defensive."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)
    second_artifact = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)
    assert second_artifact["registry_reconciliation"]["filled_gap_ids"] == artifact[
        "registry_reconciliation"
    ]["filled_gap_ids"]

    bad = {
        **artifact,
        "honest_verdict": "partial: not terminal",
        "regression_guard_passed": "true",
        "reproducible_total_levels": True,
        "reproducible_total_games": "20",
        "capstone_stamp_fix_durable": "true",
        "inference_substrate": "",
        "submitted_to_leaderboard": True,
        "random_seed": "4449",
        "reproducibility_checksum": "bad",
        "registry_reconciliation": [],
        "availability_report": [],
        "field_principles": {"honest_verdict": {"principle": "wrong"}},
        "spec_refs": [],
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with complete:/success:/passed:/shipped:" in errors
    assert "regression_guard_passed must be bare bool" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "reproducible_total_games must be bare int" in errors
    assert "capstone_stamp_fix_durable must be bare bool" in errors
    assert "inference_substrate must equal aggregation_from_upstream_artifacts" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "random_seed must be bare int" in errors
    assert "registry_reconciliation must be dict" in errors
    assert "availability_report must be dict" in errors
    assert "field_principles.regression_guard_passed.principle must match REQ-REPORT-4449" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "spec_refs must include REQ-REPORT-4449 and SCENARIO-REPORT-4449" in errors
    with pytest.raises(ValueError, match="regression_guard_passed"):
        mod.write_artifact(tmp_path, bad)
    assert "missing honest_verdict" in mod.artifact_schema_errors({})
    assert "field_principles must be dict" in mod.artifact_schema_errors({**artifact, "field_principles": []})

    assert mod._as_int("bad") == 0
    assert mod._as_float("bad") == 0.0
    assert mod.collect_gap_entries(
        {
            "4448_loo_v2": {
                "closed_residuals_by_new_operator": [
                    None,
                    {
                        "closed_by_operator": "config_rule_verifier",
                        "game": "ft09",
                        "v1_residual_delta": "missing_local_constraint_color_cycle_verifier",
                    },
                ],
                "missing_verifier_gaps": [
                    None,
                    {"game": "", "residual_delta": "missing"},
                    {"game": "x", "residual_delta": ""},
                ],
            }
        }
    ) == []
    assert mod._find_game({"games": {}}, "g50t") is None
    assert mod._find_game({"games": [{"game": "x"}]}, "g50t") is None
    assert mod._reproduced_counts({"games": {}}) == (0, 0)
    assert mod._reproduced_counts({"games": [None, {"reproducibility": "reproduced", "levels_reproduced": 2}]}) == (
        2,
        1,
    )
    assert mod._library_rows({"primitives_documented": {}}) == []
    assert mod._find_verifier({"verifiers": {}}, mod.GAP4_VERIFIER_ID) is None
    assert mod._find_verifier({"verifiers": [{"verifier_id": "other"}]}, mod.GAP4_VERIFIER_ID) is None
    assert mod._guard_current({"replayed_arc1_rule_exec": {"vote_pass2": 0.4516}})["vote_pass2"] == 0.4516
    assert mod._guard_current({}) == {}
    repo_root = str(mod.REPO_ROOT)
    original_sys_path = list(sys.path)
    sys.path[:] = [entry for entry in sys.path if entry != repo_root]
    try:
        assert mod._check_helper_import()["ok"] is True
    finally:
        sys.path[:] = original_sys_path
    created_registry, _ = mod.reconcile_verifier_registry(
        {},
        guard=_guard_ok(tmp_path),
        stamp=_stamp_ok(tmp_path),
        total_levels=39,
        total_games=20,
        filled_gap_ids=[],
        open_gap_ids=[],
        trusted={},
        excluded=[],
    )
    assert created_registry["verifiers"][0]["verifier_id"] == mod.GAP4_VERIFIER_ID
    bad_roles_registry, _ = mod.reconcile_verifier_registry(
        {"verifiers": [{"verifier_id": mod.GAP4_VERIFIER_ID, "registry_roles": "bad", "eval": {}}]},
        guard=_guard_ok(tmp_path),
        stamp=_stamp_ok(tmp_path),
        total_levels=39,
        total_games=20,
        filled_gap_ids=[],
        open_gap_ids=[],
        trusted={},
        excluded=[],
    )
    assert isinstance(bad_roles_registry["verifiers"][0]["registry_roles"], list)
    assert mod.guard_passed(
        {
            "regression_guard_passed": True,
            "arc_oracle_distinct_verifier_beats_vote": True,
            "current": {"gated_pass2": 0.4, "vote_pass2": 0.5, "vote_wins_lost": 0},
        }
    ) is False
    assert mod.guard_passed(
        {
            "gap4_execution_guard_passed": True,
            "current": {"gated_pass2": 0.6, "vote_pass2": 0.5, "vote_wins_lost": 1},
        }
    ) is False
    assert mod.stamp_fix_durable(
        {"capstone_stamp_fix_verified": True, "circular_moat_overclaim_fired": False}
    ) is True
    assert mod._yaml_mapping(tmp_path / "missing.yaml")[1]["readable"] is False
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("[]\n", encoding="utf-8")
    assert mod._yaml_mapping(list_yaml)[1]["error"] == "top-level YAML is not a mapping"
    assert mod._read_text(tmp_path / "missing.md")[1]["readable"] is False
    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    assert mod._load_json(list_json)[1]["error"] == "top-level JSON is not an object"
