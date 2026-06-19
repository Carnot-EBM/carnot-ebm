"""Tests for Exp 4461 registry and gaps hygiene.

Spec refs: REQ-REPORT-4461, SCENARIO-REPORT-4461.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4461_registry_gaps_hygiene as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_games() -> list[dict[str, Any]]:
    reproduced = [
        ("r11l", 1),
        ("ls20", 1),
        ("wa30", 1),
        ("s5i5", 1),
        ("lp85", 5),
        ("sc25", 1),
        ("cd82", 1),
        ("sp80", 1),
        ("su15", 1),
        ("tu93", 5),
        ("tn36", 7),
        ("cn04", 1),
        ("m0r0", 1),
        ("sk48", 1),
        ("ar25", 1),
        ("ka59", 1),
        ("ft09", 1),
        ("tr87", 6),
        ("g50t", 1),
        ("vc33", 1),
    ]
    rows = [
        {"game": game, "reproducibility": "reproduced", "levels_reproduced": levels}
        for game, levels in reproduced
    ]
    rows.extend(
        [
            {"game": "dc22", "reproducibility": "unsolved", "levels_reproduced": 0},
            {"game": "sb26", "reproducibility": "unsolved", "levels_reproduced": 0},
        ]
    )
    return rows


def _fixture_arc_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": "2026-06-19",
        "general_gotchas": [],
        "games": _fixture_games(),
        "reproducible_total_levels": 39,
        "reproducible_total_games": 20,
    }


def _fixture_verifier_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": mod.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "eval": {"exp4449_regression_guard_passed": True},
                "registry_roles": [{"role_id": "old", "status": "kept"}],
            }
        ]
    }


def _fixture_gaps_text() -> str:
    blocks = [
        (
            "exp4438-gap-4423-dc22-unselectable-first-contact",
            "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
        ),
        (
            "exp4438-gap-4432-loo-tr87-missing-glyph-rewrite-rule-verifier-without-tr87-adapter",
            "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER",
        ),
        (
            "exp4438-gap-4432-loo-sc25-missing-cast-grid-spell-shrink-tank-exit-verifier",
            "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
        ),
        (
            "exp4458-gap-sb26-color-match-slot-sequence",
            "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE",
        ),
    ]
    text = "# Verifier Gaps\n"
    for marker, gap_id in blocks:
        text += (
            f"\n<!-- {marker}:start -->\n"
            f"### {gap_id}: stale fixture row\n"
            "- status: open\n"
            "- evidence: stale\n"
            "- failure mode: stale\n"
            "- missing discriminator: stale\n"
            "- candidate design: stale\n"
            "- priority: high\n"
            f"<!-- {marker}:end -->\n"
        )
    return text


def _dc22_artifact(*, reproduced: bool) -> dict[str, Any]:
    return {
        "experiment": "experiment_4455_solve_dc22_cegis_config_rule",
        "honest_verdict": (
            "success: dc22_cegis_config_rule_L1_offline_reproduced"
            if reproduced
            else "blocked_baseline_pytest_coverage"
        ),
        "target_game": "dc22",
        "dc22_grounded": reproduced,
        "offline_reproduced": reproduced,
        "reproduced_levels": 1 if reproduced else 0,
        "reproducible_total_levels": 40 if reproduced else 39,
        "verifier_is_oracle": reproduced,
        "reproducibility_checksum": "5" * 64,
    }


def _glyph_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4456_generic_glyph_rewrite_operator",
        "honest_verdict": "success: tr87_generic_glyph_rewrite_L1_offline_reproduced",
        "target_game": "tr87",
        "tr87_resolved_generically": True,
        "tr87_generic_level_reproduced": 1,
        "offline_reproduced": True,
        "generic_operator_result": {
            "game": "tr87",
            "operator": "glyph_rewrite_rule_verifier",
            "target_recipe_withheld": "tr87",
            "grounded": True,
        },
        "generic_reproduction_result": {"game": "tr87", "reproduced": True, "reached_level": 1},
        "missing_verifier_gaps": [],
        "verifier_is_oracle": True,
        "reproducibility_checksum": "6" * 64,
    }


def _cast_grid_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4457_cast_grid_phase_fsm_world_model",
        "honest_verdict": "success: sc25_cast_grid_phase_fsm_L2_offline_reproduced",
        "target_game": "sc25",
        "sc25_resolved_generically": True,
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "generic_operator_result": {
            "game": "sc25",
            "operator": "cast_grid_phase_fsm_world_model",
            "target_recipe_withheld": "sc25",
            "grounded": True,
        },
        "generic_reproduction_result": {"game": "sc25", "reproduced": True, "reached_level": 2},
        "missing_verifier_gaps": [],
        "verifier_is_oracle": True,
        "reproducibility_checksum": "7" * 64,
    }


def _first_contact_artifact(*, banked: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_4458_first_contact_new_game",
        "honest_verdict": (
            "success: generic_first_contact_sb26_L1_offline_reproduced"
            if banked
            else "complete: generic_first_contact_sb26_routed_no_new_level"
        ),
        "target_game": "sb26",
        "routed_to": "s5i5",
        "offline_reproduced": banked,
        "reproduced_levels": 1 if banked else 0,
        "missing_verifier_gaps": []
        if banked
        else [
            {
                "gap_id": "GAP-4458-SB26-MISSING-COLOR-MATCH-SLOT-SEQUENCE-VERIFIER",
                "game": "sb26",
                "residual_delta": "missing_color_match_slot_sequence_verifier",
                "operator": "config_rule_verifier",
                "routed_to": "s5i5",
                "candidate_design": "add a generic ordered color-match item-slot verifier",
            }
        ],
        "verifier_is_oracle": True,
        "reproducibility_checksum": "8" * 64,
    }


def _loo_v3_artifact(*, count: int) -> dict[str, Any]:
    sc25_solved = count >= 7
    return {
        "experiment": "experiment_4459_loo_generic_solve_benchmark_v3",
        "honest_verdict": f"success: generic_loo_solve_count_v3_{count}_of_7_beats_v2_5",
        "generic_loo_solve_count_v3": count,
        "generic_loo_solve_count_v2_baseline": 5,
        "closed_residuals_by_412_operator": [
            {
                "game": "tr87",
                "closed_by_operator": "glyph_rewrite_rule_verifier",
                "v2_residual_delta": "missing_glyph_rewrite_rule_verifier_without_tr87_adapter",
            }
        ]
        + (
            [
                {
                    "game": "sc25",
                    "closed_by_operator": "cast_grid_phase_fsm_world_model",
                    "v2_residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                }
            ]
            if sc25_solved
            else []
        ),
        "per_game": [
            {
                "game": "tr87",
                "solved_without_own_recipe": True,
                "closed_by_operator": "glyph_rewrite_rule_verifier",
                "residual_delta": "none",
            },
            {
                "game": "sc25",
                "solved_without_own_recipe": sc25_solved,
                "closed_by_operator": "cast_grid_phase_fsm_world_model" if sc25_solved else "none",
                "residual_delta": "none" if sc25_solved else "missing_cast_grid_spell_shrink_tank_exit_verifier",
            },
        ],
        "offline_reproduced": True,
        "missing_verifier_gaps": []
        if sc25_solved
        else [
            {
                "game": "sc25",
                "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                "attempt_mode": "v3_412_operator_remeasurement",
            }
        ],
        "verifier_is_oracle": True,
        "reproducibility_checksum": "9" * 64,
    }


def _submission_artifact(*, total: int) -> dict[str, Any]:
    return {
        "experiment": "experiment_4460_submission_package_prep",
        "honest_verdict": f"success: submission_package_ready_{total}_levels_beats_13_quarantined_0",
        "submission_package_ready": True,
        "total_reproduced_levels_in_package": total,
        "prior_submitted_baseline_levels": 13,
        "beats_prior_baseline": True,
        "submitted_to_leaderboard": False,
        "verifier_is_oracle": True,
        "reproducibility_checksum": "a" * 64,
    }


def _write_fixture_repo(
    root: Path,
    *,
    dc22_reproduced: bool,
    cast_artifact: bool,
    loo_count: int,
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

    _write_json(root / mod.EXP4455_PATH, _dc22_artifact(reproduced=dc22_reproduced))
    _write_json(root / mod.EXP4456_PATH, _glyph_artifact())
    if cast_artifact:
        _write_json(root / "results/experiment_4457_cast_grid_phase_fsm_world_model.json", _cast_grid_artifact())
    _write_json(root / mod.EXP4458_PATH, _first_contact_artifact(banked=False))
    _write_json(root / mod.EXP4459_PATH, _loo_v3_artifact(count=loo_count))
    _write_json(root / mod.EXP4460_PATH, _submission_artifact(total=41 if dc22_reproduced else 39))


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


def test_req_report_4461_spec_declares_registry_hygiene_contract() -> None:
    """REQ-REPORT-4461: OpenSpec names the .412 hygiene contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-4461",
        "SCENARIO-REPORT-4461",
        mod.RESULT_RELATIVE_PATH,
        "regression_guard_passed",
        "reproducible_total_levels",
        "open_gap_ids",
        "aggregation_from_upstream_artifacts -- reads upstream artifacts; 100us floor",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4461_reconciles_412_ledgers_and_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4461: .412 closures, totals, guard, and stamp are reconciled."""

    _write_fixture_repo(tmp_path, dc22_reproduced=True, cast_artifact=True, loo_count=7)

    artifact = mod.run(
        tmp_path,
        gap4_guard_runner=_guard_ok,
        capstone_stamp_runner=_stamp_ok,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "complete: registry_gaps_hygiene_4461_guard_passed"
    assert type(artifact["regression_guard_passed"]) is bool
    assert artifact["regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_durable"] is True
    assert artifact["reproducible_total_levels"] == 41
    assert artifact["reproducible_total_games"] == 21
    assert artifact["open_gap_ids"] == ["GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE"]
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == 0.0001
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["no_production_verifier_edits"] is True
    assert artifact["field_principles"]["honest_verdict"]["principle"] == "terminal-prefixed"
    assert (
        artifact["field_principles"]["reproducible_total_levels"]["principle"]
        == "the reconciled authoritative count (target >= 40 after dc22 + sc25 deepening)"
    )
    assert artifact["availability_report"]["axes"]["dc22_bank"]["verdict"] is True
    assert artifact["availability_report"]["axes"]["cast_grid"]["verdict"]["sc25_levels"] == 2
    assert artifact["availability_report"]["axes"]["loo_v3"]["verdict"] == 7
    assert artifact["registry_reconciliation"]["filled_gap_ids"] == [
        "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
        "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER",
        "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
    ]

    arc_registry = yaml.safe_load((tmp_path / mod.ARC_REGISTRY_RELATIVE_PATH).read_text())
    assert arc_registry["reproducible_total_levels"] == 41
    assert arc_registry["reproducible_total_games"] == 21
    assert next(row for row in arc_registry["games"] if row["game"] == "dc22")[
        "levels_reproduced"
    ] == 1
    assert next(row for row in arc_registry["games"] if row["game"] == "sc25")[
        "levels_reproduced"
    ] == 2
    assert arc_registry["latest_hygiene_4461"]["artifact"] == mod.RESULT_RELATIVE_PATH

    gaps = (tmp_path / mod.GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT: Exp 4461 .412 registry gap hygiene" in gaps
    assert "- status: filled (exp4455_solve_dc22_cegis_config_rule)" in gaps
    assert "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE: Exp 4461 .412 registry gap hygiene" in gaps
    assert "- movement: updated_still_open" in gaps

    verifier_registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text())
    verifier = verifier_registry["verifiers"][0]
    assert verifier["eval"]["eval_exp_4461"] == mod.RESULT_RELATIVE_PATH
    assert verifier["eval"]["exp4461_regression_guard_passed"] is True
    assert verifier["eval"]["exp4461_reproducible_total_levels"] == 41
    assert verifier["registry_roles"][-1]["role_id"] == mod.V412_ROLE_ID

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert mod.artifact_schema_errors(artifact) == []
    assert importlib.import_module("scripts.capstone_aggregate_available").AxisSpec is not None


def test_req_report_4461_missing_or_blocked_inputs_do_not_erase_available_axes(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4461: missing/blocked inputs report gaps while other axes compute."""

    _write_fixture_repo(tmp_path, dc22_reproduced=False, cast_artifact=False, loo_count=6)

    artifact = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)

    assert artifact["reproducible_total_levels"] == 39
    assert artifact["reproducible_total_games"] == 20
    assert artifact["availability_report"]["axes"]["cast_grid"]["missing_artifacts"] == [
        {"axis": "cast_grid", "artifact_key": "4457_cast_grid", "experiment_id": 4457}
    ]
    assert artifact["availability_report"]["axes"]["glyph_rewrite"]["verdict"] is True
    assert artifact["availability_report"]["axes"]["loo_v3"]["verdict"] == 6
    assert artifact["registry_reconciliation"]["filled_gap_ids"] == [
        "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER"
    ]
    assert artifact["open_gap_ids"] == [
        "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
        "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
        "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE",
    ]
    assert mod.EXP4457_PATTERN in artifact["model_specs"]["upstream_artifacts"]
    assert artifact["excluded_artifacts"] == []


def test_req_report_4461_schema_and_defensive_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-4461: bare fields, guard logic, and schema checks are defensive."""

    _write_fixture_repo(tmp_path, dc22_reproduced=True, cast_artifact=True, loo_count=7)
    artifact = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)
    second = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)
    assert second["registry_reconciliation"]["filled_gap_ids"] == artifact[
        "registry_reconciliation"
    ]["filled_gap_ids"]

    bad = {
        **artifact,
        "honest_verdict": "partial: not terminal",
        "regression_guard_passed": "true",
        "reproducible_total_levels": True,
        "reproducible_total_games": "21",
        "open_gap_ids": ["ok", 3],
        "capstone_stamp_fix_durable": "true",
        "inference_substrate": "",
        "submitted_to_leaderboard": True,
        "random_seed": "4461",
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
    assert "open_gap_ids must be list[str]" in errors
    assert "capstone_stamp_fix_durable must be bare bool" in errors
    assert "inference_substrate must equal aggregation_from_upstream_artifacts" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "random_seed must be bare int" in errors
    assert "registry_reconciliation must be dict" in errors
    assert "availability_report must be dict" in errors
    assert "field_principles.regression_guard_passed.principle must match REQ-REPORT-4461" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "spec_refs must include REQ-REPORT-4461 and SCENARIO-REPORT-4461" in errors
    with pytest.raises(ValueError, match="regression_guard_passed"):
        mod.write_artifact(tmp_path, bad)
    assert "missing honest_verdict" in mod.artifact_schema_errors({})
    assert "field_principles must be dict" in mod.artifact_schema_errors({**artifact, "field_principles": []})
    assert mod.guard_passed(
        {
            "regression_guard_passed": True,
            "arc_oracle_distinct_verifier_beats_vote": True,
            "current": {"gated_pass2": 0.4, "vote_pass2": 0.5, "vote_wins_lost": 0},
        }
    ) is False
    assert mod.stamp_fix_durable({"capstone_stamp_fix_verified": True}) is True
    assert mod._sc25_levels(None) == 0
    assert mod._sc25_levels({"reproduced_levels": 3}) == 3
    assert mod._sc25_cast_grid_closed(None) is False
    assert mod._loo_closes_game({}, "sc25") is False
    assert mod._loo_closes_game(
        {
            "offline_reproduced": True,
            "per_game": [
                {
                    "game": "sc25",
                    "solved_without_own_recipe": True,
                    "closed_by_operator": "cast_grid_phase_fsm_world_model",
                    "residual_delta": "none",
                }
            ],
        },
        "sc25",
    ) is True

    open_tr87 = mod.collect_gap_entries({"4456_glyph_rewrite": {"offline_reproduced": False}}, {})
    assert any(
        gap["gap_id"]
        == "GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER"
        and gap["status"] == "open"
        for gap in open_tr87
    )
    banked_gap_entries = mod.collect_gap_entries(
        {"4458_first_contact": _first_contact_artifact(banked=True)},
        {},
    )
    assert any(
        gap["gap_id"] == "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE"
        and gap["status"] == "filled (exp4458_first_contact_new_game)"
        for gap in banked_gap_entries
    )
    banked_arc, _ = mod.reconcile_arc_registry(
        _fixture_arc_registry(),
        {"4458_first_contact": _first_contact_artifact(banked=True)},
        {},
        filled_gap_ids=[],
        open_gap_ids=[],
        excluded=[],
    )
    assert next(row for row in banked_arc["games"] if row["game"] == "sb26")[
        "levels_reproduced"
    ] == 1
