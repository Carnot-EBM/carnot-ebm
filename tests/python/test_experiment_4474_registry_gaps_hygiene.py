"""Tests for Exp 4474 registry and gaps hygiene.

Spec refs: REQ-REPORT-4474, SCENARIO-REPORT-4474.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4474_registry_gaps_hygiene as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RE86_GAP_ID = "GAP-4471-RE86-MISSING-PATTERN-MATCH-SPRITE-RESIZE-VERIFIER"
SB26_GAP_ID = "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE"
SC25_GAP_ID = "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER"
DC22_GAP_ID = "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT"


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
            {"game": "re86", "reproducibility": "unsolved", "levels_reproduced": 0},
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
        "provisional_total_levels": 5,
    }


def _fixture_verifier_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": mod.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "eval": {"exp4461_regression_guard_passed": True},
                "registry_roles": [{"role_id": "old", "status": "kept"}],
            }
        ]
    }


def _fixture_gaps_text() -> str:
    blocks = [
        ("exp4438-gap-4423-dc22-unselectable-first-contact", DC22_GAP_ID),
        (
            "exp4438-gap-4432-loo-sc25-missing-cast-grid-spell-shrink-tank-exit-verifier",
            SC25_GAP_ID,
        ),
        ("exp4458-gap-sb26-color-match-slot-sequence", SB26_GAP_ID),
        ("exp4471-gap-re86-pattern-match-sprite-resize", RE86_GAP_ID),
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


def _dc22_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4467_solve_dc22_cegis_nocov",
        "honest_verdict": "success: dc22_cegis_L1_offline_reproduced",
        "target_game": "dc22",
        "dc22_grounded": True,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "reproducible_total_levels": 40,
        "reproducible_total_games": 21,
        "submitted_to_leaderboard": False,
        "verifier_is_oracle": True,
        "reproducibility_checksum": "6" * 64,
    }


def _sc25_deep_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4468_bank_sc25_provisional_levels",
        "honest_verdict": "success: sc25_L5_offline_reproduced_banked_4_new_levels",
        "target_game": "sc25",
        "offline_reproduced": True,
        "new_sc25_levels_reproduced": 4,
        "sc25_levels_reproduced_total": 5,
        "reproduced_levels": 4,
        "reproducible_total_levels": 44,
        "reproducible_total_games": 21,
        "submitted_to_leaderboard": False,
        "verifier_is_oracle": True,
        "reproducibility_checksum": "7" * 64,
    }


def _sc25_operator_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4469_generic_cast_grid_fsm_operator",
        "honest_verdict": "success: sc25_generic_cast_grid_fsm_L1_offline_reproduced",
        "target_game": "sc25",
        "sc25_resolved_generically": True,
        "sc25_generic_level_reproduced": 1,
        "offline_reproduced": True,
        "missing_verifier_gaps": [],
        "submitted_to_leaderboard": False,
        "verifier_is_oracle": True,
        "reproducibility_checksum": "8" * 64,
    }


def _sb26_artifact(*, flagged: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_4470_color_match_slot_operator_solve_sb26",
        "honest_verdict": "success: sb26_color_match_slot_sequence_L1_offline_reproduced",
        "target_game": "sb26",
        "color_match_operator_built": True,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "counterexample_rounds": 2,
        "reproducible_total_levels": 45,
        "submitted_to_leaderboard": False,
        "missing_verifier_gaps": [],
        "flagged_adversarial": flagged,
        "verifier_is_oracle": True,
        "reproducibility_checksum": "9" * 64,
    }


def _first_contact_artifact(*, banked: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_4471_first_contact_rotated_new_game",
        "honest_verdict": (
            "success: generic_first_contact_re86_L1_offline_reproduced"
            if banked
            else "complete: generic_first_contact_re86_routed_no_new_level"
        ),
        "target_game": "re86",
        "routed_to": "tu93",
        "offline_reproduced": banked,
        "reproduced_levels": 1 if banked else 0,
        "reproducible_total_levels": 46 if banked else 45,
        "submitted_to_leaderboard": False,
        "missing_verifier_gaps": []
        if banked
        else [
            {
                "gap_id": RE86_GAP_ID,
                "game": "re86",
                "operator": "graph_astar_action_cost",
                "residual_delta": "missing_pattern_match_sprite_resize_verifier",
                "routed_to": "tu93",
                "candidate_design": "add a generic sprite-overlay pattern-match and resize/transformation verifier",
                "status": "open",
            }
        ],
        "verifier_is_oracle": True,
        "reproducibility_checksum": "a" * 64,
    }


def _variant_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4472_variant_generic_transfer_benchmark_v4",
        "honest_verdict": "success: variant_transfer_v4_remeasured",
        "generic_transfer_rate_over_variants": 0.2,
        "variants_attempted": 25,
        "variants_solved": 5,
        "generic_loo_solve_count_v4": 7,
        "generic_loo_solve_count_v3_baseline": 6,
        "offline_reproduced": True,
        "per_game": [
            {
                "game": "sc25",
                "loo_solved_without_own_recipe": True,
                "closed_by_operator": "cast_grid_phase_fsm_world_model",
                "residual_delta": "none",
            }
        ],
        "missing_verifier_gaps": [],
        "verifier_is_oracle": True,
        "reproducibility_checksum": "b" * 64,
    }


def _submission_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4473_submission_package_prep_refresh",
        "honest_verdict": "success: submission_package_ready_45_levels_beats_13_grew_vs_412_quarantined_0",
        "submission_package_ready": True,
        "total_reproduced_levels_in_package": 45,
        "grew_vs_412": True,
        "submitted_to_leaderboard": False,
        "verifier_is_oracle": True,
        "reproducibility_checksum": "c" * 64,
    }


def _write_fixture_repo(
    root: Path,
    *,
    flagged_sb26: bool = False,
    include_variant: bool = True,
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

    _write_json(root / mod.EXP4467_PATH, _dc22_artifact())
    _write_json(root / mod.EXP4468_PATH, _sc25_deep_artifact())
    _write_json(root / mod.EXP4469_PATH, _sc25_operator_artifact())
    _write_json(root / mod.EXP4470_PATH, _sb26_artifact(flagged=flagged_sb26))
    _write_json(root / mod.EXP4471_PATH, _first_contact_artifact())
    if include_variant:
        _write_json(root / mod.EXP4472_PATH, _variant_artifact())
    _write_json(root / mod.EXP4473_PATH, _submission_artifact())


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


def test_req_report_4474_spec_declares_registry_hygiene_contract() -> None:
    """REQ-REPORT-4474: OpenSpec names the .413 hygiene contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-4474",
        "SCENARIO-REPORT-4474",
        mod.RESULT_RELATIVE_PATH,
        "regression_guard_passed",
        "provisional_total_levels",
        "open_gap_ids",
        "aggregation_from_upstream_artifacts -- reads upstream artifacts; 100us floor",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4474_reconciles_413_ledgers_and_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4474: .413 closures, totals, guard, and stamp are reconciled."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        tmp_path,
        gap4_guard_runner=_guard_ok,
        capstone_stamp_runner=_stamp_ok,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "complete: registry_gaps_hygiene_4474_guard_passed"
    assert type(artifact["regression_guard_passed"]) is bool
    assert artifact["regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_durable"] is True
    assert artifact["reproducible_total_levels"] == 45
    assert artifact["reproducible_total_games"] == 22
    assert artifact["provisional_total_levels"] == 1
    assert artifact["open_gap_ids"] == [RE86_GAP_ID]
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == 0.0001
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["no_production_verifier_edits"] is True
    assert artifact["field_principles"]["honest_verdict"]["principle"] == "terminal-prefixed"
    assert (
        artifact["field_principles"]["provisional_total_levels"]["principle"]
        == "the reconciled provisional count (target < 5 after sc25 deeper levels move provisional -> reproduced)"
    )
    assert artifact["availability_report"]["axes"]["dc22_bank"]["verdict"] is True
    assert artifact["availability_report"]["axes"]["sc25_deeper_bank"]["verdict"] == {
        "new_levels": 4,
        "total_levels": 5,
        "moved_from_provisional": True,
    }
    assert artifact["availability_report"]["axes"]["variant_transfer_loo_v4"]["verdict"] == {
        "generic_loo_solve_count_v4": 7,
        "variants_attempted": 25,
        "variants_solved": 5,
    }
    assert artifact["registry_reconciliation"]["filled_gap_ids"] == [
        DC22_GAP_ID,
        SC25_GAP_ID,
        SB26_GAP_ID,
    ]

    arc_registry = yaml.safe_load((tmp_path / mod.ARC_REGISTRY_RELATIVE_PATH).read_text())
    assert arc_registry["reproducible_total_levels"] == 45
    assert arc_registry["reproducible_total_games"] == 22
    assert arc_registry["provisional_total_levels"] == 1
    assert (
        next(row for row in arc_registry["games"] if row["game"] == "dc22")["levels_reproduced"]
        == 1
    )
    assert (
        next(row for row in arc_registry["games"] if row["game"] == "sc25")["levels_reproduced"]
        == 5
    )
    assert (
        next(row for row in arc_registry["games"] if row["game"] == "sb26")["levels_reproduced"]
        == 1
    )
    assert arc_registry["latest_hygiene_4474"]["artifact"] == mod.RESULT_RELATIVE_PATH

    gaps = (tmp_path / mod.GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert f"{DC22_GAP_ID}: Exp 4474 .413 registry gap hygiene" in gaps
    assert "- status: filled (experiment_4467_solve_dc22_cegis_nocov)" in gaps
    assert f"{SC25_GAP_ID}: Exp 4474 .413 registry gap hygiene" in gaps
    assert "- status: filled (experiment_4469_generic_cast_grid_fsm_operator)" in gaps
    assert f"{RE86_GAP_ID}: Exp 4474 .413 registry gap hygiene" in gaps
    assert "- movement: updated_still_open" in gaps

    verifier_registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text())
    verifier = verifier_registry["verifiers"][0]
    assert verifier["eval"]["eval_exp_4474"] == mod.RESULT_RELATIVE_PATH
    assert verifier["eval"]["exp4474_regression_guard_passed"] is True
    assert verifier["eval"]["exp4474_reproducible_total_levels"] == 45
    assert verifier["eval"]["exp4474_provisional_total_levels"] == 1
    assert verifier["registry_roles"][-1]["role_id"] == mod.V413_ROLE_ID

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert mod.artifact_schema_errors(artifact) == []
    assert importlib.import_module("scripts.capstone_aggregate_available").AxisSpec is not None


def test_req_report_4474_flagged_or_missing_inputs_do_not_erase_available_axes(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4474: flagged/missing inputs report gaps while other axes compute."""

    _write_fixture_repo(tmp_path, flagged_sb26=True, include_variant=False)

    artifact = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)

    assert artifact["reproducible_total_levels"] == 44
    assert artifact["reproducible_total_games"] == 21
    assert artifact["provisional_total_levels"] == 1
    assert artifact["availability_report"]["axes"]["sb26_bank"]["flagged_artifacts"] == [
        {
            "axis": "sb26_bank",
            "artifact_key": "4470_sb26",
            "experiment_id": 4470,
            "reason": "flagged_adversarial",
        }
    ]
    assert artifact["availability_report"]["axes"]["variant_transfer_loo_v4"][
        "missing_artifacts"
    ] == [
        {
            "axis": "variant_transfer_loo_v4",
            "artifact_key": "4472_variant_loo_v4",
            "experiment_id": 4472,
        }
    ]
    assert artifact["availability_report"]["axes"]["dc22_bank"]["verdict"] is True
    assert artifact["availability_report"]["axes"]["sc25_generic_operator"]["verdict"] is True
    assert artifact["registry_reconciliation"]["filled_gap_ids"] == [DC22_GAP_ID, SC25_GAP_ID]
    assert artifact["open_gap_ids"] == [SB26_GAP_ID, RE86_GAP_ID]
    assert mod.EXP4472_PATH in artifact["model_specs"]["upstream_artifacts"]
    assert artifact["excluded_artifacts"] == [mod.EXP4470_PATH]


def test_req_report_4474_schema_and_defensive_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-4474: bare fields, guard logic, and schema checks are defensive."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)
    second = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)
    assert (
        second["registry_reconciliation"]["filled_gap_ids"]
        == artifact["registry_reconciliation"]["filled_gap_ids"]
    )

    bad = {
        **artifact,
        "honest_verdict": "partial: not terminal",
        "regression_guard_passed": "true",
        "reproducible_total_levels": True,
        "reproducible_total_games": "22",
        "provisional_total_levels": "1",
        "open_gap_ids": ["ok", 3],
        "capstone_stamp_fix_durable": "true",
        "inference_substrate": "",
        "submitted_to_leaderboard": True,
        "random_seed": "4474",
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
    assert "provisional_total_levels must be bare int" in errors
    assert "open_gap_ids must be list[str]" in errors
    assert "capstone_stamp_fix_durable must be bare bool" in errors
    assert "inference_substrate must equal aggregation_from_upstream_artifacts" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "random_seed must be bare int" in errors
    assert "registry_reconciliation must be dict" in errors
    assert "availability_report must be dict" in errors
    assert "field_principles.regression_guard_passed.principle must match REQ-REPORT-4474" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "spec_refs must include REQ-REPORT-4474 and SCENARIO-REPORT-4474" in errors
    with pytest.raises(ValueError, match="regression_guard_passed"):
        mod.write_artifact(tmp_path, bad)
    assert "missing honest_verdict" in mod.artifact_schema_errors({})
    assert "field_principles must be dict" in mod.artifact_schema_errors(
        {**artifact, "field_principles": []}
    )
    assert (
        mod.guard_passed(
            {
                "regression_guard_passed": True,
                "arc_oracle_distinct_verifier_beats_vote": True,
                "current": {"gated_pass2": 0.4, "vote_pass2": 0.5, "vote_wins_lost": 0},
            }
        )
        is False
    )
    assert mod.stamp_fix_durable({"capstone_stamp_fix_verified": True}) is True
    assert mod._sc25_deeper_levels(None) == (0, 0)
    assert mod._sc25_deeper_levels(
        {"new_sc25_levels_reproduced": 2, "sc25_levels_reproduced_total": 3}
    ) == (2, 3)
    assert mod._game_levels({}, "sc25") == 0
    assert mod._game_levels({"games": [{"game": "dc22", "levels_reproduced": 1}]}, "sc25") == 0
    assert mod._dc22_banked({}) is False
    assert mod._sb26_banked({}) is False
    assert mod._first_contact_banked(_first_contact_artifact(banked=True)) is True

    open_entries = mod.collect_gap_entries(
        {"4470_sb26": {**_sb26_artifact(), "offline_reproduced": False}},
        {},
    )
    assert any(gap["gap_id"] == SB26_GAP_ID and gap["status"] == "open" for gap in open_entries)
    banked_entries = mod.collect_gap_entries(
        {"4471_first_contact": _first_contact_artifact(banked=True)},
        {},
    )
    assert any(
        gap["gap_id"] == RE86_GAP_ID
        and gap["status"] == "filled (experiment_4471_first_contact_rotated_new_game)"
        for gap in banked_entries
    )
    banked_arc, report = mod.reconcile_arc_registry(
        _fixture_arc_registry(),
        {
            "4471_first_contact": _first_contact_artifact(banked=True),
            "4473_submission": _submission_artifact(),
        },
        {},
        filled_gap_ids=[],
        open_gap_ids=[],
        excluded=[],
    )
    assert (
        next(row for row in banked_arc["games"] if row["game"] == "re86")["levels_reproduced"] == 1
    )
    assert report["reproducible_total_games"] == 21
