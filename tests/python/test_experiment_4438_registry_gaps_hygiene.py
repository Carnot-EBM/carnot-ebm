"""Tests for Exp 4438 registry and gaps hygiene.

Spec refs: REQ-REPORT-4438, SCENARIO-REPORT-4438.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4438_registry_gaps_hygiene as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_arc_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated": "2026-06-18",
        "general_gotchas": [
            {"id": "offline_is_a_simulator", "note": "offline deterministic simulator"},
        ],
        "games": [
            {
                "game": "tu93",
                "reproducibility": "reproduced",
                "levels_reproduced": 4,
                "solver": "old tu93 solver",
                "gotchas": [],
            },
            {
                "game": "lp85",
                "reproducibility": "reproduced",
                "levels_reproduced": 5,
                "solver": "lp85 solver",
                "gotchas": [],
            },
            {
                "game": "g50t",
                "reproducibility": "unreproduced",
                "levels_reproduced": 0,
                "solver": "excluded if only flagged evidence exists",
                "gotchas": [],
            },
        ],
        "reproducible_total_levels": 9,
        "reproducible_total_games": 2,
    }


def _fixture_verifier_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": "gap4_program_induction_stack",
                "domain": "arc_agi2_grid",
                "eval": {
                    "metric": "rule_exec_selection",
                    "arc1_rule_exec_vote_pass2": 0.4516,
                    "arc1_rule_exec_gated_pass2": 0.5806,
                },
                "registry_roles": [],
            }
        ]
    }


def _write_fixture_repo(root: Path, *, omit_4434: bool = False) -> None:
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
    (root / mod.GAPS_RELATIVE_PATH).write_text("# Verifier Gaps\n", encoding="utf-8")

    _write_json(
        root / mod.EXP4432_PATH,
        {
            "experiment": "experiment_4432_loo_generic_solve_benchmark",
            "honest_verdict": "complete: generic_loo_solve_count_1_of_2_gate_failed",
            "generic_loo_solve_count": 1,
            "per_game": [
                {
                    "game": "tu93",
                    "solved_without_own_recipe": True,
                    "routed_to": "lp85",
                    "residual_delta": "none",
                },
                {
                    "game": "sc25",
                    "solved_without_own_recipe": False,
                    "routed_to": "tu93",
                    "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                },
            ],
            "missing_verifier_gaps": [
                {
                    "game": "sc25",
                    "routed_to": "tu93",
                    "residual_delta": "missing_cast_grid_spell_shrink_tank_exit_verifier",
                    "attempt_mode": "standing_arc_loop_adapter_withheld_graph_explore",
                }
            ],
            "offline_reproduced": True,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "1" * 64,
        },
    )
    _write_json(
        root / mod.EXP4433_PATH,
        {
            "experiment": "experiment_4433_example_conditioned_win_induction",
            "honest_verdict": "success: example_conditioned_g50t_L1_offline_reproduced",
            "flagged_adversarial": True,
            "target_game": "g50t",
            "reproduced_levels": 1,
            "offline_reproduced": True,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "2" * 64,
        },
    )
    if not omit_4434:
        _write_json(
            root / mod.EXP4434_PATH,
            {
                "experiment": "experiment_4434_example_conditioned_action_model",
                "honest_verdict": "success: example_conditioning_improved_world_model_accuracy",
                "target_game": "cn04",
                "reproduced_levels": 0,
                "offline_reproduced": False,
                "world_model_accuracy_with_examples": 1.0,
                "world_model_accuracy_cold": 0.71,
                "missing_verifier_gaps": [],
                "verifier_is_oracle": True,
                "reproducibility_checksum": "3" * 64,
            },
        )
    _write_json(
        root / mod.EXP4435_PATH,
        {
            "experiment": "experiment_4435_generic_first_contact_fixed",
            "honest_verdict": "complete: generic_first_contact_dc22_routed_no_new_level_gap_logged",
            "target_game": "dc22",
            "verdict_contract_fixed": True,
            "reproduced_levels": 0,
            "offline_reproduced": False,
            "missing_verifier_gaps": [
                {
                    "gap_id": "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                    "game": "dc22",
                    "status": "open",
                    "failure_mode": "needs_per_game_RE",
                    "missing_discriminator": "selectable verifier for the winning delta",
                    "candidate_design": "adapt config-rule predicate grounding",
                }
            ],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "4" * 64,
        },
    )
    _write_json(
        root / mod.EXP4436_PATH,
        {
            "experiment": "experiment_4436_deepen_plus_primitive_consolidation",
            "honest_verdict": "success: tu93_L5_deepened_primitives_consolidated",
            "deepened_game": "tu93",
            "reproduced_levels": 5,
            "new_levels_reproduced": 1,
            "offline_reproduced": True,
            "no_regression": True,
            "primitives_consolidated": [
                {"operator": "glyph_rewrite_matcher", "derived_from_games": ["tr87"]},
                {
                    "operator": "graph_astar_action_cost",
                    "derived_from_games": ["tu93", "lp85"],
                },
            ],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "5" * 64,
        },
    )


def _guard_ok(_root: Path) -> dict[str, Any]:
    return {
        "regression_guard_passed": True,
        "arc_oracle_distinct_verifier_beats_vote": True,
        "gap4_execution_guard_passed": True,
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


def test_req_report_4438_spec_declares_registry_hygiene_contract() -> None:
    """REQ-REPORT-4438: OpenSpec names the registry, guard, and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-4438",
        "SCENARIO-REPORT-4438",
        mod.RESULT_RELATIVE_PATH,
        "regression_guard_passed",
        "reproducible_total_levels",
        "flagged_adversarial",
        "aggregate-available",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4438_reconciles_ledgers_and_skips_flagged_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4438: residual gaps and counts reconcile without flagged evidence."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        tmp_path,
        gap4_guard_runner=_guard_ok,
        capstone_stamp_runner=_stamp_ok,
        now=lambda: 100.0,
    )

    assert artifact["honest_verdict"] == "complete: registry_gaps_hygiene_4438_guard_passed"
    assert type(artifact["regression_guard_passed"]) is bool
    assert artifact["regression_guard_passed"] is True
    assert artifact["capstone_stamp_fix_durable"] is True
    assert artifact["reproducible_total_levels"] == 10
    assert artifact["reproducible_total_games"] == 2
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["field_principles"]["honest_verdict"]["principle"] == "terminal-prefixed"
    assert (
        artifact["field_principles"]["regression_guard_passed"]["principle"]
        == "BARE bool (gated-fields-must-be-bare): the GAP-4 result did not regress"
    )
    assert (
        artifact["field_principles"]["reproducible_total_levels"]["principle"]
        == "the reconciled authoritative count"
    )
    assert artifact["availability_report"]["flagged_artifacts_excluded"] == [
        {
            "axis": "win_induction",
            "artifact_key": "4433_win_induction",
            "experiment_id": 4433,
            "reason": "flagged_adversarial",
        }
    ]
    assert artifact["availability_report"]["axes"]["primitives"]["verdict"] is True
    assert artifact["availability_report"]["axes"]["action_model"]["verdict"] is True
    assert "results/experiment_4433_example_conditioned_win_induction.json" in artifact[
        "excluded_artifacts"
    ]

    arc_registry = yaml.safe_load((tmp_path / mod.ARC_REGISTRY_RELATIVE_PATH).read_text())
    tu93 = next(row for row in arc_registry["games"] if row["game"] == "tu93")
    g50t = next(row for row in arc_registry["games"] if row["game"] == "g50t")
    assert tu93["levels_reproduced"] == 5
    assert arc_registry["reproducible_total_levels"] == 10
    assert arc_registry["reproducible_total_games"] == 2
    assert g50t["levels_reproduced"] == 0
    assert arc_registry["latest_hygiene_4438"]["excluded_artifacts"] == [
        "results/experiment_4433_example_conditioned_win_induction.json"
    ]

    gaps = (tmp_path / mod.GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "### GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER" in gaps
    assert "### GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT" in gaps
    assert "G50T" not in gaps

    verifier_registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text())
    gap4 = verifier_registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4438"] == mod.RESULT_RELATIVE_PATH
    assert gap4["eval"]["exp4438_regression_guard_passed"] is True
    assert gap4["eval"]["exp4438_reproducible_total_levels"] == 10
    assert gap4["registry_roles"][0]["role_id"] == mod.V410_ROLE_ID

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4438_aggregate_available_reports_missing_without_erasing_other_axes(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4438: a missing artifact reports a gap while other axes stay evaluable."""

    _write_fixture_repo(tmp_path, omit_4434=True)

    artifact = mod.run(
        tmp_path,
        gap4_guard_runner=_guard_ok,
        capstone_stamp_runner=_stamp_ok,
        now=lambda: 1.0,
    )

    assert artifact["availability_report"]["axes"]["action_model"]["missing_artifacts"] == [
        {"axis": "action_model", "artifact_key": "4434_action_model", "experiment_id": 4434}
    ]
    assert artifact["availability_report"]["axes"]["primitives"]["verdict"] is True
    assert artifact["availability_report"]["axes"]["loo_generic"]["verdict"] == 1
    assert artifact["registry_reconciliation"]["registries_reconciled"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_4438_reconciliation_is_idempotent(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4438: rerunning refreshes marked blocks instead of duplicating them."""

    _write_fixture_repo(tmp_path)

    first = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)
    second = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)
    gaps = (tmp_path / mod.GAPS_RELATIVE_PATH).read_text(encoding="utf-8")

    assert first["registry_reconciliation"]["gap_ids_logged"] == second["registry_reconciliation"][
        "gap_ids_logged"
    ]
    assert gaps.count("exp4438-gap-4432-loo-sc25-missing-cast-grid-spell-shrink-tank-exit-verifier:start") == 1
    assert gaps.count("exp4438-gap-4423-dc22-unselectable-first-contact:start") == 1


def test_req_report_4438_schema_rejects_non_bare_or_malformed_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-4438: schema validation protects required field shapes."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(tmp_path, gap4_guard_runner=_guard_ok, capstone_stamp_runner=_stamp_ok)

    bad = {
        **artifact,
        "honest_verdict": "partial: not allowed",
        "regression_guard_passed": "true",
        "reproducible_total_levels": True,
        "reproducible_total_games": "2",
        "submitted_to_leaderboard": True,
        "field_principles": {
            **artifact["field_principles"],
            "honest_verdict": {"principle": "wrong"},
        },
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with complete:/success:/passed:/shipped:" in errors
    assert "regression_guard_passed must be bare bool" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "reproducible_total_games must be bare int" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict.principle must match REQ-REPORT-4438" in errors
    with pytest.raises(ValueError, match="regression_guard_passed"):
        mod.write_artifact(tmp_path, bad)


def test_req_report_4438_defensive_helper_branches(tmp_path: Path) -> None:
    """REQ-REPORT-4438: malformed optional inputs are reported, not fabricated."""

    assert mod._as_int("not-int") == 0
    assert mod._as_float("not-float") == 0.0

    missing_yaml, missing_yaml_check = mod._yaml_mapping(tmp_path / "missing.yaml")
    assert missing_yaml == {}
    assert missing_yaml_check["readable"] is False
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("[]\n", encoding="utf-8")
    assert mod._yaml_mapping(list_yaml)[1]["error"] == "top-level YAML is not a mapping"
    assert mod._read_text(tmp_path / "missing.md")[1]["readable"] is False
    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    assert mod._load_json(list_json)[1]["error"] == "top-level JSON is not an object"

    assert mod._collect_loo_gaps({"missing_verifier_gaps": {}, "per_game": {}}) == []
    assert mod._collect_loo_gaps({"missing_verifier_gaps": [None]}) == []
    assert mod._collect_loo_gaps({"missing_verifier_gaps": [{"residual_delta": "none"}]}) == []
    assert mod._collect_first_contact_gaps({"missing_verifier_gaps": {}}) == []
    assert mod._collect_first_contact_gaps({"missing_verifier_gaps": [None]}) == []
    assert mod._collect_action_model_gaps({"missing_verifier_gaps": {}}) == []
    action_gaps = mod._collect_action_model_gaps(
        {
            "target_game": "cn04",
            "missing_verifier_gaps": [
                {"game": "cn04", "failure_mode": "typed_transition_gap"},
                "string_gap",
                "",
            ],
        }
    )
    assert [gap["gap_id"] for gap in action_gaps] == [
        "GAP-4434-ACTION-MODEL-CN04-TYPED-TRANSITION-GAP",
        "GAP-4434-ACTION-MODEL-CN04-STRING-GAP",
    ]
    assert mod.collect_gap_entries(
        {"4434_action_model": {"target_game": "cn04", "missing_verifier_gaps": ["raw_gap"]}}
    )[0]["gap_id"] == "GAP-4434-ACTION-MODEL-CN04-RAW-GAP"

    assert mod._find_game({"games": {}}, "x") is None
    assert mod._find_game({"games": [{"game": "a"}]}, "x") is None
    registry: dict[str, Any] = {"games": [], "general_gotchas": "bad"}
    assert mod._ensure_game(registry, "new")["game"] == "new"
    mod._ensure_general_gotcha(registry, {"operator": "", "derived_from_games": []})
    mod._ensure_general_gotcha(registry, {"operator": "op", "derived_from_games": ["g"]})
    assert registry["general_gotchas"][0]["id"] == "primitive_op"
    assert mod._reproduced_counts({"games": {}}) == (0, 0)
    assert mod._reproduced_counts({"games": [None, {"reproducibility": "reproduced", "levels_reproduced": "2"}]}) == (
        2,
        1,
    )

    updated_arc, report = mod.reconcile_arc_registry(
        {"games": [], "general_gotchas": []},
        {
            "4433_win_induction": {
                "target_game": "g50t",
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "reproducibility_checksum": "a" * 64,
            }
        },
        gap_ids=[],
        excluded=[],
    )
    assert report["reproducible_total_levels"] == 1
    assert updated_arc["games"][0]["latest_exp4433_reproduce"]["artifact"] == mod.EXP4433_PATH
    zero_level_arc, zero_level_report = mod.reconcile_arc_registry(
        {"games": [], "general_gotchas": []},
        {"4433_win_induction": {"target_game": "g50t", "offline_reproduced": True, "reproduced_levels": 0}},
        gap_ids=[],
        excluded=[],
    )
    assert zero_level_report["reproducible_total_levels"] == 0
    assert zero_level_arc["games"] == []
    missing_game_arc, missing_game_report = mod.reconcile_arc_registry(
        {"games": [], "general_gotchas": []},
        {"4435_first_contact": {"target_game": "", "offline_reproduced": True, "reproduced_levels": 1}},
        gap_ids=[],
        excluded=[],
    )
    assert missing_game_report["reproducible_total_levels"] == 0
    assert missing_game_arc["games"] == []

    assert mod._find_verifier({"verifiers": {}}, mod.GAP4_VERIFIER_ID) is None
    assert mod._find_verifier({"verifiers": [{}]}, mod.GAP4_VERIFIER_ID) is None
    assert mod._guard_current({"replayed_arc1_rule_exec": {"vote_pass2": 0.1}})["vote_pass2"] == 0.1
    assert mod._guard_current({}) == {}
    assert mod.guard_passed({"regression_guard_passed": True, "current": {"gated_pass2": 0.1, "vote_pass2": 0.2}}) is False
    assert mod.stamp_fix_durable({"capstone_stamp_fix_verified": True, "circular_moat_overclaim_fired": False}) is True

    created_registry, _ = mod.reconcile_verifier_registry(
        {},
        guard=_guard_ok(tmp_path),
        stamp=_stamp_ok(tmp_path),
        total_levels=1,
        total_games=1,
        gap_ids=["GAP-X"],
        trusted={},
        excluded=[],
    )
    assert created_registry["verifiers"][0]["verifier_id"] == mod.GAP4_VERIFIER_ID
    bad_roles_registry, _ = mod.reconcile_verifier_registry(
        {"verifiers": [{"verifier_id": mod.GAP4_VERIFIER_ID, "registry_roles": "bad", "eval": {}}]},
        guard=_guard_ok(tmp_path),
        stamp=_stamp_ok(tmp_path),
        total_levels=1,
        total_games=1,
        gap_ids=[],
        trusted={},
        excluded=[],
    )
    assert isinstance(bad_roles_registry["verifiers"][0]["registry_roles"], list)

    assert "missing honest_verdict" in mod.artifact_schema_errors({})
    malformed = {
        "honest_verdict": "complete: malformed",
        "regression_guard_passed": True,
        "reproducible_total_levels": 1,
        "reproducible_total_games": 1,
        "registry_reconciliation": [],
        "availability_report": [],
        "capstone_stamp_fix_durable": "true",
        "random_seed": "4438",
        "reproducibility_checksum": "bad",
        "field_principles": [],
        "submitted_to_leaderboard": False,
        "spec_refs": [],
    }
    malformed_errors = mod.artifact_schema_errors(malformed)
    assert "capstone_stamp_fix_durable must be bare bool" in malformed_errors
    assert "random_seed must be bare int" in malformed_errors
    assert "registry_reconciliation must be dict" in malformed_errors
    assert "availability_report must be dict" in malformed_errors
    assert "field_principles must be dict" in malformed_errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in malformed_errors
    assert "spec_refs must include REQ-REPORT-4438 and SCENARIO-REPORT-4438" in malformed_errors
