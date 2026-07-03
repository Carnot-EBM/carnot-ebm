"""Tests for Exp 5180 V474 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5180, SCENARIO-CAPSTONE-5180,
SCENARIO-CAPSTONE-5180-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5180_capstone_v474 as mod
from scripts import experiment_5180_capstone_v474 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _base(experiment: str, verdict: Any, *, flagged: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": experiment,
        "honest_verdict": verdict,
        "duration_s": 1.0,
        "inference_substrate": "fixture",
    }
    if flagged is not None:
        payload["flagged_adversarial"] = flagged
    return payload


def _payloads() -> dict[int, dict[str, Any]]:
    return {
        5168: {
            **_base(
                "experiment_5168_archive_473_activate_474",
                "complete_archive_473_closed_474_active_runtime_clean_exp5161_unquarantined",
                flagged=False,
            ),
            "v473_runtime_clean": True,
            "exp5161_unquarantine_noted": True,
        },
        5169: {
            **_base(
                "experiment_5169_adversarial_verify_qd_citation_scope_fix_v474",
                _wrap("complete: exp5156_resolves_clean_qd_citation_scope_fixed_warn_only_not_quarantine"),
            ),
            "exp5156_resolved": _wrap(True),
            "severity_handling_audit_result": _wrap("bug_found_and_fixed"),
        },
        5170: {
            **_base(
                "experiment_5170_retire_phase_d_external_text_scorer_v474",
                "complete: phase_d_external_text_scorer_scope_retired_and_hidden_state_exception_preserved",
                flagged=False,
            ),
            "exclusion_manifest_entry_added": True,
            "entry_id": "phase_d_external_text_scorer_retired_exp5163_v474",
            "false_positive_check_against_exp5178": True,
            "manifest_entry_audit": {"found": True, "errors": []},
            "current_roadmap_lint": {"passed": True, "exp5178_entry_risks": []},
        },
        5171: {
            **_base(
                "experiment_5171_harden_set_encoder_cross_corpus_n30_v474",
                "success_arc_set_encoder_cross_corpus_gate_passed_n30: gate passed at n>=30",
            ),
            "gate_passed": True,
            "held_out_task_n": 30,
            "headline_outcome": "arc_set_encoder_cross_corpus_gate_passed_n30",
            "cross_corpus_delta_n30": 0.5,
            "cross_corpus_delta_ci95_n30": [0.3333333333, 0.6666666667],
        },
        5172: {
            **_base(
                "experiment_5172_sota_ingestion_diffusion_hierarchical_search_v474",
                "complete: map_deep_read_recommends_map_pre_stage_if_phase_b_pruner_stalls",
            ),
            "bottom_line_recommendation_for_475": _wrap(
                "MAP should be prototyped next if Phase B's pruner does not fully close GAP-4891"
            ),
        },
        5173: {
            **_base(
                "experiment_5173_diffusiongemma_energy_guided_diffusion_pilot_v474",
                "blocked_diffusiongemma_meta_tensor_bug_unresolved",
                flagged=False,
            ),
            "inference_substrate": "blocked_preflight",
            "arm_rows": [],
            "preconditions": {"smoke": {"success": False}},
            "meta_tensor_bug_resolution": _wrap("blocked_diffusiongemma_meta_tensor_bug_unresolved"),
            "pass_at_1_guided": _wrap(0.0),
            "pass_at_1_unguided": _wrap(0.0),
            "pass_at_1_ar_baseline": _wrap(0.0),
        },
        5174: {
            **_base(
                "experiment_5174_gap_live_integration_reconciliation_v474",
                "complete: original three GAP-LIVE-INTEGRATION claims were stale",
            ),
            "gap_status_recommendation": {"value": "re-scoped"},
            "claim_router_dsl_unimported": {"value": False},
            "claim_target_levels_1": {"value": False},
            "claim_value_weight_0": {"value": False},
        },
        5175: {
            **_base(
                "experiment_5175_gap4891_relational_mask_pruner_ab_v474",
                "complete_relational_mask_pruner_prunes_edges_but_states_expanded_unchanged_no_level_bank_pruning_alone_does_not_close_enumeration_wall_MAP_map_then_act_next",
            ),
            "gap4891_status_recommendation": "building_with_new_lever_named",
            "levels_banked": [],
            "target_games": ["cd82", "sk48", "sp80"],
            "games_tested": ["cd82", "sk48", "sp80", "cn04"],
            "states_expanded_pruned": {"cd82": 4000, "sk48": 4000, "sp80": 4000, "cn04": 4000},
            "states_expanded_unpruned": {"cd82": 4000, "sk48": 4000, "sp80": 4000, "cn04": 4000},
            "move_pruned_edges": {"cd82": 358, "sk48": 22807, "sp80": 0, "cn04": 375},
            "next_specific_lever": "Prototype a MAP-style map-then-act / hierarchical pre-search stage.",
            "scripts_research_conductor_modified": False,
        },
        5176: {
            **_base(
                "experiment_5176_deepen_live_levelup_attempt_v474",
                "complete_blocked_no_validated_lever_from_b1_b2_zero_levels_banked",
            ),
            "target_games": [
                {"game": "cd82", "level_before": 2, "level_attempted": 3},
                {"game": "cn04", "level_before": 3, "level_attempted": 4},
            ],
            "levels_banked": [],
            "reproducible_levels_delta": 0,
            "upstream_lever_assessment": {
                "exp5175_gap4891_status_recommendation": "building_with_new_lever_named",
                "exp5175_reproduce_gated_levels": 0,
            },
        },
        5177: {
            **_base(
                "experiment_5177_gap4_scaleup_decentralization_tier_v474",
                "complete_gap4_scaleup_v474_n62_of_target180_floor_not_crossed_scale_up_recommended",
            ),
            "target_n": _wrap(180),
            "achieved_n": _wrap(62),
            "exact_test_discordant_wins": _wrap(4),
            "exact_test_discordant_losses": 0,
            "exact_test_passes_min6_rule": _wrap(False),
            "exact_test_p_value_two_sided": _wrap(0.125),
            "gap4_status_recommendation": _wrap("scale_up_recommended"),
        },
        5178: {
            **_base(
                "experiment_5178_hidden_state_verifier_pilot_v474",
                _wrap(
                    "complete_hidden_state_verifier_ties_tuned_sc_accuracy_point_lower_efficiency_loses_to_sc_extra_hidden_forward"
                ),
                flagged=True,
            ),
            "hidden_state_access_feasible": _wrap(True),
            "hidden_state_verifier_accuracy": _wrap(0.0),
            "tuned_sc_baseline_accuracy": _wrap(0.333333),
        },
        5179: {
            **_base(
                "experiment_5179_hardware_continuity_board_timing",
                "complete_hardware_continuity_board_timing_kv260:reachable_gatemate:blocked_gatemate_dirtyjtag_idcode_unresolved_after_diagnostics_polarfire:reachable_no_speedup_claim",
            ),
            "boards_reachable_count": 2,
            "hardware_speedup_claimed": False,
            "no_speedup_claim": True,
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    payloads = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, payloads[source.experiment_number])
    _write_json(
        root / "ops" / "arc_solve_registry.yaml",
        {"reproducible_total_levels": 69, "reproducible_total_games": 24, "games": []},
    )
    _write_json(
        root / "ops" / "exclusion_manifest.yaml",
        {
            "retired_extras": [
                {
                    "id": "phase_d_external_text_scorer_retired_exp5163_v474",
                    "blocked_patterns": ["phase d external text scorer rerun"],
                }
            ]
        },
    )


def _reporter(path: Path) -> dict[str, Any]:
    critical = "5178" in path.name
    info = "5177" in path.name
    if critical:
        return {
            "artifact": str(path),
            "loaded": True,
            "flag_count": 2,
            "max_severity": 2,
            "flags": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
        }
    if info:
        return {
            "artifact": str(path),
            "loaded": True,
            "flag_count": 1,
            "max_severity": 0,
            "flags": [{"kind": "IMPLAUSIBLE_PERFECT", "severity": "info"}],
        }
    return {"artifact": str(path), "loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}


def _levelup_ok() -> mod.LevelupLintResult:
    return mod.LevelupLintResult(
        exit_code=0,
        stdout="milestone: 2026.07.474  tasks: 13  level-up attempts: 1\nOK: 1 >= 1",
        structurally_satisfied=True,
    )


def _exclusion_ok() -> mod.CommandResult:
    return mod.CommandResult(
        exit_code=0,
        stdout="All violations have operator_override -- activation would proceed with warnings.",
        stderr="",
    )


def _publication_ready() -> dict[str, Any]:
    return {"paper_ready": True, "unmet_gates": [], "gates": {"G1": {"pass": True}}}


def test_req_capstone_5180_spec_declares_v474_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5180: OpenSpec declares the V474 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5180") :]

    for marker in (
        "REQ-CAPSTONE-5180",
        "SCENARIO-CAPSTONE-5180",
        "SCENARIO-CAPSTONE-5180-FIELD-PRINCIPLES",
        mod.EXPERIMENT_ID,
        str(mod.RESULT_RELATIVE_PATH),
        "diffusiongemma_pilot_reconciled",
        "acceptance_criteria_checklist",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5180_reconciles_v474_without_headlining_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5180: flagged upstreams are excluded from headline aggregation."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260703",
        duration_s=1.25,
        tests_run=["focused"],
        adversarial_reporter=_reporter,
        levelup_lint_result=_levelup_ok(),
        exclusion_lint_result=_exclusion_ok(),
        publication_gate_result=_publication_ready(),
        conductor_untouched=True,
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert len(artifact["per_task_summary"]["value"]) == 12
    assert artifact["flagged_adversarial_artifacts_excluded"]["value"] == [
        "exp5178-hidden-state-verifier-pilot-v474"
    ]
    assert "exp5178-hidden-state-verifier-pilot-v474" not in artifact["headline_eligible_task_ids"]
    assert artifact["gap4891_status_reconciled"]["value"].startswith(
        "building_with_new_lever_named_not_filled"
    )
    assert "states_expanded stayed 4000" in artifact["gap4891_status_reconciled"]["value"]
    assert artifact["gap4_status_reconciled"]["value"] == (
        "scale_up_recommended_not_filled: exp5177 achieved 62/180 rows with 4 "
        "discordant wins, 0 losses, p=0.125, and exact_test_passes_min6_rule=false."
    )
    assert artifact["diffusiongemma_pilot_reconciled"]["value"].startswith(
        "blocked_preflight_no_guided_measurement"
    )
    assert artifact["phase_d_retirement_confirmed_clean"]["value"] is True
    assert artifact["registry_reconciliation"]["reproducible_total_levels"] == 69
    assert artifact["registry_reconciliation"]["reproducible_total_games"] == 24
    assert artifact["reproducible_total_levels_delta"]["value"] == 0
    assert artifact["levelup_guarantee_structurally_satisfied"]["value"] is True
    assert artifact["levelup_guarantee_outcome_satisfied"]["value"] is False
    assert artifact["research_conductor_py_untouched_confirmed"]["value"] is True
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["flagged_adversarial"] is False

    checklist = artifact["acceptance_criteria_checklist"]["value"]
    assert [row["satisfied"] for row in checklist] == [True] * 6
    assert "exp5171" in checklist[0]["evidence"]
    assert "exp5175" in checklist[1]["evidence"]
    assert "exp5178" in checklist[2]["evidence"]
    assert "paper_ready=true" in checklist[3]["evidence"]
    assert "exp5176" in checklist[4]["evidence"]
    assert "flat at 69/24" in checklist[5]["evidence"]

    per_task = {row["task_id"]: row for row in artifact["per_task_summary"]["value"]}
    assert per_task["exp5173-diffusiongemma-energy-guided-diffusion-pilot-v474"]["headline_outcome"] == (
        "blocked_diffusiongemma_meta_tensor_bug_unresolved_no_guided_measurement"
    )
    assert per_task["exp5178-hidden-state-verifier-pilot-v474"]["headline_outcome"] == (
        "excluded_from_headline_aggregation_flagged_adversarial"
    )


def test_req_capstone_5180_validation_and_run_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5180: schema validation rejects overclaims and stale checksums."""

    _make_repo(tmp_path, omit={5179})
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260703",
        duration_s=2.0,
        tests_run=["focused"],
        adversarial_reporter=_reporter,
        levelup_lint_result=mod.LevelupLintResult(exit_code=1, stdout="FAIL", structurally_satisfied=False),
        exclusion_lint_result=mod.CommandResult(exit_code=1, stdout="HARD exp5178", stderr=""),
        publication_gate_result={"paper_ready": False, "unmet_gates": ["G2"]},
        conductor_untouched=True,
    )

    assert artifact["missing_artifacts"] == ["exp5179-hardware-continuity-board-timing-v474"]
    assert artifact["phase_d_retirement_confirmed_clean"]["value"] is False
    assert artifact["levelup_guarantee_structurally_satisfied"]["value"] is False
    assert artifact["acceptance_criteria_checklist"]["value"][2]["satisfied"] is False
    assert artifact["acceptance_criteria_checklist"]["value"][3]["satisfied"] is False
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact({key: value for key, value in artifact.items() if key != "duration_s"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "done"})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(artifact | {"inference_substrate": "live_llm_inference"})
    with pytest.raises(ValueError, match="field principle mismatch"):
        mod.validate_artifact(artifact | {"field_principles": artifact["field_principles"] | {"honest_verdict": "loose"}})
    with pytest.raises(ValueError, match="flagged_adversarial"):
        mod.validate_artifact(artifact | {"flagged_adversarial": True})
    with pytest.raises(ValueError, match="research_conductor"):
        mod.validate_artifact(artifact | {"research_conductor_py_untouched_confirmed": _wrap(False)})
    with pytest.raises(ValueError, match="levelup_guarantee_outcome_satisfied"):
        mod.validate_artifact(artifact | {"levelup_guarantee_outcome_satisfied": _wrap(True)})
    with pytest.raises(ValueError, match="headline eligible"):
        mod.validate_artifact(
            artifact
            | {
                "headline_eligible_task_ids": ["exp5178-hidden-state-verifier-pilot-v474"],
                "flagged_adversarial_artifacts_excluded": _wrap(
                    ["exp5178-hidden-state-verifier-pilot-v474"]
                ),
            }
        )
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(artifact | {"tests_run": []})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "stale"})

    assert mod.value_of(_wrap("x")) == "x"
    assert mod.value_of("x") == "x"
    assert mod.honest_verdict_text({"value": "complete_wrapped"}) == "complete_wrapped"
    assert mod.honest_verdict_text(None) == ""
    assert mod._number("3.5") == 3.5
    assert mod._number("bad") is None
    assert mod._number(True) is None
    assert mod._flag_is_critical({"severity": 2}) is True
    assert mod._banked_level_count({"levels_banked": "not-a-list"}) == 0
    assert "no flagged headline artifacts" in mod.honest_verdict_for_exclusions([])

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_mapping(malformed)[1]["error"] == "malformed_json"
    not_mapping = tmp_path / "array.json"
    not_mapping.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(not_mapping)[1]["error"] == "not_json_object"

    registry_path = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry_path.unlink()
    assert mod.read_registry_totals(tmp_path)["loadable"] is False
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("- not-a-mapping\n", encoding="utf-8")
    assert mod.read_registry_totals(tmp_path)["reproducible_total_levels"] is None
    manifest_path = tmp_path / "ops" / "exclusion_manifest.yaml"
    manifest_path.unlink()
    assert mod.phase_d_manifest_entry_present(tmp_path) is False

    unknown_source = mod.UpstreamSource(9999, "exp9999-fixture", Path("results/x.json"))
    assert mod.headline_outcome(unknown_source, {"honest_verdict": "complete_fixture"}, []) == "reconciled"
    assert mod.reproducible_level_delta({5175: {"levels_banked": [1]}, 5176: {}}) == 1
    assert mod.gap4891_status({}) == "still_open_missing_exp5175"
    assert mod.gap4891_status({5175: {"levels_banked": ["new"]}}) == "filled"
    assert mod.gap4891_status({5175: {"gap4891_status_recommendation": "retired"}}) == (
        "retired_not_filled"
    )
    assert mod.gap4_status({}, []) == "still_open_missing_exp5177"
    assert (
        mod.gap4_status(
            {
                5177: {
                    "exact_test_passes_min6_rule": _wrap(True),
                    "exact_test_discordant_wins": _wrap(6),
                    "exact_test_discordant_losses": 0,
                }
            },
            [],
        )
        == "filled"
    )
    assert (
        mod.gap4_status(
            {
                5177: {
                    "gap4_status_recommendation": "still_open",
                    "flagged_adversarial": True,
                }
            },
            [
                {
                    "task_id": "exp5177-gap4-scaleup-decentralization-tier-v474",
                    "critical_adversarial_flag": False,
                }
            ],
        )
        == "still_open_flagged_excluded_from_headline"
    )
    assert mod.diffusiongemma_pilot({}) == "still_blocked_missing_exp5173"
    assert mod.diffusiongemma_pilot({5173: {"honest_verdict": "complete_measured", "arm_rows": [1]}}) == (
        "measured_diffusiongemma_pilot"
    )

    out_path = mod.run(
        root=tmp_path,
        run_date="20260703",
        duration_s=2.5,
        tests_run=["run"],
        adversarial_reporter=_reporter,
        levelup_lint_result=_levelup_ok(),
        exclusion_lint_result=_exclusion_ok(),
        publication_gate_result=_publication_ready(),
        conductor_untouched=True,
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["duration_s"] == 2.5
    assert saved["tests_run"] == ["run"]
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)

    assert (
        script_mod.main(
            root=tmp_path,
            date="20260703",
            duration_s=3.0,
            tests_run=["script"],
            adversarial_reporter=_reporter,
            levelup_lint_result=_levelup_ok(),
            exclusion_lint_result=_exclusion_ok(),
            publication_gate_result=_publication_ready(),
            conductor_untouched=True,
        )
        == out_path
    )
    assert (
        script_mod.main(
            ["--root", str(tmp_path), "--date", "20260703"],
            duration_s=3.5,
            tests_run=["script-argv"],
            adversarial_reporter=_reporter,
            levelup_lint_result=_levelup_ok(),
            exclusion_lint_result=_exclusion_ok(),
            publication_gate_result=_publication_ready(),
            conductor_untouched=True,
        )
        == out_path
    )
