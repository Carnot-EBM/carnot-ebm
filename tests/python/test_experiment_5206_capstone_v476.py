"""Tests for Exp 5206 V476 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5206, SCENARIO-CAPSTONE-5206,
SCENARIO-CAPSTONE-5206-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5206_capstone_v476 as mod
from scripts import experiment_5206_capstone_v476 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"principle": principle, "value": value}


def _base(experiment: str | int, verdict: Any, *, flagged: bool | None = None) -> dict[str, Any]:
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
        5193: {
            **_base(
                "experiment_5193_archive_475_activate_476",
                _wrap("complete_archive_475_closed_476_active_precise_handoff_clean"),
            ),
            "clean_handoff": True,
            "roadmap_activation_check": {"activated": True, "milestone": "2026.07.476"},
        },
        5194: {
            **_base(
                "experiment_5194_poison_test_cascade_triage_module_v476",
                _wrap("complete_pretest_triage_module_ready_and_tested_wiring_documented_conductor_not_yet_patched_1of4_exact_signature"),
                flagged=False,
            ),
            "module_verification": {"tests_passed": 32, "tests_failed": 0, "new_module_coverage_pct": 100.0},
            "regression_tests_added": 32,
            "research_conductor_modified": False,
            "research_conductor_py_untouched_confirmed": True,
        },
        5195: {
            **_base(
                5195,
                "complete: retro_timing_475_false_zero_root_cause_found_and_fixed",
            ),
            "retro_timing_root_cause_found": "runtime import bug",
            "fix_applied_to": "patch file for research_conductor.py (git apply --check verified, not applied)",
            "git_apply_check_verified": True,
            "regression_test_passes_after_fix": True,
            "existing_wiring_test_stays_green": True,
            "known_issues_md_duplicate_count_before": 187,
            "known_issues_md_duplicate_count_after": 1,
            "research_conductor_py_untouched_confirmed": True,
        },
        5196: {
            **_base(
                "experiment_5196_diffusiongemma_vllm_native_retry_v476",
                "blocked_diffusiongemma_loading_exhausted_v476",
            ),
            "diffusiongemma_loadable": False,
            "forward_pass_confirmed": False,
            "loading_path_used": "both_failed",
            "retirement": "DiffusionGemma live-loading thread RETIRES per prior_failures retire_if_same_verdict=true.",
            "mitigations_tried": [{"mitigation": "vllm_native_fp8_tp2_recipe_maxseqs4"}],
        },
        5197: {
            **_base(
                "experiment_5197_gap4_scaleup_real_checkpoint_v476",
                "complete_gap4_scaleup_v476_n62_source_pool_exhausted_floor_not_crossed_scale_up_recommended",
            ),
            "n_reached": _wrap(62),
            "target_n": 180,
            "already_scored_prior_n": 62,
            "new_rows_scored": 0,
            "source_pool_exhausted_before_new_rows": True,
            "exact_test_discordant_wins": _wrap(4),
            "exact_test_discordant_losses": _wrap(0),
            "exact_test_p_value_two_sided": _wrap(0.125),
            "exact_test_passes_min6_rule": _wrap(False),
            "gap4_status_recommendation": "scale_up_recommended",
        },
        5198: {
            **_base(
                "experiment_5198_map_landmark_prestage_prototype_v476",
                "complete: MAP landmark prestage did not bank a new reproduction-gated level over pruner-only; the GAP-4891 enumeration wall persists under this lever too.",
            ),
            "lever_validated": False,
            "levels_banked": [],
            "gap4891_status_recommendation": "building_enumeration_wall_persists_under_map_prestage",
            "target_games": ["cd82", "sk48", "sp80"],
            "games_tested": ["cd82", "sk48", "sp80", "cn04"],
            "arms": ["pruner_only", "map_only", "map_plus_pruner"],
            "scripts_research_conductor_modified": False,
            "solve_provenance": "development_proxy",
        },
        5199: {
            "experiment": 5199,
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed; first failure: exp5198-map-landmark-prestage-prototype-v476.lever_validated",
            "gates_evaluated": [{"upstream": "exp5198-map-landmark-prestage-prototype-v476", "actual": False, "expected": True}],
            "duration_s": 0.0,
        },
        5200: {
            **_base(
                "experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476",
                _wrap("complete_hidden_state_probe_does_not_beat_tuned_sc_probe0.100_sc0.075_self0.075_clue0.100_rcs0.100"),
            ),
            "n_questions": _wrap(40),
            "probe_accuracy": _wrap(0.100),
            "tuned_sc_accuracy": _wrap(0.075),
            "self_certainty_accuracy": _wrap(0.075),
            "clue_accuracy": _wrap(0.100),
            "radial_consensus_score_accuracy": _wrap(0.100),
            "probe_vs_sc_delta_ci95": _wrap([0.0, 0.075]),
            "probe_vs_rcs_delta_ci95": _wrap([-0.075, 0.075]),
            "headroom_present": _wrap(True),
        },
        5201: {
            **_base(
                "experiment_5201_hardware_continuity_gatemate_diagnostic",
                "complete_hardware_continuity_gatemate_diagnostic_kv260:reachable_gatemate:blocked_gatemate_dirtyjtag_idcode_unresolved_v476_narrowed_jtag_protocol_level_polarfire:reachable_no_speedup_claim",
            ),
            "boards_reachable_count": 2,
            "gatemate_diagnostic_narrowed_to": "jtag_protocol_level",
            "hardware_speedup_claimed": False,
            "no_speedup_claim": True,
            "conductor_modified": False,
        },
        5202: {
            **_base(
                "experiment_5202_architecture_md_reconciliation_v476",
                _wrap("complete: architecture_md_reconciled_20260703_arc_phase_d_hidden_state_hardware"),
            ),
            "last_reconciled_date_updated": _wrap(True),
            "architecture_checks": {"last_reconciled": "20260703", "missing_new_sections": [], "required_topic_markers_missing": []},
            "research_conductor_modified": False,
        },
        5203: {
            **_base(
                "experiment_5203_verifier_authenticity_remediation_options_v476",
                _wrap("complete: verifier_authenticity_remediation_options_v476_ready"),
            ),
            "remediation_doc_path": _wrap("ops/verifier_remediation_options_v476.md"),
            "remediation_doc_sha256": "sha256-fixture",
            "audit_findings_independently_reconfirmed": _wrap(True),
            "failed_preconditions": [],
            "no_verifier_modified_this_task": _wrap(True),
        },
        5204: {
            **_base(
                "experiment_5204_exclusion_manifest_lint_real_bug_fix_v476",
                _wrap("success: exclusion_manifest_lint_real_bug_fixed_all_four_issues_word_boundary_principle_unwrap_general_negation_terminal_prefix"),
            ),
            "four_issues_fixed": _wrap({"word_boundary": True, "principle_unwrap": True, "general_negation": True, "terminal_prefix": True}),
            "counterexample_regression_test_passes_after_fix": _wrap(True),
            "full_adversarial_verify_test_suite_result": _wrap({"passed": 254, "failed": 0}),
            "backfill_dry_run_result": _wrap("exit_code=1 dry-run only; wrote 0 files"),
            "coverage_new_code": _wrap({"coverage_percent": 100.0}),
        },
        5205: {
            **_base(
                "experiment_5205_autopyverifier_gap1_pilot_v476",
                _wrap("complete: set_search_beats_always_on_beats_single_refuted_baseline_0.0879_best_0.2218_single_refuted_0.1506_captured_47_of_239_gap1_candidate_positive"),
            ),
            "pass_at_2_baseline_always_on_only": _wrap(0.087866),
            "pass_at_2_best_subset": _wrap(0.221757),
            "single_refuted_directional_adjacency_pass@2": 0.150628,
            "transpose_distractor_count": 239,
            "transpose_misvotes_captured": _wrap("47 out of 239"),
        },
    }


def _make_repo(root: Path, *, omit: set[int] | None = None) -> None:
    omit = omit or set()
    payloads = _payloads()
    for source in mod.UPSTREAM_SOURCES:
        if source.experiment_number not in omit:
            _write_json(root / source.relative_path, payloads[source.experiment_number])
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        "reproducible_total_levels: 69\n"
        "reproducible_total_games: 24\n"
        "games:\n"
        "  - game: lp85\n"
        "    levels_reproduced: 5\n",
        encoding="utf-8",
    )
    (root / "ops" / "known-issues.md").write_text(
        "# Known Issues\n\n### NEW Phase 4 Canonical Metric MANDATORY\n\nDiffusionGemma note.\n",
        encoding="utf-8",
    )
    (root / "_bmad" / "architecture.md").write_text(
        "# Architecture\n\n**Last Reconciled:** 20260703\n",
        encoding="utf-8",
    )
    (root / "ops" / "exclusion_manifest.yaml").write_text(
        "retired_extras:\n"
        "  - id: phase_d_external_text_scorer_retired_exp5163_v474\n"
        "    reason: hidden-state/internal-representation verifiers are outside this retired scope.\n",
        encoding="utf-8",
    )


def _reporter(path: Path) -> dict[str, Any]:
    if "5197" in path.name or "5200" in path.name:
        return {
            "artifact": str(path),
            "loaded": True,
            "flag_count": 1,
            "max_severity": 1,
            "flags": [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}],
        }
    return {"artifact": str(path), "loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}


def _levelup_ok() -> mod.LevelupLintResult:
    return mod.LevelupLintResult(
        exit_code=0,
        stdout="milestone: 2026.07.476  tasks: 14  level-up attempts: 1\nOK: 1 >= 1",
        structurally_satisfied=True,
    )


def _exclusion_ok() -> mod.CommandResult:
    return mod.CommandResult(
        exit_code=0,
        stdout="All violations have operator_override -- activation would proceed with warnings.",
        stderr="",
    )


def test_req_capstone_5206_spec_declares_v476_reconciliation_contract() -> None:
    """REQ-CAPSTONE-5206: OpenSpec declares the V476 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5206") :]

    for marker in (
        "REQ-CAPSTONE-5206",
        "SCENARIO-CAPSTONE-5206",
        "SCENARIO-CAPSTONE-5206-FIELD-PRINCIPLES",
        mod.EXPERIMENT_ID,
        str(mod.RESULT_RELATIVE_PATH),
        "diffusiongemma_arc_reconciled",
        "gated_and_skipped",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_capstone_5206_reconciles_v476_without_laundering_warnings(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5206: warning-only artifacts stay eligible; gated skips are not failures."""

    _make_repo(tmp_path)
    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260703",
        duration_s=1.25,
        tests_run=["focused"],
        adversarial_reporter=_reporter,
        levelup_lint_result=_levelup_ok(),
        exclusion_lint_result=_exclusion_ok(),
        conductor_untouched=True,
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert len(artifact["per_task_summary"]["value"]) == 13
    assert artifact["flagged_adversarial_artifacts_excluded"]["value"] == []
    assert "exp5197-gap4-scaleup-real-checkpoint-v476" in artifact["headline_eligible_task_ids"]
    assert "exp5200-hidden-state-verifier-v2-mmlu-pro-v476" in artifact["headline_eligible_task_ids"]
    assert "loading_not_achieved_thread_retired" in artifact["diffusiongemma_arc_reconciled"]["value"]
    assert "lever_validated=false" in artifact["gap4891_status_reconciled"]["value"]
    assert "n_reached=62/180" in artifact["gap4_status_reconciled"]["value"]
    assert "does_not_beat_all_controls" in artifact["hidden_state_verifier_v2_reconciled"]["value"]
    assert artifact["known_issues_md_deduped_confirmed"]["value"] is True
    assert artifact["architecture_md_reconciled"]["value"] is True
    assert "lp85 levels_reproduced=5" in artifact["lp85_registry_note_resolved"]["value"]
    assert artifact["reproducible_total_levels_delta"]["value"] == 0
    assert artifact["live_agent_self_discovery_ratio_updated"]["value"] == {
        "live_agent_self_discovery": 4,
        "development_proxy": 20,
        "total": 24,
    }
    assert artifact["levelup_guarantee_structurally_satisfied"]["value"] is True
    assert artifact["levelup_guarantee_outcome_satisfied"]["value"] is False
    assert artifact["research_conductor_py_untouched_confirmed"]["value"] is True
    assert artifact["flagged_adversarial"] is False

    per_task = {row["task_id"]: row for row in artifact["per_task_summary"]["value"]}
    assert per_task["exp5199-map-gated-levelup-attempt-v476"]["gated_and_skipped"] is True
    assert per_task["exp5199-map-gated-levelup-attempt-v476"]["headline_outcome"] == (
        "gated_skip_exp5198_lever_validated_false_not_failure"
    )
    assert per_task["exp5196-diffusiongemma-vllm-native-retry-v476"]["headline_outcome"] == (
        "diffusiongemma_loading_exhausted_thread_retired"
    )
    assert per_task["exp5204-exclusion-manifest-lint-real-bug-fix-v476"]["headline_outcome"] == (
        "exclusion_manifest_lint_real_bug_fixed_all_four_issues"
    )


def test_req_capstone_5206_validation_and_run_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5206: schema validation rejects overclaims and stale checksums."""

    _make_repo(tmp_path, omit={5201})
    payloads = _payloads()
    payloads[5205]["flagged_adversarial"] = True
    _write_json(
        tmp_path / mod.source_by_number(5205).relative_path,
        payloads[5205],
    )

    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260703",
        duration_s=2.0,
        tests_run=["focused"],
        adversarial_reporter=_reporter,
        levelup_lint_result=mod.LevelupLintResult(exit_code=1, stdout="FAIL", structurally_satisfied=False),
        exclusion_lint_result=mod.CommandResult(exit_code=1, stdout="HARD", stderr=""),
        conductor_untouched=True,
    )

    assert artifact["missing_artifacts"] == ["exp5201-hardware-continuity-gatemate-diagnostic-v476"]
    assert artifact["flagged_adversarial_artifacts_excluded"]["value"] == [
        "exp5205-autopyverifier-gap1-pilot-v476"
    ]
    assert "exp5205-autopyverifier-gap1-pilot-v476" not in artifact["headline_eligible_task_ids"]
    assert artifact["levelup_guarantee_structurally_satisfied"]["value"] is False
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
                "headline_eligible_task_ids": ["exp5205-autopyverifier-gap1-pilot-v476"],
                "flagged_adversarial_artifacts_excluded": _wrap(
                    ["exp5205-autopyverifier-gap1-pilot-v476"]
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
    assert mod.is_gated_skip({"status": "blocked", "blocked_at_layer": "conductor_pre_gate"}) is True
    assert mod.is_gated_skip({"status": "failed"}) is False
    with pytest.raises(KeyError):
        mod.source_by_number(9999)
    assert mod.gap4891_status({}) == "still_open_missing_exp5198"
    assert mod.gap4891_status({5198: {"lever_validated": True, "levels_banked": ["L2"]}}) == "filled"
    assert mod.gap4_status({}) == "still_open_missing_exp5197"
    assert mod.gap4_status({5197: {"exact_test_passes_min6_rule": True, "exact_test_discordant_wins": 6, "exact_test_discordant_losses": 0}}) == "filled"
    assert mod.diffusiongemma_reconciliation({}) == "still_blocked_missing_exp5196"
    assert mod.diffusiongemma_reconciliation(
        {5196: {"diffusiongemma_loadable": True, "forward_pass_confirmed": True}}
    ) == "loading_achieved_future_guided_vs_unguided_vs_ar_pilot_unblocked"
    assert mod.hidden_state_reconciliation({}) == "missing_exp5200"
    assert mod.live_agent_ratio(2) == {
        "live_agent_self_discovery": 6,
        "development_proxy": 20,
        "total": 26,
    }

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_mapping(malformed)[1]["error"] == "malformed_json"
    not_mapping = tmp_path / "array.json"
    not_mapping.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(not_mapping)[1]["error"] == "not_json_object"

    (tmp_path / "ops" / "arc_solve_registry.yaml").unlink()
    missing_registry = mod.read_registry(tmp_path)
    assert missing_registry["loadable"] is False
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
    assert mod.read_registry(tmp_path)["reproducible_total_levels"] is None
    (tmp_path / "ops" / "known-issues.md").unlink()
    assert mod.known_issues_phase4_count(tmp_path) == 0
    (tmp_path / "_bmad" / "architecture.md").unlink()
    assert mod.architecture_reconciled(tmp_path, {5202: _payloads()[5202]}) is False

    unknown_source = mod.UpstreamSource(9999, "exp9999-fixture", Path("results/x.json"))
    assert mod.headline_outcome(unknown_source, {"honest_verdict": "complete_fixture"}, []) == "reconciled"

    out_path = mod.run(
        root=tmp_path,
        run_date="20260703",
        duration_s=2.5,
        tests_run=["run"],
        adversarial_reporter=_reporter,
        levelup_lint_result=_levelup_ok(),
        exclusion_lint_result=_exclusion_ok(),
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
            conductor_untouched=True,
        )
        == out_path
    )
