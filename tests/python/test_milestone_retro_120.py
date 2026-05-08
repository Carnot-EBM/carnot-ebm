"""Tests for the Exp 1572 milestone .120 retrospective.

Spec: REQ-REPORT-063, SCENARIO-REPORT-063.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_120 as retro120
from carnot.reporting.milestone_retro_120 import (
    BLOCKED,
    MET,
    MISSING,
    NOT_MET,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _read_json,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _terminal_sources() -> dict[str, dict[str, object]]:
    return {
        "exp1560": {
            "status": "complete",
            "activation_manifest_complete": True,
            "thrml_scaling_sweep_lineage_retired": True,
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
            "honest_verdict": "complete: milestone_120_activation_complete",
        },
        "exp1561": {
            "status": "complete",
            "kinetic_defense_in_depth_validated": False,
            "thrml_security_parity_with_single_site_gibbs": False,
            "falsification_note": "THRML graph-color block-Gibbs hit at MH-class rates.",
            "honest_verdict": "complete_thrml_block_gibbs_falsifies_kinetic_security_parity",
        },
        "exp1562": {
            "status": "complete",
            "brain_linear_ar_rescue_validated": False,
            "factorized_vs_ar_ratio_at_k15": 1.000749,
            "best_parameterization_kl_at_k15": 0.001336,
            "phase_3_recommendation": "brain_dropped",
            "honest_verdict": "complete: falsified BRAIN+Linear-AR rescue widening",
        },
        "exp1563": {
            "status": "complete",
            "architecture_record_updated": True,
            "spec_requirements_added": 3,
            "exclusion_manifest_updated": True,
            "honest_verdict": "complete: SpecAnn rejected",
        },
        "exp1564": {
            "status": "complete",
            "thrml_vendoring_complete": True,
            "kl_to_thrml_after_vendoring": 0.0,
            "candidate_warm_start_implemented": True,
            "regression_tests_passed": False,
            "focused_regression_tests_passed": True,
            "applicable_e2e_passed": True,
            "mirror_repo_url": "ssh://mirror/example",
            "full_regression_attempt": {"completed": False, "passed": False},
            "honest_verdict": "complete: vendored THRML with full-suite caveat",
        },
        "exp1565": {
            "status": "complete",
            "soft_gibbs_residual_implemented": True,
            "hard_brs_acceptance_rate": 0.0,
            "soft_brs_decay_confirmed": True,
            "min_violation_state_found": True,
            "z_beta_curve": [{"beta": 5.0, "empirical_acceptance_rate": 0.65}],
            "honest_verdict": "complete: soft Gibbs residual implemented",
        },
        "exp1566": {
            "status": "complete",
            "candidate_warm_start_validated": True,
            "cold_start_accuracy_drop_percent_at_k100": 51.05,
            "cached_state_worse_than_cold_start": True,
            "recommended_deployment_policy": "candidate_warm_start",
            "honest_verdict": "complete: candidate warm-start validated",
        },
        "exp1567": {
            "status": "complete",
            "rho_C_curve_fitted": True,
            "rho_C_r_squared": 0.999983,
            "C_star_estimate": 32.0,
            "C_star_ci_lower": 25.0,
            "C_star_ci_upper": 46.0,
            "C_inv_estimate": 95.6,
            "C_inv_ci_lower": 91.7,
            "C_inv_ci_upper": 96.7,
            "inversion_empirically_confirmed": True,
            "srs_accepted_accuracy_at_C_above_C_inv": 0.419,
            "metadata": {"s_r_star": 0.72, "fresh_gpu_hours_consumed": 0.0},
            "honest_verdict": "complete: rho_C curve fitted",
        },
        "exp1568": {
            "status": "complete",
            "mode_collapse_audit_complete": True,
            "retained_policies_audited_count": 2,
            "retained_policy_target_count": 5,
            "retained_policy_target_met": False,
            "mode_collapse_confirmed_count": 1,
            "reversal_recommended_count": 1,
            "honest_verdict": "complete: one retained policy flagged for reversal",
        },
        "exp1569": {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "honest_verdict": "blocked_gate_check_failed",
        },
        "exp1570": {
            "status": "complete",
            "alpha_i_per_verifier": [0.12, 0.15, 0.11, 0.26, 0.05, 0.14],
            "z_beta_jensen_bound": [{"beta": 0.1, "predicted_lower": 0.91}],
            "z_beta_empirical": [{"beta": 0.1, "empirical_acceptance_rate": 0.92}],
            "jensen_bound_holds_for_all_beta": True,
            "optimal_beta_for_deployment": 0.1,
            "honest_verdict": "complete: soft-gibbs coverage bound verified",
        },
        "exp1571": {
            "status": "complete",
            "step_wise_baseline_implemented": True,
            "gradient_variance_reduction_factor": 10.45,
            "convergence_rate_matches_theorem_2": True,
            "honest_verdict": "complete: step-wise baseline passed",
        },
        "exp1573": {
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "honest_verdict": "blocked_gate_check_failed",
        },
    }


def test_req_report_063_scores_terminal_acceptance_gates() -> None:
    """REQ-REPORT-063: .120 criteria are scored from source fields."""

    artifact = build_artifact(sources=_terminal_sources(), missing_source_ids=[])

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.05.120"
    assert artifact["criteria_met"] == 10
    assert artifact["criteria_total"] == 14
    assert artifact["criteria_met_fraction"] == "10/14"
    assert artifact["criteria_results"]["kinetic_defense"]["status"] == NOT_MET
    assert artifact["criteria_results"]["brain_linear_ar_rescue"]["status"] == NOT_MET
    assert artifact["criteria_results"]["paper_v6_section_3_draft"]["status"] == BLOCKED
    assert artifact["criteria_results"]["extropic_z1_readiness"]["status"] == BLOCKED
    assert artifact["criteria_results"]["thrml_vendoring"]["status"] == MET
    assert artifact["criteria_results"]["fr11_v14_retention_audit"]["status"] == MET
    assert artifact["paper_v6_section_3_drafted"] is False
    assert artifact["all_4_carnot_contributions_validated"] is False
    assert artifact["rho_C_curve_published_ready"] is True
    assert artifact["terminal_verdicts"]["exp1569"] == "blocked_gate_check_failed"
    assert artifact["honest_verdict"].startswith("complete:")
    assert any(
        gate["gate"] == "paper_v6_section_3_finalization_after_exp1569_draft"
        for gate in artifact["carry_forward_gates_121"]
    )


def test_scenario_report_063_missing_and_repaired_branches() -> None:
    """SCENARIO-REPORT-063: missing and repaired evidence changes the score honestly."""

    sources = _terminal_sources()
    sources["exp1561"].update(
        {
            "kinetic_defense_in_depth_validated": True,
            "thrml_security_parity_with_single_site_gibbs": True,
        }
    )
    sources["exp1562"].update(
        {
            "brain_linear_ar_rescue_validated": True,
            "factorized_vs_ar_ratio_at_k15": 12.5,
            "best_parameterization_kl_at_k15": 0.05,
        }
    )
    sources["exp1569"].update(
        {
            "status": "complete",
            "section_3_drafted": True,
            "subsections_completed": 8,
            "all_4_carnot_contributions_present": True,
            "all_3_ruleouts_documented": True,
            "honest_verdict": "complete: section 3 drafted",
        }
    )
    sources["exp1573"].update(
        {
            "status": "complete",
            "z1_readiness_packet_updated": True,
            "thrml_vendor_mirroring_verified": True,
            "kv260_open_fpga_tracks_preserved": True,
            "kinetic_defense_z1_transfer_question_filed": True,
            "sampler_backend_protocol_documented": True,
            "honest_verdict": "complete: z1 packet updated",
        }
    )

    artifact = build_artifact(sources=sources, missing_source_ids=["exp1570"])

    assert artifact["criteria_met"] == 13
    assert artifact["criteria_results"]["soft_gibbs_coverage_bound"]["status"] == MISSING
    assert artifact["paper_v6_section_3_drafted"] is True
    assert artifact["contribution_validation"]["kinetic_defense_in_depth"] is True
    assert artifact["contribution_validation"]["brain_linear_ar_rescue"] is True
    assert artifact["all_4_carnot_contributions_validated"] is True


def test_run_writes_bootstrap_and_terminal_artifact_req_report_063(tmp_path: Path) -> None:
    """REQ-REPORT-063: run writes in-progress and final Exp1572 JSON."""

    out_path = tmp_path / "results" / "experiment_1572_milestone_120_retro.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    for exp_id, filename in SOURCE_FILES.items():
        _write_json(tmp_path / "results" / filename, _terminal_sources()[exp_id])

    artifact = run(root=tmp_path, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["criteria_met"] == 10
    assert written["retro_complete"] is True
    assert _read_json(tmp_path / "missing.json") is None
    assert retro120._load_sources(tmp_path / "results")[1] == []
    assert retro120._load_sources(tmp_path / "empty-results")[1] == list(SOURCE_FILES)
