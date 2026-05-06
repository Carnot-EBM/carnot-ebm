"""Tests for the Exp 1402 milestone .108 retrospective.

Spec: REQ-REPORT-009.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_108 import (
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scenario_sources() -> dict[str, dict[str, object]]:
    return {
        "exp1390": {
            "status": "complete",
            "submission_attempted": False,
            "submission_result": "manual_checklist_generated",
            "submission_method": "manual_checklist_no_credentials",
            "manual_checklist_generated": True,
            "manual_checklist_path": "docs/arxiv-manual-submission-checklist.md",
            "bundle_path": "results/arxiv_bundle_v11.tar.gz",
            "honest_verdict": "credentials_missing_manual_submission_checklist_generated",
        },
        "exp1391": {
            "status": "complete",
            "failure_analysis_complete": True,
            "top_failure_category": "CORPUS_SPECIFIC",
            "failure_categories": [{"category": "CORPUS_SPECIFIC"}],
            "fixable_failure_pct": 1.0,
            "honest_verdict": "diagnosis_complete",
        },
        "exp1392": {
            "status": "complete",
            "collection_errors_after": 0,
            "test_suite_collection_clean": True,
            "validation": {
                "full_suite_tail_result": "3 failed, 5393 passed",
                "spec_coverage_result": "failed_preexisting",
            },
            "honest_verdict": "collection_hygiene_target_met",
        },
        "exp1393": {
            "status": "complete",
            "grpo_v8_improvement_pp": 0.0,
            "unknown_rollout_rate": 1.0,
            "formal_reward_pass_rate": 0.0,
            "retire_if_same_verdict": True,
            "terminal_blocker": "ngrpo_still_zero_reward",
            "honest_verdict": "grpo_v8_ngrpo_no_improvement_all_unknown_retired",
        },
        "exp1394": {
            "status": "complete",
            "dvi_v2_deployed": True,
            "dvi_v2_auroc_delta": 0.011458,
            "dvi_v2_trained_auroc": 0.405984,
            "fresh_cases_used": 59,
            "secl_ece_reduction_pct": 45.35096,
            "honest_verdict": "dvi_v2_deployed",
        },
        "exp1395": {
            "status": "complete",
            "fresh_verified_sample_count": 1508,
            "grpo_v8_cases_integrated": 0,
            "dvi_v2_checkpoint_active": True,
            "headline_result_allowed": True,
            "self_learning_delta_overall": 1449,
            "honest_verdict": "fr11_v5_fresh_1508_grpo_0",
        },
        "exp1396": {
            "status": "complete",
            "semantic_validation_improvement_measured": True,
            "semantic_validation_pass_rate_before_fix": 0.59,
            "semantic_validation_pass_rate_after_fix": 1.0,
            "fixes_applied": [{"category": "CORPUS_SPECIFIC"}],
            "all_exp1391_failures_recovered_after_fix_count": 41,
            "honest_verdict": "semantic_fix_measured",
        },
        "exp1397": {
            "status": "complete",
            "cases_evaluated": 200,
            "certificate_parse_rate": 1.0,
            "semantic_validation_pass_rate": 1.0,
            "full_pipeline_pass_rate": 0.305,
            "headline_result_allowed": False,
            "headline_metric_gate_passed": False,
            "scheduler_accept_rate": 0.305,
            "repair_hint_precision": 1.0,
            "honest_verdict": "not_headline_full_pipeline_below_0_40",
        },
        "exp1398": {
            "status": "complete",
            "ngrpo_advantage_calibration_verified": True,
            "theory_supports_exp1393": True,
            "ngrpo_augmented_advantage_variance": 0.16,
            "original_resZero_advantage_variance": 0.0,
            "honest_verdict": "theory_supports_exp1393",
        },
        "exp1399": {
            "status": "complete",
            "bram_budget_feasible": True,
            "hardware_claim_allowed": True,
            "kv260_claim_allowed": True,
            "convergence_speedup_discrete_sb": 1.3846153846153846,
            "metadata": {"hardware_execution_performed": False, "synthesis_performed": False},
            "honest_verdict": "estimate_only",
        },
        "exp1400": {
            "status": "complete",
            "pivot_precision_delta": -0.013334,
            "retrospective_verification_viable": False,
            "forward_only_pivot_precision": 0.986667,
            "biprm_r2l_pivot_precision": 0.973333,
            "human_annotated_pivot_cases": 0,
            "honest_verdict": "not_viable_negative_r2l_pivot_precision_delta",
        },
        "exp1401": {
            "status": "complete",
            "calibration_auroc_delta": 0.18570365800551603,
            "variance_worsened": True,
            "consistency_regularization_weight": 0.0,
            "ebm_cot_v2_auroc": 0.9855748294382348,
            "paraphrase_energy_variance_before": 0.0005379585431871225,
            "paraphrase_energy_variance_after": 0.16537272058221922,
            "honest_verdict": "hinge_only_positive_variance_worsened",
        },
    }


def test_req_report_009_counts_milestone_108_from_source_fields() -> None:
    """REQ-REPORT-009: .108 criteria are computed from experiment artifacts."""

    artifact = build_artifact(
        _scenario_sources(),
        [],
        roadmap_next_present=False,
        change_proposal_present=True,
    )

    assert artifact["criteria_total"] == 13
    assert artifact["criteria_met"] == 12
    assert artifact["criteria_results"]["arxiv_submitted"]["status"] == "NOT_MET"
    assert artifact["criteria_results"]["grpo_ngrpo_measured"]["status"] == "MET"
    assert artifact["criteria_results"]["biprm_verified"]["status"] == "MET"
    assert artifact["biprm_verdict"]["retrospective_verification_viable"] is False
    assert artifact["full_pipeline_v2_verdict"]["full_pipeline_pass_rate"] == 0.305
    assert artifact["roadmap_inputs"]["missing_requested_inputs"] == ["research-roadmap-next.yaml"]
    assert (
        artifact["prior_failure_hygiene_notes"]["retirements_triggered"][0]["experiment_id"]
        == "exp1393"
    )
    assert artifact["honest_verdict"].startswith("milestone_108_12_of_13_criteria_met")


def test_req_report_009_step0_skeleton_has_required_null_fields(tmp_path: Path) -> None:
    """REQ-REPORT-009: STEP 0 writes only an in-progress skeleton."""

    out_path = tmp_path / "results" / "experiment_1402_milestone_108_retro.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["status"] == "in_progress"
    assert set(written) == set(REQUIRED_ARTIFACT_FIELDS)
    assert all(written[field] is None for field in REQUIRED_ARTIFACT_FIELDS if field != "status")


def test_req_report_009_run_marks_missing_sources(tmp_path: Path) -> None:
    """REQ-REPORT-009: missing artifacts are explicit and not fabricated."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1402_milestone_108_retro.json"
    sources = _scenario_sources()
    sources.pop("exp1400")
    for exp_id, payload in sources.items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    _write_json(tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md", {})

    artifact = run(root=tmp_path, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["experiment_statuses"]["exp1400"]["exists"] is False
    assert written["criteria_results"]["biprm_verified"]["status"] == "NOT_MET"
    assert written["prior_failure_hygiene_notes"]["missing_result_artifacts"] == ["exp1400"]
    assert any(
        item["experiment_id"] == "exp1400" and not item["exists"]
        for item in written["source_artifacts_checked"]
    )
