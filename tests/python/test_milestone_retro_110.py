"""Tests for the Exp 1438 milestone .110 retrospective.

Spec: REQ-REPORT-035, SCENARIO-REPORT-035.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_110 import (
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
        "exp1425": {
            "status": "complete",
            "carryforward_manifest_complete": True,
            "carryforward_task_count": 6,
            "honest_verdict": (
                "milestone_110_carryforward_manifest_complete_6_tracks_exp1419_"
                "exact_rerun_forbidden"
            ),
            "same_verdict_retirement_rules": [
                {
                    "experiment_id": "exp1415-dvi-v3-1508-fresh-cases",
                    "prior_verdict": "dvi_v3_blocked_nonforgetting_below_gate",
                    "retire_if_same_verdict_rule": "Retire this DVI v3 repair variant.",
                }
            ],
            "forbidden_exact_reruns": [
                {
                    "experiment_id": "exp1419-fullscale-pipeline-v3-repair-executor",
                    "forbidden_scope": (
                        "exact exp1419 200-case full-scale pipeline rerun without "
                        "nonzero accepted repair evidence"
                    ),
                    "prior_verdict": "not_headline_full_pipeline_below_0_40",
                    "retire_if_same_verdict": True,
                }
            ],
        },
        "exp1426": {
            "status": "complete",
            "failure_cluster_map_complete": True,
            "collection_clean_confirmed": True,
            "spec_coverage_debt_count": 71,
            "next_cluster_recommended": "spec_coverage_traceability_metadata",
            "honest_verdict": (
                "diagnostic_cluster_map_complete_collection_clean_spec_coverage_red_"
                "71_full_suite_not_rerun"
            ),
        },
        "exp1427": {
            "status": "complete",
            "rejection_ledger_complete": True,
            "top_rejection_reason": "missing_output_or_nonjson_response",
            "repair_v2_contract_ready": True,
            "nonzero_repair_gate_required": True,
            "honest_verdict": (
                "complete_rejection_ledger_schema_failures_dominant_"
                "repair_v2_contract_ready_nonzero_gate_required"
            ),
        },
        "exp1428": {
            "status": "complete",
            "repair_executor_v2_deployed": True,
            "repaired_case_success_rate": 1.0,
            "repaired_cases_successful": 20,
            "local_sota_model_inference_used": False,
            "honest_verdict": (
                "complete_dccd_schema_constrained_repair_v2_nonzero_repairs_"
                "prototype_no_headline_sota_claim"
            ),
        },
        "exp1429": {
            "status": "complete",
            "candidate_search_complete": True,
            "repair_success_rate_best_of_n": 1.0,
            "repair_success_rate_one_candidate": 0.0,
            "local_sota_model_inference_used": False,
            "honest_verdict": (
                "complete_mcmc_constrained_repair_candidate_search_improved_"
                "prototype_no_headline_sota_claim_live_sota_inference_not_run"
            ),
        },
        "exp1430": {
            "status": "complete",
            "prm_guided_selection_ready": True,
            "selected_repair_success_rate": 1.0,
            "raw_best_of_n_repair_success_rate": 1.0,
            "selection_improvement_pp": 0.0,
            "honest_verdict": (
                "complete_prm_guided_selector_no_improvement_prototype_candidate_pool_"
                "no_headline_claim"
            ),
        },
        "exp1431": {
            "status": "complete",
            "cases_evaluated": 50,
            "full_pipeline_pass_rate": 0.62,
            "beats_exp1419_baseline": True,
            "runtime_evidence_allows_headline_scaleup": False,
            "eligible_for_200_case_scaleup": False,
            "honest_verdict": (
                "complete_micro_validation_beats_exp1419_baseline_prototype_no_headline_scaleup"
            ),
        },
        "exp1432": {
            "status": "complete",
            "dvi_v3_deployed": True,
            "nonforgetting_rate": 1.0,
            "dvi_v3_auroc_delta": 0.011842,
            "dvi_v2_auroc_delta_baseline": 0.011458,
            "dvi_v3_auroc_nonregression_gate": True,
            "honest_verdict": "dvi_v3_deployed_replay_heldout_threshold_calibrated",
        },
        "exp1433": {
            "status": "complete",
            "dvi_v3_checkpoint_active": True,
            "headline_result_allowed": False,
            "v6_candidate_cases_evaluated": 7313,
            "v6_new_promoted_count": 0,
            "self_learning_delta_overall": 0,
            "session_memory_updated": False,
            "honest_verdict": "fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline",
        },
        "exp1434": {
            "status": "complete",
            "missing_labels_filled": 478,
            "missing_labels_remaining": 0,
            "prmv2_trained": True,
            "headline_label_coverage_ready": True,
            "honest_verdict": "prmv2_trained_all_promoted_traces_have_local_labels",
        },
        "exp1435": {
            "status": "complete",
            "headline_provenance_ready": False,
            "reranker_track_relabelled": True,
            "direct_gguf_finetune_supported": False,
            "honest_verdict": (
                "dpo_headline_not_ready_reranker_only_until_adapter_or_conversion_tooling"
            ),
        },
        "exp1436": {
            "status": "complete",
            "anchored_repair_viable": True,
            "accuracy_delta_after_planning": 0.0,
            "latent_drift_norm": 0.7071067782902382,
            "honest_verdict": "anchored_dual_path_repair_viable",
        },
        "exp1437": {
            "status": "blocked",
            "rtl_lint_complete": False,
            "simulation_complete": False,
            "hardware_claim_allowed": False,
            "hardware_execution_performed": False,
            "rtl_sources_checked": [{"path": "hardware/kv260/discrete_sb_256.v", "exists": False}],
            "honest_verdict": "blocked_missing_discrete_sb_rtl_source",
        },
    }


def test_req_report_035_scores_all_milestone_110_criteria() -> None:
    """REQ-REPORT-035: every .110 roadmap criterion is scored from source fields."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=[],
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        conductor_log_text="Exp 1437 No file changes produced",
    )

    assert artifact["criteria_total"] == 14
    assert artifact["criteria_met"] == 12
    criteria = artifact["success_criteria_results"]
    assert criteria["repair_executor_v2"]["status"] == "met"
    assert criteria["continuous_self_learning"]["status"] == "not_met"
    assert criteria["rtl_evidence"]["status"] == "blocked"
    assert criteria["retro"]["status"] == "met"
    assert artifact["repair_v2_verdict"]["accepted_repairs"] == 20
    assert artifact["dvi_fr11_verdict"]["dvi_v3_deployed"] is True
    assert artifact["dvi_fr11_verdict"]["fr11_headline_allowed"] is False
    assert artifact["prm_verdict"]["missing_labels_filled"] == 478
    assert artifact["hardware_verdict"]["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"].startswith("milestone_110_12_of_14_criteria_met")


def test_req_report_035_carry_forward_rules_preserve_prior_verdicts() -> None:
    """SCENARIO-REPORT-035: carry-forward rules include same-verdict retire gates."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=[],
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        conductor_log_text="",
    )

    tasks = {task["id"]: task for task in artifact["carry_forward_tasks"]}
    assert "fr11_positive_growth_followup" in tasks
    assert "hardware_rtl_source_before_lint_sim" in tasks
    assert tasks["fr11_positive_growth_followup"]["prior_failures"][0]["verdict"] == (
        "fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline"
    )
    assert tasks["fr11_positive_growth_followup"]["retire_if_same_verdict"] is True
    assert tasks["test_debt_spec_coverage_cluster"]["prior_failures"][0]["verdict"].startswith(
        "diagnostic_cluster_map_complete"
    )
    assert any("exp1419 200-case" in scope["scope"] for scope in artifact["retired_exact_scopes"])
    assert any("DPO headline" in scope["scope"] for scope in artifact["retired_exact_scopes"])


def test_req_report_035_step0_skeleton_has_required_null_fields(tmp_path: Path) -> None:
    """REQ-REPORT-035: STEP 0 writes an in-progress artifact with required fields."""

    out_path = tmp_path / "results" / "experiment_1438_milestone_110_retro.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["status"] == "in_progress"
    assert set(written) == set(REQUIRED_ARTIFACT_FIELDS)
    assert all(written[field] is None for field in REQUIRED_ARTIFACT_FIELDS if field != "status")


def test_req_report_035_run_marks_missing_source_as_not_run(tmp_path: Path) -> None:
    """REQ-REPORT-035: missing source artifacts are explicit not-run criteria."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1438_milestone_110_retro.json"
    sources = _scenario_sources()
    sources.pop("exp1437")
    for exp_id, payload in sources.items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "conductor-log.md").write_text("exp1438 retro", encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# roadmap",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text(
        "milestone: 2026.04.110\n",
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["criteria_met"] == 12
    assert written["success_criteria_results"]["rtl_evidence"]["status"] == "not_run"
    assert written["source_artifacts_checked"][-1] == {
        "experiment_id": "exp1437",
        "path": "results/experiment_1437_discrete_sb_kv260_rtl_lint_sim.json",
        "exists": False,
    }


def test_req_report_035_alternate_source_states_stay_auditable() -> None:
    """REQ-REPORT-035: blocked and malformed terminal values stay explicit."""

    sources = _scenario_sources()
    sources["exp1425"]["status"] = "blocked"
    sources["exp1432"]["dvi_v2_auroc_delta_baseline"] = "not numeric"
    sources["exp1437"] = {
        "status": "complete",
        "rtl_lint_complete": False,
        "simulation_complete": False,
        "hardware_claim_allowed": False,
        "hardware_execution_performed": False,
        "honest_verdict": "complete_no_lint_result",
    }

    artifact = build_artifact(
        sources,
        missing_source_ids=[],
        roadmap_doc_present=False,
        roadmap_yaml_present=False,
        conductor_log_text="",
        roadmap_next_present=True,
    )

    assert artifact["success_criteria_results"]["carry_forward_manifest"]["status"] == "blocked"
    assert artifact["success_criteria_results"]["dvi_nonforgetting"]["status"] == "not_met"
    assert artifact["success_criteria_results"]["rtl_evidence"]["status"] == "not_met"
    assert artifact["roadmap_inputs"]["requested_research_roadmap_next_present"] is True
