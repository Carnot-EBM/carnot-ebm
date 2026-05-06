"""Tests for the Exp 1424 milestone .109 retrospective.

Spec: REQ-REPORT-032.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_109 import (
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _blocked_or_missing,
    _determined,
    _number,
    _truthy_sequence_or_bool,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scenario_sources() -> dict[str, dict[str, object]]:
    return {
        "exp1412": {
            "status": "complete",
            "bundle_exists": True,
            "submission_ready_for_operator": True,
            "credentialed_submission_attempted": False,
            "honest_verdict": "operator_action_sheet_ready_no_api_attempt",
        },
        "exp1413": {
            "status": "complete",
            "repair_execution_diagnosis_complete": True,
            "repair_hint_cases_total": 100,
            "executable_hint_pct": 1.0,
            "expected_full_pipeline_pass_rate_if_50pct_repaired": 0.555,
            "honest_verdict": "repair_execution_diagnosis_complete",
        },
        "exp1414": {
            "status": "complete",
            "repair_executor_deployed": True,
            "repair_hint_cases_tested": 20,
            "repaired_cases_successful": 0,
            "repaired_case_success_rate": 0.0,
            "semantic_equivalence_pass_rate_after_repair": 0.0,
            "local_sota_model_used": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_specs": [
                {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "gpu": 0},
                {"hf_id": "unsloth/gemma-4-31B-it-GGUF", "gpu": 1},
            ],
            "honest_verdict": "complete_repair_executor_no_successful_repairs",
        },
        "exp1415": {
            "status": "blocked",
            "fresh_verified_cases_used": 1508,
            "dvi_v2_auroc_delta_baseline": 0.011458,
            "dvi_v3_auroc_delta": 0.011842,
            "dvi_v3_deployed": False,
            "nonforgetting_rate": 0.968604,
            "honest_verdict": "dvi_v3_blocked_nonforgetting_below_gate",
        },
        "exp1416": {
            "status": "complete",
            "temperature_scaling_applied": True,
            "calibration_auroc_delta_after": 0.1855,
            "paraphrase_energy_variance_after_temp_scaling": 0.1027,
            "variance_worsened": False,
            "auroc_preserved": True,
            "honest_verdict": "temperature_scaling_reduced_variance_and_preserved_auroc",
        },
        "exp1417": {
            "status": "complete",
            "latent_drift_smoke_complete": True,
            "energy_monotone": True,
            "accuracy_delta_after_planning": -0.75,
            "dual_path_decoder_required": True,
            "anchoring_required": True,
            "honest_verdict": "energy_down_accuracy_down_off_decoder_support",
        },
        "exp1419": {
            "status": "complete",
            "cases_evaluated": 200,
            "certificate_parse_rate": 1.0,
            "semantic_validation_pass_rate": 1.0,
            "repair_hint_cases_total": 100,
            "repaired_cases_successful": 0,
            "repair_success_rate": 0.0,
            "full_pipeline_pass_rate": 0.305,
            "full_pipeline_headline_gate_met": False,
            "local_sota_model_used": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_specs": [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "gpu": 0}],
            "honest_verdict": "not_headline_full_pipeline_below_0_40",
        },
        "exp1420": {
            "status": "complete",
            "verified_pairs_available": 1508,
            "dpo_full_finetune_performed": False,
            "dpo_reranker_fallback_used": True,
            "dpo_improvement_pp": 99.834437,
            "dpo_vs_baseline_auroc": 1.0,
            "local_sota_model_used": False,
            "headline_result_allowed": False,
            "honest_verdict": "gguf_dpo_unsupported_reranker_fallback_measured",
        },
        "exp1421": {
            "status": "complete",
            "collection_clean_confirmed": True,
            "execution_failures_fixed": ["embedding-store cluster fixed"],
            "spec_coverage_checked": True,
            "remaining_debt": ["full suite remains red"],
            "honest_verdict": "focused_runtime_failures_fixed_remaining_debt",
        },
        "exp1422": {
            "status": "complete",
            "rtl_spec_complete": True,
            "kv260_budget_fits": True,
            "hardware_execution_performed": False,
            "hardware_claim_allowed": False,
            "honest_verdict": "rtl_spec_complete_budget_fits_no_synthesis",
        },
        "exp1423": {
            "status": "complete",
            "training_traces_used": 1030,
            "step_labels_available": 1824,
            "prmv1_trained": True,
            "prmv1_auroc": 0.832874,
            "prmv1_step_precision": 0.380282,
            "prmv1_step_recall": 0.6,
            "honest_verdict": "prmv1_trained_with_missing_local_labels",
        },
    }


def test_req_report_032_scores_milestone_109_from_source_fields() -> None:
    """REQ-REPORT-032: .109 criteria are scored from terminal artifacts."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=["exp1418"],
        conductor_log_text="FR-11 Self-Learning v6 GATE_BLOCK upstream retired exp1415",
        roadmap_next_present=False,
        change_proposal_present=True,
    )

    assert artifact["criteria_total"] == 13
    assert artifact["criteria_met"] == 10
    criteria = artifact["success_criteria_results"]
    assert criteria["dvi_v3_improves_on_v2"]["status"] == "BLOCKED"
    assert criteria["fr11_v6_headline_allowed"]["status"] == "GATE_BLOCKED"
    assert criteria["full_pipeline_clears_headline_gate"]["status"] == "NOT_MET"
    assert criteria["retro_complete"]["status"] == "MET"
    assert artifact["retired_experiments"][0]["experiment_id"] == (
        "exp1419-fullscale-pipeline-v3-repair-executor"
    )
    assert artifact["gpu_utilization_summary"]["utilization_metrics_available"] is False
    assert artifact["honest_verdict"].startswith("milestone_109_10_of_13_criteria_met")


def test_req_report_032_carry_forwards_have_prior_failures() -> None:
    """REQ-REPORT-032: every carry-forward task has a next prior_failures entry."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=["exp1418"],
        conductor_log_text="GATE_BLOCK upstream retired exp1415",
        roadmap_next_present=False,
        change_proposal_present=True,
    )

    required = artifact["prior_failures_required_next"]
    assert {task["id"] for task in artifact["carry_forward_tasks"]} == set(required)
    for task in artifact["carry_forward_tasks"]:
        priors = required[task["id"]]
        assert priors
        assert priors == task["prior_failures"]
        assert all(prior["experiment_id"] for prior in priors)
        assert all(prior["addressed_by"] for prior in priors)


def test_req_report_032_step0_skeleton_has_required_null_fields(tmp_path: Path) -> None:
    """REQ-REPORT-032: STEP 0 writes only an in-progress skeleton."""

    out_path = tmp_path / "results" / "experiment_1424_milestone_109_retro.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["status"] == "in_progress"
    assert set(written) == set(REQUIRED_ARTIFACT_FIELDS)
    assert all(written[field] is None for field in REQUIRED_ARTIFACT_FIELDS if field != "status")


def test_req_report_032_run_marks_missing_gated_source(tmp_path: Path) -> None:
    """REQ-REPORT-032: run writes final JSON and preserves the missing gate."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1424_milestone_109_retro.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "conductor-log.md").write_text(
        "FR-11 Self-Learning v6 | GATE_BLOCK | upstream retired exp1415",
        encoding="utf-8",
    )
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# vNEXT",
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["experiment_statuses"]["exp1418"]["exists"] is False
    assert written["success_criteria_results"]["fr11_v6_headline_allowed"]["status"] == (
        "GATE_BLOCKED"
    )
    assert any(
        item["experiment_id"] == "exp1418" and not item["exists"]
        for item in written["source_artifacts_checked"]
    )


def test_req_report_032_alternate_branches_are_auditable() -> None:
    """REQ-REPORT-032: non-blocked and malformed-source branches stay explicit."""

    sources = _scenario_sources()
    sources["exp1412"] = {"status": "blocked"}
    sources["exp1415"] = {
        "status": "complete",
        "dvi_v2_auroc_delta_baseline": 0.011458,
        "dvi_v3_auroc_delta": 0.02,
        "dvi_v3_deployed": True,
    }
    sources["exp1417"] = {"status": "complete", "latent_drift_smoke_complete": False}
    sources["exp1418"] = {
        "status": "complete",
        "headline_result_allowed": True,
        "fresh_verified_sample_count": 1600,
    }
    sources["exp1420"]["model_specs"] = ["malformed"]

    artifact = build_artifact(
        sources,
        missing_source_ids=[],
        conductor_log_text="",
        roadmap_next_present=True,
        change_proposal_present=False,
    )

    assert artifact["success_criteria_results"]["arxiv_action_sheet_complete"]["status"] == (
        "BLOCKED"
    )
    assert artifact["success_criteria_results"]["dvi_v3_improves_on_v2"]["status"] == "MET"
    assert artifact["success_criteria_results"]["fr11_v6_headline_allowed"]["status"] == "MET"
    assert artifact["experiment_statuses"]["exp1417"]["honest_verdict"] is None
    assert artifact["roadmap_inputs"]["missing_requested_inputs"] == []
    assert artifact["gpu_utilization_summary"]["observed_model_assignments"]
    assert _blocked_or_missing("exp1412", {}) == "NOT_MET"
    assert _number("not numeric") is None
    assert _determined("") is False
    assert _truthy_sequence_or_bool(True) is True
    assert _truthy_sequence_or_bool(object()) is False
