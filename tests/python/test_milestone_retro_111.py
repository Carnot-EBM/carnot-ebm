"""Tests for the Exp 1452 milestone .111 retrospective.

Spec: REQ-REPORT-038, SCENARIO-REPORT-038.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_111 import (
    GATE_BLOCKED_WITH_EVIDENCE,
    MET,
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
        "exp1439": {
            "status": "complete",
            "carryforward_manifest_complete": True,
            "carryforward_task_count": 6,
            "honest_verdict": (
                "milestone_111_carryforward_manifest_complete_6_tracks_prototype_"
                "fr11_prm_rtl_exact_reruns_forbidden"
            ),
        },
        "exp1440": {
            "status": "complete",
            "spec_coverage_metadata_cluster_fixed": True,
            "initial_spec_coverage_debt_count": 71,
            "final_spec_coverage_debt_count": 0,
            "honest_verdict": (
                "metadata_cluster_fixed_spec_coverage_zero_focused_tests_green_"
                "changed_line_coverage_100_full_suite_red_101_failures_6_errors"
            ),
        },
        "exp1441": {
            "status": "complete",
            "rtl_source_created": True,
            "rtl_source_path": "hardware/kv260/discrete_sb_256.v",
            "testbench_created": True,
            "testbench_path": "hardware/kv260/tb_discrete_sb_256.v",
            "honest_verdict": (
                "rtl_source_and_testbench_created_lint_and_smoke_sim_passed_"
                "no_kv260_execution_claim"
            ),
        },
        "exp1442": {
            "status": "complete",
            "local_sota_runtime_ready": False,
            "live_sota_model_inference_used": False,
            "models_found_in_cache": [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}],
            "models_missing_from_cache": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "blockers": ["no_mandated_sota_model_completed_live_inference"],
            "honest_verdict": "blocked_no_live_sota_runtime",
        },
        "exp1444": {
            "status": "blocked",
            "gate_check_summary": (
                "1 of 1 gate(s) failed; first failure: exp1443-live-sota-"
                "dccd-semctrl-repair-v3.live_repair_candidate_pool_ready"
            ),
            "gates_evaluated": [
                {
                    "upstream": "exp1443-live-sota-dccd-semctrl-repair-v3",
                    "artifact_field": "live_repair_candidate_pool_ready",
                    "passed": False,
                    "reason": "upstream artifact not found",
                }
            ],
            "honest_verdict": "blocked_gate_check_failed",
        },
        "exp1446": {
            "status": "complete",
            "fr11_zero_growth_root_cause_identified": True,
            "fr11_zero_growth_root_cause": "asymmetric_fresh_threshold",
            "honest_verdict": (
                "fr11_v6_zero_growth_root_cause_identified_asymmetric_fresh_threshold_v7_required"
            ),
        },
        "exp1447": {
            "status": "complete",
            "self_learning_delta_overall": 156,
            "nonforgetting_preserved": True,
            "nonforgetting_rate": 1.0,
            "live_sota_inference_used": False,
            "honest_verdict": "fr11_v7_positive_verified_growth_persisted_without_forgetting",
        },
        "exp1448": {
            "status": "complete",
            "pra_selector_ready": True,
            "selection_improvement_pp": 0.0,
            "step_scores_generated": 320,
            "cases_evaluated": 20,
            "regression_against_prm_v1": False,
            "honest_verdict": (
                "complete_prmv3_no_headline_improvement_prototype_candidate_pool_no_headline_claim"
            ),
        },
        "exp1449": {
            "status": "complete",
            "ltlzinc_adapter_ready": True,
            "temporal_cases_generated": 24,
            "accepted_case_count": 12,
            "honest_verdict": "ltlzinc_temporal_adapter_ready_verified_cases_only_no_training",
        },
        "exp1450": {
            "status": "complete",
            "energy_convergence_probe_complete": True,
            "scale_recommendation": "keep_smoke_only",
            "honest_verdict": "energy_converges_but_no_decoded_quality_claim_keep_smoke_only",
        },
        "exp1451": {
            "status": "complete",
            "rtl_lint_complete": True,
            "simulation_complete": True,
            "hardware_claim_allowed": False,
            "hardware_execution_performed": False,
            "honest_verdict": "rtl_lint_and_simulation_complete_no_hardware_execution_no_kv260_claim",
        },
    }


def test_req_report_038_scores_111_successes_and_gate_blocks() -> None:
    """REQ-REPORT-038: every .111 criterion is scored without counting gates as wins."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=["exp1443", "exp1445"],
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=False,
    )

    assert artifact["criteria_total"] == 14
    assert artifact["criteria_met"] == 10
    assert artifact["ops_docs_updated"] is False
    criteria = artifact["success_criteria_results"]
    assert criteria["carry_forward_manifest"]["status"] == MET
    assert criteria["live_sota_runtime"]["status"] == GATE_BLOCKED_WITH_EVIDENCE
    assert criteria["live_repair_v3"]["status"] == GATE_BLOCKED_WITH_EVIDENCE
    assert criteria["energy_reranker"]["status"] == GATE_BLOCKED_WITH_EVIDENCE
    assert criteria["pipeline_pre_scale"]["status"] == GATE_BLOCKED_WITH_EVIDENCE
    assert criteria["continuous_self_learning"]["status"] == MET
    assert criteria["retro"]["status"] == MET
    assert artifact["missing_artifacts"] == [
        {
            "experiment_id": "exp1443",
            "path": "results/experiment_1443_live_sota_dccd_semctrl_repair_v3.json",
        },
        {
            "experiment_id": "exp1445",
            "path": "results/experiment_1445_full_pipeline_v5_100case_prescale.json",
        },
    ]
    assert artifact["honest_verdict"].startswith(
        "milestone_111_10_of_14_criteria_met_threshold_not_met"
    )
    assert any("live-SOTA repair chain" in lesson for lesson in artifact["lessons_learned"])


def test_scenario_report_038_carry_forward_rules_preserve_verdicts() -> None:
    """SCENARIO-REPORT-038: .112 carry-forward rules include exact same-verdict gates."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=["exp1443", "exp1445"],
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=False,
    )

    carry_forward = {track["id"]: track for track in artifact["carry_forward_tracks"]}
    assert carry_forward["live_sota_runtime_repair_gate"]["prior_failures"][0]["verdict"] == (
        "blocked_no_live_sota_runtime"
    )
    assert carry_forward["live_sota_runtime_repair_gate"]["retire_if_same_verdict"] is True
    assert (
        carry_forward["repair_v3_and_prescale_gated_missing"]["prior_failures"][0]["verdict"]
        == "missing_artifact_gate_blocked_by_exp1442"
    )
    assert carry_forward["prm_process_agent_no_improvement"]["prior_failures"][0][
        "verdict"
    ].startswith("complete_prmv3_no_headline_improvement")
    assert any("no-live-SOTA runtime" in item["scope"] for item in artifact["retired_variants"])
    assert any("PRM v3 no-improvement" in item["scope"] for item in artifact["retired_variants"])


def test_req_report_038_step0_skeleton_has_required_null_fields(tmp_path: Path) -> None:
    """REQ-REPORT-038: STEP 0 writes an in-progress artifact with required fields."""

    out_path = tmp_path / "results" / "experiment_1452_milestone_111_retro.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["status"] == "in_progress"
    assert set(written) == set(REQUIRED_ARTIFACT_FIELDS)
    assert all(written[field] is None for field in REQUIRED_ARTIFACT_FIELDS if field != "status")


def test_req_report_038_run_marks_missing_gated_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-038: missing gated artifacts are explicit and not counted as met."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1452_milestone_111_retro.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# roadmap",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text(
        "milestone: 2026.04.111\n",
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["criteria_met"] == 10
    assert written["roadmap_inputs"]["requested_research_roadmap_next_present"] is False
    assert written["source_artifacts_checked"][4] == {
        "experiment_id": "exp1443",
        "path": "results/experiment_1443_live_sota_dccd_semctrl_repair_v3.json",
        "exists": False,
    }
    assert (
        written["success_criteria_results"]["pipeline_pre_scale"]["status"]
        == GATE_BLOCKED_WITH_EVIDENCE
    )


def test_req_report_038_alternate_source_states_stay_auditable() -> None:
    """REQ-REPORT-038: positive, missing, and blocked alternate paths remain explicit."""

    sources = _scenario_sources()
    sources["exp1439"]["forbidden_exact_reruns"] = [
        {
            "forbidden_scope": "exact prototype-only repair scale-up",
            "prior_verdicts": ["prototype_no_headline"],
            "retire_if_same_verdict": True,
        }
    ]
    sources["exp1442"] = {
        "status": "complete",
        "local_sota_runtime_ready": True,
        "live_sota_model_inference_used": True,
        "honest_verdict": "live_sota_runtime_ready",
    }
    sources["exp1443"] = {
        "status": "complete",
        "live_sota_inference_used": True,
        "live_repair_success_rate": 0.25,
        "live_repair_candidate_pool_ready": True,
        "honest_verdict": "live_repair_v3_nonzero",
    }
    sources["exp1444"] = {
        "status": "complete",
        "energy_reranker_ready": True,
        "false_acceptance_rate_delta": 0.0,
        "honest_verdict": "energy_reranker_ready_no_false_acceptance_increase",
    }
    sources["exp1445"] = {
        "status": "complete",
        "full_pipeline_pass_rate": 0.7,
        "honest_verdict": "pipeline_v5_prescale_passed",
    }

    artifact = build_artifact(
        sources,
        missing_source_ids=[],
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=True,
    )

    criteria = artifact["success_criteria_results"]
    assert criteria["live_sota_runtime"]["status"] == MET
    assert criteria["live_repair_v3"]["status"] == MET
    assert criteria["energy_reranker"]["status"] == MET
    assert criteria["pipeline_pre_scale"]["status"] == MET
    assert artifact["criteria_met"] == 14
    assert any(
        item["scope"] == "exact prototype-only repair scale-up"
        for item in artifact["retired_variants"]
    )


def test_req_report_038_missing_and_blocked_branches_are_not_successes() -> None:
    """SCENARIO-REPORT-038: missing non-gated work and blocker artifacts are not successes."""

    sources = _scenario_sources()
    sources.pop("exp1440")
    sources.pop("exp1444")
    sources["exp1445"] = {
        "status": "complete",
        "full_pipeline_pass_rate": 0.5,
        "honest_verdict": "pipeline_v5_honest_blocker",
    }

    artifact = build_artifact(
        sources,
        missing_source_ids=["exp1440", "exp1443", "exp1444"],
        roadmap_doc_present=False,
        roadmap_yaml_present=True,
        roadmap_next_present=False,
    )

    criteria = artifact["success_criteria_results"]
    assert criteria["spec_coverage_cluster"]["status"] == "unmet"
    assert criteria["energy_reranker"]["status"] == "unmet"
    assert criteria["pipeline_pre_scale"]["status"] == GATE_BLOCKED_WITH_EVIDENCE
    assert artifact["missing_artifacts"][0] == {
        "experiment_id": "exp1440",
        "path": "results/experiment_1440_spec_coverage_traceability_metadata_fix.json",
    }
