"""Tests for the Exp 1151 milestone .89 retrospective.

Spec traces: REQ-REPORT-009, SCENARIO-REPORT-006.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1151_milestone_retro_89 as exp1151


SAMPLE_LOG = """\
| 2026-05-02 03:12 UTC | Exp 906 IterativeSelfRepair 50q old carryover | OK | before .89 |
| 2026-05-02 05:44 UTC | Milestone 2026.04.89 activated | OK | 13 tasks queued |
| 2026-05-02 05:46 UTC | arXiv Final Submission Close-Out v3 — PDF with GRP | DOOMED_RERUN_BLOCK | missing prior_failures |
| 2026-05-02 05:48 UTC | arXiv Final Submission Close-Out v3 — PDF with GRP | DOOMED_RERUN_BLOCK | missing prior_failures |
| 2026-05-02 05:50 UTC | arXiv Final Submission Close-Out v3 — PDF with GRP | DOOMED_RERUN_BLOCK | missing prior_failures |
| 2026-05-02 06:20 UTC | Roadmap Gate and Prior-Failures Audit Script v1 —  | SKIP | pre-tests failing |
| 2026-05-02 06:49 UTC | Roadmap Gate and Prior-Failures Audit Script v1 —  | SKIP | pre-tests failing |
| 2026-05-02 07:02 UTC | Roadmap Gate and Prior-Failures Audit Script v1 —  | OK | audit completed |
| 2026-05-02 08:33 UTC | GRPO Reflection Reward v3 — Energy-Delta Repair Si | FAIL | bootstrap |
| 2026-05-02 08:43 UTC | GRPO Reflection Reward v3 — Energy-Delta Repair Si | FAIL | bootstrap |
| 2026-05-02 09:05 UTC | GRPO Reflection Reward v3 — Energy-Delta Repair Si | FAIL | bootstrap |
| 2026-05-02 09:16 UTC | HardNet++-Style Projection Repair Layer for Arithm | OK | tests passed |
| 2026-05-02 09:32 UTC | MetaCluster SOS-KAN Compression — 5x Checkpoint Sh | OK | tests passed |
| 2026-05-02 09:49 UTC | KV260 Ising v5 — DC-Continuous Relaxation Diagnost | OK | tests passed |
| 2026-05-02 10:01 UTC | Extropic Z1/XTR-0 Integration Packet — THRML Parit | OK | tests passed |
| 2026-05-02 10:45 UTC | Milestone 2026.04.90 activated | OK | next milestone |
"""


def _sample_artifacts() -> dict[int, dict]:
    return {
        1139: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "1 prior failure(s) match this task's scope",
            "blocked_at_layer": "conductor_pre_gate",
        },
        1140: {
            "audit_script_written": True,
            "n_tasks_audited": 13,
            "n_prior_failures_checks": 13,
            "n_prior_failures_missing": 5,
            "roadmap_gate_audit_passed": False,
            "failure_details": [
                "PRIOR_FAILURES_COVERAGE exp1139-arxiv-final-submission-v3: prior_failures is missing",
                "PRIOR_FAILURES_COVERAGE exp1151-milestone-retro-89: prior_failures is missing",
            ],
            "honest_verdict": "prior_failures_gaps_found",
        },
        1141: {
            "slitherlink_cartridge_shipped": True,
            "canonical_puzzle_e_at_convergence": 0.0,
            "tests_passing": 5,
            "n_iterations_to_convergence": 1,
            "honest_verdict": "e0_achieved",
        },
        1142: {
            "beaver_lite_bounder_written": True,
            "beaver_lite_bound_reported": True,
            "bound_is_sound": True,
            "mock_logprobs_used": True,
            "unsafe_mass_bound": 0.4,
            "empirical_violation_rate": 0.4,
            "honest_verdict": "sound_bound_mock_logprobs",
        },
        1143: {
            "halluguard_routing_feature_measured": True,
            "halluguard_features_explain_goodfire_failures": True,
            "adaptive_tp_rate": 1.0,
            "fixed_tp_rate": 1.0,
            "accuracy_delta": 0.0,
            "cost_savings_pct": 4.4,
            "honest_verdict": "features_explain_goodfire_failures",
        },
        1144: {
            "cctu_adapter_written": True,
            "cctu_adapter_honest_result": True,
            "n_tasks_defined": 25,
            "n_tasks_evaluated": 25,
            "inference_mode": "live_gpu",
            "model_used": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "baseline_completion_rate": 0.04,
            "carnot_guided_completion_rate": 0.12,
            "carnot_delta_pp": 0.08,
            "honest_verdict": "carnot_positive_delta",
        },
        1145: {
            "n_exemplars": 36,
            "cheap_tier_tp_rate_improved": True,
            "combined_cheap_tp_before": 0.361111,
            "combined_cheap_tp_after": 0.916667,
            "false_positive_rate_after": 0.96,
            "dominant_halluguard_feature": "entropy_proxy",
            "honest_verdict": "cheap_tier_calibrated_tp_improved",
        },
        1146: {
            "dualgpu_used": True,
            "training_wall_budget_hit": False,
            "reflection_reward_integrated": True,
            "grpo_reflection_honest_result": True,
            "advantage_stdev": 0.12293561647712808,
            "evaluation_wall_budget_hit": True,
            "n_eval_questions": 35,
            "n_eval_questions_target": 50,
            "improvement_over_baseline": 0.0286,
            "honest_verdict": "positive_below_exp1129",
        },
        1147: {
            "projection_repair_written": True,
            "hardnet_projection_repair_written": True,
            "projection_repair_accuracy": 1.0,
            "projection_repair_latency_us": 117.33625,
            "speedup_factor": 76130.41274846558,
            "honest_verdict": "projection_accurate_and_fast",
        },
        1148: {
            "sos_kan_compressed": True,
            "auroc_drop_within_02": True,
            "auroc_drop": 0.018447,
            "size_reduction_factor": 5.026627,
            "energy_correlation": 0.996599,
            "honest_verdict": "compressed_within_02_auroc_5x_smaller",
        },
        1149: {
            "kv260_v5_diagnostic_complete": True,
            "energy_time_accuracy_reported": True,
            "kl_v5_below_threshold": False,
            "kl_v5_best": 0.4469032615902366,
            "kl_v4_best_prior": 0.1128,
            "kl_improvement_over_v4": -0.33410326159023657,
            "honest_verdict": "kl_unchanged_topology_wall",
        },
        1150: {
            "integration_packet_written": True,
            "thrml_backend_stub_written": True,
            "sampler_backend_interface_documented": True,
            "thrml_available": False,
            "honest_verdict": "thrml_not_available_packet_written",
        },
    }


def _write_artifacts(results_dir: Path, artifacts: dict[int, dict]) -> None:
    for exp_id, payload in artifacts.items():
        filename = exp1151.EXPERIMENT_FILES[exp_id]
        (results_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_criteria_counts_blocked_arxiv_as_unmet_req_report_009():
    criteria = exp1151.evaluate_criteria(_sample_artifacts())

    assert len(criteria) == 13
    assert criteria["arxiv_final_pdf_recompiled_and_upload_steps_provided"] is False
    assert criteria["gate_prior_failures_audit_complete"] is True
    assert criteria["kv260_v5_dc_continuous_kl_measured"] is True
    assert criteria["retro_complete"] is True
    assert sum(criteria.values()) == 12


def test_verdicts_status_and_gate_audit_gap_detection_req_report_009():
    artifacts = _sample_artifacts()

    assert exp1151.record_honest_verdicts(artifacts)["exp1139"] == "blocked_gate_check_failed"
    assert exp1151.arxiv_submission_status(artifacts) == "not_run"
    assert exp1151.roadmap_gate_audit_caught_blocking_gaps(artifacts) is True
    assert exp1151.arxiv_submission_status({1139: {"arxiv_submitted": True}}) == "submitted"
    assert exp1151.arxiv_submission_status({1139: {"arxiv_submitted": False}}) == "upload_pending"


def test_build_artifact_has_required_schema_and_honest_verdict_req_report_009():
    artifact = exp1151.build_artifact(
        artifacts=_sample_artifacts(),
        log_lines=SAMPLE_LOG.splitlines(),
    )

    required = {
        "milestone",
        "criteria_met",
        "criteria_total",
        "criteria_results",
        "notable_successes",
        "bottlenecks_identified",
        "exp906_appeared_in_slowest5",
        "arxiv_submission_status",
        "wall_time_minutes",
        "retro_complete",
        "honest_verdict",
    }
    assert required.issubset(artifact)
    assert artifact["milestone"] == "2026.04.89"
    assert artifact["criteria_met"] == 12
    assert artifact["criteria_total"] == 13
    assert artifact["honest_verdict"] == "12_of_13_criteria_met"
    assert artifact["arxiv_submission_status"] == "not_run"
    assert artifact["roadmap_gate_audit_caught_blocking_gaps"] is True
    assert artifact["wall_time_minutes"] == 257.0
    assert artifact["wall_time_improvement_vs_prior_minutes"] == -112.0


def test_slowest_5_excludes_exp906_and_uses_89_log_spans_req_report_009():
    slowest = exp1151.build_slowest_experiments(SAMPLE_LOG.splitlines())

    ids = [entry["id"] for entry in slowest]
    assert "exp906" not in ids
    assert ids[:3] == ["exp1140", "exp1146", "exp1139"]
    assert slowest[0]["duration_min"] == 42.0
    assert slowest[1]["duration_min"] == 32.0
    assert exp1151.compute_wall_time_minutes([]) == 0.0


def test_missing_artifact_and_number_fallbacks_are_explicit_scenario_report_006(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    artifacts = {1139: {"honest_verdict": "blocked_gate_check_failed"}}
    (results_dir / exp1151.EXPERIMENT_FILES[1139]).write_text(
        json.dumps(artifacts[1139]), encoding="utf-8"
    )

    loaded = exp1151.load_artifacts(results_dir)

    assert loaded[1139]["honest_verdict"] == "blocked_gate_check_failed"
    assert loaded[1140]["_missing"] is True
    assert exp1151.record_honest_verdicts(loaded)["exp1140"] == "MISSING"
    assert exp1151._float_value("not-a-number", default=7.5) == 7.5
    assert exp1151.compute_wall_time_minutes(["not a markdown table row"]) == 0.0
    assert exp1151._exp_id_for_title("Unmapped task title") is None
    assert exp1151.build_slowest_experiments(["not a markdown table row"]) == []


def test_main_writes_deliverable_for_req_report_009(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_artifacts(results_dir, _sample_artifacts())
    log_path = tmp_path / "conductor-log.md"
    log_path.write_text(SAMPLE_LOG, encoding="utf-8")
    out_path = tmp_path / "experiment_1151_milestone_retro_89.json"

    code = exp1151.main(
        [
            "--results-dir",
            str(results_dir),
            "--conductor-log",
            str(log_path),
            "--out",
            str(out_path),
        ]
    )

    assert code == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["retro_complete"] is True
    assert written["experiment_honest_verdicts"]["exp1149"] == "kl_unchanged_topology_wall"
