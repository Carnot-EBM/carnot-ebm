"""Tests for the Exp 1164 milestone .90 retrospective.

Spec traces: REQ-REPORT-010, SCENARIO-REPORT-007.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1164_milestone_retro_90 as exp1164


SAMPLE_LOG = """\
| 2026-05-02 09:01 UTC | Exp 906 IterativeSelfRepair 50q old carryover | OK | before .90 |
| 2026-05-02 10:50 UTC | Milestone 2026.04.90 activated | OK | 13 tasks queued |
| 2026-05-02 10:59 UTC | Gate Audit Pre-Activation v2 — Runs First, Fixes P | OK | tests passed |
| 2026-05-02 11:10 UTC | arXiv Final Submission v4 — Recompile PDF, Bundle, | OK | tests passed |
| 2026-05-02 11:21 UTC | Phase 3/4 Snap Validity Sweep — 10k DBAE-EBM State | OK | tests passed |
| 2026-05-02 11:36 UTC | Phase 3/4 HMC Compatibility Diagnostics D1-D4 — k= | OK | tests passed |
| 2026-05-02 11:52 UTC | Phase 3/4 HMC Sampler Conditional — Implement Regi | FAIL | bootstrap |
| 2026-05-02 11:59 UTC | Phase 3/4 HMC Sampler Conditional — Implement Regi | FAIL | bootstrap |
| 2026-05-02 12:12 UTC | Phase 3/4 HMC Sampler Conditional — Implement Regi | FAIL | stalled |
| 2026-05-02 12:46 UTC | Phase 3/4 HMC Sampler Conditional — Implement Regi | OK | exists |
| 2026-05-02 12:26 UTC | SECL-Guided Cheap-Tier Calibration — Fix FPR=0.96  | FAIL | bootstrap |
| 2026-05-02 12:46 UTC | SECL-Guided Cheap-Tier Calibration — Fix FPR=0.96 | OK | exists |
| 2026-05-02 13:26 UTC | BEAVER-lite with Real llama.cpp Logprobs — Live Pr | SKIP | pretests |
| 2026-05-02 13:41 UTC | BEAVER-lite with Real llama.cpp Logprobs — Live Pr | OK | exists |
| 2026-05-02 14:29 UTC | MARCH Multi-Agent Information-Asymmetric Claim-Che | OK | tests passed |
| 2026-05-02 14:41 UTC | KV260 v6 Sequential Gibbs Correctness Pivot — Deta | OK | tests passed |
| 2026-05-02 14:43 UTC | KANELE SOS-KAN FPGA Blueprint — LUT Specification  | DOOMED_RERUN_BLOCK | prior failures |
| 2026-05-02 15:50 UTC | KANELE SOS-KAN FPGA Blueprint — LUT Specification  | OK | tests passed |
| 2026-05-02 14:58 UTC | NRGPT Energy-Native LLM Prototype — Phase 3 Archit | FAIL | stalled |
| 2026-05-02 16:04 UTC | NRGPT Energy-Native LLM Prototype — Phase 3 Archit | OK | tests passed |
| 2026-05-02 16:10 UTC | Milestone 2026.04.91 activated | OK | next milestone |
"""


def _sample_artifacts() -> dict[int, dict]:
    return {
        1152: {
            "n_tasks_audited": 13,
            "n_prior_failures_missing": 7,
            "n_gate_upstream_failures": 0,
            "n_model_agent_coherence_failures": 0,
            "n_gate_field_cross_ref_failures": 0,
            "arxiv_task_prior_failures_complete": True,
            "roadmap_gate_audit_passed": False,
            "failure_details": ["PRIOR_FAILURES_COVERAGE exp1164"],
            "honest_verdict": "prior_failures_gaps_found",
        },
        1153: {
            "grpo_v2_result_in_paper": True,
            "projection_repair_in_paper": True,
            "metacluster_in_paper": True,
            "pdf_recompiled": True,
            "pdf_size_kb": 336.46,
            "bundle_verified": True,
            "manual_upload_steps": ["upload"],
            "arxiv_submitted": False,
            "honest_verdict": "paper_updated_recompiled",
        },
        1154: {
            "n_states_sampled": 10000,
            "snap_validity_gate_passed": True,
            "phase4_option_a_viable": True,
            "snap_validity_rate": 1.0,
            "honest_verdict": "option_a_viable_above_95pct",
        },
        1155: {
            "hmc_regime_classified": True,
            "hmc_regime": "C",
            "recommended_sampler": "blocked_gibbs",
            "honest_verdict": "regime_C_hmc_inappropriate",
        },
        1156: {
            "hmc_sampler_honest_result": True,
            "active_inference_sampler_ready": True,
            "kl_divergence_vs_boltzmann": 0.023,
            "honest_verdict": "sampler_kl_below_05_viable",
        },
        1157: {
            "cheap_tier_fpr_below_30pct": True,
            "cheap_tier_tp_above_80pct": True,
            "secl_tp_rate": 0.916667,
            "secl_fpr": 0.21,
            "honest_verdict": "calibrated_tp_above_80_fpr_below_30",
        },
        1158: {
            "beaver_lite_live_logprobs_sound_bound": True,
            "bound_is_sound_live": True,
            "bound_tighter_than_mock": True,
            "unsafe_mass_bound_live": 0.318656,
            "empirical_violation_rate_live": 0.3,
            "llama_cpp_available": False,
            "zipf_mock_used": True,
            "honest_verdict": "sound_bound_zipf_mock",
        },
        1159: {
            "grpo_v4_honest_result": True,
            "dualgpu_used": True,
            "structural_warmup_used": True,
            "improvement_over_baseline": 0.1,
            "improvement_vs_exp1129": 0.0149,
            "n_eval_questions": 50,
            "honest_verdict": "structural_warmup_above_0851",
        },
        1160: {
            "march_multiagent_honest_result": True,
            "march_tp_above_baseline": True,
            "march_tp_rate": 1.0,
            "march_fpr": 0.0,
            "honest_verdict": "march_tp_above_semenergy_baseline",
        },
        1161: {
            "kv260_v6_kl_below_threshold_sequential_gibbs": True,
            "kl_v6_vs_cpu_n8_mean": 0.0,
            "kl_threshold": 0.05,
            "rtl_spec_written": True,
            "honest_verdict": "kl_near_zero_algorithm_correct",
        },
        1162: {
            "kanele_fpga_blueprint_generated": True,
            "blueprint_written": True,
            "estimated_speedup_factor": 2408333.333333,
            "honest_verdict": "blueprint_generated_speedup_above_100x",
        },
        1163: {
            "nrgpt_phase3_prototype_honest_result": True,
            "nrgpt_above_baseline": True,
            "n_iters_monotone": False,
            "baseline_auroc": 0.887409,
            "nrgpt_auroc_n3": 0.915784,
            "honest_verdict": "nrgpt_above_baseline_energy_recurrence_helps",
        },
    }


def _write_artifacts(results_dir: Path, artifacts: dict[int, dict]) -> None:
    for exp_id, payload in artifacts.items():
        filename = exp1164.EXPERIMENT_FILES[exp_id]
        (results_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_criteria_counts_gate_audit_gap_as_unmet_req_report_010():
    criteria = exp1164.evaluate_criteria(_sample_artifacts())

    assert len(criteria) == 13
    assert criteria["arxiv_submitted_or_bundle_v4_ready"] is True
    assert criteria["gate_audit_pre_activation_passed"] is False
    assert criteria["hmc_sampler_honest_result"] is True
    assert criteria["kv260_v6_kl_below_threshold_sequential_gibbs"] is True
    assert criteria["retro_complete"] is True
    assert sum(criteria.values()) == 12


def test_milestone_specific_status_fields_are_source_derived_scenario_report_007():
    artifacts = _sample_artifacts()

    assert exp1164.arxiv_submission_status(artifacts) == "upload_pending"
    assert exp1164.arxiv_submission_status({1153: {"arxiv_submitted": True}}) == "submitted"
    assert exp1164.arxiv_submission_status({1153: {"_missing": True}}) == "not_run"
    assert exp1164.phase34_mandatory_tasks_complete(artifacts) is True
    assert exp1164.phase34_mandatory_tasks_complete({1154: artifacts[1154]}) is False
    assert exp1164.kv260_v6_kl_below_threshold(artifacts) is True
    assert exp1164.kv260_v6_kl_below_threshold({1161: {}}) is False


def test_build_artifact_has_required_schema_and_self_verdict_req_report_010():
    artifact = exp1164.build_artifact(
        artifacts=_sample_artifacts(),
        log_lines=SAMPLE_LOG.splitlines(),
        prior_wall_time_minutes=257.0,
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
        "phase34_mandatory_tasks_complete",
        "kv260_v6_kl_below_threshold",
        "wall_time_minutes",
        "retro_complete",
        "honest_verdict",
    }
    assert required.issubset(artifact)
    assert artifact["milestone"] == "2026.04.90"
    assert artifact["criteria_met"] == 12
    assert artifact["criteria_total"] == 13
    assert artifact["honest_verdict"] == "12_of_13_criteria_met"
    assert artifact["experiment_honest_verdicts"]["exp1164"] == "12_of_13_criteria_met"
    assert artifact["arxiv_submission_status"] == "upload_pending"
    assert artifact["phase34_mandatory_tasks_complete"] is True
    assert artifact["kv260_v6_kl_below_threshold"] is True
    assert artifact["wall_time_minutes"] == 314.0
    assert artifact["wall_time_improvement_vs_prior_minutes"] == -57.0
    assert len(artifact["bottlenecks_identified"]) == 3


def test_slowest_5_excludes_exp906_and_uses_90_log_spans_req_report_010():
    slowest = exp1164.build_slowest_experiments(SAMPLE_LOG.splitlines())

    ids = [entry["id"] for entry in slowest]
    assert "exp906" not in ids
    assert ids[:3] == ["exp1162", "exp1163", "exp1156"]
    assert slowest[0]["duration_min"] == 67.0
    assert slowest[1]["duration_min"] == 66.0
    assert exp1164.compute_wall_time_minutes([]) == 0.0


def test_missing_artifact_and_numeric_fallbacks_are_explicit_req_report_010(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    artifacts = {1153: {"honest_verdict": "paper_updated_recompiled"}}
    (results_dir / exp1164.EXPERIMENT_FILES[1153]).write_text(
        json.dumps(artifacts[1153]), encoding="utf-8"
    )

    loaded = exp1164.load_artifacts(results_dir)
    artifact = exp1164.build_artifact(loaded, ["not a markdown row"], 257.0)

    assert loaded[1153]["honest_verdict"] == "paper_updated_recompiled"
    assert loaded[1152]["_missing"] is True
    assert exp1164.record_honest_verdicts(loaded)["exp1152"] == "MISSING"
    assert exp1164._float_value("not-a-number", default=7.5) == 7.5
    assert exp1164._exp_id_for_title("Unmapped task title") is None
    assert exp1164.build_slowest_experiments(["not a markdown row"]) == []
    assert artifact["criteria_met"] == 1
    assert any("missing" in item.lower() for item in artifact["failures_or_partials"])


def test_main_writes_deliverable_for_req_report_010(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_artifacts(results_dir, _sample_artifacts())
    log_path = tmp_path / "conductor-log.md"
    log_path.write_text(SAMPLE_LOG, encoding="utf-8")
    out_path = tmp_path / "experiment_1164_milestone_retro_90.json"

    code = exp1164.main(
        [
            "--results-dir",
            str(results_dir),
            "--conductor-log",
            str(log_path),
            "--out",
            str(out_path),
            "--prior-wall-time-minutes",
            "257",
        ]
    )

    assert code == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["retro_complete"] is True
    assert written["experiment_honest_verdicts"]["exp1152"] == "prior_failures_gaps_found"
