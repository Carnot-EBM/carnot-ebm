"""Tests for the Exp 1138 milestone .88 retrospective.

Spec traces: REQ-REPORT-009, SCENARIO-REPORT-006.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1138_milestone_retro_88 as exp1138


SAMPLE_LOG = """\
| 2026-05-02 01:19 UTC | Milestone 2026.04.88 activated | OK | 12 tasks queued |
| 2026-05-02 01:29 UTC | SOSKANEnergyV3 Root Cause Diagnosis + k=5 AND-Comp | DOOMED_RERUN_BLOCK | missing prior_failures |
| 2026-05-02 02:04 UTC | Lagrangian Cascade v2 — Accuracy-Preserving Router | GATE_BLOCK | upstream retired |
| 2026-05-02 02:17 UTC | PRM-BiasBench Adversarial Test on k=5 AND-Compose | GATE_BLOCK | upstream retired |
| 2026-05-02 02:20 UTC | Lagrangian Cascade v2 — Accuracy-Preserving Router | GATE_BLOCK | upstream retired |
| 2026-05-02 02:44 UTC | SOS-KAN Polarity Fix v2 — k=5 Ensemble Net-Positiv | OK | tests passed |
| 2026-05-02 02:59 UTC | Zenil alpha_t Re-Measurement v2 — Post-Retrain Verifie | OK | tests passed |
| 2026-05-02 03:12 UTC | PRM-BiasBench Adversarial Test on k=5 AND-Compose | OK | deliverable exists |
| 2026-05-02 03:36 UTC | Cascade Routing v3 — Accuracy-Preserving Lagrangia | OK | tests passed |
| 2026-05-02 03:38 UTC | WOPR Slitherlink Puzzle Cartridge — Loop Constrain | DOOMED_RERUN_BLOCK | missing prior_failures |
| 2026-05-02 03:42 UTC | WOPR Slitherlink Puzzle Cartridge — Loop Constrain | DOOMED_RERUN_BLOCK | missing prior_failures |
| 2026-05-02 03:44 UTC | HF Spaces Gallery Update — Deploy Slitherlink Cart | GATE_BLOCK | upstream retired |
"""


def _sample_artifacts() -> dict[int, dict]:
    return {
        1127: {
            "pdf_compiled": True,
            "arxiv_submitted": False,
            "arxiv_bundle_verified": True,
            "honest_verdict": "pdf_compiled_upload_pending",
        },
        1128: {
            "sos_kan_root_cause_identified": True,
            "k5_ensemble_auroc_above_08": True,
            "k5_ensemble_auroc_after": 0.9402,
            "sos_kan_individual_auroc_after": 0.9902,
            "honest_verdict": "fixed_k5_above_08",
        },
        1129: {
            "grpo_v2_honest_result": True,
            "training_wall_budget_hit": False,
            "improvement_over_baseline": 0.0851,
            "n_training_questions": 100,
            "evaluation_wall_budget_hit": True,
            "honest_verdict": "positive_improvement",
        },
        1130: {
            "zenil_alpha_t_post_retrain_measured": True,
            "alpha_t_post_retrain": 0.52,
            "alpha_t_prior": 0.38,
            "honest_verdict": "alpha_t_improved",
        },
        1131: {
            "cascade_v2_accuracy_delta_above_neg05": True,
            "accuracy_delta": 0.0,
            "cost_savings_pct": 3.2,
            "honest_verdict": "savings_accuracy_both_positive",
        },
        1132: {
            "goodfire_exemplar_tp_rate_measured": True,
            "per_tier_results_logged": True,
            "z3_math_standalone_tp_rate": 0.083333,
            "honest_verdict": "mixed_results",
        },
        1133: {
            "prm_biasbench_attack_tp_measured": True,
            "k5_attack_tp_rate": 1.0,
            "honest_verdict": "z3_dominates_style_irrelevant",
        },
        1134: {
            "parameter_space_mapped": True,
            "kv260_v4_kl_below_05_or_feasibility_documented": True,
            "kl_v4_best": 0.1127718014422604,
            "kl_v4_threshold": 0.05,
            "honest_verdict": "kl_improved_not_below_threshold",
        },
        1135: {
            "position_paper_findings_updated": True,
            "honest_verdict": "fully_updated",
        },
        1136: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "5 prior failure(s) match this task's scope",
        },
        1137: {"_missing": True, "_path": "results/experiment_1137_hf_spaces_gallery_update.json"},
    }


def _write_artifacts(results_dir: Path, artifacts: dict[int, dict]) -> None:
    for exp_id, payload in artifacts.items():
        if payload.get("_missing"):
            continue
        filename = exp1138.EXPERIMENT_FILES[exp_id]
        (results_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_criteria_counts_blocked_slitherlink_as_unmet_req_report_009():
    criteria = exp1138.evaluate_criteria(_sample_artifacts())

    assert len(criteria) == 11
    assert criteria["slitherlink_cartridge_shipped"] is False
    assert criteria["retro_complete"] is True
    assert sum(criteria.values()) == 10


def test_record_honest_verdicts_includes_missing_gallery_scenario_report_006():
    verdicts = exp1138.record_honest_verdicts(_sample_artifacts())

    assert verdicts["exp1136"] == "blocked_gate_check_failed"
    assert verdicts["exp1137"] == "MISSING"


def test_build_artifact_has_required_schema_and_honest_verdict_req_report_009():
    artifact = exp1138.build_artifact(
        artifacts=_sample_artifacts(),
        log_lines=SAMPLE_LOG.splitlines(),
    )

    required = {
        "milestone",
        "criteria_met",
        "criteria_total",
        "criteria_results",
        "notable_successes",
        "failures_or_partials",
        "bottlenecks_identified",
        "exp906_appeared_in_slowest5",
        "wall_time_minutes",
        "retro_complete",
        "honest_verdict",
    }
    assert required.issubset(artifact)
    assert artifact["milestone"] == "2026.04.88"
    assert artifact["criteria_met"] == 10
    assert artifact["criteria_total"] == 11
    assert artifact["honest_verdict"] == "10_of_11_criteria_met"
    assert artifact["exp906_appeared_in_slowest5"] is False


def test_slowest_5_excludes_exp906_and_uses_log_spans_req_report_009():
    slowest = exp1138.build_slowest_experiments(SAMPLE_LOG.splitlines())

    ids = [entry["id"] for entry in slowest]
    assert "exp906" not in ids
    assert ids[0] == "exp1131"
    assert slowest[0]["duration_min"] == 92.0


def test_wall_time_and_improvement_vs_891_baseline_req_report_009():
    wall_time = exp1138.compute_wall_time_minutes(SAMPLE_LOG.splitlines())

    assert wall_time == 145.0
    assert exp1138.compute_wall_time_minutes([]) == 0.0


def test_log_and_number_fallbacks_are_explicit_req_report_009():
    noisy_log = [
        "not a markdown table row",
        "| 2026-05-02 01:10 UTC | Plan milestone 2026.04.88 | OK | pre-start |",
        "| 2026-05-02 01:19 UTC | Milestone 2026.04.88 activated | OK | start |",
        "| 2026-05-02 01:20 UTC | Unmapped task title | OK | ignored |",
    ]

    assert exp1138._float_value("not-a-number", default=7.5) == 7.5
    assert exp1138.compute_wall_time_minutes(noisy_log) == 1.0
    assert exp1138.build_slowest_experiments(noisy_log) == []


def test_main_writes_deliverable_for_req_report_009(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_artifacts(results_dir, _sample_artifacts())
    log_path = tmp_path / "conductor-log.md"
    log_path.write_text(SAMPLE_LOG, encoding="utf-8")
    out_path = tmp_path / "experiment_1138_milestone_retro_88.json"

    code = exp1138.main(
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
    assert written["experiment_honest_verdicts"]["exp1137"] == "MISSING"
