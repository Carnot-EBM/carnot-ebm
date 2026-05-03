"""Tests for the Exp 1177 milestone .91 retrospective.

Spec traces: REQ-REPORT-011, SCENARIO-REPORT-008.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1177_milestone_retro_91 as exp1177


def _sample_artifacts() -> dict[int, dict]:
    return {
        1165: {
            "prototype_operational": True,
            "n_puzzles_evaluated": 10,
            "action_count_ratio": 0.253419,
            "phase4_solved_rate": 1.0,
            "honest_verdict": "phase4_better_than_baseline",
        },
        1166: {
            "themesis_email_drafted": True,
            "leaderboard_evidence_source": "documented_fallback",
            "honest_verdict": "leaderboard_unavailable_email_drafted",
        },
        1167: {
            "section7_expanded": True,
            "phase4_results_in_paper": True,
            "pdf_recompiled": True,
            "bundle_verified": True,
            "paper_ready_for_arxiv_hold_lift": False,
            "honest_verdict": "paper_v4_phase4_section_added_fpga_figure_blocking",
            "manual_override_2026_05_02T18_40Z": {"reason": "fig3 integrity audit"},
        },
        1168: {
            "sc_energy_auroc_above_threshold": True,
            "sc_energy_auroc": 1.0,
            "all_r_below_0.5": True,
            "honest_verdict": "sc_energy_viable_k6_ready",
        },
        1169: {
            "fover_sota_pairs_v6_above_500": True,
            "n_new_pairs": 1000,
            "honest_verdict": "corpus_expanded_labels_complete",
        },
        1170: {
            "mock_logprobs_used": False,
            "bound_is_sound": True,
            "honest_verdict": "live_logprobs_sound_bound",
        },
        1171: {
            "dot_inference_pareto_measured": True,
            "monotone_improvement": False,
            "honest_verdict": "non_monotone_diminishing_returns",
        },
        1172: {
            "nrgpt_per_token_energy_above_batch": True,
            "per_token_auroc": 0.998199,
            "batch_auroc_baseline": 0.887409,
            "honest_verdict": "per_token_improves_auroc",
        },
        1173: {
            "status": "blocked",
            "inference_mode": "blocked_no_dualgpu",
            "blocked_reason": "llama.cpp runtime lacks GPU offload support",
            "dualgpu_confirmed": False,
            "grpo_v5_honest_result": False,
            "honest_verdict": "training_wall_hit",
        },
        1174: {
            "bika_hardware_analysis_complete": True,
            "bika_resource_reduction_pct": 39.634,
            "npu_feasibility_verdict": "npu_feasible",
            "honest_verdict": "bika_feasible_for_npu",
        },
        1175: {
            "cartridge_shipped": True,
            "n_tests_passing": 10,
            "honest_verdict": "cartridge_shipped_e0_at_convergence",
        },
        1176: {
            "k6_and_compose_auroc_measured": True,
            "k6_above_k5": False,
            "k6_auroc": 0.897344,
            "k5_auroc_on_eval": 0.92403,
            "honest_verdict": "k6_no_improvement",
        },
    }


def _write_artifacts(results_dir: Path, artifacts: dict[int, dict]) -> None:
    for exp_id, payload in artifacts.items():
        filename = exp1177.EXPERIMENT_FILES[exp_id]
        (results_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_criteria_scores_all_13_source_fields_req_report_011() -> None:
    """REQ-REPORT-011: every planned .91 criterion is evaluated from source fields."""
    criteria = exp1177.evaluate_criteria(_sample_artifacts())
    status = exp1177.criteria_status(criteria)

    assert len(criteria) == 13
    assert status == {
        "phase4_prototype_operational": "MET",
        "themesis_leaderboard_comparison_documented": "MET",
        "paper_v4_phase4_section_integrated": "NOT_MET",
        "sc_energy_7th_verifier_auroc_above_threshold": "MET",
        "fover_sota_pairs_v6_above_500": "MET",
        "beaver_live_logprobs_sound_bound": "MET",
        "dot_inference_pareto_measured": "MET",
        "nrgpt_per_token_energy_above_batch": "MET",
        "grpo_v5_honest_result": "GATE_BLOCKED",
        "bika_hardware_analysis_complete": "MET",
        "connect_four_cartridge_shipped": "MET",
        "k6_and_compose_auroc_measured": "MET",
        "retro_complete": "MET",
    }
    assert exp1177.criteria_met_count(criteria) == 11


def test_build_artifact_uses_exp1167_override_for_hold_lift_scenario_report_008() -> None:
    """SCENARIO-REPORT-008: Exp1167 false keeps publication hold-lift readiness false."""
    artifact = exp1177.build_artifact(_sample_artifacts())

    assert artifact["milestone"] == "2026.04.91"
    assert artifact["criteria_total"] == 13
    assert artifact["criteria_met"] == 11
    assert artifact["criteria_status"]["paper_v4_phase4_section_integrated"] == "NOT_MET"
    assert artifact["criteria_status"]["grpo_v5_honest_result"] == "GATE_BLOCKED"
    assert artifact["phase4_hold_lift_ready"] is False
    assert artifact["phase4_hold_lift_note"].startswith(
        "Phase 4 hold-lift prerequisites are not met"
    )
    assert artifact["honest_verdict"] == "11_of_13_criteria_met"
    assert len(artifact["top_3_successes"]) == 3
    assert len(artifact["top_3_gaps"]) == 3
    assert len(artifact["open_items_for_92"]) >= 3
    assert any("figure-integrity" in item for item in artifact["open_items_for_92"])


def test_hold_lift_true_adds_required_operator_note_req_report_011() -> None:
    """REQ-REPORT-011: ready Exp1167 emits the exact operator-review note."""
    artifacts = _sample_artifacts()
    artifacts[1167] = {
        **artifacts[1167],
        "paper_ready_for_arxiv_hold_lift": True,
        "honest_verdict": "paper_v4_phase4_complete_arxiv_ready",
    }

    artifact = exp1177.build_artifact(artifacts)

    assert artifact["phase4_hold_lift_ready"] is True
    assert artifact["phase4_hold_lift_note"] == (
        "Phase 4 hold-lift prerequisites are met; operator should review "
        "docs/arxiv-paper/main.pdf and carnot-arxiv-v5.tar.gz for arXiv submission."
    )
    assert artifact["criteria_status"]["paper_v4_phase4_section_integrated"] == "MET"
    assert artifact["criteria_met"] == 12


def test_missing_artifacts_are_not_fabricated_as_success_req_report_011(tmp_path: Path) -> None:
    """REQ-REPORT-011: absent source JSONs score NOT_MET and appear in source verdicts."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    payload = _sample_artifacts()[1165]
    (results_dir / exp1177.EXPERIMENT_FILES[1165]).write_text(json.dumps(payload), encoding="utf-8")

    loaded = exp1177.load_artifacts(results_dir)
    artifact = exp1177.build_artifact(loaded)

    assert loaded[1166]["_missing"] is True
    assert artifact["criteria_status"]["themesis_leaderboard_comparison_documented"] == "NOT_MET"
    assert artifact["experiment_honest_verdicts"]["exp1166"] == "MISSING"
    assert artifact["criteria_met"] == 2


def test_main_writes_required_deliverable_schema_req_report_011(tmp_path: Path) -> None:
    """REQ-REPORT-011: main writes the machine-readable Exp1177 deliverable."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_artifacts(results_dir, _sample_artifacts())
    out_path = tmp_path / "experiment_1177_milestone_retro_91.json"

    code = exp1177.main(["--results-dir", str(results_dir), "--out", str(out_path)])

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0
    assert written["criteria_met"] == 11
    assert written["criteria_total"] == 13
    assert written["criteria_status"]["retro_complete"] == "MET"
    assert written["honest_verdict"] == "11_of_13_criteria_met"
