"""Tests for the Exp 1190 milestone .92 retrospective.

Spec traces: REQ-REPORT-012, SCENARIO-REPORT-009.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1190_milestone_retro_92 as exp1190


SAMPLE_LOG = """\
| 2026-05-02 22:27 UTC | Milestone 2026.04.92 activated | OK | 13 tasks queued |
| 2026-05-02 22:40 UTC | Pytest Memory Watchdog -- Per-Test RSS Monitoring + | OK | pass |
| 2026-05-02 22:49 UTC | llama.cpp GPU Offload Fix -- Rebuild with LLAMA_CUD | SKIP | pretests |
| 2026-05-02 23:31 UTC | llama.cpp GPU Offload Fix -- Rebuild with LLAMA_CUD | SKIP | pretests |
| 2026-05-02 23:49 UTC | llama.cpp GPU Offload Fix -- Rebuild with LLAMA_CUD | SKIP | pretests |
| 2026-05-03 00:31 UTC | Paper v5 Integrity -- Critical ISSUE-1 to ISSUE-5 + | SKIP | pretests |
| 2026-05-03 01:19 UTC | Paper v5 Integrity -- Critical ISSUE-1 to ISSUE-5 + | SKIP | pretests |
| 2026-05-03 05:47 UTC | WOPR Hex Game Cartridge -- Constraint-Satisfaction  | OK | pass |
"""


def _sample_artifacts() -> dict[int, dict]:
    return {
        1178: {
            "watchdog_operational": True,
            "sample_test_run_passed": True,
            "honest_verdict": "watchdog_operational",
            "duration_s": 2.0,
        },
        1181: {
            "high_severity_fixed": 5,
            "4_test_passes_high": True,
            "honest_verdict": "all_5_high_resolved",
            "duration_s": 10.0,
        },
        1182: {
            "medium_low_issues_fixed": 8,
            "paper_claim_audit_script_active": True,
            "honest_verdict": "all_8_medium_low_resolved",
            "duration_s": 12.0,
        },
        1183: {
            "arxiv_bundle_v6_ready": False,
            "4_test_full_pass": False,
            "pdf_compiles_without_error": False,
            "prerequisites_met": False,
            "honest_verdict": "audit_failures_remain",
            "duration_s": 3.0,
        },
        1184: {
            "status": "blocked",
            "training_completed": False,
            "dualgpu_confirmed": False,
            "grpo_v5_delta_pp": -0.26,
            "honest_verdict": "gpu_offload_prerequisite_not_met",
            "duration_s": 1.0,
        },
        1185: {
            "sc_energy_regularized": True,
            "overfit_resolved": True,
            "k6_viable_for_production": False,
            "retire_k6": True,
            "k6_regularized_auroc": 0.902707,
            "k5_auroc_on_eval": 0.92403,
            "honest_verdict": "overfit_resolved_but_k6_still_regresses",
            "duration_s": 1.903,
        },
        1186: {
            "token_gradient_norms_near_zero": True,
            "redesign_implemented": True,
            "redesigned_dot_auroc": 0.4699,
            "retire_dot": True,
            "honest_verdict": "dot_retired",
            "duration_s": 3.24,
        },
        1187: {
            "latent_grpo_implemented": True,
            "latent_grpo_delta_pp": 0.0,
            "one_sided_noise_applied": True,
            "honest_verdict": "latent_grpo_no_delta",
            "duration_s": 1.0,
        },
        1188: {
            "hex_game_operational": True,
            "energy_player_beats_random": True,
            "random_vs_gibbs_win_rate": 0.9,
            "honest_verdict": "hex_operational_energy_player_wins",
            "duration_s": 10.839,
        },
        1189: {
            "stronger_baseline_implemented": True,
            "free_energy_values_all_puzzles": True,
            "phase4_vs_bfs_delta_reported": True,
            "honest_verdict": "phase4_tied_with_bfs",
            "duration_s": 4.0,
        },
    }


def _write_artifacts(results_dir: Path, artifacts: dict[int, dict]) -> None:
    for exp_id, payload in artifacts.items():
        filename = exp1190.EXPERIMENT_FILES[exp_id]
        (results_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_criteria_counts_missing_gates_as_failures_req_report_012() -> None:
    """REQ-REPORT-012: all 13 .92 criteria are derived from source artifacts."""
    criteria = exp1190.evaluate_criteria(_sample_artifacts())
    status = exp1190.criteria_status(criteria)

    assert len(criteria) == 13
    assert status["exp1178_watchdog_operational"] == "PASS"
    assert status["exp1179_gpu_offload_verified"] == "FAIL"
    assert status["exp1180_critical_issues_fixed"] == "FAIL"
    assert status["exp1183_arxiv_bundle_v6_ready"] == "FAIL"
    assert status["exp1184_grpo_v5_result_honest"] == "PASS"
    assert status["exp1186_dot_diagnosis_complete"] == "PASS"
    assert exp1190.criteria_met_count(criteria) == 10
    assert exp1190.criteria_results(criteria)["exp1180_critical_issues_fixed"] is False


def test_publication_hold_remains_until_full_audit_passes_scenario_report_009() -> None:
    """SCENARIO-REPORT-009: missing critical artifacts keep publication hold active."""
    artifact = exp1190.build_artifact(_sample_artifacts(), SAMPLE_LOG.splitlines())

    assert artifact["milestone"] == "2026.04.92"
    assert artifact["criteria_met"] == 10
    assert artifact["criteria_score_pct"] == 76.92
    assert artifact["paper_integrity_issues_resolved"] == 13
    assert artifact["publication_hold_lifted"] is False
    assert artifact["grpo_v5_result"] == "gpu_offload_prerequisite_not_met"
    assert artifact["k6_viable"] is False
    assert artifact["k6_retired"] is True
    assert artifact["dot_retired"] is True
    assert artifact["latent_grpo_delta_pp"] == 0.0
    assert artifact["hex_operational"] is True
    assert artifact["phase4_stronger_baseline_result"] == "phase4_tied_with_bfs"
    assert artifact["slowest_task_id"] == "exp1179"
    assert artifact["honest_verdict"] == "milestone_partial"
    assert any("exp1179" in item for item in artifact["open_items_for_93"])
    assert any("exp1180" in item for item in artifact["open_items_for_93"])


def test_publication_hold_lift_true_requires_operator_approval_req_report_012() -> None:
    """REQ-REPORT-012: all issue and audit gates plus operator approval lift the hold."""
    artifacts = _sample_artifacts()
    artifacts[1179] = {
        "llama_cpp_gpu_offload_verified": True,
        "throughput_tokens_per_sec": 55.0,
        "honest_verdict": "gpu_offload_verified",
    }
    artifacts[1180] = {
        "critical_issues_fixed": 5,
        "figure_integrity_script_active": True,
        "4_test_passes_critical": True,
        "honest_verdict": "all_5_critical_resolved",
    }
    artifacts[1183] = {
        **artifacts[1183],
        "arxiv_bundle_v6_ready": True,
        "4_test_full_pass": True,
        "pdf_compiles_without_error": True,
        "operator_explicit_approval": True,
    }

    artifact = exp1190.build_artifact(artifacts, [])

    assert exp1190.publication_hold_lifted(artifacts) is True
    assert artifact["criteria_met"] == 13
    assert artifact["publication_hold_lifted"] is True
    assert artifact["honest_verdict"] == "milestone_complete"
    assert artifact["slowest_task_id"] == "exp1182"


def test_empty_artifacts_are_failed_and_missing_verdicts_are_explicit_req_report_012() -> None:
    """REQ-REPORT-012: missing source JSONs are explicit failures, not fabricated wins."""
    artifacts: dict[int, dict] = {}
    artifact = exp1190.build_artifact(artifacts, [])

    assert artifact["criteria_met"] == 1
    assert artifact["honest_verdict"] == "milestone_failed"
    assert artifact["experiment_honest_verdicts"]["exp1178"] == "MISSING"
    assert artifact["slowest_task_id"] == "unknown"
    assert exp1190.paper_integrity_issues_resolved(artifacts) == 0


def test_slowest_task_log_parser_handles_boundaries_req_report_012() -> None:
    """REQ-REPORT-012: conductor spans ignore invalid, unmapped, and out-of-window rows."""
    log_lines = [
        "| 2026-05-02 22:20 UTC | Pytest Memory Watchdog -- before milestone | OK | ignored |",
        "not a markdown row",
        "| 2026-05-02 22:27 UTC | Milestone 2026.04.92 activated | OK | start |",
        "| 2026-05-02 22:30 UTC | Unmapped .92 task | OK | ignored |",
        "| 2026-05-02 22:40 UTC | Pytest Memory Watchdog -- Per-Test RSS Monitoring + | OK | pass |",
        "| 2026-05-02 22:45 UTC | Pytest Memory Watchdog -- Per-Test RSS Monitoring + | OK | pass |",
        "| 2026-05-02 22:50 UTC | Milestone 2026.04.93 activated | OK | stop |",
        "| 2026-05-02 23:50 UTC | llama.cpp GPU Offload Fix -- outside window | OK | ignored |",
    ]

    slowest = exp1190.build_slowest_tasks(log_lines)

    assert slowest == [{"rank": 1, "id": "exp1178", "duration_min": 5.0, "attempts_seen": 2}]
    assert exp1190._exp_id_for_title("Unmapped .92 task") is None


def test_load_artifacts_and_main_write_required_schema_req_report_012(tmp_path: Path) -> None:
    """REQ-REPORT-012: main writes the machine-readable Exp1190 deliverable."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_artifacts(results_dir, _sample_artifacts())
    log_path = tmp_path / "conductor-log.md"
    log_path.write_text(SAMPLE_LOG, encoding="utf-8")
    out_path = tmp_path / "experiment_1190_milestone_retro_92.json"

    loaded = exp1190.load_artifacts(results_dir)
    code = exp1190.main(
        [
            "--results-dir",
            str(results_dir),
            "--conductor-log",
            str(log_path),
            "--out",
            str(out_path),
        ]
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded[1179]["_missing"] is True
    assert code == 0
    assert written["criteria_total"] == 13
    assert written["criteria_met"] == 10
    assert written["criteria_status"]["exp1179_gpu_offload_verified"] == "FAIL"
    assert written["publication_hold_lifted"] is False
