"""Tests for the Exp 1202 milestone .93 retrospective.

Spec traces: REQ-REPORT-013, SCENARIO-REPORT-010.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1202_milestone_retro_93 as exp1202


SAMPLE_LOG = """\
| 2026-05-03 06:51 UTC | Milestone 2026.04.93 activated | OK | 12 tasks queued |
| 2026-05-03 07:36 UTC | prlimit Memory Cap -- resource.setrlimit RLIMIT_AS  | FAIL | Post-tests failed |
| 2026-05-03 07:38 UTC | prlimit Memory Cap -- resource.setrlimit RLIMIT_AS | OK | Deliverable already exists |
| 2026-05-03 07:43 UTC | llama.cpp GPU Offload Fix v2 -- Pre-built CUDA Whee | SKIP | pretests |
| 2026-05-03 08:08 UTC | llama.cpp GPU Offload Fix v2 -- Pre-built CUDA Whee | SKIP | pretests |
| 2026-05-03 09:11 UTC | llama.cpp GPU Offload Fix v2 -- Pre-built CUDA Whee | SKIP | pretests |
| 2026-05-03 10:13 UTC | Paper v5 Critical ISSUE-1 to ISSUE-5 -- Retry with | SKIP | pretests |
| 2026-05-03 11:42 UTC | Paper v5 Critical ISSUE-1 to ISSUE-5 -- Retry with | SKIP | pretests |
| 2026-05-03 12:13 UTC | Phase 4 ARC-AGI-3 Harder Puzzles -- BFS-Intractable | SKIP | pretests |
| 2026-05-03 12:48 UTC | Phase 4 ARC-AGI-3 Harder Puzzles -- BFS-Intractable | SKIP | pretests |
| 2026-05-03 13:19 UTC | KANtize SOS-KAN 4-bit Quantization -- Edge Deployme | OK | pass |
"""


def _sample_artifacts() -> dict[int, dict]:
    return {
        1191: {
            "rlimit_as_set": True,
            "rlimit_as_limit_bytes": 8589934592,
            "honest_verdict": "prlimit_active",
        },
        1196: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        1198: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        1199: {
            "kantize_auroc_maintained_above_0p97": True,
            "soskan_4bit_auroc": 0.990137,
            "honest_verdict": "4bit_auroc_above_threshold",
        },
        1200: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        1201: {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
    }


def _write_artifacts(results_dir: Path, artifacts: dict[int, dict]) -> None:
    for exp_id, payload in artifacts.items():
        filename = exp1202.EXPERIMENT_FILES[exp_id]
        (results_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_criteria_results_follow_source_fields_req_report_013() -> None:
    """REQ-REPORT-013: all 12 criteria are derived from milestone source fields."""
    criteria = exp1202.evaluate_criteria(_sample_artifacts())
    results = exp1202.criteria_results(criteria)

    assert len(results) == 12
    assert results["prlimit_memory_cap_active"] is True
    assert results["llama_cpp_gpu_offload_verified"] is False
    assert results["critical_issues_fixed_5_of_5"] is False
    assert results["arxiv_bundle_v7_ready"] is False
    assert results["grpo_v5_honest_result"] is False
    assert results["grpo_vps_step_delta_measured"] is False
    assert results["phase4_bfs_intractable_fraction_above_50pct"] is False
    assert results["fover_v7_pairs_above_500"] is False
    assert results["kantize_auroc_maintained_above_0p97"] is True
    assert results["tier1_online_addition_honest_verdict"] is False
    assert results["nonogram_cartridge_shipped"] is False
    assert results["retro_complete"] is True
    assert exp1202.criteria_met_count(criteria) == 3


def test_build_artifact_marks_missing_gates_unmet_scenario_report_010() -> None:
    """SCENARIO-REPORT-010: missing and blocked metric gates stay failed."""
    artifact = exp1202.build_artifact(_sample_artifacts(), SAMPLE_LOG.splitlines())

    assert artifact["milestone"] == "2026.04.93"
    assert artifact["criteria_total"] == 12
    assert artifact["criteria_met"] == 3
    assert artifact["criteria_score_pct"] == 25.0
    assert artifact["publication_hold_status"] == "active"
    assert artifact["dualgpu_utilization"]["available"] is False
    assert artifact["dualgpu_utilization"]["reason"] == "MISSING"
    assert artifact["grpo_trajectory"]["v3_improvement_pp"] == 2.86
    assert artifact["grpo_trajectory"]["v4_improvement_pp"] == 10.0
    assert artifact["grpo_trajectory"]["v5_result"] == "MISSING"
    assert len(artifact["significant_findings"]) == 3
    assert len(artifact["open_items_for_94"]) == 5
    assert artifact["retro_complete"] is True
    assert artifact["honest_verdict"] == "milestone_failed"
    assert artifact["experiment_honest_verdicts"]["exp1195"] == "MISSING"
    assert artifact["experiment_honest_verdicts"]["exp1201"] == "blocked_gate_check_failed"


def test_slowest_tasks_rank_top_five_from_milestone_window_req_report_013() -> None:
    """REQ-REPORT-013: slowest_tasks comes from .93 conductor-log spans."""
    slowest = exp1202.build_slowest_tasks(SAMPLE_LOG.splitlines())

    assert slowest == [
        {"rank": 1, "id": "exp1193", "duration_min": 89.0, "attempts_seen": 2},
        {"rank": 2, "id": "exp1192", "duration_min": 88.0, "attempts_seen": 3},
        {"rank": 3, "id": "exp1197", "duration_min": 35.0, "attempts_seen": 2},
        {"rank": 4, "id": "exp1191", "duration_min": 2.0, "attempts_seen": 2},
        {"rank": 5, "id": "exp1199", "duration_min": 0.0, "attempts_seen": 1},
    ]
    assert exp1202._exp_id_for_title("unmapped task") is None


def test_defensive_branches_for_logs_fallback_and_verdicts_req_report_013() -> None:
    """REQ-REPORT-013: defensive log and fallback paths stay deterministic."""
    log_lines = [
        "not a markdown row",
        "| 2026-05-03 06:40 UTC | prlimit Memory Cap -- before start | OK | ignored |",
        "| 2026-05-03 06:51 UTC | Milestone 2026.04.93 activated | OK | start |",
        "| 2026-05-03 06:52 UTC | unknown task | OK | ignored |",
        "| 2026-05-03 06:53 UTC | Milestone 2026.04.94 activated | OK | stop |",
        "| 2026-05-03 06:54 UTC | prlimit Memory Cap -- after stop | OK | ignored |",
    ]

    assert exp1202.build_slowest_tasks(log_lines) == []
    assert exp1202._artifact_slowest_tasks(
        {
            1191: {"duration_s": 120.0},
            1192: {"_missing": True},
            1193: {"honest_verdict": "no_duration"},
        }
    ) == [{"rank": 1, "id": "exp1191", "duration_min": 2.0, "attempts_seen": 1}]
    assert exp1202.dualgpu_utilization(
        {1195: {"honest_verdict": "training_wall_hit", "dualgpu_gpu0_utilization_pct": 5.0}}
    ) == {
        "available": False,
        "gpu0_utilization_pct": 5.0,
        "gpu1_utilization_pct": None,
        "source": "exp1195",
        "reason": "training_wall_hit",
    }
    assert exp1202._honest_verdict(12) == "milestone_complete"
    assert exp1202._honest_verdict(6) == "milestone_partial"
    assert exp1202._parse_log_line("not a markdown row") is None


def test_dualgpu_utilization_records_ran_artifact_req_report_013() -> None:
    """REQ-REPORT-013: Exp1195 GPU utilization is retained when available."""
    artifacts = _sample_artifacts()
    artifacts[1195] = {
        "honest_verdict": "improvement_below_v4",
        "training_completed": True,
        "dualgpu_gpu0_utilization_pct": 71.5,
        "dualgpu_gpu1_utilization_pct": 69.25,
        "improvement_over_baseline_pp": -1.5,
    }

    artifact = exp1202.build_artifact(artifacts, [])

    assert artifact["criteria_results"]["grpo_v5_honest_result"] is True
    assert artifact["dualgpu_utilization"] == {
        "available": True,
        "gpu0_utilization_pct": 71.5,
        "gpu1_utilization_pct": 69.25,
        "source": "exp1195",
    }
    assert artifact["grpo_trajectory"]["v5_result"] == "-1.5pp_vs_v4"


def test_load_artifacts_and_main_write_required_schema_req_report_013(tmp_path: Path) -> None:
    """REQ-REPORT-013: main writes the machine-readable Exp1202 deliverable."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_artifacts(results_dir, _sample_artifacts())
    log_path = tmp_path / "conductor-log.md"
    log_path.write_text(SAMPLE_LOG, encoding="utf-8")
    out_path = tmp_path / "experiment_1202_milestone_retro_93.json"

    loaded = exp1202.load_artifacts(results_dir)
    code = exp1202.main(
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
    assert loaded[1192]["_missing"] is True
    assert code == 0
    assert written["criteria_total"] == 12
    assert written["criteria_met"] == 3
    assert written["criteria_results"]["kantize_auroc_maintained_above_0p97"] is True
    assert written["slowest_tasks"][0]["id"] == "exp1193"
