"""Tests for the Exp 1665 `.127` operational retrospective.

Spec: REQ-REPORT-069, SCENARIO-REPORT-069.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1665_retro as exp


def _roadmap() -> dict[str, object]:
    return {
        "milestone": "2026.05.127",
        "tasks": [
            {
                "id": "exp1652-archive-126",
                "title": "Exp 1652: Archive .126 and initialize .127",
                "deliverable": "results/experiment_1652_archive.json",
            },
            {
                "id": "exp1653-nsvif-sota-integration",
                "title": "Exp 1653: NSVIF DSL SOTA GGUF Integration",
                "deliverable": "python/carnot/pipeline/nsvif_sota.py",
            },
            {
                "id": "exp1654-energy-guided-decoding",
                "title": "Exp 1654: Energy-Guided Decoding via STATIC CSR",
                "deliverable": "python/carnot/samplers/energy_guided.py",
            },
            {
                "id": "exp1655-e2e-guided-decoding-eval",
                "title": "Exp 1655: gated on Exp 1653, 1654: E2E Guided Decoding Eval",
                "deliverable": "results/experiment_1655_e2e_guided_decoding_eval.json",
            },
            {
                "id": "exp1656-ebrm-trace-scorer",
                "title": "Exp 1656: EBRM Trace Scorer",
                "deliverable": "results/experiment_1656_ebrm_trace_scorer.json",
            },
            {
                "id": "exp1657-kv260-ebrm-integration",
                "title": "Exp 1657: KV260 EBRM Hardware Offload",
                "deliverable": "results/experiment_1657_kv260_ebrm_binding.json",
            },
            {
                "id": "exp1658-hardware-trace-scoring-eval",
                "title": "Exp 1658: gated on Exp 1656, 1657: Hardware Trace Eval",
                "deliverable": "results/experiment_1658_hw_eval.json",
            },
            {
                "id": "exp1659-smgi-certified-updates",
                "title": "Exp 1659: SMGI Certified Updates for FR-11",
                "deliverable": "results/experiment_1659_smgi_certified_updates.json",
            },
            {
                "id": "exp1660-ltlzinc-benchmark",
                "title": "Exp 1660: LTLZinc Temporal Benchmark",
                "deliverable": "data/ltlzinc_benchmark.json",
            },
            {
                "id": "exp1661-fr11-smgi-learning",
                "title": "Exp 1661: gated on Exp 1659, 1660: FR-11 SMGI Continuous Learning",
                "deliverable": "results/experiment_1661_fr11_smgi_learning.json",
            },
            {
                "id": "exp1662-pinet-projection-layer",
                "title": "Exp 1662: Pi-net Differentiable Projection",
                "deliverable": "results/experiment_1662_pinet_layer.json",
            },
            {
                "id": "exp1663-pinet-vs-tskm",
                "title": "Exp 1663: gated on Exp 1662: Pi-net vs T-SKM Eval",
                "deliverable": "results/experiment_1663_pinet_vs_tskm.json",
            },
            {
                "id": "exp1664-update-e2e-plan",
                "title": "Exp 1664: Update E2E Test Plan for SMGI/EBRM",
                "deliverable": "results/experiment_1664_e2e_plan.json",
            },
            {
                "id": "exp1665-retro-127",
                "title": "Exp 1665: Milestone .127 Retrospective",
                "deliverable": "results/operational_retro_2026_05_127.json",
            },
        ],
    }


def _conductor_log() -> str:
    return """
| 2026-05-09 21:03 UTC | Plan milestone 2026.05.127 | OK | 14 tasks proposed |
| 2026-05-09 21:05 UTC | Milestone 2026.05.127 activated | OK | 14 tasks queued |
| 2026-05-09 21:18 UTC | Exp 1652: Archive .126 and initialize .127 | OK | 81 passed, 1 warning in 5.67s |
| 2026-05-09 22:13 UTC | Exp 1653: NSVIF DSL SOTA GGUF Integration | OK | 81 passed, 1 warning in 4.71s |
| 2026-05-09 22:15 UTC | Exp 1654: Energy-Guided Decoding via STATIC CSR | DOOMED_RERUN_BLOCK | 2 prior failure(s) match this task's scope |
| 2026-05-09 22:17 UTC | Exp 1654: Energy-Guided Decoding via STATIC CSR | DOOMED_RERUN_BLOCK | 2 prior failure(s) match this task's scope |
| 2026-05-09 22:19 UTC | Exp 1654: Energy-Guided Decoding via STATIC CSR | DOOMED_RERUN_BLOCK | 2 prior failure(s) match this task's scope |
| 2026-05-09 22:21 UTC | Exp 1655: gated on Exp 1653, 1654: E2E Guided Decoding Eval | GATE_BLOCK | upstream artifact not found for task id 'exp1653-nsvif-sota-integration' |
| 2026-05-09 22:23 UTC | Exp 1655: gated on Exp 1653, 1654: E2E Guided Decoding Eval | GATE_BLOCK | upstream artifact not found for task id 'exp1653-nsvif-sota-integration' |
| 2026-05-09 22:25 UTC | Exp 1655: gated on Exp 1653, 1654: E2E Guided Decoding Eval | GATE_BLOCK | upstream artifact not found for task id 'exp1653-nsvif-sota-integration' |
| 2026-05-09 23:11 UTC | Exp 1656: EBRM Trace Scorer | OK | 81 passed, 1 warning in 5.20s |
| 2026-05-10 00:10 UTC | Exp 1657: KV260 EBRM Hardware Offload | OK | cache hit: 81 passed, 1 warning in 5.20s |
| 2026-05-10 00:22 UTC | Exp 1658: gated on Exp 1656, 1657: Hardware Trace Eval | OK | 81 passed, 1 warning in 5.14s |
| 2026-05-10 01:08 UTC | Exp 1659: SMGI Certified Updates for FR-11 | OK | 81 passed, 1 warning in 3.92s |
| 2026-05-10 01:18 UTC | Exp 1660: LTLZinc Temporal Benchmark | OK | 81 passed, 1 warning in 4.61s |
| 2026-05-10 01:20 UTC | Exp 1661: gated on Exp 1659, 1660: FR-11 SMGI Continuous Learning | DOOMED_RERUN_BLOCK | 3 prior failure(s) match this task's scope |
| 2026-05-10 01:22 UTC | Exp 1661: gated on Exp 1659, 1660: FR-11 SMGI Continuous Learning | DOOMED_RERUN_BLOCK | 3 prior failure(s) match this task's scope |
| 2026-05-10 01:24 UTC | Exp 1661: gated on Exp 1659, 1660: FR-11 SMGI Continuous Learning | DOOMED_RERUN_BLOCK | 3 prior failure(s) match this task's scope |
| 2026-05-10 02:46 UTC | Exp 1662: Pi-net Differentiable Projection | FAIL | Gemini CLI error: Hard wall-clock cap after 4804s |
| 2026-05-10 02:48 UTC | Exp 1662: Pi-net Differentiable Projection | OK | Deliverable already exists in repo |
| 2026-05-10 02:48 UTC | Exp 1663: gated on Exp 1662: Pi-net vs T-SKM Eval | DOOMED_RERUN_BLOCK | 1 prior failure(s) match this task's scope |
| 2026-05-10 02:50 UTC | Exp 1663: gated on Exp 1662: Pi-net vs T-SKM Eval | DOOMED_RERUN_BLOCK | 1 prior failure(s) match this task's scope |
| 2026-05-10 02:52 UTC | Exp 1663: gated on Exp 1662: Pi-net vs T-SKM Eval | DOOMED_RERUN_BLOCK | 1 prior failure(s) match this task's scope |
| 2026-05-10 03:03 UTC | Exp 1664: Update E2E Test Plan for SMGI/EBRM | OK | 81 passed, 1 warning in 5.05s |
"""


def _source_payloads() -> dict[str, dict[str, object]]:
    return {
        "results/experiment_1652_archive.json": {"status": "complete"},
        "results/experiment_1655_e2e_guided_decoding_eval.json": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        "results/experiment_1656_ebrm_trace_scorer.json": {
            "status": "complete",
            "score_accuracy": 1.0,
            "honest_verdict": "complete: EBRM trace scorer separates traces",
        },
        "results/experiment_1657_kv260_ebrm_binding.json": {
            "status": "complete",
            "hardware_execution_available": False,
            "software_fallback_used": True,
        },
        "results/experiment_1658_hw_eval.json": {
            "status": "complete",
            "hardware_execution_available": False,
            "max_score_delta": 0.0,
            "scoring_delta_within_tolerance": True,
        },
        "results/experiment_1659_smgi_certified_updates.json": {
            "status": "complete",
            "certified_update_success": True,
            "smgi_certified_update_ready": True,
        },
        "results/experiment_1661_fr11_smgi_learning.json": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        "results/experiment_1662_pinet_layer.json": {"status": "complete"},
        "results/experiment_1663_pinet_vs_tskm.json": {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
        "results/experiment_1664_e2e_plan.json": {"status": "complete", "plan_updated": True},
    }


def _deliverable_exists() -> dict[str, bool]:
    return {
        "results/experiment_1652_archive.json": True,
        "python/carnot/pipeline/nsvif_sota.py": True,
        "python/carnot/samplers/energy_guided.py": False,
        "results/experiment_1655_e2e_guided_decoding_eval.json": True,
        "results/experiment_1656_ebrm_trace_scorer.json": True,
        "results/experiment_1657_kv260_ebrm_binding.json": True,
        "results/experiment_1658_hw_eval.json": True,
        "results/experiment_1659_smgi_certified_updates.json": True,
        "data/ltlzinc_benchmark.json": True,
        "results/experiment_1661_fr11_smgi_learning.json": True,
        "results/experiment_1662_pinet_layer.json": True,
        "results/experiment_1663_pinet_vs_tskm.json": True,
        "results/experiment_1664_e2e_plan.json": True,
    }


def test_scenario_report_069_summarizes_127_findings() -> None:
    """SCENARIO-REPORT-069: source events drive the .127 retrospective."""

    artifact = exp.build_artifact(
        active_roadmap=_roadmap(),
        conductor_log_text=_conductor_log(),
        source_payloads=_source_payloads(),
        deliverable_exists=_deliverable_exists(),
        protected_files_unchanged=True,
        generated_at="2026-05-10T03:05:00Z",
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "success"
    assert artifact["schema"] == "carnot.operational_retro.v64"
    assert artifact["milestone"] == "2026.05.127"
    assert artifact["total_wall_time_minutes"] == 358
    assert artifact["experiments_completed"] == 9
    assert artifact["task_attempts"] == 22
    assert artifact["completed_task_count"] == 9
    assert artifact["blocked_task_count"] == 4
    assert artifact["task_outcomes"]["exp1654-energy-guided-decoding"]["outcome"] == "blocked"
    assert artifact["task_outcomes"]["exp1656-ebrm-trace-scorer"]["outcome"] == "complete"
    assert artifact["task_outcomes"]["exp1662-pinet-projection-layer"]["failed_then_completed"] is True
    assert artifact["blocked_tasks"] == [
        "exp1654-energy-guided-decoding",
        "exp1655-e2e-guided-decoding-eval",
        "exp1661-fr11-smgi-learning",
        "exp1663-pinet-vs-tskm",
    ]
    assert artifact["failed_then_completed_tasks"] == ["exp1662-pinet-projection-layer"]
    assert artifact["slowest_experiments"][0].startswith("1662: Pi-net Differentiable Projection (82min)")
    assert artifact["hardware_execution_claimed"] is False
    assert artifact["software_fallback_used"] is True
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["honest_verdict"] == "milestone_127_operational_retro_success_9_of_13_tasks_complete_4_blocked"


def test_req_report_069_blocks_wrong_roadmap_and_missing_events() -> None:
    """REQ-REPORT-069: wrong roadmap or missing .127 events blocks success."""

    artifact = exp.build_artifact(
        active_roadmap={"milestone": "2026.05.128", "tasks": []},
        conductor_log_text="",
        source_payloads={},
        deliverable_exists={},
        protected_files_unchanged=False,
        generated_at="2026-05-10T03:05:00Z",
    )

    assert artifact["status"] == "blocked"
    assert artifact["completed_task_count"] == 0
    assert artifact["blocked_task_count"] == 0
    assert artifact["task_attempts"] == 0
    assert "active roadmap is not 2026.05.127" in artifact["blocked_reasons"]
    assert "conductor log has no Exp 1652-1664 .127 task events" in artifact["blocked_reasons"]
    assert "protected files changed" in artifact["blocked_reasons"]


def test_req_report_069_writes_skeleton_then_terminal_file(tmp_path: Path) -> None:
    """REQ-REPORT-069: run() writes in-progress first and then terminal JSON."""

    root = tmp_path
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "research-roadmap.yaml").write_text(
        """
milestone: 2026.05.127
tasks:
  - id: exp1652-archive-126
    title: 'Exp 1652: Archive .126 and initialize .127'
    deliverable: results/experiment_1652_archive.json
  - id: exp1665-retro-127
    title: 'Exp 1665: Milestone .127 Retrospective'
    deliverable: results/operational_retro_2026_05_127.json
""",
        encoding="utf-8",
    )
    (root / "ops" / "conductor-log.md").write_text(
        """
| 2026-05-09 21:05 UTC | Milestone 2026.05.127 activated | OK | 14 tasks queued |
| 2026-05-09 21:18 UTC | Exp 1652: Archive .126 and initialize .127 | OK | 81 passed |
""",
        encoding="utf-8",
    )
    (root / "results" / "experiment_1652_archive.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )

    out_path = root / "results" / "operational_retro_2026_05_127.json"
    artifact = exp.run(
        project_root=root,
        output_path=out_path,
        generated_at="2026-05-10T03:06:00Z",
        gpu_snapshot=[{"index": 0, "name": "RTX 3090", "memory_used_mb": 4, "utilization_gpu_pct": 0}],
        protected_files_unchanged=True,
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "success"
    assert artifact["gpu_snapshot"][0]["name"] == "RTX 3090"
    assert artifact["completed_task_count"] == 1
    assert artifact["honest_verdict"] == "milestone_127_operational_retro_success_1_of_1_tasks_complete_0_blocked"


def test_req_report_069_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-REPORT-069: helper edge cases stay deterministic for coverage."""

    assert exp._now_z().endswith("Z")
    assert exp._task_number("not-an-exp-id") is None
    assert exp._read_text(tmp_path / "missing.txt") == ""
    assert exp._load_yaml(tmp_path / "missing.yaml") == {}
    assert exp._read_json(tmp_path / "missing.json") == {}
    assert exp._fallback_result_path(tmp_path, "python/carnot/models/foo.py") is None

    roadmap = {
        "milestone": "2026.05.127",
        "tasks": [
            "ignore-me",
            {"id": "no-exp-id", "deliverable": ""},
            {"id": "exp1652-one", "title": "Exp 1652", "deliverable": ""},
            {"id": "exp1653-two", "title": "Exp 1653", "deliverable": "python/carnot/models/foo.py"},
            {"id": "exp1654-three", "title": "Exp 1654", "deliverable": "results/experiment_1654_short.json"},
        ],
    }
    tasks = exp._roadmap_tasks(roadmap)
    assert [task["id"] for task in tasks] == ["exp1652-one", "exp1653-two", "exp1654-three"]

    log_text = """
| 2026-05-09 21:05 UTC | Milestone 2026.05.127 activated | OK | 14 tasks queued |
| 2026-05-09 21:07 UTC | Exp 1669: Outside current scope | OK | ignored |
| 2026-05-09 21:10 UTC | Exp 1652: Failed task | FAIL | failed |
"""
    artifact = exp.build_artifact(
        active_roadmap=roadmap,
        conductor_log_text=log_text,
        source_payloads={},
        deliverable_exists={},
        protected_files_unchanged=True,
        generated_at="2026-05-10T03:07:00Z",
    )

    assert artifact["task_outcomes"]["exp1652-one"]["outcome"] == "failed"
    assert artifact["task_outcomes"]["exp1653-two"]["outcome"] == "missing"
    assert artifact["estimated_time_savings_pct"] == 15
    assert artifact["bottlenecks_identified"][0].startswith("No conductor pre-gate")

    (tmp_path / "results").mkdir()
    fallback = tmp_path / "results" / "experiment_1654_actual.json"
    fallback.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    payloads = exp._source_payloads_for_tasks(tmp_path, tasks)
    exists = exp._deliverable_exists_map(tmp_path, tasks)

    assert payloads["results/experiment_1654_short.json"] == {"status": "blocked"}
    assert "" not in exists
    assert exists["python/carnot/models/foo.py"] is False
    assert exists["results/experiment_1654_short.json"] is True
