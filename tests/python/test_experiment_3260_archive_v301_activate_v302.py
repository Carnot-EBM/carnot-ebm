"""Tests for Exp 3260 archive .301 and activate .302.

Spec refs: REQ-REPORT-3260, SCENARIO-REPORT-3260.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v301_activate_v302_3260 as mod


REQUIRED_FIELDS = {
    "archive_v301_activate_v302_ready",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capstone_payload(
    *,
    ready: bool = True,
    paper_ready: bool = False,
    blockers: int = 106,
    next_gap: str = mod.PRIOR_NEXT_TOP_GAP,
) -> dict[str, Any]:
    return {
        "schema_version": "carnot.milestone_capstone.v301_matrix_v34_closeout.v1",
        "experiment_id": "exp3258",
        "task_id": "exp3258-capstone-v301",
        "milestone": "2026.05.301",
        "capstone_v301_ready": ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": blockers,
        "blocker_delta_from_v300": 0,
        "next_top_gap": next_gap,
        "honest_verdict": (
            "complete: capstone_v301_ready=true; paper_ready=false; "
            "publication_blocker_count=106; next_top_gap="
            "keep_exp3248_blocked_repair_cuda_runtime"
        ),
    }


def _operational_retro_payload() -> dict[str, Any]:
    return {
        "schema": "carnot.operational_retro.v64",
        "milestone": "2026.05.301",
        "generated_at": "2026-05-28T08:55:00Z",
        "retro_type": "operational_full",
        "total_wall_time_minutes": 133,
        "experiments_completed": 13,
        "compute_bound_experiments_count": 0,
        "summary": ".301 operational closeout.",
    }


def _research_complete_yaml() -> str:
    task_lines: list[str] = []
    for task in mod.PRIOR_TASKS:
        task_lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {task['title']}",
                f"    deliverable: {task['deliverable']}",
                "    result: OK (conductor)",
            ]
        )
    return "\n".join(
        [
            "- id: 2026.05.301",
            "  title: Selected-Python CUDA Repair",
            "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
            "  completed: '2026-05-28'",
            "  tasks:",
            *task_lines,
            "",
        ]
    )


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.302",
    first_task: str = mod.TASK_ID,
    last_task: str = mod.EXPECTED_QUEUE_LAST_TASK,
) -> str:
    tasks = [
        first_task,
        "exp3261-cuda-recovery-confirmation-smoke-v1",
        "exp3262-llama-cpp-cuda-receipt-smoke-v4",
        "exp3263-sota-gguf-receipt-v9",
        "exp3264-prompt-injection-teacher-label-shard-v3",
        "exp3265-prompt-injection-kan-train-eval-shard-v3",
        last_task,
    ]
    lines = [
        f'milestone: "{milestone}"',
        'milestone_title: "CUDA-Recovered SOTA Receipt"',
        f'milestone_doc: "{mod.VNEXT_DOC_REL_PATH.as_posix()}"',
        "tasks:",
    ]
    for task_id in tasks:
        lines.extend(
            [
                f'  - id: "{task_id}"',
                f'    milestone: "{milestone}"',
                f'    deliverable: "results/{task_id}.json"',
            ]
        )
    return "\n".join(lines) + "\n"


def _write_sources(root: Path, *, include_retro: bool = True) -> None:
    _write_json(root, mod.CAPSTONE_V301_REL_PATH, _capstone_payload())
    if include_retro:
        _write_json(root, mod.OPERATIONAL_RETRO_V301_REL_PATH, _operational_retro_payload())
    _write_text(root, mod.RESEARCH_COMPLETE_REL_PATH, _research_complete_yaml())
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(
        root,
        mod.CONDUCTOR_LOG_REL_PATH,
        "| 2026-05-28 08:53 UTC | Milestone 2026.05.302 activated | OK | 7 tasks queued |\n",
    )
    _write_text(root, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT\n")
    _write_text(root, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")


def test_req_report_3260_spec_anchor_exists() -> None:
    """REQ-REPORT-3260: OpenSpec declares the handoff contract before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3260" in spec
    assert "SCENARIO-REPORT-3260" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert Path(mod.__file__).exists()


def test_scenario_report_3260_builds_active_queue_handoff(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3260: active .302 roadmap can confirm the handoff."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.75)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    sources = {row["role"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3260"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.302"
    assert artifact["prior_milestone"] == "2026.05.301"
    assert artifact["inference_substrate"] == "artifact_aggregation_only"
    assert artifact["archive_v301_activate_v302_ready"] is True
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 106
    assert artifact["prior_next_top_gap"] == mod.PRIOR_NEXT_TOP_GAP
    assert artifact["prior_operational_retro_present"] is True
    assert artifact["prior_operational_retro_summary"]["experiments_completed"] == 13
    assert artifact["research_complete_prior_summary_present"] is True
    assert artifact["prior_task_summary"]["task_count"] == len(mod.PRIOR_TASKS)
    assert artifact["prior_task_summary"]["first_task"] == mod.PRIOR_TASKS[0]["id"]
    assert artifact["prior_task_summary"]["last_task"] == mod.PRIOR_TASKS[-1]["id"]
    assert artifact["queue_paths"]["selected_queue_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["queue_paths"]["queue_first_task"] == mod.TASK_ID
    assert artifact["queue_paths"]["queue_last_task"] == mod.EXPECTED_QUEUE_LAST_TASK
    assert artifact["queue_paths"]["queue_task_count"] == 7
    assert artifact["activation_already_observed"] is True
    assert artifact["protected_files_untouched"] == {
        "research-roadmap.yaml": True,
        "scripts/research_conductor.py": True,
    }
    assert artifact["no_new_cuda_probe"] is True
    assert artifact["no_new_kan_training"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["duration_s"] == pytest.approx(1.75)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=true" not in artifact["honest_verdict"]
    assert sources["capstone_v301"]["sha256"] == _sha256(tmp_path / mod.CAPSTONE_V301_REL_PATH)
    assert sources["operational_retro_v301"]["sha256"] == _sha256(
        tmp_path / mod.OPERATIONAL_RETRO_V301_REL_PATH
    )


def test_req_report_3260_writer_appends_missing_research_complete_entry(tmp_path: Path) -> None:
    """REQ-REPORT-3260: writer materializes a missing .301 summary once."""

    _write_json(tmp_path, mod.CAPSTONE_V301_REL_PATH, _capstone_payload())
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(
        tmp_path,
        mod.CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.302 activated\n",
    )
    _write_text(tmp_path, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT\n")
    _write_text(tmp_path, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=6.5)
    saved = json.loads(output.read_text(encoding="utf-8"))
    archive_text = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    ensure_result = mod.ensure_research_complete_entry(tmp_path)

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v301_activate_v302_ready"] is True
    assert saved["prior_operational_retro_present"] is False
    assert saved["prior_operational_retro_summary"] == {}
    assert saved["research_complete_prior_summary_present"] is True
    assert archive_text.count("- id: 2026.05.301") == 1
    assert ensure_result == {
        "path": mod.RESEARCH_COMPLETE_REL_PATH.as_posix(),
        "appended": False,
        "already_present": True,
    }


def test_req_report_3260_fail_closed_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3260: malformed inputs remain explicit and non-fabricated."""

    _write_text(tmp_path, mod.RESEARCH_COMPLETE_REL_PATH, "[")
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone="2026.05.303"))
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("tasks: [", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=3.0, now_s=2.0)

    assert artifact["archive_v301_activate_v302_ready"] is False
    assert artifact["duration_s"] == 0.0
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 0
    assert artifact["prior_operational_retro_present"] is False
    assert "capstone_v301 authority is missing or malformed" in artifact["blocked_reasons"]
    assert "research-complete.yaml does not contain the .301 task summary" in artifact[
        "blocked_reasons"
    ]
    assert artifact["honest_verdict"].startswith("complete:")

    direct_reasons = mod._blocked_reasons(
        capstone={"capstone_v301_ready": False, "paper_ready": True},
        capstone_ready=False,
        prior_paper_ready=True,
        prior_publication_blocker_count=105,
        prior_next_top_gap="wrong",
        research_complete_present=True,
        queue_paths={
            "selected_queue_milestone": mod.MILESTONE,
            "queue_first_task": "wrong-first-task",
            "queue_last_task": "wrong-last-task",
            "milestone_doc": "wrong-doc.md",
        },
        activation_already_observed=False,
        vnext_doc_exists=False,
    )
    assert "capstone_v301 authority is not ready" in direct_reasons
    assert "prior paper_ready must remain false" in direct_reasons
    assert "prior publication blocker count is not 106" in direct_reasons
    assert "prior next_top_gap does not preserve the .301 runtime block" in direct_reasons
    assert "selected queue first task is not exp3260-archive-v301-activate-v302" in direct_reasons
    assert "selected queue last task is not exp3266-capstone-v302" in direct_reasons
    assert "selected queue milestone_doc is not the vNEXT document" in direct_reasons
    assert "milestone 2026.05.302 activation is not observed" in direct_reasons
    assert "openspec/change-proposals/research-roadmap-vNEXT.md is missing" in direct_reasons

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_document(bad_yaml) == {}
    assert mod.read_yaml_document(tmp_path / "missing.yaml") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._milestone_entries("not-yaml-shape") == []
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._int_value(True) == 0
    assert mod._int_value("7") == 7
    assert mod._int_value("bad") == 0
