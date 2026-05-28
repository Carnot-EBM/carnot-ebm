"""Tests for Exp 3221 archive .298 and activate .299.

Spec refs: REQ-REPORT-3221, SCENARIO-REPORT-3221.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v298_activate_v299_3221 as mod


REQUIRED_FIELDS = {
    "archive_v298_activate_v299_ready",
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


def _capstone_payload(*, ready: bool = True, blocker_count: int = 100) -> dict[str, Any]:
    return {
        "schema_version": "carnot.milestone_capstone.v298_matrix_v32_terminal_aggregation.v1",
        "experiment_id": "exp3232",
        "milestone": "2026.05.298",
        "capstone_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": blocker_count,
        "blocker_delta_from_v31": 8,
        "next_top_gap": mod.PRIOR_NEXT_TOP_GAP,
        "honest_verdict": "complete: capstone_ready=true",
    }


def _operational_retro_payload() -> dict[str, Any]:
    return {
        "schema": "carnot.operational_retro.v64",
        "milestone": "2026.05.298",
        "generated_at": "2026-05-28T02:40:00Z",
        "retro_type": "operational_full",
        "total_wall_time_minutes": 54,
        "experiments_completed": 14,
        "compute_bound_experiments_count": 1,
        "summary": "Operational closeout for .298.",
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
            "- id: 2026.05.298",
            "  title: Hermetic CUDA Receipt Repair",
            "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
            "  completed: '2026-05-28'",
            "  tasks:",
            *task_lines,
            "",
        ]
    )


def _roadmap_yaml(*, milestone: str = "2026.05.299", first_task: str | None = None) -> str:
    first = first_task or mod.TASK_ID
    return "\n".join(
        [
            f'milestone: "{milestone}"',
            'milestone_title: "Prompt-Injection KAN v4 Single-Focus Milestone"',
            f'milestone_doc: "{mod.VNEXT_DOC_REL_PATH.as_posix()}"',
            "tasks:",
            f"  - id: {first}",
            f'    milestone: "{milestone}"',
            f'    deliverable: "{mod.OUTPUT_REL_PATH.as_posix()}"',
            "  - id: exp3222-prompt-injection-kan-distill-v4-15k",
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3222_prompt_injection_kan_distill_v4_15k.json"',
            "",
        ]
    )


def _write_sources(tmp_path: Path, *, include_retro: bool = True) -> None:
    _write_json(tmp_path, mod.CAPSTONE_V298_REL_PATH, _capstone_payload())
    if include_retro:
        _write_json(tmp_path, mod.OPERATIONAL_RETRO_V298_REL_PATH, _operational_retro_payload())
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_REL_PATH, _research_complete_yaml())
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(
        tmp_path,
        mod.CONDUCTOR_LOG_REL_PATH,
        "| 2026-05-28 02:35 UTC | Milestone 2026.05.299 activated | OK | 3 tasks queued |\n",
    )
    _write_text(tmp_path, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT\n")


def test_req_report_3221_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3221: OpenSpec declares the archive/activation contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3221" in spec
    assert "SCENARIO-REPORT-3221" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3221_builds_activation_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3221: .298 archive carries readiness into .299 activation."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=3.25)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    sources = {row["role"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3221"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.299"
    assert artifact["prior_milestone"] == "2026.05.298"
    assert artifact["archive_v298_activate_v299_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 100
    assert artifact["prior_capstone_ready"] is True
    assert artifact["prior_next_top_gap"] == mod.PRIOR_NEXT_TOP_GAP
    assert artifact["prior_operational_retro_present"] is True
    assert artifact["prior_operational_retro_summary"]["experiments_completed"] == 14
    assert artifact["research_complete_prior_summary_present"] is True
    assert artifact["queue_paths"]["selected_queue_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["queue_paths"]["selected_queue_first_task"] == mod.TASK_ID
    assert artifact["conductor_activation_observed"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "archive_v298_activate_v299_ready=true" in artifact["honest_verdict"]
    assert sources["capstone_v298"]["sha256"] == _sha256(tmp_path / mod.CAPSTONE_V298_REL_PATH)
    assert sources["operational_retro_v298"]["sha256"] == _sha256(
        tmp_path / mod.OPERATIONAL_RETRO_V298_REL_PATH
    )


def test_req_report_3221_writer_appends_missing_research_complete_entry(tmp_path: Path) -> None:
    """REQ-REPORT-3221: writer materializes a missing .298 archive summary once."""

    _write_json(tmp_path, mod.CAPSTONE_V298_REL_PATH, _capstone_payload())
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(
        tmp_path,
        mod.CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.299 activated\n",
    )
    _write_text(tmp_path, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT\n")

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.5)
    saved = json.loads(output.read_text(encoding="utf-8"))
    archive_text = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    ensure_result = mod.ensure_research_complete_entry(tmp_path)

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v298_activate_v299_ready"] is True
    assert saved["prior_operational_retro_present"] is False
    assert saved["prior_operational_retro_summary"] == {}
    assert saved["research_complete_prior_summary_present"] is True
    assert archive_text.count("- id: 2026.05.298") == 1
    assert ensure_result == {
        "path": mod.RESEARCH_COMPLETE_REL_PATH.as_posix(),
        "appended": False,
        "already_present": True,
    }


def test_req_report_3221_fail_closed_and_helper_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3221: malformed inputs remain explicit and non-fabricated."""

    _write_text(tmp_path, mod.RESEARCH_COMPLETE_REL_PATH, "[")
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml(milestone="2026.05.300"))
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.0)

    assert artifact["archive_v298_activate_v299_ready"] is False
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 0
    assert artifact["prior_operational_retro_present"] is False
    assert "capstone_v298 authority is missing or malformed" in artifact["blocked_reasons"]
    assert "research-complete.yaml does not contain the .298 task summary" in artifact[
        "blocked_reasons"
    ]
    assert artifact["honest_verdict"].startswith("complete:")

    direct_reasons = mod._blocked_reasons(
        capstone={"capstone_ready": False},
        capstone_ready=False,
        research_complete_present=True,
        queue_paths={
            "selected_queue_milestone": mod.MILESTONE,
            "selected_queue_first_task": "wrong-first-task",
            "milestone_doc": "wrong-doc.md",
        },
        root_path=tmp_path / "no_activation_sources",
    )
    assert "capstone_v298 authority is not ready" in direct_reasons
    assert "selected queue first task is not exp3221-archive-v298-activate-v299" in direct_reasons
    assert "selected queue milestone_doc is not the vNEXT document" in direct_reasons

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    _write_text(tmp_path, Path("not_mapping.json"), "[]")
    assert mod.read_json_object(tmp_path / "not_mapping.json") == {}
    assert mod.read_yaml_document(tmp_path / "missing.yaml") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.expected_next_milestone("2026.05.298") == "2026.05.299"
    assert mod.expected_next_milestone("bad") == ""
    assert mod._milestone_entries("not-yaml-shape") == []
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._int_value(True) == 0
    assert mod._int_value("7") == 7
    assert mod._int_value("bad") == 0
    assert mod._duration(3.0, 2.0) == 0.0
