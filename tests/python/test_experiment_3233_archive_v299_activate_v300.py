"""Tests for Exp 3233 archive .299 and activate .300.

Spec refs: REQ-REPORT-3233, SCENARIO-REPORT-3233.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import archive_v299_activate_v300_3233 as mod


REQUIRED_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "archive_v299_activate_v300_ready",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_v4_outcome",
    "missing_v4_artifact_path",
    "next_top_gap",
    "queue_first_task",
    "queue_last_task",
    "protected_files_untouched",
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


def _archive_payload() -> dict[str, Any]:
    return {
        "schema_version": "carnot.archive_activation.v298_to_v299.v1",
        "experiment_id": "exp3221",
        "task_id": "exp3221-archive-v298-activate-v299",
        "milestone": "2026.05.299",
        "archive_v298_activate_v299_ready": True,
        "prior_paper_ready": False,
        "prior_publication_blocker_count": 100,
        "honest_verdict": "complete: archive_v298_activate_v299_ready=true",
    }


def _capstone_payload(
    *,
    ready: bool = True,
    v4_outcome: str = mod.PRIOR_V4_OUTCOME,
    blockers: int = 100,
) -> dict[str, Any]:
    return {
        "schema_version": "carnot.milestone_capstone.v299_single_focus_aggregation.v1",
        "experiment_id": "exp3223",
        "task_id": "exp3223-capstone-v299-single-focus",
        "milestone": "2026.05.299",
        "capstone_v299_ready": ready,
        "paper_ready": False,
        "publication_blocker_count": blockers,
        "v4_outcome": v4_outcome,
        "next_top_gap": mod.NEXT_TOP_GAP,
        "v4_result_artifact": mod.MISSING_V4_ARTIFACT_REL_PATH.as_posix(),
        "honest_verdict": (
            "complete: capstone_v299_ready=true; paper_ready=false; "
            "publication_blocker_count=100; v4_outcome=blocked_missing_exp3222_result"
        ),
    }


def _roadmap_yaml(
    *,
    milestone: str = "2026.05.300",
    first_task: str = mod.TASK_ID,
    last_task: str = mod.EXPECTED_QUEUE_LAST_TASK,
) -> str:
    return "\n".join(
        [
            f'milestone: "{milestone}"',
            'milestone_title: "Runtime Receipt Recovery"',
            f'milestone_doc: "{mod.VNEXT_DOC_REL_PATH.as_posix()}"',
            "tasks:",
            f'  - id: "{first_task}"',
            f'    milestone: "{milestone}"',
            f'    deliverable: "{mod.OUTPUT_REL_PATH.as_posix()}"',
            '  - id: "exp3234-cli-backend-failure-root-cause-ledger-v1"',
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3234_cli_backend_failure_root_cause_ledger_v1.json"',
            f'  - id: "{last_task}"',
            f'    milestone: "{milestone}"',
            '    deliverable: "results/experiment_3245_capstone_v300.json"',
            "",
        ]
    )


def _conductor_log(*, failures: int = 3, activated: bool = True) -> str:
    lines = [
        (
            "| 2026-05-28 02:53 UTC | Prompt-Injection KAN Distillation v4 - "
            "15k Corpus | FAIL | Codex CLI error: backend exited before artifact |\n"
        )
        for _ in range(failures)
    ]
    if activated:
        lines.append(
            "| 2026-05-28 03:51 UTC | Milestone 2026.05.300 activated | OK | 13 tasks queued |\n"
        )
    return "".join(lines)


def _write_sources(tmp_path: Path, *, staged: bool = False) -> None:
    _write_json(tmp_path, mod.PRIOR_ARCHIVE_REL_PATH, _archive_payload())
    _write_json(tmp_path, mod.CAPSTONE_V299_REL_PATH, _capstone_payload())
    _write_text(tmp_path, mod.CONDUCTOR_LOG_REL_PATH, _conductor_log())
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    if staged:
        _write_text(tmp_path, mod.STAGED_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(tmp_path, mod.VNEXT_DOC_REL_PATH, "# Research Roadmap vNEXT\n")
    _write_text(tmp_path, mod.CONDUCTOR_REL_PATH, "# protected conductor\n")


def test_req_report_3233_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3233: OpenSpec declares the handoff contract before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3233" in spec
    assert "SCENARIO-REPORT-3233" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3233_builds_active_queue_handoff(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3233: active .300 roadmap can confirm the archive handoff."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.5)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    sources = {row["role"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3233"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.300"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["archive_v299_activate_v300_ready"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 100
    assert artifact["prior_v4_outcome"] == mod.PRIOR_V4_OUTCOME
    assert artifact["missing_v4_artifact_path"] == mod.MISSING_V4_ARTIFACT_REL_PATH.as_posix()
    assert "did not produce" in artifact["missing_v4_artifact_note"]
    assert artifact["missing_v4_artifact_exists"] is False
    assert artifact["next_top_gap"] == mod.NEXT_TOP_GAP
    assert artifact["queue_first_task"] == mod.TASK_ID
    assert artifact["queue_last_task"] == mod.EXPECTED_QUEUE_LAST_TASK
    assert artifact["queue_task_count"] == 3
    assert artifact["queue_paths"]["selected_queue_path"] == mod.ACTIVE_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_pre_activation_shape_observed"] is False
    assert artifact["activation_already_observed"] is True
    assert artifact["exp3222_failure_count"] == 3
    assert artifact["protected_files"] == [
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    ]
    assert artifact["protected_files_untouched"] == {
        "research-roadmap.yaml": True,
        "scripts/research_conductor.py": True,
    }
    assert artifact["principle_annotations"]["honest_verdict"].startswith("Terminal")
    assert artifact["no_new_kan_training"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "paper_ready=true" not in artifact["honest_verdict"]
    assert sources["capstone_v299"]["sha256"] == _sha256(tmp_path / mod.CAPSTONE_V299_REL_PATH)


def test_req_report_3233_prefers_staged_queue_when_present(tmp_path: Path) -> None:
    """REQ-REPORT-3233: staged .300 roadmap is accepted before activation."""

    _write_sources(tmp_path, staged=True)
    _write_text(
        tmp_path,
        mod.ACTIVE_ROADMAP_REL_PATH,
        _roadmap_yaml(
            milestone="2026.05.299",
            first_task="exp3221-archive-v298-activate-v299",
            last_task="exp3223-capstone-v299",
        ),
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["archive_v299_activate_v300_ready"] is True
    assert artifact["queue_paths"]["selected_queue_path"] == mod.STAGED_ROADMAP_REL_PATH.as_posix()
    assert artifact["roadmap_pre_activation_shape_observed"] is True
    assert artifact["queue_first_task"] == mod.TASK_ID
    assert artifact["queue_last_task"] == mod.EXPECTED_QUEUE_LAST_TASK


def test_req_report_3233_writer_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3233: writer persists JSON and malformed inputs stay explicit."""

    _write_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=6.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["archive_v299_activate_v300_ready"] is True
    assert saved["duration_s"] == pytest.approx(2.0)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_document(tmp_path / "missing.yaml") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._int_value(True) == 0
    assert mod._int_value("9") == 9
    assert mod._duration(3.0, 2.0) == 0.0

    empty = mod.build_artifact(tmp_path / "empty", started_s=3.0, now_s=2.0)
    assert empty["archive_v299_activate_v300_ready"] is False
    assert empty["honest_verdict"].startswith("complete:")
    assert "capstone_v299 authority is missing or malformed" in empty["blocked_reasons"]

    _write_json(
        tmp_path,
        mod.PRIOR_ARCHIVE_REL_PATH,
        {**_archive_payload(), "archive_v298_activate_v299_ready": False},
    )
    _write_json(tmp_path, mod.CAPSTONE_V299_REL_PATH, _capstone_payload(ready=False))
    not_ready = mod.build_artifact(tmp_path)
    assert "archive_v298_activate_v299 authority is not ready" in not_ready["blocked_reasons"]
    assert "capstone_v299 authority is not ready" in not_ready["blocked_reasons"]


def test_req_report_3233_blocks_when_v4_artifact_unexpectedly_exists(tmp_path: Path) -> None:
    """REQ-REPORT-3233: missing-v4 handoff cannot hide a present conflicting artifact."""

    _write_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.MISSING_V4_ARTIFACT_REL_PATH,
        {"experiment_id": "exp3222", "honest_verdict": "complete: unexpected"},
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["archive_v299_activate_v300_ready"] is False
    assert artifact["missing_v4_artifact_exists"] is True
    assert "expected missing exp3222 v4 artifact is present" in artifact["blocked_reasons"]
