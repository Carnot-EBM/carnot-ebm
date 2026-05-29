"""Tests for Exp 3321 archive .306 and activate .307 handoff.

Spec refs: REQ-REPORT-3321, SCENARIO-REPORT-3321.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.reporting import archive_v306_activate_v307_3321 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _blocked_capstone() -> dict[str, Any]:
    return {
        "experiment": 3320,
        "schema": "blocked_gate_check_v1",
        "run_date": "2026-05-29",
        "duration_s": 0.0,
        "status": "blocked",
        "title": "Capstone v306",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": (
            "1 of 1 gate(s) failed; first failure: exp3319-evidence-matrix-v38.matrix_v38_ready"
        ),
        "gates_evaluated": [
            {
                "upstream": "exp3319-evidence-matrix-v38",
                "artifact_field": "matrix_v38_ready",
                "op": "==",
                "expected": True,
                "actual": None,
                "passed": False,
                "reason": "upstream artifact not found for task id 'exp3319-evidence-matrix-v38'",
            }
        ],
        "blocked_at_layer": "conductor_pre_gate",
    }


def _operational_retro() -> dict[str, Any]:
    return {
        "schema": "operational_retro_v1",
        "milestone": "2026.05.306",
        "experiments_completed": 0,
        "total_wall_time_minutes": 0,
        "compute_bound_experiments_count": 0,
        "summary": "No authoritative timing rows were available for .306.",
    }


def _research_complete_yaml(*, archived: bool = True) -> str:
    if not archived:
        return "milestones: []\n"
    lines = [
        "milestones:",
        "- id: 2026.05.306",
        "  title: DataFlip + Quality-Flag Cleanup For Publication-Ready Evidence",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-05-29'",
        "  finding: See conductor log for per-experiment results.",
        "  tasks:",
    ]
    for task in mod.PRIOR_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {task['title']}",
                f"    deliverable: {task['deliverable']}",
                "    result: OK (conductor)",
            ]
        )
    return "\n".join(lines) + "\n"


def _roadmap_yaml(milestone: str = "2026.05.307") -> str:
    return (
        f'milestone: "{milestone}"\n'
        'milestone_title: "Phase-3 Path De-Risking - the two existential link tests"\n'
        'milestone_doc: "openspec/change-proposals/research-roadmap-vNEXT.md"\n'
        "tasks:\n"
        '  - id: "exp3321-archive-v306-activate-v307"\n'
        f'    milestone: "{milestone}"\n'
        '    deliverable: "results/experiment_3321_archive_v306_activate_v307.json"\n'
        '  - id: "exp3322-energy-descent-vs-autoregressive-premise-v1"\n'
        f'    milestone: "{milestone}"\n'
    )


def _conductor_log() -> str:
    statuses = {
        "exp3317-repair-headline-evidence-audit-v2": "GATE_BLOCK",
        "exp3318-fr11-failure-targeted-curriculum-replay-v3": "GATE_BLOCK",
        "exp3319-evidence-matrix-v38": "GATE_BLOCK",
        "exp3320-capstone-v306": "GATE_BLOCK",
    }
    lines = []
    for index, task in enumerate(mod.PRIOR_TASKS, start=26):
        status = statuses.get(str(task["id"]), "OK")
        detail = "1 of 1 gate(s) failed" if status == "GATE_BLOCK" else "81 passed in 3.00s"
        lines.append(
            f"| 2026-05-29 05:{index:02d} UTC | {task['log_title']} | {status} | {detail} |"
        )
    lines.append("| 2026-05-29 05:09 UTC | Milestone 2026.05.307 activated | OK | 4 tasks queued |")
    return "\n".join(lines) + "\n"


def _write_sources(root: Path, *, archived: bool = True, with_retro: bool = True) -> None:
    _write_json(root, mod.CAPSTONE_V306_REL_PATH, _blocked_capstone())
    if with_retro:
        _write_json(root, mod.OPERATIONAL_RETRO_V306_REL_PATH, _operational_retro())
    _write_text(root, mod.RESEARCH_COMPLETE_REL_PATH, _research_complete_yaml(archived=archived))
    _write_text(root, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml())
    _write_text(root, mod.CONDUCTOR_LOG_REL_PATH, _conductor_log())
    _write_text(
        root,
        mod.NORTH_STAR_REL_PATH,
        "## 2. THE STABLE PUBLICATION GATE\nG1 Headline measured; G2 reproduced.\n",
    )


def test_req_report_3321_spec_anchor_declares_activation_schema() -> None:
    """REQ-REPORT-3321: OpenSpec declares the .306/.307 handoff first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3321" in spec
    assert "SCENARIO-REPORT-3321" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_3321_archived_blocked_capstone_opens_v307(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3321: blocked .306 capstone is archived without overclaiming."""

    _write_sources(tmp_path)
    before_complete = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    before_roadmap = (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=3.5)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    output = mod.write_artifact(tmp_path, started_s=4.0, now_s=5.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3321"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["run_date"] == "20260529"
    assert artifact["source_milestone"] == "2026.05.306"
    assert artifact["target_milestone"] == "2026.05.307"
    assert artifact["archive_v306_activate_v307_ready"] is True
    assert artifact["v306_closed_v307_opened"] is True
    assert artifact["prior_paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 1
    assert artifact["prior_publication_blocker_source"] == "failed_capstone_gates"
    assert artifact["prior_capstone_status"] == "blocked"
    assert artifact["prior_capstone_gate_failures"][0]["artifact_field"] == "matrix_v38_ready"
    assert artifact["prior_capstone_terminal"] is True
    assert artifact["operational_retrospective_summary"]["milestone"] == "2026.05.306"
    assert artifact["research_complete_update"] == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }
    assert artifact["research_complete_source_summary"]["task_count"] == len(mod.PRIOR_TASKS)
    assert artifact["v307_queue"]["selected_queue_milestone"] == "2026.05.307"
    assert artifact["v307_queue"]["queue_first_task"] == mod.TASK_ID
    assert artifact["v307_activation_observed"] is True
    assert artifact["conductor_log_terminal_status_counts"] == {"OK": 10, "GATE_BLOCK": 4}
    assert artifact["source_checksums"][mod.CAPSTONE_V306_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.CAPSTONE_V306_REL_PATH
    )
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert saved["duration_s"] == pytest.approx(1.25)
    assert saved["honest_verdict"].startswith("complete:")
    assert "prior_paper_ready=false" in saved["honest_verdict"]
    assert (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(
        encoding="utf-8"
    ) == before_complete
    assert (tmp_path / mod.ACTIVE_ROADMAP_REL_PATH).read_text(encoding="utf-8") == before_roadmap
    mod.validate_artifact(artifact)


def test_req_report_3321_appends_missing_archive_once(tmp_path: Path) -> None:
    """REQ-REPORT-3321: missing .306 archive is materialized exactly once."""

    _write_sources(tmp_path, archived=False, with_retro=False)

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=3.0)
    saved = json.loads(output.read_text(encoding="utf-8"))
    archive_text = (tmp_path / mod.RESEARCH_COMPLETE_REL_PATH).read_text(encoding="utf-8")
    ensure_result = mod.ensure_research_complete_entry(tmp_path)

    assert saved["archive_v306_activate_v307_ready"] is True
    assert saved["operational_retrospective_summary"] == {}
    assert saved["research_complete_update"]["appended"] is True
    assert saved["research_complete_update"]["already_present"] is False
    assert archive_text.count("- id: 2026.05.306") == 1
    assert ensure_result == {
        "path": "research-complete.yaml",
        "appended": False,
        "already_present": True,
    }


def test_req_report_3321_fail_closed_helpers_and_validation(tmp_path: Path) -> None:
    """REQ-REPORT-3321: malformed inputs stay explicit and non-fabricated."""

    _write_sources(tmp_path, archived=False, with_retro=False)
    _write_text(tmp_path, mod.ACTIVE_ROADMAP_REL_PATH, _roadmap_yaml("2026.05.306"))
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_REL_PATH, "[")
    _write_text(
        tmp_path,
        mod.CONDUCTOR_LOG_REL_PATH,
        "| 2026-05-29 05:07 UTC | Capstone v306 | GATE_BLOCK | fixture |\n",
    )
    _write_json(
        tmp_path,
        mod.CAPSTONE_V306_REL_PATH,
        {
            "experiment_id": "exp3320",
            "task_id": "exp3320-capstone-v306",
            "paper_ready": True,
            "publication_blocker_count": 0,
            "honest_verdict": "complete: fixture",
            "gates_evaluated": [{"passed": True}],
        },
    )
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("tasks: [", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=8.0, now_s=3.0)

    assert artifact["archive_v306_activate_v307_ready"] is False
    assert artifact["v306_closed_v307_opened"] is False
    assert artifact["duration_s"] == 0.0
    assert artifact["prior_paper_ready"] is True
    assert artifact["prior_publication_blocker_count"] == 0
    assert (
        "research-complete.yaml does not contain the .306 task summary"
        in artifact["blocked_reasons"]
    )
    assert "selected queue milestone is not 2026.05.307" in artifact["blocked_reasons"]
    assert "milestone 2026.05.307 activation is not observed" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("complete:")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_yaml_document(bad_yaml) == {}
    assert mod.read_yaml_document(tmp_path / "missing.yaml") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._as_mapping([]) == {}
    assert mod._as_list({}) == []
    assert mod._int_value(True) == 0
    assert mod._int_value(7) == 7
    assert mod._task_ids({"tasks": ["bad", {"id": "x"}]}) == ["x"]
    assert mod._milestone_entries("bad") == []
    assert mod._milestone_entries([{"id": "x"}, "bad"]) == [{"id": "x"}]
    assert mod._terminal_prefix_ok("shipped: done") is True
    assert mod._terminal_prefix_ok("blocked") is False
    assert mod._parse_conductor_line("not a conductor row") == {}
    assert all(
        row["status"] == "missing"
        for row in mod._conductor_log_terminal_rows(tmp_path / "missing-log-root")
    )

    empty_archive = tmp_path / "empty" / mod.RESEARCH_COMPLETE_REL_PATH
    mod._append_research_complete_entry(empty_archive)
    assert empty_archive.read_text(encoding="utf-8").startswith("milestones:\n- id: 2026.05.306")
    list_archive = tmp_path / "list" / mod.RESEARCH_COMPLETE_REL_PATH
    list_archive.parent.mkdir(parents=True)
    list_archive.write_text("milestones: []\n", encoding="utf-8")
    mod._append_research_complete_entry(list_archive)
    assert list_archive.read_text(encoding="utf-8").count("- id: 2026.05.306") == 1
    no_newline_archive = tmp_path / "no-newline" / mod.RESEARCH_COMPLETE_REL_PATH
    no_newline_archive.parent.mkdir(parents=True)
    no_newline_archive.write_text("milestones:\n- id: 2026.05.305\n  tasks: []", encoding="utf-8")
    mod._append_research_complete_entry(no_newline_archive)
    assert no_newline_archive.read_text(encoding="utf-8").count("- id: 2026.05.306") == 1

    summary_root = tmp_path / "summary"
    _write_text(
        summary_root,
        mod.RESEARCH_COMPLETE_REL_PATH,
        "milestones:\n- id: 2026.05.305\n  tasks: []\n- id: 2026.05.306\n  tasks: []\n",
    )
    assert mod._research_complete_task_summary(summary_root)["task_count"] == 0
    assert mod._capstone_terminal({}) is False
    assert mod._file_contains(tmp_path / "missing.log", "needle") is False

    good_artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.0)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(good_artifact | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(good_artifact | {"task_id": "bad"})
    with pytest.raises(ValueError, match="source_milestone"):
        mod.validate_artifact(good_artifact | {"source_milestone": "bad"})
    with pytest.raises(ValueError, match="target_milestone"):
        mod.validate_artifact(good_artifact | {"target_milestone": "bad"})
    with pytest.raises(ValueError, match="random_seed"):
        mod.validate_artifact(good_artifact | {"random_seed": 0})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(good_artifact | {"inference_substrate": "live"})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(good_artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="prior_publication_blocker_count"):
        mod.validate_artifact(good_artifact | {"prior_publication_blocker_count": -1})
    with pytest.raises(ValueError, match="no_push"):
        mod.validate_artifact(good_artifact | {"no_push": False})
