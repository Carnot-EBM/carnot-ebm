"""Build the Exp 1601 `.122` archive and `.123` state artifact.

Spec: REQ-REPORT-066, SCENARIO-REPORT-066.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260509"
PREDECESSOR_MILESTONE = "2026.05.122"
TARGET_MILESTONE = "2026.05.123"
EXPERIMENT = "1601_archive"
SCHEMA = "milestone_123_archive_v1"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1601_archive.json"
ROADMAP_DOC_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

EXPECTED_122_TASK_IDS = [
    "exp1588-nsvif-dsl",
    "exp1589-dsl-sota-validation",
    "exp1590-csr-mask",
    "exp1591-dccd-adapter",
    "exp1592-dccd-repair-sota",
    "exp1593-cdg-repair",
    "exp1594-cerce-ledger",
    "exp1595-cerce-bounds",
    "exp1596-fr11-v16",
    "exp1597-inertial-ising",
    "exp1598-z1-drift",
    "exp1599-kanele-audit",
    "exp1600-ot-rewrite",
]

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "predecessor_milestone",
    "predecessor_archived",
    "predecessor_task_count",
    "predecessor_tasks_terminal",
    "active_roadmap_milestone",
    "active_roadmap_task_count",
    "first_active_task_id",
    "status_moved_to_changelog",
    "setup_123_state",
    "missing_task_deliverables",
    "research_roadmap_yaml_modified",
    "scripts_research_conductor_modified",
    "honest_verdict",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-066: persist a started marker before evidence reads."""

    artifact: dict[str, Any] = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "status": "in_progress",
            "milestone": TARGET_MILESTONE,
            "predecessor_milestone": PREDECESSOR_MILESTONE,
            "predecessor_archived": False,
            "predecessor_tasks_terminal": False,
            "status_moved_to_changelog": False,
            "setup_123_state": False,
            "missing_task_deliverables": [],
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
            "honest_verdict": "in_progress",
        }
    )
    return _write_json(Path(out_path), artifact)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _relative_path(path: Path) -> str:
    parts = path.parts
    for marker in ("results", "ops", "openspec"):
        if marker in parts:
            return str(Path(*parts[parts.index(marker) :]))
    return path.name


def _find_milestone(research_complete: Mapping[str, Any], milestone: str) -> dict[str, Any]:
    for entry in research_complete.get("milestones", []):
        if isinstance(entry, Mapping) and str(entry.get("id")) == milestone:
            return dict(entry)
    return {}


def _task_rows(archive_entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in archive_entry.get("tasks", []):
        if isinstance(task, Mapping):
            rows.append(
                {
                    "id": str(task.get("id") or ""),
                    "title": str(task.get("title") or ""),
                    "deliverable": str(task.get("deliverable") or ""),
                    "result": str(task.get("result") or ""),
                }
            )
    return rows


def _tasks_terminal(rows: Sequence[Mapping[str, Any]]) -> bool:
    terminal_tokens = ("ok", "complete", "blocked", "retired", "failed")
    return bool(rows) and all(
        any(token in str(row.get("result") or "").lower() for token in terminal_tokens)
        for row in rows
    )


def _missing_deliverables(
    rows: Sequence[Mapping[str, Any]],
    deliverable_exists: Mapping[str, bool],
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for row in rows:
        deliverable = str(row.get("deliverable") or "")
        if deliverable.startswith("results/") and not deliverable_exists.get(deliverable, False):
            missing.append({"task_id": str(row.get("id") or ""), "deliverable": deliverable})
    return missing


def _roadmap_task_ids(active_roadmap: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for task in active_roadmap.get("tasks", []):
        if isinstance(task, Mapping):
            ids.append(str(task.get("id") or ""))
    return ids


def _context_has_all(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def _protected_files_clean(root: Path) -> bool:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "--",
            "research-roadmap.yaml",
            "scripts/research_conductor.py",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _artifact_exists_map(root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    exists: dict[str, bool] = {}
    for row in rows:
        deliverable = str(row.get("deliverable") or "")
        if deliverable.startswith("results/"):
            exists[deliverable] = (root / deliverable).exists()
    return exists


def build_artifact(
    *,
    research_complete: Mapping[str, Any],
    active_roadmap: Mapping[str, Any],
    roadmap_doc_text: str,
    changelog_text: str,
    deliverable_exists: Mapping[str, bool],
    protected_files_unchanged: bool,
) -> dict[str, Any]:
    """REQ-REPORT-066: build the terminal `.122` archive and `.123` state artifact."""

    archive = _find_milestone(research_complete, PREDECESSOR_MILESTONE)
    archived_tasks = _task_rows(archive)
    archived_task_ids = [task["id"] for task in archived_tasks]
    expected_ids_present = archived_task_ids == EXPECTED_122_TASK_IDS
    predecessor_archived = bool(archive) and expected_ids_present
    predecessor_tasks_terminal = _tasks_terminal(archived_tasks)

    roadmap_task_ids = _roadmap_task_ids(active_roadmap)
    active_roadmap_milestone = str(active_roadmap.get("milestone") or "")
    first_active_task_id = roadmap_task_ids[0] if roadmap_task_ids else ""
    active_roadmap_is_123 = active_roadmap_milestone == TARGET_MILESTONE
    first_task_is_archive = first_active_task_id == "exp1601-archive-122"
    status_moved_to_changelog = PREDECESSOR_MILESTONE in changelog_text

    formal_kan_track_ready = _context_has_all(roadmap_doc_text, ["Exact-Rational KAN", "Z3"])
    ebcn_latent_track_ready = _context_has_all(roadmap_doc_text, ["EBCN", "latent gradient"])
    cerce_scale_track_ready = _context_has_all(roadmap_doc_text, ["CerCE", "FR-11"])
    dccd_dsl_scale_track_ready = _context_has_all(roadmap_doc_text, ["DCCD", "DSL"])
    hardware_accounting_track_ready = _context_has_all(roadmap_doc_text, ["hardware", "RKAN"])
    setup_123_state = bool(
        active_roadmap_is_123
        and first_task_is_archive
        and formal_kan_track_ready
        and ebcn_latent_track_ready
        and cerce_scale_track_ready
        and dccd_dsl_scale_track_ready
        and hardware_accounting_track_ready
    )

    blocked_reasons: list[str] = []
    if not archive:
        blocked_reasons.append("research-complete.yaml has no 2026.05.122 archive")
    if archive and not expected_ids_present:
        blocked_reasons.append("2026.05.122 archive task ids do not match exp1588-exp1600")
    if archived_tasks and not predecessor_tasks_terminal:
        blocked_reasons.append("not every 2026.05.122 archive task has a terminal result")
    if not active_roadmap_is_123:
        blocked_reasons.append("active roadmap is not 2026.05.123")
    if active_roadmap_is_123 and not first_task_is_archive:
        blocked_reasons.append("active roadmap does not start with exp1601-archive-122")
    if not status_moved_to_changelog:
        blocked_reasons.append("ops/changelog.md lacks the 2026.05.122 status entry")
    if active_roadmap_is_123 and not setup_123_state:
        blocked_reasons.append("2026.05.123 roadmap state lacks one or more expected tracks")
    if not protected_files_unchanged:
        blocked_reasons.append("protected files changed")

    status = "complete" if not blocked_reasons else "blocked"
    missing_task_deliverables = _missing_deliverables(archived_tasks, deliverable_exists)
    task_result_summary = [
        {
            "id": task["id"],
            "title": task["title"],
            "deliverable": task["deliverable"],
            "result": task["result"],
            "deliverable_exists": deliverable_exists.get(task["deliverable"], False),
        }
        for task in archived_tasks
    ]

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": status,
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "predecessor_archived": predecessor_archived,
        "predecessor_title": str(archive.get("title") or ""),
        "predecessor_task_count": len(archived_tasks),
        "predecessor_task_ids": archived_task_ids,
        "predecessor_tasks_terminal": predecessor_tasks_terminal,
        "archived_task_results": task_result_summary,
        "active_roadmap_milestone": active_roadmap_milestone,
        "active_roadmap_title": str(active_roadmap.get("milestone_title") or ""),
        "active_roadmap_task_count": len(roadmap_task_ids),
        "first_active_task_id": first_active_task_id,
        "status_moved_to_changelog": status_moved_to_changelog,
        "setup_123_state": setup_123_state,
        "formal_kan_track_ready": formal_kan_track_ready,
        "ebcn_latent_track_ready": ebcn_latent_track_ready,
        "cerce_scale_track_ready": cerce_scale_track_ready,
        "dccd_dsl_scale_track_ready": dccd_dsl_scale_track_ready,
        "hardware_accounting_track_ready": hardware_accounting_track_ready,
        "missing_task_deliverables": missing_task_deliverables,
        "missing_task_deliverable_count": len(missing_task_deliverables),
        "blocked_reasons": blocked_reasons,
        "research_roadmap_yaml_modified": not protected_files_unchanged,
        "scripts_research_conductor_modified": not protected_files_unchanged,
        "honest_verdict": (
            "complete: milestone_123_state_initialized_122_archived"
            if status == "complete"
            else "blocked: milestone_123_archive_missing_or_unsafe_state"
        ),
    }


def _source_inputs(root: Path) -> dict[str, dict[str, bool]]:
    paths = [
        Path("research-complete.yaml"),
        Path("research-roadmap.yaml"),
        Path("ops/changelog.md"),
        ROADMAP_DOC_PATH,
    ]
    return {str(path): {"exists": (root / path).exists()} for path in paths}


def run(
    *,
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-066: write the `.122` archive and `.123` state artifact."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)

    research_complete = _load_yaml(root_path / "research-complete.yaml")
    active_roadmap = _load_yaml(root_path / "research-roadmap.yaml")
    archive = _find_milestone(research_complete, PREDECESSOR_MILESTONE)
    archived_tasks = _task_rows(archive)
    protected_clean = (
        _protected_files_clean(root_path)
        if protected_files_unchanged is None
        else protected_files_unchanged
    )

    artifact = build_artifact(
        research_complete=research_complete,
        active_roadmap=active_roadmap,
        roadmap_doc_text=_read_text(root_path / ROADMAP_DOC_PATH),
        changelog_text=_read_text(root_path / "ops" / "changelog.md"),
        deliverable_exists=_artifact_exists_map(root_path, archived_tasks),
        protected_files_unchanged=protected_clean,
    )
    artifact["source_inputs_read"] = _source_inputs(root_path)
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
