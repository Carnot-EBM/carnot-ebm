"""Build the Exp 1652 `.126` archive and `.127` initialization artifact.

Spec: REQ-REPORT-068, SCENARIO-REPORT-068.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260509"
PREDECESSOR_MILESTONE = "2026.05.126"
TARGET_MILESTONE = "2026.05.127"
EXPERIMENT = "1652_archive"
SCHEMA = "milestone_127_initialization_v1"

DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1652_archive.json"
ROADMAP_DOC_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

EXPECTED_126_TASK_IDS = [
    "exp1640-nsvif-dsl-parser",
    "exp1641-nsvif-live-sota",
    "exp1642-llguidance-adapter",
    "exp1643-static-csr-mask",
    "exp1644-cerce-ledger",
    "exp1645-fr11-cerce-learning",
    "exp1646-ebcn-prototype",
    "exp1647-rkan-lean4-export",
    "exp1648-sparse-kan-clustering",
    "exp1649-vivado-potts-synthesis",
    "exp1650-kv260-potts-bringup",
    "exp1651-milestone-retro",
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
    "setup_127_state",
    "nsvif_dsl_landed",
    "kv260_potts_synthesis_landed",
    "cerce_ledger_landed",
    "missing_task_deliverables",
    "research_roadmap_yaml_modified",
    "scripts_research_conductor_modified",
    "honest_verdict",
}

NSVIF_DSL_PATH = "results/experiment_1640_nsvif_dsl.json"
NSVIF_SOTA_PATH = "results/experiment_1641_nsvif_sota.json"
CERCE_LEDGER_PATH = "results/experiment_1644_cerce_ledger.json"
KV260_POTTS_SYNTHESIS_PATH = "results/experiment_1649_vivado_synthesis.json"
KV260_POTTS_BRINGUP_PATH = "results/experiment_1650_kv260_bringup.json"
RETRO_PATH = "results/experiment_1651_retro.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(output_path: Path | str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-068: persist a started marker before reading archive evidence."""

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
            "setup_127_state": False,
            "nsvif_dsl_landed": False,
            "kv260_potts_synthesis_landed": False,
            "cerce_ledger_landed": False,
            "missing_task_deliverables": [],
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
            "honest_verdict": "in_progress",
        }
    )
    return _write_json(Path(output_path), artifact)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")) or {}


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
    terminal_tokens = ("ok", "complete", "blocked", "retired", "failed", "research finding")
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
    return [
        str(task.get("id") or "")
        for task in active_roadmap.get("tasks", [])
        if isinstance(task, Mapping)
    ]


def _context_has_all(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def _archive_mentions(rows: Sequence[Mapping[str, Any]], terms: Sequence[str]) -> bool:
    haystack = "\n".join(
        f"{row.get('id', '')} {row.get('title', '')} {row.get('result', '')}" for row in rows
    ).lower()
    return all(term.lower() in haystack for term in terms)


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


def _artifact_payloads(root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for row in rows:
        deliverable = str(row.get("deliverable") or "")
        if deliverable.startswith("results/"):
            payloads[deliverable] = _read_json(root / deliverable)
    return payloads


def _source_inputs(root: Path) -> dict[str, dict[str, bool]]:
    paths = [
        Path("research-complete.yaml"),
        Path("research-roadmap.yaml"),
        Path("ops/changelog.md"),
        ROADMAP_DOC_PATH,
    ]
    return {str(path): {"exists": (root / path).exists()} for path in paths}


def build_artifact(
    *,
    research_complete: Mapping[str, Any],
    active_roadmap: Mapping[str, Any],
    roadmap_doc_text: str,
    changelog_text: str,
    artifact_payloads: Mapping[str, Mapping[str, Any]],
    deliverable_exists: Mapping[str, bool],
    protected_files_unchanged: bool,
) -> dict[str, Any]:
    """REQ-REPORT-068: build the terminal `.126` archive and `.127` state artifact."""

    archive = _find_milestone(research_complete, PREDECESSOR_MILESTONE)
    archived_tasks = _task_rows(archive)
    archived_task_ids = [task["id"] for task in archived_tasks]
    expected_ids_present = archived_task_ids == EXPECTED_126_TASK_IDS
    predecessor_archived = bool(archive) and expected_ids_present
    predecessor_tasks_terminal = _tasks_terminal(archived_tasks)

    roadmap_task_ids = _roadmap_task_ids(active_roadmap)
    active_roadmap_milestone = str(active_roadmap.get("milestone") or "")
    first_active_task_id = roadmap_task_ids[0] if roadmap_task_ids else ""
    active_roadmap_is_127 = active_roadmap_milestone == TARGET_MILESTONE
    first_task_is_archive = first_active_task_id == "exp1652-archive-126"
    status_moved_to_changelog = PREDECESSOR_MILESTONE in changelog_text

    nsvif_track_ready = _context_has_all(roadmap_doc_text, ["NSVIF", "SOTA"])
    energy_guided_track_ready = _context_has_all(
        roadmap_doc_text, ["Energy-Guided Decoding", "STATIC CSR"]
    )
    ebrm_trace_track_ready = _context_has_all(roadmap_doc_text, ["EBRM", "Trace Scoring"])
    smgi_track_ready = _context_has_all(roadmap_doc_text, ["SMGI", "CerCE"])
    kv260_potts_track_ready = _context_has_all(roadmap_doc_text, ["KV260", "Potts"])
    setup_127_state = bool(
        active_roadmap_is_127
        and first_task_is_archive
        and nsvif_track_ready
        and energy_guided_track_ready
        and ebrm_trace_track_ready
        and smgi_track_ready
        and kv260_potts_track_ready
    )

    nsvif_parser = artifact_payloads.get(NSVIF_DSL_PATH, {})
    nsvif_sota = artifact_payloads.get(NSVIF_SOTA_PATH, {})
    cerce_ledger = artifact_payloads.get(CERCE_LEDGER_PATH, {})
    potts_synthesis = artifact_payloads.get(KV260_POTTS_SYNTHESIS_PATH, {})
    potts_bringup = artifact_payloads.get(KV260_POTTS_BRINGUP_PATH, {})
    retro = artifact_payloads.get(RETRO_PATH, {})

    nsvif_zero_false_accepts = (
        nsvif_parser.get("false_accept_rate") == 0.0 and nsvif_sota.get("false_accepts") == 0
    )
    nsvif_dsl_landed = bool(
        _archive_mentions(archived_tasks, ["nsvif"])
        and nsvif_track_ready
        and nsvif_parser.get("status") == "complete"
        and nsvif_parser.get("parser_success") is True
        and nsvif_sota.get("status") == "complete"
        and nsvif_zero_false_accepts
    )
    cerce_ledger_landed = bool(
        _archive_mentions(archived_tasks, ["cerce", "ledger"])
        and smgi_track_ready
        and cerce_ledger.get("status") == "complete"
        and cerce_ledger.get("cerce_ledger_ready") is True
        and cerce_ledger.get("ledger_implemented") is True
    )
    kv260_potts_synthesis_landed = bool(
        _archive_mentions(archived_tasks, ["kv260", "potts", "synthesis"])
        and kv260_potts_track_ready
        and bool(potts_synthesis)
    )
    kv260_potts_vivado_success = bool(potts_synthesis.get("synthesis_success"))
    kv260_potts_vivado_available = bool(potts_synthesis.get("vivado_available"))
    hardware_execution_claimed = bool(
        potts_bringup.get("hardware_execution_confirmed")
        or potts_bringup.get("kv260_board_bringup_success")
        or potts_bringup.get("hardware_bringup_success")
    )

    blocked_reasons: list[str] = []
    if not archive:
        blocked_reasons.append("research-complete.yaml has no 2026.05.126 archive")
    if archive and not expected_ids_present:
        blocked_reasons.append("2026.05.126 archive task ids do not match exp1640-exp1651")
    if archived_tasks and not predecessor_tasks_terminal:
        blocked_reasons.append("not every 2026.05.126 archive task has a terminal result")
    if not active_roadmap_is_127:
        blocked_reasons.append("active roadmap is not 2026.05.127")
    if active_roadmap_is_127 and not first_task_is_archive:
        blocked_reasons.append("active roadmap does not start with exp1652-archive-126")
    if not status_moved_to_changelog:
        blocked_reasons.append("ops/changelog.md lacks the 2026.05.126 status entry")
    if active_roadmap_is_127 and not setup_127_state:
        blocked_reasons.append("2026.05.127 roadmap state lacks one or more expected tracks")
    if not nsvif_dsl_landed:
        blocked_reasons.append("NSVIF DSL landing evidence incomplete")
    if not kv260_potts_synthesis_landed:
        blocked_reasons.append("KV260 Potts synthesis landing evidence incomplete")
    if not cerce_ledger_landed:
        blocked_reasons.append("CerCE ledger landing evidence incomplete")
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
        "setup_127_state": setup_127_state,
        "nsvif_track_ready": nsvif_track_ready,
        "energy_guided_track_ready": energy_guided_track_ready,
        "ebrm_trace_track_ready": ebrm_trace_track_ready,
        "smgi_track_ready": smgi_track_ready,
        "kv260_potts_track_ready": kv260_potts_track_ready,
        "nsvif_dsl_landed": nsvif_dsl_landed,
        "nsvif_zero_false_accepts": nsvif_zero_false_accepts,
        "cerce_ledger_landed": cerce_ledger_landed,
        "kv260_potts_synthesis_landed": kv260_potts_synthesis_landed,
        "kv260_potts_vivado_available": kv260_potts_vivado_available,
        "kv260_potts_vivado_success": kv260_potts_vivado_success,
        "hardware_execution_claimed": hardware_execution_claimed,
        "kv260_potts_evidence_note": (
            "archived_result_with_vivado_unavailable_no_hardware_execution_claim"
            if kv260_potts_synthesis_landed and not kv260_potts_vivado_success
            else "vivado_synthesis_successful"
        ),
        "milestone_126_retro": {
            "status": retro.get("status"),
            "criteria_met": retro.get("criteria_met"),
            "criteria_total": retro.get("criteria_total"),
            "honest_verdict": retro.get("honest_verdict"),
        },
        "source_artifact_status": {
            NSVIF_DSL_PATH: {
                "exists": deliverable_exists.get(NSVIF_DSL_PATH, False),
                "status": nsvif_parser.get("status"),
                "honest_verdict": nsvif_parser.get("honest_verdict"),
            },
            NSVIF_SOTA_PATH: {
                "exists": deliverable_exists.get(NSVIF_SOTA_PATH, False),
                "status": nsvif_sota.get("status"),
                "honest_verdict": nsvif_sota.get("honest_verdict"),
            },
            CERCE_LEDGER_PATH: {
                "exists": deliverable_exists.get(CERCE_LEDGER_PATH, False),
                "status": cerce_ledger.get("status"),
                "honest_verdict": cerce_ledger.get("honest_verdict"),
            },
            KV260_POTTS_SYNTHESIS_PATH: {
                "exists": deliverable_exists.get(KV260_POTTS_SYNTHESIS_PATH, False),
                "status": potts_synthesis.get("status"),
                "honest_verdict": potts_synthesis.get("honest_verdict"),
                "synthesis_success": potts_synthesis.get("synthesis_success"),
            },
            KV260_POTTS_BRINGUP_PATH: {
                "exists": deliverable_exists.get(KV260_POTTS_BRINGUP_PATH, False),
                "status": potts_bringup.get("status"),
                "honest_verdict": potts_bringup.get("honest_verdict"),
            },
        },
        "missing_task_deliverables": missing_task_deliverables,
        "missing_task_deliverable_count": len(missing_task_deliverables),
        "blocked_reasons": blocked_reasons,
        "research_roadmap_yaml_modified": not protected_files_unchanged,
        "scripts_research_conductor_modified": not protected_files_unchanged,
        "honest_verdict": (
            "complete: milestone_127_initialized_126_archived_nsvif_cerce_kv260_potts_landed_no_hardware_execution_claim"
            if status == "complete"
            else "blocked: milestone_127_initialization_missing_or_unsafe_state"
        ),
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    protected_files_unchanged: bool | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-068: write the `.126` archive and `.127` initialization artifact."""

    root_path = Path(root)
    output = Path(output_path)
    write_in_progress_artifact(output)

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
        artifact_payloads=_artifact_payloads(root_path, archived_tasks),
        deliverable_exists=_artifact_exists_map(root_path, archived_tasks),
        protected_files_unchanged=protected_clean,
    )
    artifact["source_inputs_read"] = _source_inputs(root_path)
    return _write_json(output, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
