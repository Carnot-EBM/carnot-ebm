"""Generate the Exp 3026 archive artifact for milestone 2026.05.283.

Spec refs: REQ-REPORT-3026, SCENARIO-REPORT-3026.

This module is intentionally narrow bookkeeping. It reads the completed .283
capstone, the matrix that fed it, the completed-milestone ledger, and the .284
roadmap state. It then writes an auditable activation artifact without editing
the active roadmap, the conductor, historical results, or ops reconciliation
files.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.archive_activation.v2"
EXPERIMENT_ID = "exp3026-archive-v283-activate-v284"
ARCHIVED_MILESTONE = "2026.05.283"
NEXT_MILESTONE = "2026.05.284"
RUN_DATE = "20260525"
CAPSTONE_SOURCE = "results/experiment_3025_capstone_v283.json"
MATRIX_SOURCE = "results/experiment_3024_cross_corpus_matrix_v17.json"
DEFAULT_OUTPUT_PATH = Path("results/experiment_3026_archive_v283_activate_v284.json")
MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"
REQUESTED_STAGED_ROADMAP = "research-roadmap-next.yaml"
ACTIVE_ROADMAP = "research-roadmap.yaml"
PROTECTED_FILES = (
    ACTIVE_ROADMAP,
    REQUESTED_STAGED_ROADMAP,
    "scripts/research_conductor.py",
)
INFERENCE_SUBSTRATE = {
    "mode": "aggregation",
    "source": "checked_in_artifacts",
    "live_inference": False,
    "llm_calls": False,
    "gpu_required": False,
}
REQUIRED_ARTIFACT_FIELDS = {
    "milestone_archived",
    "next_milestone",
    "next_roadmap_path",
    "capstone_ready",
    "previous_paper_ready",
    "carry_forward_blockers",
    "protected_files_unchanged",
    "inference_substrate",
    "honest_verdict",
}
STATUS_ROWS = (
    ("flagged", "flagged_rows"),
    ("blocked", "blocked_rows"),
    ("gated-skipped", "gated_skipped_rows"),
    ("projection-only", "projection_only_rows"),
    ("pilot-only", "pilot_only_rows"),
    ("missing", "missing_rows"),
)
STATUS_COUNT_KEYS = {
    "blocked": "blocked_count",
    "clean": "clean_count",
    "flagged": "flagged_count",
    "gated-skipped": "gated_skipped_count",
    "missing": "missing_count",
    "pilot-only": "pilot_only_count",
    "projection-only": "projection_only_count",
}
STATUS_SUMMARY_KEYS = {
    "blocked": "blocked",
    "clean": "clean",
    "flagged": "flagged",
    "gated-skipped": "gated_skipped",
    "missing": "missing",
    "pilot-only": "pilot_only",
    "projection-only": "projection_only",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    if not text.strip():
        return {}
    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _as_str_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _as_int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }


def _task_ids(payload: Mapping[str, Any]) -> list[str]:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [str(task["id"]) for task in tasks if isinstance(task, dict) and "id" in task]


def _matrix_rows(value: Any) -> list[dict[str, Any]]:
    rows = value.get("rows") if isinstance(value, Mapping) else None
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _sha256_path(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protected_state(root: Path) -> dict[str, dict[str, Any]]:
    return {
        rel_path: {
            "exists": (root / rel_path).exists(),
            "sha256": _sha256_path(root / rel_path),
        }
        for rel_path in PROTECTED_FILES
    }


def _protected_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> bool:
    return dict(before) == dict(after)


def _archive_summary(root: Path) -> dict[str, Any]:
    payload = _load_yaml_mapping(root / "research-complete.yaml")
    rows = payload.get("milestones")
    milestone_rows = rows if isinstance(rows, list) else []
    archive_rows = [
        row
        for row in milestone_rows
        if isinstance(row, dict) and str(row.get("id")) == ARCHIVED_MILESTONE
    ]
    completed_rows = [row for row in archive_rows if "completed" in row]
    task_counts = [
        len(row.get("tasks"))
        for row in completed_rows
        if isinstance(row.get("tasks"), list)
    ]
    n_tasks_archived = max(task_counts, default=0)
    return {
        "archive_path": "research-complete.yaml",
        "archive_row_count": len(archive_rows),
        "completed_archive_row_count": len(completed_rows),
        "n_tasks_archived": n_tasks_archived,
        "milestone_archived": len(completed_rows) == 1 and n_tasks_archived >= 14,
    }


def _roadmap_activation(root: Path) -> dict[str, Any]:
    staged_path = root / REQUESTED_STAGED_ROADMAP
    active_payload = _load_yaml_mapping(root / ACTIVE_ROADMAP)
    if staged_path.exists():
        source_name = REQUESTED_STAGED_ROADMAP
        payload = _load_yaml_mapping(staged_path)
        used_fallback = False
    else:
        source_name = ACTIVE_ROADMAP
        payload = active_payload
        used_fallback = True

    tasks = _task_ids(payload)
    milestone_doc = str(payload.get("milestone_doc") or "")
    milestone = str(payload.get("milestone") or "")
    return {
        "requested_staged_roadmap_path": REQUESTED_STAGED_ROADMAP,
        "requested_staged_roadmap_exists": staged_path.exists(),
        "source_path": source_name,
        "source_exists": (root / source_name).exists(),
        "used_active_roadmap_fallback": used_fallback,
        "active_roadmap_milestone": str(active_payload.get("milestone") or ""),
        "observed_milestone": milestone,
        "expected_milestone": NEXT_MILESTONE,
        "milestone_matches": milestone == NEXT_MILESTONE,
        "observed_milestone_doc": milestone_doc,
        "expected_milestone_doc": MILESTONE_DOC,
        "milestone_doc_matches": milestone_doc == MILESTONE_DOC,
        "milestone_title": str(payload.get("milestone_title") or ""),
        "n_tasks": len(tasks),
        "non_empty_tasks": bool(tasks),
        "task_ids": tasks,
    }


def _count_for_status(
    status: str,
    *,
    matrix: Mapping[str, Any],
    capstone_counts: Mapping[str, int],
) -> int:
    matrix_key = STATUS_COUNT_KEYS[status]
    matrix_value = matrix.get(matrix_key)
    if isinstance(matrix_value, int) and not isinstance(matrix_value, bool):
        return matrix_value
    return capstone_counts.get(status, 0)


def _adversarial_rows(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _matrix_rows(matrix):
        flags = _as_str_list(row.get("upstream_flags"))
        if not flags:
            continue
        rows.append(
            {
                "row_id": str(row.get("row_id") or ""),
                "source_experiment_id": str(row.get("source_experiment_id") or ""),
                "status": str(row.get("status") or ""),
                "source_honest_verdict": str(row.get("source_honest_verdict") or ""),
                "upstream_flags": flags,
            }
        )
    return rows


def _milestone_summary(
    *,
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> dict[str, Any]:
    capstone_counts = _as_int_mapping(capstone.get("matrix_status_counts"))
    matrix_status_summary = {
        STATUS_SUMMARY_KEYS[status]: _count_for_status(
            status,
            matrix=matrix,
            capstone_counts=capstone_counts,
        )
        for status in STATUS_SUMMARY_KEYS
    }
    matrix_status_summary["paper_ready"] = bool(capstone.get("paper_ready", False))
    adversarial = _adversarial_rows(matrix)
    return {
        "matrix_v17_ready": bool(matrix.get("matrix_v17_ready", False)),
        "matrix_status_summary": matrix_status_summary,
        "capstone_matrix_status_counts": capstone_counts,
        "capstone_task_classification_counts": _as_int_mapping(
            capstone.get("task_classification_counts")
        ),
        "clean_task_rows": _as_str_list(capstone.get("clean_task_rows")),
        "flagged_task_rows": _as_str_list(capstone.get("flagged_task_rows")),
        "blocked_task_rows": _as_str_list(capstone.get("blocked_task_rows")),
        "gated_skipped_task_rows": _as_str_list(capstone.get("gated_skipped_task_rows")),
        "projection_only_task_rows": _as_str_list(capstone.get("projection_only_task_rows")),
        "pilot_only_task_rows": _as_str_list(capstone.get("pilot_only_task_rows")),
        "missing_task_rows": _as_str_list(capstone.get("missing_task_rows")),
        "adversarially_flagged": bool(
            capstone.get("flagged_adversarial") or matrix.get("flagged_adversarial")
        ),
        "adversarially_flagged_count": len(adversarial),
        "adversarially_flagged_rows": adversarial,
    }


def _carry_forward_blockers(
    *,
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = [
        {
            "kind": "paper_ready_blockers",
            "source": CAPSTONE_SOURCE,
            "rows": _as_str_list(capstone.get("paper_ready_blockers")),
        }
    ]
    for status, field in STATUS_ROWS:
        blockers.append(
            {
                "kind": "status_bucket",
                "status": status,
                "source": CAPSTONE_SOURCE,
                "rows": _as_str_list(capstone.get(field)),
            }
        )
    blockers.extend(
        [
            {
                "kind": "adversarial_flags",
                "source": MATRIX_SOURCE,
                "rows": _adversarial_rows(matrix),
            },
            {
                "kind": "matrix_still_blocked_claims",
                "source": MATRIX_SOURCE,
                "rows": _as_str_list(matrix.get("still_blocked_claims"))
                or _as_str_list(capstone.get("matrix_still_blocked_claims")),
            },
            {
                "kind": "recommended_next_actions",
                "source": MATRIX_SOURCE,
                "rows": _as_str_list(matrix.get("recommended_next_actions"))
                or _as_str_list(capstone.get("matrix_recommended_next_actions")),
            },
        ]
    )
    return blockers


def _honest_verdict(
    *,
    capstone_present: bool,
    capstone_ready: bool,
    archive: Mapping[str, Any],
    roadmap: Mapping[str, Any],
    protected_files_unchanged: bool,
) -> tuple[str, list[str]]:
    blocked_reasons: list[str] = []
    if not capstone_present:
        blocked_reasons.append("Exp 3025 capstone source missing or invalid")
    elif not capstone_ready:
        blocked_reasons.append("Exp 3025 capstone is not capstone_ready=true")
    if not archive.get("milestone_archived"):
        blocked_reasons.append("research-complete.yaml does not contain completed 2026.05.283 archive")
    if not roadmap.get("milestone_matches"):
        blocked_reasons.append("roadmap milestone is not 2026.05.284")
    if not roadmap.get("milestone_doc_matches"):
        blocked_reasons.append("roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md")
    if not roadmap.get("non_empty_tasks"):
        blocked_reasons.append("roadmap for 2026.05.284 has no tasks")
    if not protected_files_unchanged:
        blocked_reasons.append("protected files changed during archive activation")

    if blocked_reasons:
        return "blocked: " + "; ".join(blocked_reasons), blocked_reasons
    return (
        "complete: milestone_archived=true; next_milestone=2026.05.284; "
        "capstone_ready=true; previous_paper_ready=false; protected_files_unchanged=true",
        blocked_reasons,
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """SCENARIO-REPORT-3026: write the .283 archive and .284 activation JSON."""

    start_s = clock()
    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    protected_before = _protected_state(root_path)
    capstone = _read_json_mapping(root_path / CAPSTONE_SOURCE)
    matrix = _read_json_mapping(root_path / MATRIX_SOURCE)
    archive = _archive_summary(root_path)
    roadmap = _roadmap_activation(root_path)
    protected_after = _protected_state(root_path)
    protected_files_unchanged = _protected_unchanged(protected_before, protected_after)
    capstone_ready = bool(capstone.get("capstone_ready", False))
    previous_paper_ready = bool(capstone.get("paper_ready", False))
    honest_verdict, blocked_reasons = _honest_verdict(
        capstone_present=bool(capstone),
        capstone_ready=capstone_ready,
        archive=archive,
        roadmap=roadmap,
        protected_files_unchanged=protected_files_unchanged,
    )
    duration_s = round(clock() - start_s, 6)

    artifact = {
        "id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": duration_s,
        "milestone_archived": bool(archive["milestone_archived"]),
        "archived_milestone": ARCHIVED_MILESTONE,
        "next_milestone": NEXT_MILESTONE,
        "next_roadmap_path": str(roadmap["source_path"]),
        "capstone_source": CAPSTONE_SOURCE,
        "matrix_source": MATRIX_SOURCE,
        "capstone_ready": capstone_ready,
        "previous_paper_ready": previous_paper_ready,
        "carry_forward_blockers": _carry_forward_blockers(capstone=capstone, matrix=matrix),
        "protected_files_unchanged": protected_files_unchanged,
        "protected_file_state_before": protected_before,
        "protected_file_state_after": protected_after,
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "honest_verdict": honest_verdict,
        "blocked_reasons": blocked_reasons,
        "archive": archive,
        "roadmap_activation": roadmap,
        "next_execution_order": list(roadmap["task_ids"]),
        "milestone_283_summary": _milestone_summary(matrix=matrix, capstone=capstone),
        "source_artifacts_read": [
            CAPSTONE_SOURCE if bool(capstone) else "",
            MATRIX_SOURCE if bool(matrix) else "",
        ],
        "no_new_llm_call": True,
        "no_new_inference": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_push": True,
        "notes": [
            "research-roadmap-next.yaml was preferred when present.",
            "research-roadmap.yaml was used read-only when the staged roadmap was absent.",
            "research-roadmap.yaml and scripts/research_conductor.py were not modified.",
            "Historical result rows were carried forward without repair.",
        ],
    }
    return _write_json(output, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
