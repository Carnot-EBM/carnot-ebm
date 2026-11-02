"""Build the Exp 3205 archive and .297 activation artifact.

Spec refs: REQ-REPORT-3205, SCENARIO-REPORT-3205.

This module is a small provenance accountant. It reads the completed `.296`
capstone, matrix, archived task ledger, and current roadmap queue metadata, then
writes a machine-readable handoff record for `.297`. It does not run the
conductor or mutate protected roadmap files because this experiment is meant to
record the activation boundary, not perform the activation itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
SCHEMA_VERSION = "carnot.archive_activation.v296_to_v297.v1"
EXPERIMENT_ID = "exp3205"
MILESTONE = "2026.05.297"
PRIOR_MILESTONE = "2026.05.296"
PRIOR_TASK_RANGE = ["exp3191", "exp3204"]
TOP_UNRESOLVED_GAP = "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
FIRST_V297_TASK_ID = "exp3205-archive-v296-activate-v297"
ARTIFACT = "experiment_3205_archive_v296_activate_v297"

OUTPUT_REL_PATH = Path("results/experiment_3205_archive_v296_activate_v297.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3205_archive_v296_activate_v297.py"
CAPSTONE_V296_REL_PATH = Path("results/experiment_3204_capstone_v296.json")
MATRIX_V30_REL_PATH = Path("results/experiment_3203_cross_corpus_matrix_v30.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
CHANGELOG_REL_PATH = Path("ops/changelog.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

EXPECTED_PRIOR_EXPERIMENT_IDS = tuple(f"exp{experiment}" for experiment in range(3191, 3205))
CRITICAL_PATH_IDS = (
    "exp3206-cuda-env-forensics-ledger-v1",
    "exp3207-llama-cpp-cuda-rebuild-clean-subprocess-v1",
    "exp3208-full-local-sota-receipt-v5",
    "exp3209-clean-live-sota-verifier-rerun-v12",
    "exp3212-structured-repair-proposal-preflight-v1",
    "exp3213-repair-gate-decision-v6",
    "exp3214-multi-turn-repair-ladder-v7",
    "exp3217-cross-corpus-matrix-v31",
    "exp3218-capstone-v297",
)


@dataclass(frozen=True)
class SourceSpec:
    """One source file whose exact bytes support the activation artifact."""

    role: str
    path: Path


SOURCE_SPECS = (
    SourceSpec("capstone_v296_authority", CAPSTONE_V296_REL_PATH),
    SourceSpec("matrix_v30_authority", MATRIX_V30_REL_PATH),
    SourceSpec("research_complete_archive", RESEARCH_COMPLETE_REL_PATH),
    SourceSpec("staged_roadmap_queue", STAGED_ROADMAP_REL_PATH),
    SourceSpec("active_roadmap_queue", ACTIVE_ROADMAP_REL_PATH),
    SourceSpec("vnext_milestone_doc", VNEXT_DOC_REL_PATH),
    SourceSpec("conductor_log_authority", CONDUCTOR_LOG_REL_PATH),
    SourceSpec("changelog_authority", CHANGELOG_REL_PATH),
    SourceSpec("protected_research_conductor", CONDUCTOR_REL_PATH),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and return empty evidence when the file is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_mapping(path: Path) -> JsonDict:
    """Read YAML while preserving root lists used by `research-complete.yaml`."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}
    if isinstance(payload, Mapping):
        return dict(payload) if all(isinstance(key, str) for key in payload) else {}
    if isinstance(payload, list):
        return {"_root_list": payload}
    return {}


def sha256_file(path: Path) -> str | None:
    """Checksum source files so the archive can be reproduced byte-for-byte."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_next_milestone(current: str) -> str:
    """Increment the final CalVer component while preserving its zero padding."""

    parts = current.split(".")
    if len(parts) != 3 or not parts[-1].isdigit():
        return ""
    width = len(parts[-1])
    parts[-1] = f"{int(parts[-1]) + 1:0{width}d}"
    return ".".join(parts)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3205: synthesize the .296 archive and .297 activation record."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V296_REL_PATH)
    matrix = read_json_object(root_path / MATRIX_V30_REL_PATH)
    research_complete = read_yaml_mapping(root_path / RESEARCH_COMPLETE_REL_PATH)
    staged = read_yaml_mapping(root_path / STAGED_ROADMAP_REL_PATH)
    active = read_yaml_mapping(root_path / ACTIVE_ROADMAP_REL_PATH)

    queue_paths = _queue_paths(root_path, staged, active)
    prior_terminal_statuses = _prior_terminal_statuses(research_complete)
    capstone_ready = capstone.get("capstone_v296_ready") is True
    matrix_ready = matrix.get("cross_corpus_matrix_v30_ready") is True
    prior_blockers = _int_value(capstone.get("publication_blocker_count"))
    prior_next_gap = str(capstone.get("next_top_gap") or "")
    blocked_reasons = _blocked_reasons(
        capstone=capstone,
        matrix=matrix,
        capstone_ready=capstone_ready,
        matrix_ready=matrix_ready,
        prior_next_gap=prior_next_gap,
        prior_terminal_statuses=prior_terminal_statuses,
        queue_paths=queue_paths,
        vnext_doc_exists=(root_path / VNEXT_DOC_REL_PATH).is_file(),
    )
    activation_ready = not blocked_reasons
    source_artifacts = _source_artifacts(root_path)

    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "prior_task_range": list(PRIOR_TASK_RANGE),
        "prior_task_count": len(EXPECTED_PRIOR_EXPERIMENT_IDS),
        "capstone_artifact": CAPSTONE_V296_REL_PATH.as_posix(),
        "matrix_artifact": MATRIX_V30_REL_PATH.as_posix(),
        "prior_capstone_ready": capstone_ready,
        "prior_matrix_ready": matrix_ready,
        "prior_paper_ready": capstone.get("paper_ready") is True,
        "prior_publication_blocker_count": prior_blockers,
        "prior_next_top_gap": prior_next_gap,
        "activation_ready": activation_ready,
        "research_roadmap_next_exists": (root_path / STAGED_ROADMAP_REL_PATH).is_file(),
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "protected_files_modified_by_this_task": {
            "scripts/research_conductor.py": False,
            "research-roadmap.yaml": False,
        },
        "queue_paths": queue_paths,
        "new_milestone_document": VNEXT_DOC_REL_PATH.as_posix(),
        "conductor_activation_observed": _file_contains(
            root_path / CONDUCTOR_LOG_REL_PATH,
            "Milestone 2026.05.297 activated",
        ),
        "changelog_mentions_milestone": _file_contains(
            root_path / CHANGELOG_REL_PATH,
            "Milestone 2026.05.297",
        ),
        "prior_terminal_statuses": prior_terminal_statuses,
        "terminal_status_counts": _terminal_status_counts(prior_terminal_statuses),
        "critical_path": _critical_path(queue_paths),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "blocked_reasons": blocked_reasons,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_repair_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3205 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _queue_paths(root: Path, staged: Mapping[str, Any], active: Mapping[str, Any]) -> JsonDict:
    staged_exists = (root / STAGED_ROADMAP_REL_PATH).is_file()
    active_exists = (root / ACTIVE_ROADMAP_REL_PATH).is_file()
    selected_path = STAGED_ROADMAP_REL_PATH if staged_exists else ACTIVE_ROADMAP_REL_PATH
    selected_payload = staged if staged_exists else active
    task_ids = _task_ids(selected_payload)
    return {
        "staged_roadmap_path": STAGED_ROADMAP_REL_PATH.as_posix(),
        "staged_roadmap_exists": staged_exists,
        "active_roadmap_path": ACTIVE_ROADMAP_REL_PATH.as_posix(),
        "active_roadmap_exists": active_exists,
        "selected_queue_path": selected_path.as_posix(),
        "selected_queue_milestone": str(selected_payload.get("milestone") or ""),
        "selected_queue_first_task": task_ids[0] if task_ids else "",
        "selected_queue_task_count": len(task_ids),
        "milestone_doc": str(selected_payload.get("milestone_doc") or ""),
        "milestone_doc_exists": (root / VNEXT_DOC_REL_PATH).is_file(),
    }


def _prior_terminal_statuses(research_complete: Mapping[str, Any]) -> list[JsonDict]:
    milestone = _find_milestone_entry(research_complete, PRIOR_MILESTONE)
    tasks = _as_list(milestone.get("tasks"))
    found: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id") or "")
        exp_id = task_id.split("-", 1)[0]
        if exp_id in EXPECTED_PRIOR_EXPERIMENT_IDS:
            found[exp_id] = task

    rows: list[JsonDict] = []
    for exp_id in EXPECTED_PRIOR_EXPERIMENT_IDS:
        task = found.get(exp_id, {})
        status = str(task.get("result") or "missing")
        rows.append(
            {
                "experiment_id": exp_id,
                "task_id": str(task.get("id") or exp_id),
                "title": str(task.get("title") or ""),
                "deliverable": str(task.get("deliverable") or ""),
                "terminal_status": status,
                "terminal": _is_terminal_status(status),
            }
        )
    return rows


def _find_milestone_entry(research_complete: Mapping[str, Any], milestone: str) -> JsonDict:
    candidates = _as_list(research_complete.get("_root_list"))
    candidates.extend(_as_list(research_complete.get("milestones")))
    for entry in candidates:
        if isinstance(entry, Mapping) and str(entry.get("id") or "") == milestone:
            return dict(entry)
    if str(research_complete.get("id") or "") == milestone:
        return dict(research_complete)
    return {}


def _is_terminal_status(status: str) -> bool:
    return status.startswith(("OK", "FAIL", "SKIP", "BLOCKED")) or status.endswith("(conductor)")


def _terminal_status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("terminal_status") or "missing") for row in rows)
    return dict(sorted(counts.items()))


def _prior_task_range_complete(rows: list[Mapping[str, Any]]) -> bool:
    observed = [str(row.get("experiment_id") or "") for row in rows]
    return observed == list(EXPECTED_PRIOR_EXPERIMENT_IDS) and all(
        row.get("terminal") is True for row in rows
    )


def _blocked_reasons(
    *,
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
    capstone_ready: bool,
    matrix_ready: bool,
    prior_next_gap: str,
    prior_terminal_statuses: list[Mapping[str, Any]],
    queue_paths: Mapping[str, Any],
    vnext_doc_exists: bool,
) -> list[str]:
    reasons: list[str] = []
    if not capstone:
        reasons.append("capstone_v296 authority is missing or malformed")
    elif not capstone_ready:
        reasons.append("capstone_v296 authority is not ready")
    if not matrix:
        reasons.append("matrix_v30 authority is missing or malformed")
    elif not matrix_ready:
        reasons.append("matrix_v30 authority is not ready")
    if _int_value(capstone.get("publication_blocker_count")) != _int_value(
        matrix.get("publication_blocker_count")
    ):
        reasons.append("capstone and matrix publication blocker counts disagree")
    if prior_next_gap != TOP_UNRESOLVED_GAP:
        reasons.append("prior next_top_gap does not match the .297 critical path")
    if matrix and str(matrix.get("next_top_gap") or "") != prior_next_gap:
        reasons.append("matrix_v30 next_top_gap disagrees with capstone authority")
    if not _prior_task_range_complete(prior_terminal_statuses):
        reasons.append("prior terminal statuses do not cover exp3191 through exp3204")
    if expected_next_milestone(PRIOR_MILESTONE) != MILESTONE:
        reasons.append("expected CalVer sequence does not produce 2026.05.297")
    if queue_paths.get("selected_queue_milestone") != MILESTONE:
        reasons.append("selected queue milestone is not 2026.05.297")
    if queue_paths.get("selected_queue_first_task") != FIRST_V297_TASK_ID:
        reasons.append("selected queue first task is not exp3205-archive-v296-activate-v297")
    if queue_paths.get("milestone_doc") != VNEXT_DOC_REL_PATH.as_posix():
        reasons.append("selected queue milestone_doc is not the vNEXT document")
    if not queue_paths.get("selected_queue_task_count"):
        reasons.append("selected queue has no tasks")
    if not vnext_doc_exists:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


def _critical_path(queue_paths: Mapping[str, Any]) -> JsonDict:
    return {
        "top_unresolved_gap": TOP_UNRESOLVED_GAP,
        "first_milestone_task": str(queue_paths.get("selected_queue_first_task") or ""),
        "unblock_sequence": list(CRITICAL_PATH_IDS),
        "terminal_aggregation_tasks": [
            "exp3217-cross-corpus-matrix-v31",
            "exp3218-capstone-v297",
        ],
    }


def _source_artifacts(root: Path) -> list[JsonDict]:
    return [
        {
            "role": spec.role,
            "path": spec.path.as_posix(),
            "present": (root / spec.path).is_file(),
            "sha256": sha256_file(root / spec.path),
        }
        for spec in SOURCE_SPECS
    ]


def _inference_substrate() -> JsonDict:
    return {
        "kind": "archive_activation_aggregation_from_checked_in_artifacts",
        "source": "capstone_v296_matrix_v30_research_complete_roadmap_queue_and_ops_logs",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("activation_ready") is True:
        return (
            "complete: archive_v296_activate_v297_ready=true; "
            "prior_task_range=exp3191-exp3204; "
            f"prior_paper_ready={str(artifact.get('prior_paper_ready')).lower()}; "
            f"prior_publication_blocker_count={artifact.get('prior_publication_blocker_count')}; "
            f"prior_next_top_gap={artifact.get('prior_next_top_gap')}; "
            f"queue_path={_as_mapping(artifact.get('queue_paths')).get('selected_queue_path')}"
        )
    return (
        "blocked_activation_not_ready: "
        "prior_task_range=exp3191-exp3204; "
        f"reasons={'; '.join(str(reason) for reason in _as_list(artifact.get('blocked_reasons')))}"
    )


def _file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except OSError:
        return False


def _task_ids(payload: Mapping[str, Any]) -> list[str]:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [
        str(task["id"])
        for task in tasks
        if isinstance(task, Mapping) and task.get("id") not in (None, "")
    ]


def _duration(start: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - start), 6)


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
