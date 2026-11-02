"""Build the Exp 3219 archive and .298 activation artifact.

Spec refs: REQ-REPORT-3219, SCENARIO-REPORT-3219.

This module records the boundary between milestones `.297` and `.298`. It
reads the checked-in capstone and matrix artifacts, then writes a compact
handoff ledger. It does not run the conductor or touch protected roadmap files;
the point is to make the activation state auditable, not to perform activation.
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
SCHEMA_VERSION = "carnot.archive_activation.v297_to_v298.v1"
EXPERIMENT_ID = "exp3219"
MILESTONE = "2026.05.298"
PRIOR_MILESTONE = "2026.05.297"
PRIOR_TASK_RANGE = ["exp3205", "exp3218"]
TOP_UNRESOLVED_GAP = "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
FIRST_V298_TASK_ID = "exp3219-archive-v297-activate-v298"
ARTIFACT = "experiment_3219_archive_v297_activate_v298"

OUTPUT_REL_PATH = Path("results/experiment_3219_archive_v297_activate_v298.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3219_archive_v297_activate_v298.py"
CAPSTONE_V297_REL_PATH = Path("results/experiment_3218_capstone_v297.json")
MATRIX_V31_REL_PATH = Path("results/experiment_3217_cross_corpus_matrix_v31.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
CHANGELOG_REL_PATH = Path("ops/changelog.md")
STATUS_REL_PATH = Path("ops/status.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

EXPECTED_PRIOR_EXPERIMENT_IDS = tuple(f"exp{experiment}" for experiment in range(3205, 3219))
ARTIFACT_STATUSES = {"clean", "blocked", "gated_skipped", "diagnostic_only", "missing"}
CRITICAL_PATH_IDS = (
    "exp3220-hermetic-cuda-runtime-repair-ledger-v1",
    "exp3221-llama-cpp-cuda-offload-receipt-smoke-v1",
    "exp3222-full-local-sota-receipt-v6",
    "exp3225-clean-live-sota-verifier-rerun-v13",
    "exp3226-structured-repair-proposal-preflight-v2",
    "exp3227-repair-gate-decision-v7",
    "exp3228-multi-turn-repair-ladder-v8",
    "exp3231-cross-corpus-matrix-v32",
    "exp3232-capstone-v298",
)


@dataclass(frozen=True)
class SourceSpec:
    """One checked-in source whose bytes support the activation artifact."""

    role: str
    path: Path


SOURCE_SPECS = (
    SourceSpec("capstone_v297_authority", CAPSTONE_V297_REL_PATH),
    SourceSpec("matrix_v31_authority", MATRIX_V31_REL_PATH),
    SourceSpec("research_complete_archive", RESEARCH_COMPLETE_REL_PATH),
    SourceSpec("staged_roadmap_queue", STAGED_ROADMAP_REL_PATH),
    SourceSpec("active_roadmap_queue", ACTIVE_ROADMAP_REL_PATH),
    SourceSpec("vnext_milestone_doc", VNEXT_DOC_REL_PATH),
    SourceSpec("conductor_log_authority", CONDUCTOR_LOG_REL_PATH),
    SourceSpec("changelog_authority", CHANGELOG_REL_PATH),
    SourceSpec("status_authority", STATUS_REL_PATH),
    SourceSpec("protected_research_conductor", CONDUCTOR_REL_PATH),
)


def read_json_object(path: Path) -> JsonDict:
    """Read JSON source evidence and fail closed on absent or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_mapping(path: Path) -> JsonDict:
    """Read YAML evidence while preserving root-list milestone ledgers."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, list):
        return {"_root_list": payload}
    return {}


def sha256_file(path: Path) -> str | None:
    """Checksum source evidence so the handoff can be reproduced exactly."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_next_milestone(current: str) -> str:
    """Increment the final CalVer sequence component while preserving padding."""

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
    """REQ-REPORT-3219: synthesize the .297 archive and .298 activation record."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V297_REL_PATH)
    matrix = read_json_object(root_path / MATRIX_V31_REL_PATH)
    research_complete = read_yaml_mapping(root_path / RESEARCH_COMPLETE_REL_PATH)
    staged = read_yaml_mapping(root_path / STAGED_ROADMAP_REL_PATH)
    active = read_yaml_mapping(root_path / ACTIVE_ROADMAP_REL_PATH)

    queue_paths = _queue_paths(root_path, staged, active)
    prior_terminal_statuses = _prior_terminal_statuses(matrix, capstone, research_complete)
    capstone_ready = capstone.get("capstone_v297_ready") is True
    matrix_ready = matrix.get("cross_corpus_matrix_v31_ready") is True
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
        "prior_task_range_expanded": list(EXPECTED_PRIOR_EXPERIMENT_IDS),
        "prior_task_count": len(EXPECTED_PRIOR_EXPERIMENT_IDS),
        "capstone_artifact": CAPSTONE_V297_REL_PATH.as_posix(),
        "matrix_artifact": MATRIX_V31_REL_PATH.as_posix(),
        "prior_capstone_ready": capstone_ready,
        "prior_matrix_ready": matrix_ready,
        "prior_paper_ready": capstone.get("paper_ready") is True,
        "prior_publication_blocker_count": prior_blockers,
        "prior_next_top_gap": prior_next_gap,
        "activation_ready": activation_ready,
        "research_roadmap_next_exists": (root_path / STAGED_ROADMAP_REL_PATH).is_file(),
        "inference_substrate": "artifact_aggregation_only",
        "inference_substrate_details": _inference_substrate_details(),
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "protected_files_modified_by_this_task": {
            "scripts/research_conductor.py": False,
            "research-roadmap.yaml": False,
        },
        "queue_paths": queue_paths,
        "new_milestone_document": VNEXT_DOC_REL_PATH.as_posix(),
        "new_milestone_document_exists": (root_path / VNEXT_DOC_REL_PATH).is_file(),
        "conductor_activation_observed": _file_contains(
            root_path / CONDUCTOR_LOG_REL_PATH,
            "Milestone 2026.05.298 activated",
        ),
        "prior_terminal_statuses": prior_terminal_statuses,
        "terminal_status_counts": _terminal_status_counts(prior_terminal_statuses),
        "gate_blocked_prior_tasks": _gate_blocked_tasks(prior_terminal_statuses),
        "critical_path": _critical_path(queue_paths),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "blocked_reasons": blocked_reasons,
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
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
    """Build and persist the Exp 3219 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _prior_terminal_statuses(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    research_complete: Mapping[str, Any],
) -> list[JsonDict]:
    upstream = {
        str(row.get("experiment_id") or ""): row
        for row in _as_list(matrix.get("upstream_artifacts"))
        if isinstance(row, Mapping)
    }
    conductor_tasks = _prior_task_meta(research_complete)
    rows: list[JsonDict] = []
    for exp_id in EXPECTED_PRIOR_EXPERIMENT_IDS:
        task = conductor_tasks.get(exp_id, {})
        if exp_id == "exp3217":
            source = {
                "path": MATRIX_V31_REL_PATH.as_posix(),
                "role": "cross_corpus_matrix_v31",
                "status": _artifact_status(matrix, matrix.get("cross_corpus_matrix_v31_ready") is True),
                "honest_verdict": str(matrix.get("honest_verdict") or ""),
            }
        elif exp_id == "exp3218":
            source = {
                "path": CAPSTONE_V297_REL_PATH.as_posix(),
                "role": "capstone_v297",
                "status": _artifact_status(capstone, capstone.get("capstone_v297_ready") is True),
                "honest_verdict": str(capstone.get("honest_verdict") or ""),
            }
        else:
            source = dict(upstream.get(exp_id, {}))
        artifact_status = str(source.get("status") or "missing")
        if artifact_status not in ARTIFACT_STATUSES:
            artifact_status = "missing"
        conductor_status = _conductor_status(conductor_tasks, exp_id)
        gate_blocked = _gate_blocked(source, artifact_status)
        rows.append(
            {
                "experiment_id": exp_id,
                "task_id": str(task.get("id") or exp_id),
                "title": str(task.get("title") or ""),
                "deliverable": str(task.get("deliverable") or source.get("path") or ""),
                "role": str(source.get("role") or ""),
                "artifact_status": artifact_status,
                "terminal_status": artifact_status,
                "status_rationale": str(source.get("status_rationale") or ""),
                "conductor_terminal_status": conductor_status,
                "honest_verdict": str(source.get("honest_verdict") or ""),
                "gate_blocked": gate_blocked,
                "gated_skip_evidence": _as_mapping(source.get("gated_skip_evidence")),
                "terminal": _is_terminal_status(conductor_status),
            }
        )
    return rows


def _prior_task_meta(research_complete: Mapping[str, Any]) -> dict[str, JsonDict]:
    milestone = _find_milestone_entry(research_complete, PRIOR_MILESTONE)
    tasks = _as_list(milestone.get("tasks"))
    found: dict[str, JsonDict] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id") or "")
        exp_id = task_id.split("-", 1)[0]
        if exp_id in EXPECTED_PRIOR_EXPERIMENT_IDS:
            found[exp_id] = dict(task)
    return found


def _find_milestone_entry(research_complete: Mapping[str, Any], milestone: str) -> JsonDict:
    candidates = _as_list(research_complete.get("_root_list"))
    candidates.extend(_as_list(research_complete.get("milestones")))
    for entry in candidates:
        if isinstance(entry, Mapping) and str(entry.get("id") or "") == milestone:
            return dict(entry)
    if str(research_complete.get("id") or "") == milestone:
        return dict(research_complete)
    return {}


def _artifact_status(payload: Mapping[str, Any], ready: bool) -> str:
    if ready and payload:
        return "clean"
    return "blocked"


def _conductor_status(task_meta: Mapping[str, Mapping[str, Any]], exp_id: str) -> str:
    return str(_as_mapping(task_meta.get(exp_id)).get("result") or "")


def _gate_blocked(source: Mapping[str, Any], artifact_status: str) -> bool:
    evidence = _as_mapping(source.get("gated_skip_evidence"))
    return artifact_status == "gated_skipped" or evidence.get("status") == "gated_skipped"


def _is_terminal_status(status: str) -> bool:
    return status.startswith(("OK", "FAIL", "SKIP", "BLOCKED", "GATE_BLOCK"))


def _terminal_status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("artifact_status") or "missing") for row in rows)
    return dict(sorted(counts.items()))


def _gate_blocked_tasks(rows: list[Mapping[str, Any]]) -> list[str]:
    return [
        str(row.get("experiment_id"))
        for row in rows
        if row.get("gate_blocked") is True and row.get("experiment_id")
    ]


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
        reasons.append("capstone_v297 authority is missing or malformed")
    elif not capstone_ready:
        reasons.append("capstone_v297 authority is not ready")
    if not matrix:
        reasons.append("matrix_v31 authority is missing or malformed")
    elif not matrix_ready:
        reasons.append("matrix_v31 authority is not ready")
    if capstone and matrix and _int_value(capstone.get("publication_blocker_count")) != _int_value(
        matrix.get("publication_blocker_count")
    ):
        reasons.append("capstone and matrix publication blocker counts disagree")
    if prior_next_gap != TOP_UNRESOLVED_GAP:
        reasons.append("prior next_top_gap does not match the .298 critical path")
    if matrix and str(matrix.get("next_top_gap") or "") != prior_next_gap:
        reasons.append("matrix_v31 next_top_gap disagrees with capstone authority")
    if not _prior_task_range_complete(prior_terminal_statuses):
        reasons.append("prior terminal statuses do not cover exp3205 through exp3218")
    if expected_next_milestone(PRIOR_MILESTONE) != MILESTONE:
        reasons.append("expected CalVer sequence does not produce 2026.05.298")
    if queue_paths.get("selected_queue_milestone") != MILESTONE:
        reasons.append("selected queue milestone is not 2026.05.298")
    if queue_paths.get("selected_queue_first_task") != FIRST_V298_TASK_ID:
        reasons.append("selected queue first task is not exp3219-archive-v297-activate-v298")
    if queue_paths.get("milestone_doc") != VNEXT_DOC_REL_PATH.as_posix():
        reasons.append("selected queue milestone_doc is not the vNEXT document")
    if not queue_paths.get("selected_queue_task_count"):
        reasons.append("selected queue has no tasks")
    if not vnext_doc_exists:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


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


def _critical_path(queue_paths: Mapping[str, Any]) -> JsonDict:
    return {
        "top_unresolved_gap": TOP_UNRESOLVED_GAP,
        "first_milestone_task": str(queue_paths.get("selected_queue_first_task") or ""),
        "unblock_sequence": list(CRITICAL_PATH_IDS),
        "parallel_support_tracks": [
            "exp3223-distributional-ebm-exact-row-uncertainty-sidecar-v2",
            "exp3224-logitext-partial-smt-context-coverage-pilot-v1",
            "exp3229-fr11-nonforgetting-promotion-controller-v3",
            "exp3230-kan-cl-certificate-boundary-audit-v2",
        ],
        "terminal_aggregation_tasks": [
            "exp3231-cross-corpus-matrix-v32",
            "exp3232-capstone-v298",
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


def _inference_substrate_details() -> JsonDict:
    return {
        "kind": "archive_activation_aggregation_from_checked_in_artifacts",
        "source": "capstone_v297_matrix_v31_roadmap_queue_and_ops_logs",
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
            "complete: archive_v297_activate_v298_ready=true; "
            "prior_task_range=exp3205-exp3218; "
            f"prior_paper_ready={str(artifact.get('prior_paper_ready')).lower()}; "
            f"prior_publication_blocker_count={artifact.get('prior_publication_blocker_count')}; "
            f"prior_next_top_gap={artifact.get('prior_next_top_gap')}; "
            f"queue_path={_as_mapping(artifact.get('queue_paths')).get('selected_queue_path')}"
        )
    return (
        "blocked_activation_not_ready: "
        "prior_task_range=exp3205-exp3218; "
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
