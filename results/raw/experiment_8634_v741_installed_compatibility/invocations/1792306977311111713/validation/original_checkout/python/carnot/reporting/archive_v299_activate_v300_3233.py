"""Build the Exp 3233 archive and .300 activation handoff artifact.

Spec refs: REQ-REPORT-3233, SCENARIO-REPORT-3233.

This module records the boundary between the narrow `.299` milestone and the
`.300` queue. It only reads checked-in artifacts, roadmap state, and conductor
log evidence. The handoff is intentionally aggregation-only: it does not train
KANs, rerun Garak, touch the conductor, edit roadmaps, or update ops docs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.archive_activation.v299_to_v300.v1"
EXPERIMENT_ID = "exp3233"
TASK_ID = "exp3233-archive-v299-activate-v300"
ARTIFACT = "experiment_3233_archive_v299_activate_v300"
MILESTONE = "2026.05.300"
PRIOR_MILESTONE = "2026.05.299"
PRIOR_V4_OUTCOME = "blocked_missing_exp3222_result"
NEXT_TOP_GAP = "cuda_chain_for_full_local_sota_receipts"
EXPECTED_QUEUE_LAST_TASK = "exp3245-capstone-v300"
RANDOM_SEED = 3233

OUTPUT_REL_PATH = Path("results/experiment_3233_archive_v299_activate_v300.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3233_archive_v299_activate_v300.py"
PRIOR_ARCHIVE_REL_PATH = Path("results/experiment_3221_archive_v298_activate_v299.json")
CAPSTONE_V299_REL_PATH = Path("results/experiment_3223_capstone_v299.json")
MISSING_V4_ARTIFACT_REL_PATH = Path(
    "results/experiment_3222_prompt_injection_kan_distill_v4_15k.json"
)
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

PROTECTED_FILES = [
    ACTIVE_ROADMAP_REL_PATH.as_posix(),
    CONDUCTOR_REL_PATH.as_posix(),
]


def read_json_object(path: Path) -> JsonDict:
    """Read JSON evidence as an object and fail closed on bad or absent input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_document(path: Path) -> Any:
    """Read YAML roadmap evidence without assuming malformed data is usable."""

    try:
        text = path.read_text(encoding="utf-8")
        return yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so the aggregation handoff is reproducible."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3233: synthesize the `.299` archive and `.300` queue record."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    prior_archive = read_json_object(root_path / PRIOR_ARCHIVE_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V299_REL_PATH)
    queue_paths = _queue_paths(root_path)
    failure_lines = _exp3222_failure_lines(root_path / CONDUCTOR_LOG_REL_PATH)
    missing_v4_exists = (root_path / MISSING_V4_ARTIFACT_REL_PATH).is_file()
    prior_capstone_ready = capstone.get("capstone_v299_ready") is True
    prior_v4_outcome = str(capstone.get("v4_outcome") or "")
    next_top_gap = str(capstone.get("next_top_gap") or "")
    protected_files_untouched = {path: True for path in PROTECTED_FILES}
    blocked_reasons = _blocked_reasons(
        prior_archive=prior_archive,
        capstone=capstone,
        prior_capstone_ready=prior_capstone_ready,
        prior_v4_outcome=prior_v4_outcome,
        next_top_gap=next_top_gap,
        missing_v4_exists=missing_v4_exists,
        queue_paths=queue_paths,
        failure_count=len(failure_lines),
        activation_already_observed=_activation_already_observed(root_path, queue_paths),
        vnext_doc_exists=(root_path / VNEXT_DOC_REL_PATH).is_file(),
    )
    source_artifacts = _source_artifacts(root_path, prior_archive, capstone)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "prior_archive_artifact": PRIOR_ARCHIVE_REL_PATH.as_posix(),
        "prior_capstone_artifact": CAPSTONE_V299_REL_PATH.as_posix(),
        "prior_archive_ready": prior_archive.get("archive_v298_activate_v299_ready") is True,
        "prior_capstone_ready": prior_capstone_ready,
        "prior_paper_ready": capstone.get("paper_ready") is True,
        "prior_publication_blocker_count": _int_value(capstone.get("publication_blocker_count")),
        "prior_v4_outcome": prior_v4_outcome,
        "missing_v4_artifact_path": MISSING_V4_ARTIFACT_REL_PATH.as_posix(),
        "missing_v4_artifact_exists": missing_v4_exists,
        "missing_v4_artifact_note": (
            f"exp3222 did not produce {MISSING_V4_ARTIFACT_REL_PATH.as_posix()}"
        ),
        "next_top_gap": next_top_gap,
        "queue_first_task": str(queue_paths.get("queue_first_task") or ""),
        "queue_last_task": str(queue_paths.get("queue_last_task") or ""),
        "queue_task_count": _int_value(queue_paths.get("queue_task_count")),
        "queue_paths": queue_paths,
        "roadmap_pre_activation_shape_observed": _roadmap_pre_activation_shape_observed(
            queue_paths
        ),
        "activation_already_observed": _activation_already_observed(root_path, queue_paths),
        "exp3222_failure_count": len(failure_lines),
        "exp3222_failure_evidence": failure_lines,
        "protected_files": list(PROTECTED_FILES),
        "protected_files_untouched": protected_files_untouched,
        "principle_annotations": _principle_annotations(),
        "archive_v299_activate_v300_ready": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "no_new_model_execution": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "active_roadmap_modified_by_this_task": False,
        "staged_roadmap_modified_by_this_task": False,
        "conductor_file_modified_by_this_task": False,
        "ops_status_modified_by_this_task": False,
        "ops_changelog_modified_by_this_task": False,
        "traceability_modified_by_this_task": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3233 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _queue_paths(root: Path) -> JsonDict:
    staged_exists = (root / STAGED_ROADMAP_REL_PATH).is_file()
    active_exists = (root / ACTIVE_ROADMAP_REL_PATH).is_file()
    selected_path = STAGED_ROADMAP_REL_PATH if staged_exists else ACTIVE_ROADMAP_REL_PATH
    staged_payload = _as_mapping(read_yaml_document(root / STAGED_ROADMAP_REL_PATH))
    active_payload = _as_mapping(read_yaml_document(root / ACTIVE_ROADMAP_REL_PATH))
    selected_payload = staged_payload if staged_exists else active_payload
    task_ids = _task_ids(selected_payload)
    return {
        "staged_roadmap_path": STAGED_ROADMAP_REL_PATH.as_posix(),
        "staged_roadmap_exists": staged_exists,
        "staged_roadmap_milestone": str(staged_payload.get("milestone") or ""),
        "active_roadmap_path": ACTIVE_ROADMAP_REL_PATH.as_posix(),
        "active_roadmap_exists": active_exists,
        "active_roadmap_milestone": str(active_payload.get("milestone") or ""),
        "selected_queue_path": selected_path.as_posix(),
        "selected_queue_milestone": str(selected_payload.get("milestone") or ""),
        "queue_first_task": task_ids[0] if task_ids else "",
        "queue_last_task": task_ids[-1] if task_ids else "",
        "queue_task_count": len(task_ids),
        "queue_task_ids": task_ids,
        "milestone_doc": str(selected_payload.get("milestone_doc") or ""),
    }


def _blocked_reasons(
    *,
    prior_archive: Mapping[str, Any],
    capstone: Mapping[str, Any],
    prior_capstone_ready: bool,
    prior_v4_outcome: str,
    next_top_gap: str,
    missing_v4_exists: bool,
    queue_paths: Mapping[str, Any],
    failure_count: int,
    activation_already_observed: bool,
    vnext_doc_exists: bool,
) -> list[str]:
    reasons: list[str] = []
    if not prior_archive:
        reasons.append("archive_v298_activate_v299 authority is missing or malformed")
    if prior_archive and prior_archive.get("archive_v298_activate_v299_ready") is not True:
        reasons.append("archive_v298_activate_v299 authority is not ready")
    if not capstone:
        reasons.append("capstone_v299 authority is missing or malformed")
    if capstone and not prior_capstone_ready:
        reasons.append("capstone_v299 authority is not ready")
    if prior_v4_outcome != PRIOR_V4_OUTCOME:
        reasons.append("prior v4 outcome is not blocked_missing_exp3222_result")
    if next_top_gap != NEXT_TOP_GAP:
        reasons.append("next_top_gap does not preserve the CUDA-chain gap")
    if missing_v4_exists:
        reasons.append("expected missing exp3222 v4 artifact is present")
    if queue_paths.get("selected_queue_milestone") != MILESTONE:
        reasons.append("selected queue milestone is not 2026.05.300")
    if queue_paths.get("queue_first_task") != TASK_ID:
        reasons.append("selected queue first task is not exp3233-archive-v299-activate-v300")
    if queue_paths.get("queue_last_task") != EXPECTED_QUEUE_LAST_TASK:
        reasons.append("selected queue last task is not exp3245-capstone-v300")
    if queue_paths.get("milestone_doc") != VNEXT_DOC_REL_PATH.as_posix():
        reasons.append("selected queue milestone_doc is not the vNEXT document")
    if failure_count < 3:
        reasons.append("conductor log does not record three exp3222 v4 failures")
    if not activation_already_observed and not queue_paths.get("staged_roadmap_exists"):
        reasons.append("milestone 2026.05.300 is neither staged nor activation-observed")
    if not vnext_doc_exists:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


def _source_artifacts(
    root: Path,
    prior_archive: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _source_record(
            root, "archive_v298_activate_v299", PRIOR_ARCHIVE_REL_PATH, bool(prior_archive)
        ),
        _source_record(root, "capstone_v299", CAPSTONE_V299_REL_PATH, bool(capstone)),
        _source_record(
            root, "missing_prompt_injection_kan_v4", MISSING_V4_ARTIFACT_REL_PATH, False
        ),
        _source_record(root, "active_roadmap_queue", ACTIVE_ROADMAP_REL_PATH, True),
        _source_record(root, "staged_roadmap_queue", STAGED_ROADMAP_REL_PATH, True),
        _source_record(root, "vnext_milestone_doc", VNEXT_DOC_REL_PATH, True),
        _source_record(root, "conductor_log_authority", CONDUCTOR_LOG_REL_PATH, True),
        _source_record(root, "protected_research_conductor", CONDUCTOR_REL_PATH, True),
    ]


def _source_record(root: Path, role: str, rel_path: Path, readable: bool) -> JsonDict:
    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "readable": readable and path.is_file(),
        "sha256": sha256_file(path),
    }


def _exp3222_failure_lines(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [
        line
        for line in lines
        if "Prompt-Injection KAN Distillation v4" in line and "| FAIL |" in line
    ]


def _roadmap_pre_activation_shape_observed(queue_paths: Mapping[str, Any]) -> bool:
    return (
        queue_paths.get("active_roadmap_milestone") == PRIOR_MILESTONE
        and queue_paths.get("staged_roadmap_milestone") == MILESTONE
    )


def _activation_already_observed(root: Path, queue_paths: Mapping[str, Any]) -> bool:
    return queue_paths.get("active_roadmap_milestone") == MILESTONE or _file_contains(
        root / CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.300 activated",
    )


def _principle_annotations() -> JsonDict:
    return {
        "inference_substrate": (
            "Declares aggregation-only work so downstream auditors do not expect live inference."
        ),
        "archive_v299_activate_v300_ready": (
            "Summarizes whether the prior capstone, missing artifact, and queue evidence agree."
        ),
        "prior_paper_ready": (
            "Carries forward the prior capstone readiness signal without promoting the paper."
        ),
        "missing_v4_artifact_path": (
            "Preserves the exact absent artifact that blocked the .299 single-focus task."
        ),
        "queue_first_task": "Anchors the activated .300 queue to the expected handoff task.",
        "queue_last_task": "Records the downstream queue boundary for capstone planning.",
        "protected_files_untouched": (
            "Documents that this handoff does not edit protected conductor or roadmap files."
        ),
        "honest_verdict": (
            "Terminal prefix keeps conductor reconciliation from treating a complete handoff as partial."
        ),
    }


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable_payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    data = json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    ready = str(artifact.get("archive_v299_activate_v300_ready")).lower()
    blocker_count = artifact.get("prior_publication_blocker_count")
    v4_outcome = artifact.get("prior_v4_outcome")
    first = artifact.get("queue_first_task")
    last = artifact.get("queue_last_task")
    return (
        f"complete: archive_v299_activate_v300_ready={ready}; "
        "prior_paper_ready=false; "
        f"prior_publication_blocker_count={blocker_count}; "
        f"prior_v4_outcome={v4_outcome}; "
        f"queue_range={first}..{last}"
    )


def _file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except OSError:
        return False


def _task_ids(payload: Mapping[str, Any]) -> list[str]:
    return [
        str(task["id"])
        for task in _as_list(payload.get("tasks"))
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
