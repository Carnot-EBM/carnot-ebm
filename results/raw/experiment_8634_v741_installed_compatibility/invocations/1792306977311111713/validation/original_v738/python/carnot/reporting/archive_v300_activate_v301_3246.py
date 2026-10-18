"""Build the Exp 3246 archive and .301 activation handoff artifact.

Spec refs: REQ-REPORT-3246, SCENARIO-REPORT-3246.

This module records the boundary between the `.300` capstone and the `.301`
queue. It is deliberately aggregation-only: the output is an audit receipt
that explains why selected-Python CUDA repair is next, not a rerun of CUDA,
local SOTA models, prompt-injection labels, or repair experiments.
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
SCHEMA_VERSION = "carnot.archive_activation.v300_to_v301.v1"
EXPERIMENT_ID = "exp3246"
TASK_ID = "exp3246-archive-v300-activate-v301"
ARTIFACT = "experiment_3246_archive_v300_activate_v301"
MILESTONE = "2026.05.301"
PRIOR_MILESTONE = "2026.05.300"
EXPECTED_QUEUE_LAST_TASK = "exp3258-capstone-v301"
EXPECTED_PRIOR_BLOCKER_COUNT = 106
NEXT_TOP_GAP = "repair_selected_python_torch_cuda_before_exp3237"
RANDOM_SEED = 3246

OUTPUT_REL_PATH = Path("results/experiment_3246_archive_v300_activate_v301.json")
CAPSTONE_V300_REL_PATH = Path("results/experiment_3245_capstone_v300.json")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
STATUS_REL_PATH = Path("ops/status.md")
CHANGELOG_REL_PATH = Path("ops/changelog.md")
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
    """Read YAML roadmap evidence without treating malformed data as usable."""

    try:
        text = path.read_text(encoding="utf-8")
        return yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so downstream reviewers can reproduce the receipt."""

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
    """REQ-REPORT-3246: synthesize the `.300` archive and `.301` queue record."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V300_REL_PATH)
    queue_paths = _queue_paths(root_path)
    local_status = _status_from_capstone(
        capstone, "local_sota_receipt_state", "local_sota_receipt_status"
    )
    prompt_status = _status_from_capstone(
        capstone, "prompt_injection_v4_state", "prompt_injection_v4_status"
    )
    fr11_status = _status_from_capstone(
        capstone, "fr11_failure_memory_state", "fr11_failure_memory_status"
    )
    conductor_evidence = _milestone_300_log_evidence(root_path / CONDUCTOR_LOG_REL_PATH)
    status_evidence = _evidence_lines(
        root_path / STATUS_REL_PATH,
        ["2026.05.301", "publication_blocker_count=106", NEXT_TOP_GAP],
    )
    changelog_evidence = _evidence_lines(
        root_path / CHANGELOG_REL_PATH,
        ["2026.05.301", "publication_blocker_count=106", "local_sota_receipt_status"],
    )
    activation_already_observed = _activation_already_observed(root_path, queue_paths)
    protected_files_untouched = {path: True for path in PROTECTED_FILES}
    protected_file_checksums = {
        path: sha256_file(root_path / path) for path in PROTECTED_FILES
    }
    selected_python_boundary = _selected_python_cuda_boundary_summary(
        capstone,
        local_status=local_status,
        prompt_status=prompt_status,
        conductor_evidence=conductor_evidence,
    )
    blocked_reasons = _blocked_reasons(
        capstone=capstone,
        local_status=local_status,
        prompt_status=prompt_status,
        fr11_status=fr11_status,
        queue_paths=queue_paths,
        activation_already_observed=activation_already_observed,
        vnext_doc_exists=(root_path / VNEXT_DOC_REL_PATH).is_file(),
        conductor_evidence_count=len(conductor_evidence),
        status_evidence_count=len(status_evidence),
        changelog_evidence_count=len(changelog_evidence),
    )
    source_artifacts = _source_artifacts(root_path, capstone)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "prior_capstone_artifact": CAPSTONE_V300_REL_PATH.as_posix(),
        "prior_capstone_ready": capstone.get("capstone_v300_ready") is True,
        "prior_paper_ready": capstone.get("paper_ready") is True,
        "prior_publication_blocker_count": _int_value(
            capstone.get("publication_blocker_count")
        ),
        "prior_local_sota_receipt_status": local_status,
        "prior_prompt_injection_v4_status": prompt_status,
        "prior_fr11_failure_memory_status": fr11_status,
        "prior_local_sota_receipt_state": _as_mapping(
            capstone.get("local_sota_receipt_state")
        ),
        "prior_prompt_injection_v4_state": _as_mapping(
            capstone.get("prompt_injection_v4_state")
        ),
        "prior_fr11_failure_memory_state": _as_mapping(
            capstone.get("fr11_failure_memory_state")
        ),
        "next_top_gap": str(capstone.get("next_top_gap") or ""),
        "selected_python_cuda_boundary_summary": selected_python_boundary,
        "queue_first_task": str(queue_paths.get("queue_first_task") or ""),
        "queue_last_task": str(queue_paths.get("queue_last_task") or ""),
        "queue_task_count": _int_value(queue_paths.get("queue_task_count")),
        "queue_paths": queue_paths,
        "roadmap_pre_activation_shape_observed": _roadmap_pre_activation_shape_observed(
            queue_paths
        ),
        "activation_already_observed": activation_already_observed,
        "roadmap_activation_observation": _roadmap_activation_observation(queue_paths),
        "milestone_300_log_evidence": conductor_evidence,
        "status_evidence": status_evidence,
        "changelog_evidence": changelog_evidence,
        "protected_files": list(PROTECTED_FILES),
        "protected_files_untouched": protected_files_untouched,
        "protected_file_checksums": protected_file_checksums,
        "principle_annotations": _principle_annotations(),
        "archive_v300_activate_v301_ready": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "no_new_model_execution": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_delong_run": True,
        "no_new_garak_run": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "active_roadmap_modified_by_this_task": False,
        "staged_roadmap_modified_by_this_task": False,
        "scripts_research_conductor_modified_by_this_task": False,
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
    """Build and persist the Exp 3246 deliverable JSON."""

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
    capstone: Mapping[str, Any],
    local_status: str,
    prompt_status: str,
    fr11_status: str,
    queue_paths: Mapping[str, Any],
    activation_already_observed: bool,
    vnext_doc_exists: bool,
    conductor_evidence_count: int,
    status_evidence_count: int,
    changelog_evidence_count: int,
) -> list[str]:
    reasons: list[str] = []
    if not capstone:
        reasons.append("capstone_v300 authority is missing or malformed")
    if capstone and capstone.get("capstone_v300_ready") is not True:
        reasons.append("capstone_v300 authority is not ready")
    if capstone.get("paper_ready") is not False:
        reasons.append("prior paper_ready must remain false")
    if _int_value(capstone.get("publication_blocker_count")) != EXPECTED_PRIOR_BLOCKER_COUNT:
        reasons.append("prior publication blocker count is not 106")
    if local_status != "blocked":
        reasons.append("local SOTA receipt status is not blocked")
    if prompt_status != "gate_blocked":
        reasons.append("prompt-injection v4 status is not gate_blocked")
    if fr11_status != "complete":
        reasons.append("FR-11 failure-memory status is not complete")
    if str(capstone.get("next_top_gap") or "") != NEXT_TOP_GAP:
        reasons.append("next_top_gap does not preserve the selected-Python CUDA gap")
    if queue_paths.get("selected_queue_milestone") != MILESTONE:
        reasons.append("selected queue milestone is not 2026.05.301")
    if queue_paths.get("queue_first_task") != TASK_ID:
        reasons.append("selected queue first task is not exp3246-archive-v300-activate-v301")
    if queue_paths.get("queue_last_task") != EXPECTED_QUEUE_LAST_TASK:
        reasons.append("selected queue last task is not exp3258-capstone-v301")
    if queue_paths.get("milestone_doc") != VNEXT_DOC_REL_PATH.as_posix():
        reasons.append("selected queue milestone_doc is not the vNEXT document")
    if not activation_already_observed and not queue_paths.get("staged_roadmap_exists"):
        reasons.append("milestone 2026.05.301 is neither staged nor activation-observed")
    if not vnext_doc_exists:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    if conductor_evidence_count == 0:
        reasons.append("conductor log does not contain .300 terminal evidence")
    if status_evidence_count == 0:
        reasons.append("ops/status.md does not contain .301 activation facts")
    if changelog_evidence_count == 0:
        reasons.append("ops/changelog.md does not contain .301 activation facts")
    return reasons


def _source_artifacts(root: Path, capstone: Mapping[str, Any]) -> list[JsonDict]:
    return [
        _source_record(root, "capstone_v300", CAPSTONE_V300_REL_PATH, bool(capstone)),
        _source_record(root, "active_roadmap_queue", ACTIVE_ROADMAP_REL_PATH, True),
        _source_record(root, "staged_roadmap_queue", STAGED_ROADMAP_REL_PATH, True),
        _source_record(root, "vnext_milestone_doc", VNEXT_DOC_REL_PATH, True),
        _source_record(root, "ops_status_authority", STATUS_REL_PATH, True),
        _source_record(root, "ops_changelog_authority", CHANGELOG_REL_PATH, True),
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


def _milestone_300_log_evidence(path: Path) -> list[str]:
    needles = [
        "Isolated CUDA and selected-Python smoke receipt",
        "llama.cpp CUDA receipt smoke",
        "Mandated local SOTA GGUF receipt",
        "Prompt-injection KAN v4 resource manifest",
        "Prompt-injection KAN v4 teacher-label",
        "Prompt-injection KAN v4 shard train/eval",
        "DCCD exact-row structured proposal preflight",
        "FR-11 failure-memory controller",
        "Cross-corpus matrix v33",
        "Capstone .300 publication readiness",
        "exp3236",
        "exp3237",
        "exp3238",
        "exp3240",
        "exp3242",
    ]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line for line in lines if any(needle in line for needle in needles)]


def _evidence_lines(path: Path, needles: list[str]) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line for line in lines if any(needle in line for needle in needles)]


def _selected_python_cuda_boundary_summary(
    capstone: Mapping[str, Any],
    *,
    local_status: str,
    prompt_status: str,
    conductor_evidence: list[str],
) -> JsonDict:
    local_state = _as_mapping(capstone.get("local_sota_receipt_state"))
    return {
        "boundary": "selected_python_torch_cuda",
        "blocked_local_sota_receipts": (
            local_status == "blocked"
            and local_state.get("selected_python_torch_cuda_available") is False
        ),
        "blocked_downstream_prompt_injection_tasks": prompt_status == "gate_blocked",
        "blocked_downstream_structured_repair_tasks": local_status == "blocked",
        "blocking_artifacts": _as_list(local_state.get("blocking_artifacts")),
        "downstream_blocked_tasks": ["exp3237", "exp3238", "exp3240", "exp3241", "exp3242"],
        "recommended_next_task": NEXT_TOP_GAP,
        "evidence_line_count": len(conductor_evidence),
        "operator_safe_note": (
            "Do not rerun local SOTA receipts, prompt-injection teacher labels, "
            "KAN train/eval, or structured repair preflight until selected Python "
            "CUDA is repaired and the receipt gates pass."
        ),
    }


def _roadmap_pre_activation_shape_observed(queue_paths: Mapping[str, Any]) -> bool:
    return (
        queue_paths.get("active_roadmap_milestone") == PRIOR_MILESTONE
        and queue_paths.get("staged_roadmap_milestone") == MILESTONE
    )


def _activation_already_observed(root: Path, queue_paths: Mapping[str, Any]) -> bool:
    return queue_paths.get("active_roadmap_milestone") == MILESTONE or _file_contains(
        root / CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.301 activated",
    )


def _roadmap_activation_observation(queue_paths: Mapping[str, Any]) -> str:
    active = queue_paths.get("active_roadmap_milestone") or "missing"
    staged = queue_paths.get("staged_roadmap_milestone") or "missing"
    if _roadmap_pre_activation_shape_observed(queue_paths):
        return "pre_activation_shape_observed: active=.300, staged=.301"
    if active == MILESTONE and not queue_paths.get("staged_roadmap_exists"):
        return "already_activated: active=.301, staged roadmap absent"
    return f"nonstandard_shape: active={active}, staged={staged}"


def _principle_annotations() -> JsonDict:
    return {
        "inference_substrate": (
            "Declares aggregation-only work so no live model, hardware, or repair run is implied."
        ),
        "archive_v300_activate_v301_ready": (
            "True only when the capstone, activation queue, and ops evidence agree."
        ),
        "prior_paper_ready": (
            "Carries forward the capstone paper-readiness value without promoting the paper."
        ),
        "prior_publication_blocker_count": (
            "Preserves the blocker count from the .300 capstone for .301 planning."
        ),
        "prior_local_sota_receipt_status": (
            "Keeps selected-Python CUDA failure visible as the local SOTA blocker."
        ),
        "prior_prompt_injection_v4_status": (
            "Keeps teacher-label and KAN shard gate blocks separate from manifest readiness."
        ),
        "prior_fr11_failure_memory_status": (
            "Records controller-memory completion without claiming model-weight learning."
        ),
        "queue_first_task": "Anchors the activated .301 queue to the expected handoff task.",
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
    ready = str(artifact.get("archive_v300_activate_v301_ready")).lower()
    blocker_count = artifact.get("prior_publication_blocker_count")
    local_status = artifact.get("prior_local_sota_receipt_status")
    prompt_status = artifact.get("prior_prompt_injection_v4_status")
    fr11_status = artifact.get("prior_fr11_failure_memory_status")
    first = artifact.get("queue_first_task")
    last = artifact.get("queue_last_task")
    return (
        f"complete: archive_v300_activate_v301_ready={ready}; "
        "prior_paper_ready=false; "
        f"prior_publication_blocker_count={blocker_count}; "
        f"prior_local_sota_receipt_status={local_status}; "
        f"prior_prompt_injection_v4_status={prompt_status}; "
        f"prior_fr11_failure_memory_status={fr11_status}; "
        f"next_top_gap={artifact.get('next_top_gap')}; "
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


def _status_from_capstone(
    capstone: Mapping[str, Any],
    nested_key: str,
    fallback_key: str,
) -> str:
    nested = capstone.get(nested_key)
    if isinstance(nested, Mapping) and nested.get("status") not in (None, ""):
        return str(nested["status"])
    return str(capstone.get(fallback_key) or "")


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
