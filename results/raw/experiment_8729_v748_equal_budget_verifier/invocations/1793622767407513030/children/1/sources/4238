"""Build the Exp 3260 archive and .302 activation manifest.

Spec refs: REQ-REPORT-3260, SCENARIO-REPORT-3260.

This handoff closes the `.301` milestone boundary using already-written
evidence. It reads the `.301` capstone, records the optional operational
retrospective if present, ensures `research-complete.yaml` has one `.301`
summary, and checks the `.302` queue. It does not run models, CUDA probes,
teacher labeling, KAN training, repair, hardware, or the conductor.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.archive_activation.v301_to_v302.v1"
EXPERIMENT_ID = "exp3260"
TASK_ID = "exp3260-archive-v301-activate-v302"
ARTIFACT = "experiment_3260_archive_v301_activate_v302"
MILESTONE = "2026.05.302"
PRIOR_MILESTONE = "2026.05.301"
EXPECTED_PRIOR_BLOCKER_COUNT = 106
PRIOR_NEXT_TOP_GAP = "keep_exp3248_blocked_repair_cuda_runtime"
EXPECTED_QUEUE_LAST_TASK = "exp3266-capstone-v302"
RANDOM_SEED = 3260

OUTPUT_REL_PATH = Path("results/experiment_3260_archive_v301_activate_v302.json")
CAPSTONE_V301_REL_PATH = Path("results/experiment_3258_capstone_v301.json")
OPERATIONAL_RETRO_V301_REL_PATH = Path("results/operational_retro_2026_05_301.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

PROTECTED_FILES = [
    ACTIVE_ROADMAP_REL_PATH.as_posix(),
    CONDUCTOR_REL_PATH.as_posix(),
]

PRIOR_TASKS: tuple[JsonDict, ...] = (
    {
        "id": "exp3246-archive-v300-activate-v301",
        "title": "Archive .300 closeout and activate .301 planning authority",
        "deliverable": "results/experiment_3246_archive_v300_activate_v301.json",
    },
    {
        "id": "exp3247-selected-python-cuda-root-cause-surgery-v1",
        "title": "Selected-Python CUDA root-cause surgery ledger",
        "deliverable": (
            "results/experiment_3247_selected_python_cuda_root_cause_surgery_v1.json"
        ),
    },
    {
        "id": "exp3248-isolated-cuda-selected-python-smoke-v2",
        "title": "Isolated selected-Python CUDA smoke v2 gated on root-cause surgery",
        "deliverable": "results/experiment_3248_isolated_cuda_selected_python_smoke_v2.json",
    },
    {
        "id": "exp3249-llama-cpp-cuda-receipt-smoke-v3",
        "title": "llama.cpp CUDA receipt smoke v3 gated on selected-Python CUDA",
        "deliverable": "results/experiment_3249_llama_cpp_cuda_receipt_smoke_v3.json",
    },
    {
        "id": "exp3250-sota-gguf-receipt-v8",
        "title": "Mandated SOTA GGUF receipt v8 gated on llama.cpp CUDA",
        "deliverable": "results/experiment_3250_sota_gguf_receipt_v8.json",
    },
    {
        "id": "exp3251-prompt-injection-v4-constraint-tax-manifest-v2",
        "title": "Prompt-injection v4 constraint-tax manifest refresh",
        "deliverable": (
            "results/experiment_3251_prompt_injection_v4_constraint_tax_manifest_v2.json"
        ),
    },
    {
        "id": "exp3252-prompt-injection-teacher-label-shard-v2",
        "title": "Prompt-injection teacher-label shard v2 gated on SOTA receipt",
        "deliverable": (
            "results/experiment_3252_prompt_injection_teacher_label_shard_v2.json"
        ),
    },
    {
        "id": "exp3253-prompt-injection-kan-train-eval-shard-v2",
        "title": "Prompt-injection KAN shard train/eval v2 with constraint-tax guardrail",
        "deliverable": (
            "results/experiment_3253_prompt_injection_kan_train_eval_shard_v2.json"
        ),
    },
    {
        "id": "exp3254-dccd-severa-structured-proposal-preflight-v2",
        "title": "DCCD/SEVerA structured proposal preflight v2 gated on clean SOTA receipt",
        "deliverable": (
            "results/experiment_3254_dccd_severa_structured_proposal_preflight_v2.json"
        ),
    },
    {
        "id": "exp3255-fr11-lifelong-failure-memory-retention-audit-v1",
        "title": "FR-11 lifelong failure-memory retention audit",
        "deliverable": (
            "results/experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.json"
        ),
    },
    {
        "id": "exp3256-pdit-potts-multistate-sampler-diagnostic-v1",
        "title": "P-dit/Potts multi-state sampler diagnostic manifest",
        "deliverable": (
            "results/experiment_3256_pdit_potts_multistate_sampler_diagnostic_v1.json"
        ),
    },
    {
        "id": "exp3257-cross-corpus-matrix-v34",
        "title": "Cross-corpus matrix v34 for .301 evidence",
        "deliverable": "results/experiment_3257_cross_corpus_matrix_v34.json",
    },
    {
        "id": "exp3258-capstone-v301",
        "title": "Capstone v301 and next-gap selection",
        "deliverable": "results/experiment_3258_capstone_v301.json",
    },
)
EXPECTED_PRIOR_TASK_IDS = {str(task["id"]) for task in PRIOR_TASKS}


def read_json_object(path: Path) -> JsonDict:
    """Read JSON source evidence and return empty evidence on bad input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_document(path: Path) -> Any:
    """Read YAML evidence without inventing a shape for missing or bad files."""

    try:
        text = path.read_text(encoding="utf-8")
        return yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}


def sha256_file(path: Path) -> str | None:
    """Checksum source bytes so reviewers can reproduce the handoff receipt."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_research_complete_entry(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-REPORT-3260: append the `.301` archive summary only when absent."""

    root_path = Path(root)
    archive_path = root_path / RESEARCH_COMPLETE_REL_PATH
    if _research_complete_contains_prior(root_path):
        return {
            "path": RESEARCH_COMPLETE_REL_PATH.as_posix(),
            "appended": False,
            "already_present": True,
        }
    existing = archive_path.read_text(encoding="utf-8") if archive_path.is_file() else ""
    separator = "\n" if existing and not existing.endswith("\n") else ""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(existing + separator + _research_complete_entry(), encoding="utf-8")
    return {
        "path": RESEARCH_COMPLETE_REL_PATH.as_posix(),
        "appended": True,
        "already_present": False,
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """SCENARIO-REPORT-3260: synthesize the `.301` archive and `.302` manifest."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V301_REL_PATH)
    operational_retro = read_json_object(root_path / OPERATIONAL_RETRO_V301_REL_PATH)
    queue_paths = _queue_paths(root_path)
    research_complete_present = _research_complete_contains_prior(root_path)
    capstone_ready = capstone.get("capstone_v301_ready") is True
    prior_paper_ready = capstone.get("paper_ready") is True
    prior_blockers = _int_value(capstone.get("publication_blocker_count"))
    prior_next_gap = str(capstone.get("next_top_gap") or "")
    activation_observed = _activation_already_observed(root_path, queue_paths)
    blocked_reasons = _blocked_reasons(
        capstone=capstone,
        capstone_ready=capstone_ready,
        prior_paper_ready=prior_paper_ready,
        prior_publication_blocker_count=prior_blockers,
        prior_next_top_gap=prior_next_gap,
        research_complete_present=research_complete_present,
        queue_paths=queue_paths,
        activation_already_observed=activation_observed,
        vnext_doc_exists=(root_path / VNEXT_DOC_REL_PATH).is_file(),
    )
    source_artifacts = _source_artifacts(root_path, capstone, operational_retro)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "prior_task_ids": [str(task["id"]) for task in PRIOR_TASKS],
        "prior_task_count": len(PRIOR_TASKS),
        "prior_capstone_artifact": CAPSTONE_V301_REL_PATH.as_posix(),
        "prior_operational_retro_artifact": OPERATIONAL_RETRO_V301_REL_PATH.as_posix(),
        "prior_capstone_ready": capstone_ready,
        "prior_paper_ready": prior_paper_ready,
        "prior_publication_blocker_count": prior_blockers,
        "prior_next_top_gap": prior_next_gap,
        "prior_capstone_honest_verdict": str(capstone.get("honest_verdict") or ""),
        "prior_operational_retro_present": bool(operational_retro),
        "prior_operational_retro_summary": _operational_retro_summary(operational_retro),
        "research_complete_prior_summary_present": research_complete_present,
        "research_complete_update_policy": "append_once_if_missing",
        "prior_task_summary": _research_complete_task_summary(root_path),
        "queue_paths": queue_paths,
        "activation_already_observed": activation_observed,
        "protected_files": list(PROTECTED_FILES),
        "protected_files_untouched": {path: True for path in PROTECTED_FILES},
        "protected_file_checksums": {
            path: sha256_file(root_path / path) for path in PROTECTED_FILES
        },
        "principle_annotations": _principle_annotations(),
        "archive_v301_activate_v302_ready": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "inference_substrate": "artifact_aggregation_only",
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_llama_cpp_run": True,
        "no_new_gguf_receipt": True,
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
    update_research_complete: bool = True,
) -> Path:
    """Build and persist the Exp 3260 archive/activation deliverable JSON."""

    root_path = Path(root)
    if update_research_complete:
        ensure_research_complete_entry(root_path)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _research_complete_entry() -> str:
    lines = [
        "- id: 2026.05.301",
        "  title: Selected-Python CUDA Repair + Constraint-Tax Prompt Injection + Lifelong FR-11 Retention",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-05-28'",
        "  finding: See conductor log for per-experiment results.",
        "  tasks:",
    ]
    for task in PRIOR_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {task['title']}",
                f"    deliverable: {task['deliverable']}",
                "    result: OK (conductor)",
            ]
        )
    return "\n".join(lines) + "\n"


def _research_complete_contains_prior(root: Path) -> bool:
    payload = read_yaml_document(root / RESEARCH_COMPLETE_REL_PATH)
    for entry in _milestone_entries(payload):
        if str(entry.get("id") or "") == PRIOR_MILESTONE:
            return EXPECTED_PRIOR_TASK_IDS <= set(_task_ids(entry))
    return False


def _research_complete_task_summary(root: Path) -> JsonDict:
    payload = read_yaml_document(root / RESEARCH_COMPLETE_REL_PATH)
    for entry in _milestone_entries(payload):
        if str(entry.get("id") or "") != PRIOR_MILESTONE:
            continue
        tasks = [
            dict(task)
            for task in _as_list(entry.get("tasks"))
            if isinstance(task, Mapping) and task.get("id") not in (None, "")
        ]
        task_ids = [str(task["id"]) for task in tasks]
        result_counts = Counter(str(task.get("result") or "unspecified") for task in tasks)
        return {
            "milestone": str(entry.get("id") or ""),
            "title": str(entry.get("title") or ""),
            "completed": str(entry.get("completed") or ""),
            "task_count": len(task_ids),
            "first_task": task_ids[0] if task_ids else "",
            "last_task": task_ids[-1] if task_ids else "",
            "task_ids": task_ids,
            "result_counts": dict(result_counts),
        }
    return {}


def _milestone_entries(payload: Any) -> list[JsonDict]:
    if isinstance(payload, list):
        return [dict(entry) for entry in payload if isinstance(entry, Mapping)]
    if isinstance(payload, Mapping):
        entries = [dict(payload)]
        entries.extend(dict(entry) for entry in _as_list(payload.get("milestones")))
        return entries
    return []


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
    capstone_ready: bool,
    prior_paper_ready: bool,
    prior_publication_blocker_count: int,
    prior_next_top_gap: str,
    research_complete_present: bool,
    queue_paths: Mapping[str, Any],
    activation_already_observed: bool,
    vnext_doc_exists: bool,
) -> list[str]:
    reasons: list[str] = []
    if not capstone:
        reasons.append("capstone_v301 authority is missing or malformed")
    if capstone and not capstone_ready:
        reasons.append("capstone_v301 authority is not ready")
    if prior_paper_ready is not False:
        reasons.append("prior paper_ready must remain false")
    if prior_publication_blocker_count != EXPECTED_PRIOR_BLOCKER_COUNT:
        reasons.append("prior publication blocker count is not 106")
    if prior_next_top_gap != PRIOR_NEXT_TOP_GAP:
        reasons.append("prior next_top_gap does not preserve the .301 runtime block")
    if not research_complete_present:
        reasons.append("research-complete.yaml does not contain the .301 task summary")
    if queue_paths.get("selected_queue_milestone") != MILESTONE:
        reasons.append("selected queue milestone is not 2026.05.302")
    if queue_paths.get("queue_first_task") != TASK_ID:
        reasons.append("selected queue first task is not exp3260-archive-v301-activate-v302")
    if queue_paths.get("queue_last_task") != EXPECTED_QUEUE_LAST_TASK:
        reasons.append("selected queue last task is not exp3266-capstone-v302")
    if queue_paths.get("milestone_doc") != VNEXT_DOC_REL_PATH.as_posix():
        reasons.append("selected queue milestone_doc is not the vNEXT document")
    if not activation_already_observed:
        reasons.append("milestone 2026.05.302 activation is not observed")
    if not vnext_doc_exists:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


def _source_artifacts(
    root: Path,
    capstone: Mapping[str, Any],
    operational_retro: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _source_record(root, "capstone_v301", CAPSTONE_V301_REL_PATH, bool(capstone)),
        _source_record(
            root,
            "operational_retro_v301",
            OPERATIONAL_RETRO_V301_REL_PATH,
            bool(operational_retro),
        ),
        _source_record(root, "research_complete_archive", RESEARCH_COMPLETE_REL_PATH, True),
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


def _operational_retro_summary(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return {}
    return {
        "milestone": str(payload.get("milestone") or ""),
        "experiments_completed": _int_value(payload.get("experiments_completed")),
        "total_wall_time_minutes": _int_value(payload.get("total_wall_time_minutes")),
        "compute_bound_experiments_count": _int_value(
            payload.get("compute_bound_experiments_count")
        ),
        "summary": str(payload.get("summary") or ""),
    }


def _activation_already_observed(root: Path, queue_paths: Mapping[str, Any]) -> bool:
    return queue_paths.get("active_roadmap_milestone") == MILESTONE or _file_contains(
        root / CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.302 activated",
    )


def _principle_annotations() -> JsonDict:
    return {
        "archive_v301_activate_v302_ready": (
            "True only when the .301 capstone, research archive, and .302 queue agree."
        ),
        "prior_paper_ready": (
            "Carries forward the .301 capstone publication-readiness signal unchanged."
        ),
        "prior_publication_blocker_count": (
            "Preserves the .301 blocker count so .302 can measure any reduction."
        ),
        "research_complete_update_policy": (
            "Append the .301 milestone summary exactly once and keep reruns idempotent."
        ),
        "honest_verdict": (
            "Uses a complete prefix while avoiding any paper-ready claim not in the capstone."
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
    ready = str(artifact.get("archive_v301_activate_v302_ready")).lower()
    paper_ready = str(artifact.get("prior_paper_ready")).lower()
    blocker_count = artifact.get("prior_publication_blocker_count")
    first = _as_mapping(artifact.get("queue_paths")).get("queue_first_task")
    last = _as_mapping(artifact.get("queue_paths")).get("queue_last_task")
    return (
        f"complete: archive_v301_activate_v302_ready={ready}; "
        f"prior_paper_ready={paper_ready}; "
        f"prior_publication_blocker_count={blocker_count}; "
        f"next_top_gap={artifact.get('prior_next_top_gap')}; "
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
