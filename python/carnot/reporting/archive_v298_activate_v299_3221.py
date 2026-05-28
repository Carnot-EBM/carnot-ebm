"""Build the Exp 3221 archive and .299 activation manifest.

Spec refs: REQ-REPORT-3221, SCENARIO-REPORT-3221.

This module closes the `.298` milestone boundary using already-written
evidence. It reads the `.298` capstone, records whether the optional
operational retrospective is present, checks the `.299` queue, and writes a
machine-readable handoff artifact. It does not run models, verifiers, repair
steps, solvers, hardware commands, or the conductor.
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
SCHEMA_VERSION = "carnot.archive_activation.v298_to_v299.v1"
EXPERIMENT_ID = "exp3221"
TASK_ID = "exp3221-archive-v298-activate-v299"
ARTIFACT = "experiment_3221_archive_v298_activate_v299"
MILESTONE = "2026.05.299"
PRIOR_MILESTONE = "2026.05.298"
PRIOR_TASK_RANGE = ["exp3219", "exp3232"]
PRIOR_NEXT_TOP_GAP = "repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt"
RANDOM_SEED = 3221

OUTPUT_REL_PATH = Path("results/experiment_3221_archive_v298_activate_v299.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3221_archive_v298_activate_v299.py"
CAPSTONE_V298_REL_PATH = Path("results/experiment_3232_capstone_v298.json")
OPERATIONAL_RETRO_V298_REL_PATH = Path("results/operational_retro_2026_05_298.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

PRIOR_TASKS: tuple[JsonDict, ...] = (
    {
        "id": "exp3219-archive-v297-activate-v298",
        "title": "Archive .297 closeout and activate .298 planning authority",
        "deliverable": "results/experiment_3219_archive_v297_activate_v298.json",
    },
    {
        "id": "exp3220-hermetic-cuda-runtime-repair-ledger-v1",
        "title": "Hermetic CUDA runtime repair ledger for selected Python and isolated CUDA venv",
        "deliverable": "results/experiment_3220_hermetic_cuda_runtime_repair_ledger_v1.json",
    },
    {
        "id": "exp3221-llama-cpp-cuda-offload-receipt-smoke-v1",
        "title": "llama.cpp CUDA offload receipt smoke gated on hermetic runtime repair",
        "deliverable": "results/experiment_3221_llama_cpp_cuda_offload_receipt_smoke_v1.json",
    },
    {
        "id": "exp3222-full-local-sota-receipt-v6",
        "title": "Full local SOTA GGUF receipt v6 gated on llama.cpp CUDA readiness",
        "deliverable": "results/experiment_3222_full_local_sota_receipt_v6.json",
    },
    {
        "id": "exp3223-distributional-ebm-exact-row-uncertainty-sidecar-v2",
        "title": "Distributional EBM exact-row uncertainty sidecar for context and constraint fixtures",
        "deliverable": (
            "results/experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.json"
        ),
    },
    {
        "id": "exp3224-logitext-partial-smt-context-coverage-pilot-v1",
        "title": "Logitext-style partial SMT coverage pilot for context-dependent constraints",
        "deliverable": (
            "results/experiment_3224_logitext_partial_smt_context_coverage_pilot_v1.json"
        ),
    },
    {
        "id": "exp3225-clean-live-sota-verifier-rerun-v13",
        "title": "Clean live SOTA verifier rerun v13 using exact-row triage",
        "deliverable": "results/experiment_3225_clean_live_sota_verifier_rerun_v13.json",
    },
    {
        "id": "exp3226-structured-repair-proposal-preflight-v2",
        "title": "Structured repair proposal preflight v2 with schema-constrained decoding",
        "deliverable": (
            "results/experiment_3226_structured_repair_proposal_preflight_v2.json"
        ),
    },
    {
        "id": "exp3227-repair-gate-decision-v7",
        "title": "Repair gate decision v7 after receipt, clean verifier, and structured preflight",
        "deliverable": "results/experiment_3227_repair_gate_decision_v7.json",
    },
    {
        "id": "exp3228-multi-turn-repair-ladder-v8",
        "title": "Multi-turn repair ladder v8 gated on repair gate unblocked",
        "deliverable": "results/experiment_3228_multi_turn_repair_ladder_v8.json",
    },
    {
        "id": "exp3229-fr11-nonforgetting-promotion-controller-v3",
        "title": "FR-11 nonforgetting promotion controller v3 with rollback policy",
        "deliverable": (
            "results/experiment_3229_fr11_nonforgetting_promotion_controller_v3.json"
        ),
    },
    {
        "id": "exp3230-kan-cl-certificate-boundary-audit-v2",
        "title": "KAN-CL certificate boundary audit v2 for FR-11 sidecar promotion",
        "deliverable": "results/experiment_3230_kan_cl_certificate_boundary_audit_v2.json",
    },
    {
        "id": "exp3231-cross-corpus-matrix-v32",
        "title": "Cross-corpus matrix v32 for .298 runtime, verifier, repair, and FR-11 evidence",
        "deliverable": "results/experiment_3231_cross_corpus_matrix_v32.json",
    },
    {
        "id": "exp3232-capstone-v298",
        "title": "Capstone .298 publication readiness and next-gap decision",
        "deliverable": "results/experiment_3232_capstone_v298.json",
    },
)
EXPECTED_PRIOR_TASK_IDS = {str(task["id"]) for task in PRIOR_TASKS}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON source artifact and return an empty dict on bad evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_document(path: Path) -> Any:
    """Read YAML source evidence without inventing a shape for malformed data."""

    try:
        text = path.read_text(encoding="utf-8")
        return yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}


def sha256_file(path: Path) -> str | None:
    """Checksum a source file so the handoff can be reproduced."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_next_milestone(current: str) -> str:
    """Increment the final CalVer component while preserving its width."""

    parts = current.split(".")
    if len(parts) != 3 or not parts[-1].isdigit():
        return ""
    width = len(parts[-1])
    parts[-1] = f"{int(parts[-1]) + 1:0{width}d}"
    return ".".join(parts)


def ensure_research_complete_entry(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-REPORT-3221: append the `.298` archive summary only when absent."""

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
    """SCENARIO-REPORT-3221: synthesize the `.298` archive and `.299` manifest."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V298_REL_PATH)
    operational_retro = read_json_object(root_path / OPERATIONAL_RETRO_V298_REL_PATH)
    queue_paths = _queue_paths(root_path)
    research_complete_present = _research_complete_contains_prior(root_path)
    capstone_ready = capstone.get("capstone_ready") is True
    blocked_reasons = _blocked_reasons(
        capstone=capstone,
        capstone_ready=capstone_ready,
        research_complete_present=research_complete_present,
        queue_paths=queue_paths,
        root_path=root_path,
    )

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "prior_task_range": list(PRIOR_TASK_RANGE),
        "prior_task_ids": [str(task["id"]) for task in PRIOR_TASKS],
        "prior_capstone_artifact": CAPSTONE_V298_REL_PATH.as_posix(),
        "prior_operational_retro_artifact": OPERATIONAL_RETRO_V298_REL_PATH.as_posix(),
        "prior_capstone_ready": capstone_ready,
        "prior_paper_ready": capstone.get("paper_ready") is True,
        "prior_publication_blocker_count": _int_value(capstone.get("publication_blocker_count")),
        "prior_next_top_gap": str(capstone.get("next_top_gap") or ""),
        "prior_operational_retro_present": bool(operational_retro),
        "prior_operational_retro_summary": _operational_retro_summary(operational_retro),
        "research_complete_prior_summary_present": research_complete_present,
        "queue_paths": queue_paths,
        "conductor_activation_observed": _file_contains(
            root_path / CONDUCTOR_LOG_REL_PATH,
            "Milestone 2026.05.299 activated",
        ),
        "archive_v298_activate_v299_ready": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
        "source_artifacts": _source_artifacts(root_path, capstone, operational_retro),
        "source_checksums": {
            row["path"]: row["sha256"]
            for row in _source_artifacts(root_path, capstone, operational_retro)
        },
        "inference_substrate": "artifact_aggregation_only",
        "conductor_file_modified": False,
        "active_roadmap_modified_by_this_task": False,
        "ops_status_modified_by_this_task": False,
        "ops_changelog_modified_by_this_task": False,
        "traceability_modified_by_this_task": False,
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
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
    """Build and persist the Exp 3221 archive/activation deliverable JSON."""

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
        "- id: 2026.05.298",
        "  title: Hermetic CUDA Receipt Repair + Distributional Exact-Row Triage + FR-11 Nonforgetting Promotion",
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
    selected_payload = _as_mapping(read_yaml_document(root / selected_path))
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
    }


def _blocked_reasons(
    *,
    capstone: Mapping[str, Any],
    capstone_ready: bool,
    research_complete_present: bool,
    queue_paths: Mapping[str, Any],
    root_path: Path,
) -> list[str]:
    reasons: list[str] = []
    if not capstone:
        reasons.append("capstone_v298 authority is missing or malformed")
    if capstone and not capstone_ready:
        reasons.append("capstone_v298 authority is not ready")
    if not research_complete_present:
        reasons.append("research-complete.yaml does not contain the .298 task summary")
    if queue_paths.get("selected_queue_milestone") != MILESTONE:
        reasons.append("selected queue milestone is not 2026.05.299")
    if queue_paths.get("selected_queue_first_task") != TASK_ID:
        reasons.append("selected queue first task is not exp3221-archive-v298-activate-v299")
    if queue_paths.get("milestone_doc") != VNEXT_DOC_REL_PATH.as_posix():
        reasons.append("selected queue milestone_doc is not the vNEXT document")
    if not _file_contains(root_path / CONDUCTOR_LOG_REL_PATH, "Milestone 2026.05.299 activated"):
        reasons.append("conductor log does not record 2026.05.299 activation")
    if not (root_path / VNEXT_DOC_REL_PATH).is_file():
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


def _source_artifacts(
    root: Path,
    capstone: Mapping[str, Any],
    operational_retro: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _source_record(root, "capstone_v298", CAPSTONE_V298_REL_PATH, bool(capstone)),
        _source_record(
            root,
            "operational_retro_v298",
            OPERATIONAL_RETRO_V298_REL_PATH,
            bool(operational_retro),
        ),
        _source_record(root, "research_complete_archive", RESEARCH_COMPLETE_REL_PATH, True),
        _source_record(root, "active_roadmap_queue", ACTIVE_ROADMAP_REL_PATH, True),
        _source_record(root, "staged_roadmap_queue", STAGED_ROADMAP_REL_PATH, True),
        _source_record(root, "conductor_log_authority", CONDUCTOR_LOG_REL_PATH, True),
    ]


def _source_record(root: Path, role: str, path: Path, readable: bool) -> JsonDict:
    return {
        "role": role,
        "path": path.as_posix(),
        "present": (root / path).is_file(),
        "readable": readable,
        "sha256": sha256_file(root / path),
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
    ready = str(artifact.get("archive_v298_activate_v299_ready")).lower()
    paper_ready = str(artifact.get("prior_paper_ready")).lower()
    blocker_count = artifact.get("prior_publication_blocker_count")
    queue_path = _as_mapping(artifact.get("queue_paths")).get("selected_queue_path")
    return (
        f"complete: archive_v298_activate_v299_ready={ready}; "
        f"prior_paper_ready={paper_ready}; "
        f"prior_publication_blocker_count={blocker_count}; "
        f"queue_path={queue_path}"
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
