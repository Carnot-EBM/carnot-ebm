"""Build the Exp 3337 archive-.308/open-.309 handoff artifact.

Spec refs: REQ-REPORT-3337, SCENARIO-REPORT-3337.

This module records a milestone boundary. It reads the `.308` result files and
conductor log, keeps missing and blocked work explicit, and emits a `.309`
activation receipt without running any model, verifier, hardware, or conductor
work. The point is to preserve what happened, including failed operational
handoffs, rather than to repair the milestone history in-place.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260529"
SCHEMA_VERSION = "carnot.archive_activation.v308_to_v309_runtime_recovery_handoff.v1"
EXPERIMENT_ID = "exp3337"
TASK_ID = "exp3337-archive-v308-activate-v309"
ARTIFACT = "experiment_3337_archive_v308_activate_v309"
ARCHIVED_MILESTONE = "2026.05.308"
ACTIVATED_MILESTONE = "2026.05.309"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 3337
VNEXT_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"
NEXT_TOP_GAP = "recover_sota_gguf_tokenizer_runtime_then_rerun_energy_descent_bootstrap"

OUTPUT_REL_PATH = Path("results/experiment_3337_archive_v308_activate_v309.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
VNEXT_PROPOSAL_REL_PATH = Path(VNEXT_DOC)
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
OPERATIONAL_RETRO_REL_PATH = Path("results/operational_retro_2026_05_308.json")

TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")
REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "files_updated",
    "archived_milestone",
    "activated_milestone",
    "completed_artifacts",
    "blocked_artifacts",
    "missing_artifacts",
    "duration_flagged_artifacts",
    "next_top_gap",
}
FILES_UPDATED = [
    "openspec/capabilities/research-reporting/spec.md",
    "python/carnot/reporting/archive_v308_activate_v309_3337.py",
    "scripts/experiment_3337_archive_v308_activate_v309.py",
    "tests/python/test_experiment_3337_archive_v308_activate_v309.py",
    OUTPUT_REL_PATH.as_posix(),
]

PRIOR_TASKS: tuple[JsonDict, ...] = (
    {
        "id": "exp3325-archive-v307-activate-v308",
        "title": "Archive milestone .307 honestly and activate .308",
        "deliverable": "results/experiment_3325_archive_v307_activate_v308.json",
        "log_title": "Archive milestone .307 honestly and activate .308",
        "archive_result": "missing_artifact_operational_failure",
    },
    {
        "id": "exp3326-phase3-path-preflight-manifest-v1",
        "title": "Phase-3 path preflight manifest for recovered upstream tests",
        "deliverable": "results/experiment_3326_phase3_path_preflight_manifest_v1.json",
        "log_title": "Phase-3 path preflight manifest for recovered upst",
        "archive_result": "missing_artifact_operational_failure",
    },
    {
        "id": "exp3327-energy-descent-substrate-bootstrap-v1",
        "title": "Energy-descent substrate bootstrap smoke for SOTA GGUF recovery",
        "deliverable": "results/experiment_3327_energy_descent_substrate_bootstrap_v1.json",
        "log_title": "Energy-descent substrate bootstrap smoke for SOTA",
        "archive_result": "blocked_gpu_setup_tokenizer_runtime",
    },
    {
        "id": "exp3328-energy-descent-vs-ar-sota-panel-v2",
        "title": "Energy-descent versus autoregressive SOTA panel v2",
        "deliverable": "results/experiment_3328_energy_descent_vs_ar_sota_panel_v2.json",
        "log_title": "Energy-descent versus autoregressive SOTA panel v2",
        "archive_result": "gate_blocked_by_exp3327",
    },
    {
        "id": "exp3329-verifier-ensemble-diversity-audit-v2",
        "title": "Verifier ensemble diversity and lambda_min audit v2",
        "deliverable": "results/experiment_3329_verifier_ensemble_diversity_audit_v2.json",
        "log_title": "Verifier ensemble diversity and lambda_min audit v",
        "archive_result": "usable_evidence",
    },
    {
        "id": "exp3330-verifier-diversity-remediation-plan-v1",
        "title": "Verifier diversity remediation plan",
        "deliverable": "results/experiment_3330_verifier_diversity_remediation_plan_v1.json",
        "log_title": "Verifier diversity remediation plan (gated on exp3",
        "archive_result": "missing_artifact_operational_failure",
    },
    {
        "id": "exp3331-ebt-sidecar-adapter-smoke-v2",
        "title": "EBT sidecar adapter smoke against exact verifier scores",
        "deliverable": "results/experiment_3331_ebt_sidecar_adapter_smoke_v2.json",
        "log_title": "EBT sidecar adapter smoke against exact verifier s",
        "archive_result": "usable_evidence",
    },
    {
        "id": "exp3332-interwhen-monitor-pilot-v1",
        "title": "Interwhen-style monitor pilot for intermediate candidate scoring",
        "deliverable": "results/experiment_3332_interwhen_monitor_pilot_v1.json",
        "log_title": "Interwhen-style monitor pilot for intermediate can",
        "archive_result": "usable_evidence",
    },
    {
        "id": "exp3333-energy-guided-ttscaling-sota-ablation-v1",
        "title": "Energy-guided test-time scaling SOTA ablation",
        "deliverable": "results/experiment_3333_energy_guided_ttscaling_sota_ablation_v1.json",
        "log_title": "Energy-guided test-time scaling SOTA ablation unde",
        "archive_result": "usable_diagnostic_duration_flagged",
    },
    {
        "id": "exp3334-fr11-online-verifier-memory-nonforgetting-v4",
        "title": "FR-11 online verifier memory nonforgetting v4",
        "deliverable": "results/experiment_3334_fr11_online_verifier_memory_nonforgetting_v4.json",
        "log_title": "FR-11 online verifier memory nonforgetting v4",
        "archive_result": "usable_evidence",
    },
    {
        "id": "exp3335-reproducer-pack-and-evidence-matrix-v39",
        "title": "Independent reproducer pack and evidence matrix v39",
        "deliverable": "results/experiment_3335_reproducer_pack_and_evidence_matrix_v39.json",
        "log_title": "Independent reproducer pack and evidence matrix v3",
        "archive_result": "missing_artifact_operational_failure",
    },
    {
        "id": "exp3336-capstone-v308",
        "title": "Milestone .308 capstone and next-gap decision",
        "deliverable": "results/experiment_3336_capstone_v308.json",
        "log_title": "Milestone .308 capstone and next-gap decision",
        "archive_result": "missing_artifact_operational_failure",
    },
)
EXPECTED_PRIOR_TASK_IDS = {str(task["id"]) for task in PRIOR_TASKS}

SUMMARY_KEYS = (
    "status",
    "honest_verdict",
    "blocked_reasons",
    "n_cases",
    "effective_k",
    "lambda_min_sigma",
    "collapsed_pairs",
    "adapter_ready",
    "monitor_pilot_ready",
    "ttscaling_ablation_ready",
    "flagged_adversarial",
    "fr11_nonforgetting_ready",
    "new_task_delta",
    "old_task_delta",
    "rollback_count",
)


def read_json_object(path: Path) -> JsonDict:
    """Read JSON evidence, returning empty evidence for missing or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_document(path: Path) -> Any:
    """Read YAML evidence without inventing a successful structure on errors."""

    try:
        text = path.read_text(encoding="utf-8")
        return yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so the archive can be audited without rerunning work."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_research_complete_entry(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-REPORT-3337: archive `.308` once and preserve existing entries."""

    root_path = Path(root)
    if _research_complete_contains_prior(root_path):
        return {
            "path": RESEARCH_COMPLETE_REL_PATH.as_posix(),
            "appended": False,
            "already_present": True,
        }
    _append_research_complete_entry(root_path / RESEARCH_COMPLETE_REL_PATH)
    return {
        "path": RESEARCH_COMPLETE_REL_PATH.as_posix(),
        "appended": True,
        "already_present": False,
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    research_complete_update: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """SCENARIO-REPORT-3337: synthesize the .308 archive and .309 receipt."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    terminal_rows = _conductor_log_terminal_rows(root_path)
    classifications = _classify_artifacts(root_path, terminal_rows)
    research_complete_present = _research_complete_contains_prior(root_path)
    archive_update = (
        dict(research_complete_update)
        if research_complete_update
        else {
            "path": RESEARCH_COMPLETE_REL_PATH.as_posix(),
            "appended": False,
            "already_present": research_complete_present,
        }
    )
    archive_present_after_update = bool(
        research_complete_present
        or archive_update.get("already_present") is True
        or archive_update.get("appended") is True
    )
    roadmap_validation = _roadmap_validation(root_path)
    blocked_reasons = _blocked_reasons(archive_present_after_update, roadmap_validation)
    source_artifacts = _source_artifacts(root_path)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "milestone": ACTIVATED_MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "archive_v308_activate_v309_ready": not blocked_reasons,
        "completed_artifacts": classifications["completed"],
        "blocked_artifacts": classifications["blocked"],
        "gate_blocked_artifacts": classifications["gate_blocked"],
        "missing_artifacts": classifications["missing"],
        "duration_flagged_artifacts": classifications["duration_flagged"],
        "next_top_gap": NEXT_TOP_GAP,
        "roadmap_validation": roadmap_validation,
        "operational_retrospective_summary": _operational_retrospective_summary(root_path),
        "research_complete_update": archive_update,
        "research_complete_source_summary": _research_complete_task_summary(root_path),
        "research_complete_existing_entry_overstates_success": _research_complete_overstates_success(
            root_path, classifications
        ),
        "conductor_log_terminal_rows": terminal_rows,
        "conductor_log_terminal_status_counts": dict(
            Counter(str(row.get("status") or "missing") for row in terminal_rows)
        ),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "blocked_reasons": blocked_reasons,
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_verifier_scoring": True,
        "no_new_fr11_update": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "research_roadmap_modified_by_this_task": False,
        "scripts_research_conductor_modified_by_this_task": False,
        "ops_status_modified_by_this_task": False,
        "ops_changelog_modified_by_this_task": False,
        "traceability_modified_by_this_task": False,
        "files_updated": list(FILES_UPDATED),
        "random_seed": RANDOM_SEED,
        "duration_s": _duration(start, now_s),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    update_research_complete: bool = True,
) -> Path:
    """Build and persist the Exp 3337 handoff JSON."""

    root_path = Path(root)
    archive_update = (
        ensure_research_complete_entry(root_path)
        if update_research_complete
        else {
            "path": RESEARCH_COMPLETE_REL_PATH.as_posix(),
            "appended": False,
            "already_present": _research_complete_contains_prior(root_path),
        }
    )
    artifact = build_artifact(
        root_path,
        research_complete_update=archive_update,
        started_s=started_s,
        now_s=now_s,
    )
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject archive JSON that omits required fields or claims unsafe actions."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3337")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        raise ValueError("archived_milestone must be 2026.05.308")
    if artifact.get("activated_milestone") != ACTIVATED_MILESTONE:
        raise ValueError("activated_milestone must be 2026.05.309")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3337")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if artifact.get("no_push") is not True:
        raise ValueError("no_push must remain true")
    if not _as_list(artifact.get("files_updated")):
        raise ValueError("files_updated must be non-empty")


def _append_research_complete_entry(path: Path) -> None:
    entry = _research_complete_entry()
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not existing.strip():
        path.write_text("milestones:\n" + entry, encoding="utf-8")
        return
    if re.search(r"(?m)^milestones:\s*\[\]\s*$", existing):
        path.write_text(
            re.sub(r"(?m)^milestones:\s*\[\]\s*$", "milestones:\n" + entry.rstrip(), existing)
            + "\n",
            encoding="utf-8",
        )
        return
    separator = "" if existing.endswith("\n") else "\n"
    path.write_text(existing + separator + entry, encoding="utf-8")


def _research_complete_entry() -> str:
    lines = [
        "- id: 2026.05.308",
        "  title: Phase-3 Path Recovery, Verifier Grounding, and FR-11 Nonforgetting",
        f"  doc: {VNEXT_DOC}",
        "  completed: '2026-05-29'",
        (
            "  finding: exp3327 blocked on SOTA GGUF tokenizer/runtime setup; "
            "exp3328 gate-blocked; exp3330/3335/3336 missing; "
            "exp3329 and exp3331-exp3334 provide bounded usable evidence."
        ),
        "  tasks:",
    ]
    for task in PRIOR_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {task['title']}",
                f"    deliverable: {task['deliverable']}",
                f"    result: {task['archive_result']}",
            ]
        )
    return "\n".join(lines) + "\n"


def _research_complete_contains_prior(root: Path) -> bool:
    payload = read_yaml_document(root / RESEARCH_COMPLETE_REL_PATH)
    for entry in _milestone_entries(payload):
        if str(entry.get("id") or "") == ARCHIVED_MILESTONE:
            return EXPECTED_PRIOR_TASK_IDS <= set(_task_ids(entry))
    return False


def _research_complete_task_summary(root: Path) -> JsonDict:
    payload = read_yaml_document(root / RESEARCH_COMPLETE_REL_PATH)
    for entry in _milestone_entries(payload):
        if str(entry.get("id") or "") != ARCHIVED_MILESTONE:
            continue
        tasks = [
            dict(task)
            for task in _as_list(entry.get("tasks"))
            if isinstance(task, Mapping) and task.get("id") not in (None, "")
        ]
        task_ids = [str(task["id"]) for task in tasks]
        return {
            "milestone": str(entry.get("id") or ""),
            "title": str(entry.get("title") or ""),
            "completed": str(entry.get("completed") or ""),
            "finding": str(entry.get("finding") or ""),
            "task_count": len(task_ids),
            "task_ids": task_ids,
            "result_counts": dict(Counter(str(task.get("result") or "") for task in tasks)),
        }
    return {}


def _research_complete_overstates_success(root: Path, classifications: Mapping[str, Any]) -> bool:
    summary = _research_complete_task_summary(root)
    result_counts = _as_mapping(summary.get("result_counts"))
    has_non_success = any(
        _as_list(classifications.get(key))
        for key in ("blocked", "gate_blocked", "missing", "duration_flagged")
    )
    return result_counts.get("OK (conductor)") == len(PRIOR_TASKS) and has_non_success


def _milestone_entries(payload: Any) -> list[JsonDict]:
    if isinstance(payload, list):
        return [dict(entry) for entry in payload if isinstance(entry, Mapping)]
    if isinstance(payload, Mapping):
        entries = [dict(payload)] if payload.get("id") is not None else []
        entries.extend(dict(entry) for entry in _as_list(payload.get("milestones")))
        return entries
    return []


def _roadmap_validation(root: Path) -> JsonDict:
    active = _roadmap_state(root / ACTIVE_ROADMAP_REL_PATH)
    staged = _roadmap_state(root / STAGED_ROADMAP_REL_PATH)
    activation_seen = _file_contains(
        root / CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.309 activated",
    )
    return {
        "active": active,
        "staged": staged,
        "activated_milestone_confirmed": (
            active.get("milestone") == ACTIVATED_MILESTONE or activation_seen
        ),
        "research_roadmap_next_absent_after_activation": staged.get("exists") is False,
    }


def _roadmap_state(path: Path) -> JsonDict:
    exists = path.is_file()
    payload: Any = {}
    yaml_parse_ok = False
    if exists:
        try:
            text = path.read_text(encoding="utf-8")
            payload = yaml.safe_load(text) if text.strip() else {}
            yaml_parse_ok = True
        except (OSError, yaml.YAMLError):
            payload = {}
    mapping = _as_mapping(payload)
    task_ids = _task_ids(mapping)
    milestone_doc = str(mapping.get("milestone_doc") or "")
    return {
        "path": _relative_path(path),
        "exists": exists,
        "yaml_parse_ok": yaml_parse_ok,
        "milestone": str(mapping.get("milestone") or ""),
        "milestone_doc": milestone_doc,
        "points_to_vnext": milestone_doc == VNEXT_DOC,
        "queue_first_task": task_ids[0] if task_ids else "",
        "queue_task_count": len(task_ids),
        "task_ids": task_ids,
    }


def _conductor_log_terminal_rows(root: Path) -> list[JsonDict]:
    try:
        lines = (root / CONDUCTOR_LOG_REL_PATH).read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    rows: list[JsonDict] = []
    for task in PRIOR_TASKS:
        matches = [line for line in lines if str(task["log_title"]) in line]
        parsed = _parse_conductor_line(matches[-1]) if matches else {}
        rows.append(
            {
                "experiment_id": str(task["id"]).split("-", maxsplit=1)[0],
                "task_id": task["id"],
                "title": task["title"],
                "line": matches[-1] if matches else "",
                "timestamp_utc": str(parsed.get("timestamp_utc") or ""),
                "status": str(parsed.get("status") or "missing"),
                "details": str(parsed.get("details") or ""),
            }
        )
    return rows


def _parse_conductor_line(line: str) -> JsonDict:
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    if len(parts) < 4:
        return {}
    return {
        "timestamp_utc": parts[0],
        "title": parts[1],
        "status": parts[2],
        "details": parts[3],
    }


def _classify_artifacts(root: Path, terminal_rows: list[Mapping[str, Any]]) -> JsonDict:
    row_by_task = {str(row.get("task_id") or ""): dict(row) for row in terminal_rows}
    classifications: JsonDict = {
        "completed": [],
        "blocked": [],
        "gate_blocked": [],
        "missing": [],
        "duration_flagged": [],
    }
    for task in PRIOR_TASKS:
        record = _artifact_record(root, task, row_by_task.get(str(task["id"]), {}))
        if record["duration_flags"]:
            classifications["duration_flagged"].append(record)
        if record["conductor_status"] == "GATE_BLOCK" or str(task["archive_result"]).startswith(
            "gate_blocked"
        ):
            classifications["gate_blocked"].append(record)
        elif not record["artifact_present"]:
            classifications["missing"].append(record)
        elif _record_is_blocked(record):
            classifications["blocked"].append(record)
        else:
            classifications["completed"].append(record)
    return classifications


def _artifact_record(root: Path, task: Mapping[str, Any], terminal_row: Mapping[str, Any]) -> JsonDict:
    rel_path = Path(str(task["deliverable"]))
    full_path = root / rel_path
    payload = read_json_object(full_path)
    summary = {key: payload[key] for key in SUMMARY_KEYS if key in payload}
    duration_flags = _duration_flag_reasons(payload)
    return {
        "experiment_id": str(task["id"]).split("-", maxsplit=1)[0],
        "task_id": str(task["id"]),
        "title": str(task["title"]),
        "deliverable": rel_path.as_posix(),
        "artifact_present": full_path.is_file() and bool(payload),
        "conductor_status": str(terminal_row.get("status") or "missing"),
        "conductor_details": str(terminal_row.get("details") or ""),
        "archive_result": str(task.get("archive_result") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "status": str(payload.get("status") or ""),
        "duration_s": payload.get("duration_s"),
        "duration_flags": duration_flags,
        "summary": summary,
        "sha256": sha256_file(full_path),
    }


def _record_is_blocked(record: Mapping[str, Any]) -> bool:
    verdict = str(record.get("honest_verdict") or "").lower()
    status = str(record.get("status") or "").lower()
    archive_result = str(record.get("archive_result") or "").lower()
    return status == "blocked" or verdict.startswith("blocked") or archive_result.startswith("blocked")


def _duration_flag_reasons(payload: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    for item in _as_list(payload.get("corrigendum_pending")):
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind") or "")
        detail = str(item.get("detail") or "")
        if kind == "DURATION_TOO_SHORT" or "duration" in detail.lower():
            flags.append(f"{kind}: {detail}".strip(": "))
    if not flags and payload.get("flagged_adversarial") is True:
        flags.append("flagged_adversarial=true")
    return flags


def _operational_retrospective_summary(root: Path) -> JsonDict:
    payload = read_json_object(root / OPERATIONAL_RETRO_REL_PATH)
    return {
        key: payload[key]
        for key in (
            "milestone",
            "total_wall_time_minutes",
            "experiments_completed",
            "compute_bound_experiments_count",
            "slowest_experiments",
            "summary",
        )
        if key in payload
    }


def _source_artifacts(root: Path) -> list[JsonDict]:
    paths: list[tuple[str, Path, bool]] = [
        ("research_complete", RESEARCH_COMPLETE_REL_PATH, True),
        ("active_roadmap", ACTIVE_ROADMAP_REL_PATH, True),
        ("staged_roadmap", STAGED_ROADMAP_REL_PATH, False),
        ("vnext_proposal", VNEXT_PROPOSAL_REL_PATH, True),
        ("conductor_log", CONDUCTOR_LOG_REL_PATH, True),
        ("operational_retro", OPERATIONAL_RETRO_REL_PATH, False),
    ]
    paths.extend((str(task["id"]), Path(str(task["deliverable"])), False) for task in PRIOR_TASKS)
    records = []
    for label, rel_path, required in paths:
        full_path = root / rel_path
        records.append(
            {
                "label": label,
                "path": rel_path.as_posix(),
                "exists": full_path.is_file(),
                "required": required,
                "sha256": sha256_file(full_path),
            }
        )
    return records


def _blocked_reasons(archive_present: bool, roadmap_validation: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    active = _as_mapping(roadmap_validation.get("active"))
    if not archive_present:
        reasons.append("research-complete.yaml does not contain the .308 task summary")
    if active.get("milestone") != ACTIVATED_MILESTONE:
        reasons.append("active roadmap milestone is not 2026.05.309")
    if active.get("points_to_vnext") is not True:
        reasons.append(f"active roadmap does not point to {VNEXT_DOC}")
    if roadmap_validation.get("activated_milestone_confirmed") is not True:
        reasons.append("milestone 2026.05.309 activation is not observed")
    return reasons


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    ready = artifact.get("archive_v308_activate_v309_ready") is True
    return (
        "complete: archive_v308_activate_v309_ready="
        f"{str(ready).lower()}; completed={len(_as_list(artifact.get('completed_artifacts')))}; "
        f"blocked={len(_as_list(artifact.get('blocked_artifacts')))}; "
        f"gate_blocked={len(_as_list(artifact.get('gate_blocked_artifacts')))}; "
        f"missing={len(_as_list(artifact.get('missing_artifacts')))}; "
        f"duration_flagged={len(_as_list(artifact.get('duration_flagged_artifacts')))}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _task_ids(payload: Mapping[str, Any]) -> list[str]:
    return [
        str(task["id"])
        for task in _as_list(payload.get("tasks"))
        if isinstance(task, Mapping) and task.get("id") not in (None, "")
    ]


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _terminal_prefix_ok(verdict: str) -> bool:
    return verdict.startswith(TERMINAL_PREFIXES)


def _file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except OSError:
        return False


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
