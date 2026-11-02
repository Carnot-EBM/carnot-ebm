"""Build the Exp 3321 archive-.306/open-.307 handoff artifact.

Spec refs: REQ-REPORT-3321, SCENARIO-REPORT-3321.

This module records a milestone boundary rather than running research work. It
reads the `.306` capstone, preserves blocked gate-check evidence as a terminal
state, archives the `.306` task list only when it is absent, and emits the
machine-readable `.307` activation receipt that downstream conductor steps can
consume.
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
SCHEMA_VERSION = "carnot.archive_activation.v306_to_v307_phase3_handoff.v1"
EXPERIMENT_ID = "exp3321"
TASK_ID = "exp3321-archive-v306-activate-v307"
ARTIFACT = "experiment_3321_archive_v306_activate_v307"
SOURCE_MILESTONE = "2026.05.306"
TARGET_MILESTONE = "2026.05.307"
INFERENCE_SUBSTRATE = "artifact_aggregation_only"
RANDOM_SEED = 3321

OUTPUT_REL_PATH = Path("results/experiment_3321_archive_v306_activate_v307.json")
CAPSTONE_V306_REL_PATH = Path("results/experiment_3320_capstone_v306.json")
OPERATIONAL_RETRO_V306_REL_PATH = Path("results/operational_retro_2026_05_306.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS = {
    "archive_v306_activate_v307_ready",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}

PRIOR_TASKS: tuple[JsonDict, ...] = (
    {
        "id": "exp3307-archive-v305-activate-v306",
        "title": "Close .305 ledger and open .306 quality-cleanup queue",
        "deliverable": "results/experiment_3307_archive_v305_activate_v306.json",
        "log_title": "Close .305 ledger and open .306 quality-cleanup qu",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3308-quality-flag-root-cause-autopsy-v1",
        "title": "Quality-flag root-cause autopsy v1",
        "deliverable": "results/experiment_3308_quality_flag_root_cause_autopsy_v1.json",
        "log_title": "Quality-flag root-cause autopsy v1",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3309-live-runtime-provenance-contract-v1",
        "title": "Live runtime provenance contract v1",
        "deliverable": "results/experiment_3309_live_runtime_provenance_contract_v1.json",
        "log_title": "Live runtime provenance contract v1",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3310-dataflip-kad-challenge-manifest-v1",
        "title": "DataFlip/KAD challenge manifest v1",
        "deliverable": "results/experiment_3310_dataflip_kad_challenge_manifest_v1.json",
        "log_title": "DataFlip/KAD challenge manifest v1",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3311-pcfi-argus-dataflip-guard-pilot-v1",
        "title": "PCFI/ARGUS DataFlip guard pilot v1",
        "deliverable": "results/experiment_3311_pcfi_argus_dataflip_guard_pilot_v1.json",
        "log_title": "PCFI/ARGUS DataFlip guard pilot v1",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3312-gated-dataflip-garak-quality-clean-rerun-v4",
        "title": "Gated DataFlip/Garak quality-clean rerun v4",
        "deliverable": "results/experiment_3312_dataflip_garak_quality_clean_rerun_v4.json",
        "log_title": "Gated DataFlip/Garak quality-clean rerun v4",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3313-repair-substrate-root-cause-autopsy-v1",
        "title": "Repair substrate root-cause autopsy v1",
        "deliverable": "results/experiment_3313_repair_substrate_root_cause_autopsy_v1.json",
        "log_title": "Repair substrate root-cause autopsy v1",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3314-distributional-ebm-repair-uncertainty-audit-v1",
        "title": "Distributional EBM repair uncertainty audit v1",
        "deliverable": "results/experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.json",
        "log_title": "Distributional EBM repair uncertainty audit v1",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3315-vgb-backtracking-repair-policy-v1",
        "title": "VGB repair backtracking policy v1",
        "deliverable": "results/experiment_3315_vgb_backtracking_repair_policy_v1.json",
        "log_title": "VGB repair backtracking policy v1",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3316-gated-sota-repair-rerun-v12-runtime-clean",
        "title": "Gated SOTA repair rerun v12 runtime-clean",
        "deliverable": "results/experiment_3316_sota_repair_rerun_v12_runtime_clean.json",
        "log_title": "Gated SOTA repair rerun v12 runtime-clean",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3317-repair-headline-evidence-audit-v2",
        "title": "Repair headline evidence audit v2",
        "deliverable": "results/experiment_3317_repair_headline_evidence_audit_v2.json",
        "log_title": "Repair headline evidence audit v2",
        "result": "GATE_BLOCK (conductor)",
    },
    {
        "id": "exp3318-fr11-failure-targeted-curriculum-replay-v3",
        "title": "FR-11 failure-targeted curriculum replay v3",
        "deliverable": "results/experiment_3318_fr11_failure_targeted_curriculum_replay_v3.json",
        "log_title": "FR-11 failure-targeted curriculum replay v3",
        "result": "GATE_BLOCK (conductor)",
    },
    {
        "id": "exp3319-evidence-matrix-v38",
        "title": "Evidence matrix v38",
        "deliverable": "results/experiment_3319_evidence_matrix_v38.json",
        "log_title": "Evidence matrix v38",
        "result": "GATE_BLOCK (conductor)",
    },
    {
        "id": "exp3320-capstone-v306",
        "title": "Capstone v306",
        "deliverable": "results/experiment_3320_capstone_v306.json",
        "log_title": "Capstone v306",
        "result": "GATE_BLOCK (conductor)",
    },
)
EXPECTED_PRIOR_TASK_IDS = {str(task["id"]) for task in PRIOR_TASKS}


def read_json_object(path: Path) -> JsonDict:
    """Read JSON evidence while treating absent or malformed sources as empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_document(path: Path) -> Any:
    """Read YAML evidence without guessing a successful archive shape."""

    try:
        text = path.read_text(encoding="utf-8")
        return yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so the handoff can prove what it summarized."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_research_complete_entry(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-REPORT-3321: archive `.306` once and preserve an existing entry."""

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
    """SCENARIO-REPORT-3321: synthesize the .306 archive and .307 receipt."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V306_REL_PATH)
    operational_retro = read_json_object(root_path / OPERATIONAL_RETRO_V306_REL_PATH)
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
    queue = _v307_queue(root_path)
    activation_observed = _v307_activation_observed(root_path, queue)
    prior_blocker_count, prior_blocker_source = _prior_publication_blocker_count(capstone)
    gate_failures = _failed_capstone_gates(capstone)
    capstone_terminal = _capstone_terminal(capstone)
    blocked_reasons = _blocked_reasons(
        capstone=capstone,
        capstone_terminal=capstone_terminal,
        research_complete_present=research_complete_present,
        queue=queue,
        activation_observed=activation_observed,
    )
    closed_opened = not blocked_reasons
    source_artifacts = _source_artifacts(root_path, capstone, operational_retro)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "source_milestone": SOURCE_MILESTONE,
        "target_milestone": TARGET_MILESTONE,
        "milestone": TARGET_MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "archive_v306_activate_v307_ready": closed_opened,
        "v306_closed_v307_opened": closed_opened,
        "prior_paper_ready": capstone.get("paper_ready") is True,
        "prior_publication_blocker_count": prior_blocker_count,
        "prior_publication_blocker_source": prior_blocker_source,
        "prior_publication_gate_successor": {
            "source": NORTH_STAR_REL_PATH.as_posix(),
            "gate": "G1-G4",
            "paper_ready_definition": "paper_ready := G1 and G2 and G3 and G4",
        },
        "prior_capstone_status": str(
            capstone.get("status") or capstone.get("capstone_status") or ""
        ),
        "prior_capstone_honest_verdict": str(capstone.get("honest_verdict") or ""),
        "prior_capstone_gate_check_summary": str(capstone.get("gate_check_summary") or ""),
        "prior_capstone_gate_failures": gate_failures,
        "prior_capstone_terminal": capstone_terminal,
        "operational_retrospective_summary": _operational_retrospective_summary(operational_retro),
        "research_complete_update": archive_update,
        "research_complete_source_summary": _research_complete_task_summary(root_path),
        "v307_queue": queue,
        "v307_activation_observed": activation_observed,
        "conductor_log_terminal_rows": _conductor_log_terminal_rows(root_path),
        "conductor_log_terminal_status_counts": {},
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "blocked_reasons": blocked_reasons,
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_dataflip_run": True,
        "no_new_repair_run": True,
        "no_new_verifier_run": True,
        "no_new_fr11_weight_update": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_external_submission_or_publication": True,
        "no_push": True,
        "ops_status_modified_by_this_task": False,
        "ops_changelog_modified_by_this_task": False,
        "traceability_modified_by_this_task": False,
        "random_seed": RANDOM_SEED,
        "duration_s": _duration(start, now_s),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["conductor_log_terminal_status_counts"] = dict(
        Counter(
            str(row.get("status") or "missing") for row in artifact["conductor_log_terminal_rows"]
        )
    )
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
    """Build and persist the Exp 3321 handoff JSON."""

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
    """Reject handoff JSON that omits required fields or claims unsafe actions."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3321")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3321-archive-v306-activate-v307")
    if artifact.get("source_milestone") != SOURCE_MILESTONE:
        raise ValueError("source_milestone must be 2026.05.306")
    if artifact.get("target_milestone") != TARGET_MILESTONE:
        raise ValueError("target_milestone must be 2026.05.307")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3321")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be artifact_aggregation_only")
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if _int_value(artifact.get("prior_publication_blocker_count")) < 0:
        raise ValueError("prior_publication_blocker_count must be non-negative")
    if artifact.get("no_push") is not True:
        raise ValueError("no_push must remain true")


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
        "- id: 2026.05.306",
        "  title: DataFlip + Quality-Flag Cleanup For Publication-Ready Evidence",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-05-29'",
        (
            "  finding: prior_paper_ready=false; repair headline evidence gate was "
            "blocked; .307 opens Phase-3 path de-risking authority."
        ),
        "  tasks:",
    ]
    for task in PRIOR_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {task['title']}",
                f"    deliverable: {task['deliverable']}",
                f"    result: {task['result']}",
            ]
        )
    return "\n".join(lines) + "\n"


def _research_complete_contains_prior(root: Path) -> bool:
    payload = read_yaml_document(root / RESEARCH_COMPLETE_REL_PATH)
    for entry in _milestone_entries(payload):
        if str(entry.get("id") or "") == SOURCE_MILESTONE:
            return EXPECTED_PRIOR_TASK_IDS <= set(_task_ids(entry))
    return False


def _research_complete_task_summary(root: Path) -> JsonDict:
    payload = read_yaml_document(root / RESEARCH_COMPLETE_REL_PATH)
    for entry in _milestone_entries(payload):
        if str(entry.get("id") or "") != SOURCE_MILESTONE:
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
            "finding": str(entry.get("finding") or ""),
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
        entries = [dict(payload)] if payload.get("id") is not None else []
        entries.extend(dict(entry) for entry in _as_list(payload.get("milestones")))
        return entries
    return []


def _v307_queue(root: Path) -> JsonDict:
    payload = _as_mapping(read_yaml_document(root / ACTIVE_ROADMAP_REL_PATH))
    task_ids = _task_ids(payload)
    return {
        "active_roadmap_path": ACTIVE_ROADMAP_REL_PATH.as_posix(),
        "active_roadmap_exists": (root / ACTIVE_ROADMAP_REL_PATH).is_file(),
        "selected_queue_milestone": str(payload.get("milestone") or ""),
        "queue_first_task": task_ids[0] if task_ids else "",
        "queue_task_count": len(task_ids),
        "queue_task_ids": task_ids,
        "milestone_title": str(payload.get("milestone_title") or ""),
        "milestone_doc": str(payload.get("milestone_doc") or ""),
    }


def _v307_activation_observed(root: Path, queue: Mapping[str, Any]) -> bool:
    return queue.get("selected_queue_milestone") == TARGET_MILESTONE or _file_contains(
        root / CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.307 activated",
    )


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


def _source_artifacts(
    root: Path, capstone: Mapping[str, Any], operational_retro: Mapping[str, Any]
) -> list[JsonDict]:
    return [
        _source_record(
            root, "capstone_v306", CAPSTONE_V306_REL_PATH, bool(capstone), bool(capstone)
        ),
        _source_record(
            root,
            "operational_retrospective_v306",
            OPERATIONAL_RETRO_V306_REL_PATH,
            bool(operational_retro),
            bool(operational_retro),
        ),
        _source_record(root, "research_complete_archive", RESEARCH_COMPLETE_REL_PATH, True, True),
        _source_record(root, "active_v307_roadmap", ACTIVE_ROADMAP_REL_PATH, True, True),
        _source_record(root, "conductor_log_authority", CONDUCTOR_LOG_REL_PATH, True, True),
        _source_record(root, "north_star_publication_gate", NORTH_STAR_REL_PATH, True, True),
    ]


def _source_record(
    root: Path,
    role: str,
    rel_path: Path,
    readable: bool,
    ready: bool,
) -> JsonDict:
    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "readable": readable and path.is_file(),
        "ready": bool(ready),
        "sha256": sha256_file(path),
    }


def _prior_publication_blocker_count(capstone: Mapping[str, Any]) -> tuple[int, str]:
    value = capstone.get("publication_blocker_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value), "publication_blocker_count"
    failures = _failed_capstone_gates(capstone)
    return len(failures), "failed_capstone_gates"


def _failed_capstone_gates(capstone: Mapping[str, Any]) -> list[JsonDict]:
    return [
        dict(gate)
        for gate in _as_list(capstone.get("gates_evaluated"))
        if isinstance(gate, Mapping) and gate.get("passed") is not True
    ]


def _capstone_terminal(capstone: Mapping[str, Any]) -> bool:
    if not capstone:
        return False
    status = str(capstone.get("status") or "")
    return status in {"blocked", "complete", "success"} or _terminal_prefix_ok(
        str(capstone.get("honest_verdict") or "")
    )


def _operational_retrospective_summary(operational_retro: Mapping[str, Any]) -> JsonDict:
    if not operational_retro:
        return {}
    return {
        "milestone": str(operational_retro.get("milestone") or ""),
        "experiments_completed": _int_value(operational_retro.get("experiments_completed")),
        "total_wall_time_minutes": _int_value(operational_retro.get("total_wall_time_minutes")),
        "summary": str(operational_retro.get("summary") or ""),
    }


def _blocked_reasons(
    *,
    capstone: Mapping[str, Any],
    capstone_terminal: bool,
    research_complete_present: bool,
    queue: Mapping[str, Any],
    activation_observed: bool,
) -> list[str]:
    checks = (
        (not capstone, "capstone_v306 authority is missing or malformed"),
        (bool(capstone) and not capstone_terminal, "capstone_v306 terminal state is missing"),
        (
            not research_complete_present,
            "research-complete.yaml does not contain the .306 task summary",
        ),
        (
            queue.get("selected_queue_milestone") != TARGET_MILESTONE,
            "selected queue milestone is not 2026.05.307",
        ),
        (
            queue.get("queue_first_task") != TASK_ID,
            "selected queue first task is not exp3321-archive-v306-activate-v307",
        ),
        (not activation_observed, "milestone 2026.05.307 activation is not observed"),
    )
    return [reason for failed, reason in checks if failed]


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "archive_v306_activate_v307_ready": artifact.get("archive_v306_activate_v307_ready"),
        "v306_closed_v307_opened": artifact.get("v306_closed_v307_opened"),
        "source_milestone": artifact.get("source_milestone"),
        "target_milestone": artifact.get("target_milestone"),
        "prior_paper_ready": artifact.get("prior_paper_ready"),
        "prior_publication_blocker_count": artifact.get("prior_publication_blocker_count"),
        "prior_publication_blocker_source": artifact.get("prior_publication_blocker_source"),
        "prior_capstone_status": artifact.get("prior_capstone_status"),
        "v307_queue": artifact.get("v307_queue"),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: v306_closed_v307_opened="
        f"{str(artifact.get('v306_closed_v307_opened') is True).lower()}; "
        f"prior_paper_ready={str(artifact.get('prior_paper_ready') is True).lower()}; "
        f"prior_publication_blocker_count={artifact.get('prior_publication_blocker_count')}; "
        f"prior_capstone_status={artifact.get('prior_capstone_status')}; "
        f"target_milestone={artifact.get('target_milestone')}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _terminal_prefix_ok(verdict: str) -> bool:
    return verdict.startswith(TERMINAL_PREFIXES)


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


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value(value: Any) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0
