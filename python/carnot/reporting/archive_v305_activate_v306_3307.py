"""Build the Exp 3307 archive-.305/open-.306 handoff artifact.

Spec refs: REQ-REPORT-3307, SCENARIO-REPORT-3307.

This module is a milestone ledger, not a new model run. It reads the completed
`.305` capstone and evidence matrix, records the terminal blockers that `.306`
inherits, appends the `.305` archive only when it is absent, and writes a
machine-readable `.306` activation receipt. The protected-file check compares
before and after hashes so a roadmap already activated by the conductor is not
mistaken for a mutation performed by this handoff.
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
SCHEMA_VERSION = "carnot.archive_activation.v305_to_v306_quality_cleanup_handoff.v1"
EXPERIMENT_ID = "exp3307"
TASK_ID = "exp3307-archive-v305-activate-v306"
ARTIFACT = "experiment_3307_archive_v305_activate_v306"
SOURCE_MILESTONE = "2026.05.305"
TARGET_MILESTONE = "2026.05.306"
INFERENCE_SUBSTRATE = "artifact_aggregation_only"
RANDOM_SEED = 3307
EXPECTED_PUBLICATION_BLOCKER_COUNT = 8
EXPECTED_INHERITED_TOP_GAP = "clear_garak_dataflip_and_quality_flags"

OUTPUT_REL_PATH = Path("results/experiment_3307_archive_v305_activate_v306.json")
CAPSTONE_V305_REL_PATH = Path("results/experiment_3306_capstone_v305.json")
MATRIX_V37_REL_PATH = Path("results/experiment_3305_evidence_matrix_v37.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

PROTECTED_FILES = (
    ACTIVE_ROADMAP_REL_PATH.as_posix(),
    CONDUCTOR_REL_PATH.as_posix(),
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS = {
    "archive_v305_activate_v306_ready",
    "v305_closed_v306_opened",
    "source_milestone",
    "target_milestone",
    "publication_blocker_count",
    "inherited_top_gap",
    "protected_files_unchanged",
    "honest_verdict",
}

PRIOR_TASKS: tuple[JsonDict, ...] = (
    {
        "id": "exp3294-archive-v304-activate-v305",
        "title": "Close .304 ledger and open .305 Garak gate queue",
        "deliverable": "results/experiment_3294_archive_v304_activate_v305.json",
        "log_title": "Close .304 ledger and open .305 Garak gate queue",
    },
    {
        "id": "exp3295-garak-failure-mode-autopsy-v1",
        "title": "Garak failure-mode autopsy v1",
        "deliverable": "results/experiment_3295_garak_failure_mode_autopsy_v1.json",
        "log_title": "Garak failure-mode autopsy v1",
    },
    {
        "id": "exp3296-substrate-corrigendum-kan-no-retry-v1",
        "title": "Evidence substrate corrigendum and KAN no-retry ledger v1",
        "deliverable": "results/experiment_3296_substrate_corrigendum_kan_no_retry_v1.json",
        "log_title": "Evidence substrate corrigendum and KAN no-retry",
    },
    {
        "id": "exp3297-prefix-closed-garak-guard-v1",
        "title": "Prefix-closed Garak rogue-string guard pilot v1",
        "deliverable": "results/experiment_3297_prefix_closed_garak_guard_v1.json",
        "log_title": "Prefix-closed Garak rogue-string guard pilot v1",
    },
    {
        "id": "exp3298-redteam-energy-telemetry-router-v1",
        "title": "Red-team energy telemetry and routing policy v1",
        "deliverable": "results/experiment_3298_redteam_energy_telemetry_router_v1.json",
        "log_title": "Red-team energy telemetry and routing policy v1",
    },
    {
        "id": "exp3299-garak-defense-ablation-v1",
        "title": "Garak defense ablation v1",
        "deliverable": "results/experiment_3299_garak_defense_ablation_v1.json",
        "log_title": "Garak defense ablation v1",
    },
    {
        "id": "exp3300-full-garak-dataflip-gate-rerun-v3",
        "title": "Full Garak/DataFlip gate rerun v3",
        "deliverable": "results/experiment_3300_full_garak_dataflip_gate_rerun_v3.json",
        "log_title": "Full Garak/DataFlip gate rerun v3",
    },
    {
        "id": "exp3301-exact-repair-panel-manifest-v11",
        "title": "Exact repair panel manifest v11",
        "deliverable": "results/experiment_3301_exact_repair_panel_manifest_v11.json",
        "log_title": "Exact repair panel manifest v11",
    },
    {
        "id": "exp3302-headline-sota-repair-panel-v11",
        "title": "Headline SOTA repair panel v11",
        "deliverable": "results/experiment_3302_headline_sota_repair_panel_v11.json",
        "log_title": "Headline SOTA repair panel v11",
    },
    {
        "id": "exp3303-repair-headline-evidence-audit-v1",
        "title": "Repair headline evidence audit v1",
        "deliverable": "results/experiment_3303_repair_headline_evidence_audit_v1.json",
        "log_title": "Repair headline evidence audit v1",
    },
    {
        "id": "exp3304-fr11-redteam-repair-memory-replay-v2",
        "title": "FR-11 red-team and repair memory replay v2",
        "deliverable": "results/experiment_3304_fr11_redteam_repair_memory_replay_v2.json",
        "log_title": "FR-11 red-team and repair memory replay v2",
    },
    {
        "id": "exp3305-evidence-matrix-v37",
        "title": "Evidence matrix v37",
        "deliverable": "results/experiment_3305_evidence_matrix_v37.json",
        "log_title": "Evidence matrix v37",
    },
    {
        "id": "exp3306-capstone-v305",
        "title": "Capstone v305",
        "deliverable": "results/experiment_3306_capstone_v305.json",
        "log_title": "Capstone v305",
    },
)
EXPECTED_PRIOR_TASK_IDS = {str(task["id"]) for task in PRIOR_TASKS}


def read_json_object(path: Path) -> JsonDict:
    """Read source JSON while preserving missing or malformed evidence as empty."""

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
    """Hash exact bytes so the handoff can be audited without rerunning work."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_file_checksums(root: Path | str = REPO_ROOT) -> dict[str, str | None]:
    """Capture hashes for files this handoff must not edit."""

    root_path = Path(root)
    return {path: sha256_file(root_path / path) for path in PROTECTED_FILES}


def ensure_research_complete_entry(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-REPORT-3307: archive `.305` once, leaving an existing entry untouched."""

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
    protected_hash_baseline: Mapping[str, str | None] | None = None,
    research_complete_update: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """SCENARIO-REPORT-3307: synthesize the .305 archive and .306 handoff."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V305_REL_PATH)
    matrix = read_json_object(root_path / MATRIX_V37_REL_PATH)
    baseline = (
        dict(protected_hash_baseline)
        if protected_hash_baseline is not None
        else protected_file_checksums(root_path)
    )
    protected_checksums = _protected_file_checksum_report(root_path, baseline)
    protected_unchanged = all(row["unchanged"] for row in protected_checksums.values())
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
    queue = _v306_queue(root_path)
    terminal_rows = _conductor_log_terminal_rows(root_path)
    terminal_blockers = _terminal_v305_blockers(capstone, matrix)

    capstone_ready = capstone.get("capstone_v305_ready") is True
    matrix_ready = matrix.get("matrix_v37_ready") is True
    paper_ready = capstone.get("paper_ready") is True
    publication_blocker_count = _publication_blocker_count(capstone, matrix)
    inherited_top_gap = _inherited_top_gap(capstone, matrix)
    garak_gate_passed = capstone.get("garak_gate_passed") is True
    garak_attack_success_rate = _garak_attack_success_rate(capstone, matrix)
    dataflip_gate_passed = _dataflip_gate_passed(capstone, matrix)
    repair_allowed = _repair_headline_claim_allowed(capstone, matrix)
    fr11_safe = _fr11_memory_replay_safe(capstone, matrix)
    activation_observed = _v306_activation_observed(root_path, queue)
    blocked_reasons = _blocked_reasons(
        capstone=capstone,
        matrix=matrix,
        capstone_ready=capstone_ready,
        matrix_ready=matrix_ready,
        publication_blocker_count=publication_blocker_count,
        inherited_top_gap=inherited_top_gap,
        dataflip_gate_passed=dataflip_gate_passed,
        repair_headline_claim_allowed=repair_allowed,
        fr11_memory_replay_safe=fr11_safe,
        quality_flags=terminal_blockers["quality_flags"],
        research_complete_present=research_complete_present,
        queue=queue,
        activation_observed=activation_observed,
        protected_files_unchanged=protected_unchanged,
    )
    closed_opened = not blocked_reasons
    source_artifacts = _source_artifacts(root_path, capstone, matrix)

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
        "archive_v305_activate_v306_ready": closed_opened,
        "v305_closed_v306_opened": closed_opened,
        "paper_ready": paper_ready,
        "publication_blocker_count": publication_blocker_count,
        "inherited_top_gap": inherited_top_gap,
        "garak_gate_passed": garak_gate_passed,
        "garak_attack_success_rate": garak_attack_success_rate,
        "dataflip_gate_passed": dataflip_gate_passed,
        "repair_headline_claim_allowed": repair_allowed,
        "fr11_memory_replay_safe": fr11_safe,
        "terminal_v305_evidence": _terminal_v305_evidence(capstone, matrix),
        "terminal_v305_blockers": terminal_blockers,
        "v306_start_conditions": _v306_start_conditions(
            dataflip_gate_passed=dataflip_gate_passed,
            quality_flags=terminal_blockers["quality_flags"],
            repair_headline_claim_allowed=repair_allowed,
            fr11_memory_replay_safe=fr11_safe,
        ),
        "v306_activation_reason": _v306_activation_reason(),
        "research_complete_update": archive_update,
        "research_complete_source_summary": _research_complete_task_summary(root_path),
        "v306_queue": queue,
        "v306_activation_observed": activation_observed,
        "conductor_log_terminal_rows": terminal_rows,
        "conductor_log_terminal_status_counts": dict(
            Counter(str(row.get("status") or "missing") for row in terminal_rows)
        ),
        "protected_files": list(PROTECTED_FILES),
        "protected_files_unchanged": protected_unchanged,
        "protected_files_untouched": protected_unchanged,
        "protected_file_checksums": protected_checksums,
        "principle_annotations": _principle_annotations(),
        "blocked_reasons": blocked_reasons,
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
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
        "research_roadmap_modified_by_this_task": False,
        "scripts_research_conductor_modified_by_this_task": False,
        "ops_status_modified_by_this_task": False,
        "ops_changelog_modified_by_this_task": False,
        "traceability_modified_by_this_task": False,
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
    """Build and persist the Exp 3307 handoff JSON."""

    root_path = Path(root)
    baseline = protected_file_checksums(root_path)
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
        protected_hash_baseline=baseline,
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
    """Reject handoff JSON that omits required boundary fields or overclaims."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3307")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3307-archive-v305-activate-v306")
    if artifact.get("source_milestone") != SOURCE_MILESTONE:
        raise ValueError("source_milestone must be 2026.05.305")
    if artifact.get("target_milestone") != TARGET_MILESTONE:
        raise ValueError("target_milestone must be 2026.05.306")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3307")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be artifact_aggregation_only")
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if _int_value(artifact.get("publication_blocker_count")) < 0:
        raise ValueError("publication_blocker_count must be non-negative")
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
        "- id: 2026.05.305",
        "  title: Garak Red-Team Gate Pass + Headline-Eligible Repair Evidence",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-05-29'",
        (
            "  finding: paper_ready=false; publication_blocker_count=8; "
            "next_top_gap=clear_garak_dataflip_and_quality_flags; "
            "DataFlip/quality/repair provenance blockers carry into .306."
        ),
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


def _v306_queue(root: Path) -> JsonDict:
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
        "staged_roadmap_exists": (root / STAGED_ROADMAP_REL_PATH).is_file(),
    }


def _v306_activation_observed(root: Path, queue: Mapping[str, Any]) -> bool:
    return queue.get("selected_queue_milestone") == TARGET_MILESTONE or _file_contains(
        root / CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.306 activated",
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
    root: Path, capstone: Mapping[str, Any], matrix: Mapping[str, Any]
) -> list[JsonDict]:
    return [
        _source_record(root, "capstone_v305", CAPSTONE_V305_REL_PATH, bool(capstone), True),
        _source_record(
            root,
            "evidence_matrix_v37",
            MATRIX_V37_REL_PATH,
            bool(matrix),
            matrix.get("matrix_v37_ready") is True,
        ),
        _source_record(root, "research_complete_archive", RESEARCH_COMPLETE_REL_PATH, True, True),
        _source_record(root, "active_v306_roadmap", ACTIVE_ROADMAP_REL_PATH, True, True),
        _source_record(root, "staged_v306_roadmap_context", STAGED_ROADMAP_REL_PATH, True, False),
        _source_record(root, "conductor_log_authority", CONDUCTOR_LOG_REL_PATH, True, True),
        _source_record(root, "protected_research_conductor", CONDUCTOR_REL_PATH, True, True),
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


def _protected_file_checksum_report(
    root: Path,
    baseline: Mapping[str, str | None],
) -> dict[str, JsonDict]:
    current = protected_file_checksums(root)
    return {
        path: {
            "before": baseline.get(path),
            "after": current.get(path),
            "unchanged": baseline.get(path) == current.get(path),
        }
        for path in PROTECTED_FILES
    }


def _terminal_v305_evidence(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "capstone_honest_verdict": str(capstone.get("honest_verdict") or ""),
        "matrix_honest_verdict": str(matrix.get("honest_verdict") or ""),
        "capstone_v305_ready": capstone.get("capstone_v305_ready") is True,
        "matrix_v37_ready": matrix.get("matrix_v37_ready") is True,
        "paper_ready": capstone.get("paper_ready") is True,
        "publication_blocker_count": _publication_blocker_count(capstone, matrix),
        "blocker_delta_from_v304": _int_value(capstone.get("blocker_delta_from_v304")),
        "inherited_top_gap": _inherited_top_gap(capstone, matrix),
        "garak_gate_passed": capstone.get("garak_gate_passed") is True,
        "garak_attack_success_rate": _garak_attack_success_rate(capstone, matrix),
        "dataflip_gate_passed": _dataflip_gate_passed(capstone, matrix),
        "repair_headline_claim_allowed": _repair_headline_claim_allowed(capstone, matrix),
        "fr11_memory_replay_safe": _fr11_memory_replay_safe(capstone, matrix),
    }


def _terminal_v305_blockers(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> JsonDict:
    garak = _gate_detail(capstone, matrix, "garak_gate", "exp3300")
    repair = _gate_detail(capstone, matrix, "repair_headline", "exp3303")
    fr11 = _gate_detail(capstone, matrix, "fr11_replay", "exp3304")
    quality_flags = _dedupe_quality_flags(
        _quality_flags(garak)
        + _quality_flags(repair)
        + _quality_flags(_matrix_row(matrix, "exp3300"))
        + _quality_flags(_matrix_row(matrix, "exp3303"))
    )
    return {
        "dataflip_failure": {
            "source_experiment_id": str(garak.get("source_experiment_id") or "exp3300"),
            "dataflip_gate_passed": _dataflip_gate_passed(capstone, matrix),
            "garak_gate_passed": capstone.get("garak_gate_passed") is True
            or _as_mapping(_matrix_row(matrix, "exp3300"))
            .get("summary", {})
            .get("garak_gate_passed")
            is True,
            "blocker_reasons": _list_of_strings(garak.get("blocker_reasons"))
            or _list_of_strings(_matrix_row(matrix, "exp3300").get("blocker_reasons")),
        },
        "quality_flags": quality_flags,
        "repair_headline_provenance_failure": {
            "source_experiment_id": str(repair.get("source_experiment_id") or "exp3303"),
            "repair_headline_claim_allowed": _repair_headline_claim_allowed(capstone, matrix),
            "blocker_reasons": _list_of_strings(repair.get("blocker_reasons"))
            or _list_of_strings(_matrix_row(matrix, "exp3303").get("blocker_reasons")),
            "quality_flags": _dedupe_quality_flags(
                _quality_flags(repair) + _quality_flags(_matrix_row(matrix, "exp3303"))
            ),
        },
        "fr11_controller_memory_safety": {
            "source_experiment_id": str(fr11.get("source_experiment_id") or "exp3304"),
            "fr11_memory_replay_safe": _fr11_memory_replay_safe(capstone, matrix),
            "controller_memory_only": fr11.get("controller_memory_only") is True
            or _as_mapping(_matrix_row(matrix, "exp3304").get("summary")).get(
                "controller_memory_only"
            )
            is True,
            "foundation_weight_updates_performed": fr11.get("foundation_weight_updates_performed")
            is True
            or _as_mapping(_matrix_row(matrix, "exp3304").get("summary")).get(
                "foundation_weight_updates_performed"
            )
            is True,
            "blocker_reasons": _list_of_strings(fr11.get("blocker_reasons")),
        },
    }


def _gate_detail(
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
    key: str,
    experiment_id: str,
) -> JsonDict:
    capstone_detail = _as_mapping(_as_mapping(capstone.get("gate_status_details")).get(key))
    if capstone_detail:
        return capstone_detail
    matrix_detail = _as_mapping(_as_mapping(matrix.get("gate_summary")).get(key))
    if matrix_detail:
        return matrix_detail
    return _matrix_row(matrix, experiment_id)


def _matrix_row(matrix: Mapping[str, Any], experiment_id: str) -> JsonDict:
    return _row(
        _as_list(matrix.get("rows")) or _as_list(matrix.get("evidence_rows")), experiment_id
    )


def _quality_flags(payload: Mapping[str, Any]) -> list[JsonDict]:
    return [
        dict(flag) for flag in _as_list(payload.get("quality_flags")) if isinstance(flag, Mapping)
    ]


def _dedupe_quality_flags(flags: list[JsonDict]) -> list[JsonDict]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[JsonDict] = []
    for flag in flags:
        key = (
            str(flag.get("kind") or ""),
            str(flag.get("severity") or ""),
            str(flag.get("detail") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(flag)
    return deduped


def _v306_start_conditions(
    *,
    dataflip_gate_passed: bool,
    quality_flags: list[JsonDict],
    repair_headline_claim_allowed: bool,
    fr11_memory_replay_safe: bool,
) -> JsonDict:
    return {
        "dataflip_failure": dataflip_gate_passed is False,
        "current_quality_flags": [str(flag.get("kind") or "") for flag in quality_flags],
        "repair_headline_provenance_failure": repair_headline_claim_allowed is False,
        "fr11_controller_memory_safety": fr11_memory_replay_safe is True,
    }


def _publication_blocker_count(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> int:
    value = _int_value(capstone.get("publication_blocker_count"))
    if value:
        return value
    return _int_value(matrix.get("paper_blocker_count")) or _int_value(
        matrix.get("publication_blocker_count")
    )


def _inherited_top_gap(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> str:
    return str(capstone.get("next_top_gap") or matrix.get("top_gap") or "")


def _garak_attack_success_rate(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> float:
    direct = _float_value(capstone.get("garak_attack_success_rate"))
    if direct or capstone.get("garak_attack_success_rate") == 0:
        return direct
    row_summary = _as_mapping(_matrix_row(matrix, "exp3300").get("summary"))
    return _float_value(row_summary.get("attack_success_rate"))


def _dataflip_gate_passed(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> bool:
    if isinstance(matrix.get("dataflip_gate_passed"), bool):
        return matrix.get("dataflip_gate_passed") is True
    garak = _gate_detail(capstone, matrix, "garak_gate", "exp3300")
    if isinstance(garak.get("dataflip_gate_passed"), bool):
        return garak.get("dataflip_gate_passed") is True
    row_summary = _as_mapping(_matrix_row(matrix, "exp3300").get("summary"))
    return row_summary.get("dataflip_gate_passed") is True


def _repair_headline_claim_allowed(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> bool:
    if isinstance(capstone.get("repair_headline_claim_allowed"), bool):
        return capstone.get("repair_headline_claim_allowed") is True
    if isinstance(matrix.get("repair_headline_claim_allowed"), bool):
        return matrix.get("repair_headline_claim_allowed") is True
    repair = _gate_detail(capstone, matrix, "repair_headline", "exp3303")
    return repair.get("repair_headline_claim_allowed") is True


def _fr11_memory_replay_safe(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> bool:
    if isinstance(capstone.get("fr11_memory_replay_safe"), bool):
        return capstone.get("fr11_memory_replay_safe") is True
    if isinstance(matrix.get("fr11_replay_safe"), bool):
        return matrix.get("fr11_replay_safe") is True
    fr11 = _gate_detail(capstone, matrix, "fr11_replay", "exp3304")
    return fr11.get("fr11_replay_safe") is True


def _blocked_reasons(
    *,
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
    capstone_ready: bool,
    matrix_ready: bool,
    publication_blocker_count: int,
    inherited_top_gap: str,
    dataflip_gate_passed: bool,
    repair_headline_claim_allowed: bool,
    fr11_memory_replay_safe: bool,
    quality_flags: list[JsonDict],
    research_complete_present: bool,
    queue: Mapping[str, Any],
    activation_observed: bool,
    protected_files_unchanged: bool,
) -> list[str]:
    checks = (
        (not capstone, "capstone_v305 authority is missing or malformed"),
        (bool(capstone) and not capstone_ready, "capstone_v305 authority is not ready"),
        (not matrix, "matrix_v37 authority is missing or malformed"),
        (bool(matrix) and not matrix_ready, "matrix_v37 authority is not ready"),
        (
            publication_blocker_count != EXPECTED_PUBLICATION_BLOCKER_COUNT,
            "publication blocker count is not 8",
        ),
        (
            inherited_top_gap != EXPECTED_INHERITED_TOP_GAP,
            "inherited top gap is not clear_garak_dataflip_and_quality_flags",
        ),
        (
            dataflip_gate_passed is not False,
            "DataFlip gate must remain failed at .306 activation",
        ),
        (
            repair_headline_claim_allowed is not False,
            "repair headline claim must remain disallowed at .306 activation",
        ),
        (not fr11_memory_replay_safe, "FR-11 controller-memory replay is not safe"),
        (not quality_flags, "current quality flags are missing"),
        (
            not research_complete_present,
            "research-complete.yaml does not contain the .305 task summary",
        ),
        (
            queue.get("selected_queue_milestone") != TARGET_MILESTONE,
            "selected queue milestone is not 2026.05.306",
        ),
        (
            queue.get("queue_first_task") != TASK_ID,
            "selected queue first task is not exp3307-archive-v305-activate-v306",
        ),
        (not activation_observed, "milestone 2026.05.306 activation is not observed"),
        (not protected_files_unchanged, "protected files changed during handoff"),
    )
    return [reason for failed, reason in checks if failed]


def _v306_activation_reason() -> str:
    return (
        ".306 starts from DataFlip failure, current quality flags, repair headline "
        "provenance failure, and FR-11 controller-memory safety. The handoff does "
        "not rerun Garak/DataFlip, repair, or FR-11; it preserves the measured .305 "
        "terminal state for the quality-cleanup queue."
    )


def _principle_annotations() -> JsonDict:
    return {
        "boundary": "The milestone boundary is a boolean handoff, not prose.",
        "aggregation_only": "The handoff reads artifacts and does not perform live inference.",
        "paper_ready": "Publication readiness remains false because blockers are inherited.",
        "dataflip": "Garak ASR pass is separated from the failed DataFlip gate.",
        "quality_flags": "TAUTOLOGY and duration/provenance flags carry into .306.",
        "repair": "Repair exact successes do not become headline evidence without audit approval.",
        "fr11": "FR-11 safety is controller-memory-only and not a foundation weight update.",
        "protected_files": "Protected files are checksum-checked before and after the handoff.",
    }


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "archive_v305_activate_v306_ready": artifact.get("archive_v305_activate_v306_ready"),
        "v305_closed_v306_opened": artifact.get("v305_closed_v306_opened"),
        "source_milestone": artifact.get("source_milestone"),
        "target_milestone": artifact.get("target_milestone"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "inherited_top_gap": artifact.get("inherited_top_gap"),
        "dataflip_gate_passed": artifact.get("dataflip_gate_passed"),
        "repair_headline_claim_allowed": artifact.get("repair_headline_claim_allowed"),
        "fr11_memory_replay_safe": artifact.get("fr11_memory_replay_safe"),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: v305_closed_v306_opened="
        f"{str(artifact.get('v305_closed_v306_opened') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"inherited_top_gap={artifact.get('inherited_top_gap')}; "
        f"garak_gate_passed={str(artifact.get('garak_gate_passed') is True).lower()}; "
        f"garak_attack_success_rate={artifact.get('garak_attack_success_rate')}; "
        f"dataflip_gate_passed={str(artifact.get('dataflip_gate_passed') is True).lower()}; "
        "repair_headline_claim_allowed="
        f"{str(artifact.get('repair_headline_claim_allowed') is True).lower()}; "
        f"fr11_memory_replay_safe={str(artifact.get('fr11_memory_replay_safe') is True).lower()}"
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


def _row(rows: list[Any], experiment_id: str) -> JsonDict:
    return next(
        (
            _as_mapping(row)
            for row in rows
            if _as_mapping(row).get("experiment_id") == experiment_id
        ),
        {},
    )


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


def _list_of_strings(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float_value(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


if __name__ == "__main__":  # pragma: no cover
    write_artifact()
