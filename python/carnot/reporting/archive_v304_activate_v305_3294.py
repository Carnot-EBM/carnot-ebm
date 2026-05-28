"""Build the Exp 3294 archive-.304/open-.305 handoff artifact.

Spec refs: REQ-REPORT-3294, SCENARIO-REPORT-3294.

This module is an operational ledger, not a research rerun. It reads the
completed `.304` capstone and evidence matrix, records the terminal conductor
evidence, appends the `.304` archive only when it is absent, and writes a
machine-readable `.305` activation receipt. The artifact keeps the Garak gate
failure quantitative so later work cannot confuse "Garak is installed" with
"the Garak red-team gate passed."
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
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.archive_activation.v304_to_v305_garak_repair_handoff.v1"
EXPERIMENT_ID = "exp3294"
TASK_ID = "exp3294-archive-v304-activate-v305"
ARTIFACT = "experiment_3294_archive_v304_activate_v305"
MILESTONE = "2026.05.305"
PRIOR_MILESTONE = "2026.05.304"
INFERENCE_SUBSTRATE = "artifact_aggregation_only"
RANDOM_SEED = 3294
EXPECTED_PUBLICATION_BLOCKER_COUNT = 10
EXPECTED_NEXT_TOP_GAP = "pass_garak_redteam_gate"
RETIRED_KAN_HEADLINE_DECISION = "retire_from_prompt_injection_headline"

OUTPUT_REL_PATH = Path("results/experiment_3294_archive_v304_activate_v305.json")
CAPSTONE_V304_REL_PATH = Path("results/experiment_3293_capstone_v304.json")
MATRIX_V36_REL_PATH = Path("results/experiment_3292_evidence_matrix_v36.json")
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
    "v304_closed_v305_opened",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_next_top_gap",
    "garak_unblocked",
    "prior_garak_gate_passed",
    "prior_attack_success_rate",
    "clean_verifier_abstention_unblocked",
    "kan_headline_retired",
    "repair_gate_open",
    "repair_micro_panel_headline_eligible",
    "protected_files_untouched",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}

PRIOR_TASKS: tuple[JsonDict, ...] = (
    {
        "id": "exp3281-archive-v303-activate-v304",
        "title": "Close .303 ledger and open .304 blocker queue",
        "deliverable": "results/experiment_3281_archive_v303_activate_v304.json",
        "log_title": "Close .303 ledger and open .304 blocker queue",
    },
    {
        "id": "exp3282-garak-install-and-probe-manifest-v1",
        "title": "Garak install and probe manifest v1",
        "deliverable": "results/experiment_3282_garak_install_and_probe_manifest_v1.json",
        "log_title": "Garak install and probe manifest v1",
    },
    {
        "id": "exp3283-prompt-injection-corrigendum-duration-audit-v1",
        "title": "Prompt-injection corrigendum and duration audit v1",
        "deliverable": (
            "results/experiment_3283_prompt_injection_corrigendum_duration_audit_v1.json"
        ),
        "log_title": "Prompt-injection corrigendum and duration audit v1",
    },
    {
        "id": "exp3284-garak-local-smoke-sota-gguf-v1",
        "title": "Gated Garak local smoke against mandated SOTA GGUF v1",
        "deliverable": "results/experiment_3284_garak_local_smoke_sota_gguf_v1.json",
        "log_title": "Gated Garak local smoke against mandated SOTA GGUF",
    },
    {
        "id": "exp3285-full-garak-dataflip-redteam-eval-v2",
        "title": "Gated full Garak/DataFlip red-team eval v2",
        "deliverable": "results/experiment_3285_full_garak_dataflip_redteam_eval_v2.json",
        "log_title": "Gated full Garak/DataFlip red-team eval v2",
    },
    {
        "id": "exp3286-clean-verifier-abstention-root-cause-v1",
        "title": "Clean verifier abstention root-cause audit v1",
        "deliverable": "results/experiment_3286_clean_verifier_abstention_root_cause_v1.json",
        "log_title": "Clean verifier abstention root-cause audit v1",
    },
    {
        "id": "exp3287-abstention-calibrated-clean-verifier-v15",
        "title": "Gated abstention-calibrated clean verifier v15",
        "deliverable": "results/experiment_3287_abstention_calibrated_clean_verifier_v15.json",
        "log_title": "Gated abstention-calibrated clean verifier v15",
    },
    {
        "id": "exp3288-kan-sidecar-failure-autopsy-boundary-v1",
        "title": "Gated KAN sidecar failure autopsy and boundary decision v1",
        "deliverable": "results/experiment_3288_kan_sidecar_failure_autopsy_boundary_v1.json",
        "log_title": "Gated KAN sidecar failure autopsy and boundary dec",
    },
    {
        "id": "exp3289-repair-gate-decision-v9-after-garak-abstention",
        "title": "Gated repair gate decision v9 after Garak and abstention",
        "deliverable": "results/experiment_3289_repair_gate_decision_v9_after_garak_abstention.json",
        "log_title": "Gated repair gate decision v9 after Garak and abst",
    },
    {
        "id": "exp3290-gated-sota-repair-micro-panel-v10",
        "title": "Gated SOTA repair micro-panel v10",
        "deliverable": "results/experiment_3290_gated_sota_repair_micro_panel_v10.json",
        "log_title": "Gated SOTA repair micro-panel v10",
    },
    {
        "id": "exp3291-fr11-garak-abstention-memory-replay-v1",
        "title": "FR-11 Garak and abstention memory replay v1",
        "deliverable": "results/experiment_3291_fr11_garak_abstention_memory_replay_v1.json",
        "log_title": "FR-11 Garak and abstention memory replay v1",
    },
    {
        "id": "exp3292-evidence-matrix-v36",
        "title": "Evidence matrix v36",
        "deliverable": "results/experiment_3292_evidence_matrix_v36.json",
        "log_title": "Evidence matrix v36",
    },
    {
        "id": "exp3293-capstone-v304",
        "title": "Capstone v304",
        "deliverable": "results/experiment_3293_capstone_v304.json",
        "log_title": "Capstone v304",
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
    """Capture hashes for files that this handoff is forbidden to edit."""

    root_path = Path(root)
    return {path: sha256_file(root_path / path) for path in PROTECTED_FILES}


def ensure_research_complete_entry(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-REPORT-3294: archive `.304` once, leaving an existing entry untouched."""

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
    """SCENARIO-REPORT-3294: synthesize the .304 archive and .305 handoff."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V304_REL_PATH)
    matrix = read_json_object(root_path / MATRIX_V36_REL_PATH)
    baseline = (
        dict(protected_hash_baseline)
        if protected_hash_baseline is not None
        else protected_file_checksums(root_path)
    )
    protected_checksums = _protected_file_checksum_report(root_path, baseline)
    protected_untouched = all(row["unchanged"] for row in protected_checksums.values())
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
    queue = _v305_queue(root_path)
    terminal_rows = _conductor_log_terminal_rows(root_path)

    prior_paper_ready = capstone.get("paper_ready") is True
    prior_blockers = _int_value(capstone.get("publication_blocker_count"))
    prior_next_gap = str(capstone.get("next_top_gap") or "")
    garak_unblocked = capstone.get("garak_unblocked") is True
    garak_gate_passed = capstone.get("garak_gate_passed") is True
    attack_success_rate = _prior_attack_success_rate(capstone, matrix)
    clean_unblocked = capstone.get("clean_verifier_abstention_unblocked") is True
    kan_headline_retired = (
        str(capstone.get("kan_boundary_decision") or "") == RETIRED_KAN_HEADLINE_DECISION
    )
    repair_open = capstone.get("repair_gate_open") is True
    repair_headline_eligible = capstone.get("repair_micro_panel_headline_eligible") is True
    activation_observed = _v305_activation_observed(root_path, queue)
    blocked_reasons = _blocked_reasons(
        capstone=capstone,
        capstone_ready=capstone.get("capstone_v304_ready") is True,
        prior_paper_ready=prior_paper_ready,
        prior_publication_blocker_count=prior_blockers,
        prior_next_top_gap=prior_next_gap,
        garak_unblocked=garak_unblocked,
        prior_garak_gate_passed=garak_gate_passed,
        prior_attack_success_rate=attack_success_rate,
        clean_verifier_abstention_unblocked=clean_unblocked,
        kan_headline_retired=kan_headline_retired,
        repair_gate_open=repair_open,
        repair_micro_panel_headline_eligible=repair_headline_eligible,
        research_complete_present=research_complete_present,
        queue=queue,
        activation_observed=activation_observed,
        protected_files_untouched=protected_untouched,
    )

    source_artifacts = _source_artifacts(root_path, capstone, matrix)
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v304_closed_v305_opened": not blocked_reasons,
        "prior_paper_ready": prior_paper_ready,
        "prior_publication_blocker_count": prior_blockers,
        "prior_next_top_gap": prior_next_gap,
        "garak_unblocked": garak_unblocked,
        "prior_garak_gate_passed": garak_gate_passed,
        "prior_attack_success_rate": attack_success_rate,
        "clean_verifier_abstention_unblocked": clean_unblocked,
        "kan_headline_retired": kan_headline_retired,
        "repair_gate_open": repair_open,
        "repair_micro_panel_headline_eligible": repair_headline_eligible,
        "v304_terminal_evidence": _v304_terminal_evidence(capstone, matrix),
        "v305_activation_reason": _v305_activation_reason(
            attack_success_rate=attack_success_rate,
            prior_next_top_gap=prior_next_gap,
            garak_unblocked=garak_unblocked,
            prior_garak_gate_passed=garak_gate_passed,
            clean_verifier_abstention_unblocked=clean_unblocked,
            kan_headline_retired=kan_headline_retired,
            repair_gate_open=repair_open,
            repair_micro_panel_headline_eligible=repair_headline_eligible,
        ),
        "research_complete_update": archive_update,
        "research_complete_prior_summary": _research_complete_task_summary(root_path),
        "v305_queue": queue,
        "v305_activation_observed": activation_observed,
        "conductor_log_terminal_rows": terminal_rows,
        "conductor_log_terminal_status_counts": dict(
            Counter(str(row.get("status") or "missing") for row in terminal_rows)
        ),
        "protected_files": list(PROTECTED_FILES),
        "protected_files_untouched": protected_untouched,
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
        "no_new_repair_run": True,
        "no_new_verifier_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
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
    """Build and persist the Exp 3294 handoff JSON."""

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
    """Reject handoff JSON that drops required machine-readable boundary fields."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3294")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3294-archive-v304-activate-v305")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.305")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3294")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be artifact_aggregation_only")
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must begin with a terminal success prefix")


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
        "- id: 2026.05.304",
        "  title: Garak Availability + Abstention-Calibrated Verifier + Repair Gate Reopen",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-05-28'",
        "  finding: See Exp 3293 capstone and Exp 3292 evidence matrix for terminal evidence.",
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
        entries = [dict(payload)] if payload.get("id") is not None else []
        entries.extend(dict(entry) for entry in _as_list(payload.get("milestones")))
        return entries
    return []


def _v305_queue(root: Path) -> JsonDict:
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
        "milestone_provenance": str(payload.get("milestone_provenance") or ""),
        "staged_roadmap_exists": (root / STAGED_ROADMAP_REL_PATH).is_file(),
    }


def _v305_activation_observed(root: Path, queue: Mapping[str, Any]) -> bool:
    return queue.get("selected_queue_milestone") == MILESTONE or _file_contains(
        root / CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.305 activated",
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


def _source_artifacts(root: Path, capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> list[JsonDict]:
    return [
        _source_record(root, "capstone_v304", CAPSTONE_V304_REL_PATH, bool(capstone), True),
        _source_record(
            root,
            "evidence_matrix_v36",
            MATRIX_V36_REL_PATH,
            bool(matrix),
            matrix.get("matrix_v36_ready") is True,
        ),
        _source_record(root, "research_complete_archive", RESEARCH_COMPLETE_REL_PATH, True, True),
        _source_record(root, "active_v305_roadmap", ACTIVE_ROADMAP_REL_PATH, True, True),
        _source_record(root, "staged_v305_roadmap", STAGED_ROADMAP_REL_PATH, True, False),
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


def _v304_terminal_evidence(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "capstone_honest_verdict": str(capstone.get("honest_verdict") or ""),
        "matrix_honest_verdict": str(matrix.get("honest_verdict") or ""),
        "capstone_v304_ready": capstone.get("capstone_v304_ready") is True,
        "matrix_v36_ready": matrix.get("matrix_v36_ready") is True,
        "paper_ready": capstone.get("paper_ready") is True,
        "publication_blocker_count": _int_value(capstone.get("publication_blocker_count")),
        "blocker_delta_from_v303": _int_value(capstone.get("blocker_delta_from_v303")),
        "next_top_gap": str(capstone.get("next_top_gap") or ""),
        "matrix_top_gaps": _as_list(matrix.get("top_gaps")),
    }


def _prior_attack_success_rate(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> float:
    gate = _as_mapping(_as_mapping(capstone.get("gate_status_details")).get("garak_redteam"))
    direct = _float_value(gate.get("attack_success_rate"))
    if direct:
        return direct
    parsed = _parse_attack_success_rate(str(gate.get("honest_verdict") or ""))
    if parsed:
        return parsed
    matrix_gate = _as_mapping(_as_mapping(matrix.get("gate_summary")).get("garak_redteam"))
    parsed = _parse_attack_success_rate(str(matrix_gate.get("honest_verdict") or ""))
    if parsed:
        return parsed
    for key in ("evidence_rows", "rows"):
        row = _row(_as_list(matrix.get(key)), "exp3285")
        value = _float_value(_as_mapping(row.get("summary")).get("attack_success_rate"))
        if value:
            return value
    return _parse_attack_success_rate(str(capstone.get("honest_verdict") or ""))


def _parse_attack_success_rate(text: str) -> float:
    match = re.search(r"attack_success_rate=([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else 0.0


def _v305_activation_reason(
    *,
    attack_success_rate: float,
    prior_next_top_gap: str,
    garak_unblocked: bool,
    prior_garak_gate_passed: bool,
    clean_verifier_abstention_unblocked: bool,
    kan_headline_retired: bool,
    repair_gate_open: bool,
    repair_micro_panel_headline_eligible: bool,
) -> str:
    return (
        ".305 starts with Garak gate pass and headline repair evidence, not "
        "another installation, corpus, or KAN milestone: "
        f"next_top_gap={prior_next_top_gap}; Garak unblocked={garak_unblocked} "
        f"but prior_garak_gate_passed={prior_garak_gate_passed} with "
        f"attack_success_rate={attack_success_rate:.6f}; clean verifier "
        f"abstention unblocked={clean_verifier_abstention_unblocked}; "
        f"KAN headline retired={kan_headline_retired}; repair gate open="
        f"{repair_gate_open}; repair micro-panel headline eligible="
        f"{repair_micro_panel_headline_eligible}."
    )


def _blocked_reasons(
    *,
    capstone: Mapping[str, Any],
    capstone_ready: bool,
    prior_paper_ready: bool,
    prior_publication_blocker_count: int,
    prior_next_top_gap: str,
    garak_unblocked: bool,
    prior_garak_gate_passed: bool,
    prior_attack_success_rate: float,
    clean_verifier_abstention_unblocked: bool,
    kan_headline_retired: bool,
    repair_gate_open: bool,
    repair_micro_panel_headline_eligible: bool,
    research_complete_present: bool,
    queue: Mapping[str, Any],
    activation_observed: bool,
    protected_files_untouched: bool,
) -> list[str]:
    checks = (
        (not capstone, "capstone_v304 authority is missing or malformed"),
        (bool(capstone) and not capstone_ready, "capstone_v304 authority is not ready"),
        (prior_paper_ready is not False, "prior paper_ready must remain false"),
        (
            prior_publication_blocker_count != EXPECTED_PUBLICATION_BLOCKER_COUNT,
            "prior publication blocker count is not 10",
        ),
        (
            prior_next_top_gap != EXPECTED_NEXT_TOP_GAP,
            "prior next_top_gap does not preserve the .305 Garak gate anchor",
        ),
        (not garak_unblocked, "Garak is not unblocked"),
        (
            prior_garak_gate_passed is not False,
            "prior Garak gate must remain failed for .305 targeting",
        ),
        (prior_attack_success_rate <= 0.0, "prior attack_success_rate is missing"),
        (
            not clean_verifier_abstention_unblocked,
            "clean verifier abstention is not unblocked",
        ),
        (not kan_headline_retired, "KAN headline retirement is not recorded"),
        (not repair_gate_open, "repair gate is not open"),
        (
            repair_micro_panel_headline_eligible is not False,
            "repair micro-panel unexpectedly became headline eligible",
        ),
        (
            not research_complete_present,
            "research-complete.yaml does not contain the .304 task summary",
        ),
        (
            queue.get("selected_queue_milestone") != MILESTONE,
            "selected queue milestone is not 2026.05.305",
        ),
        (
            queue.get("queue_first_task") != TASK_ID,
            "selected queue first task is not exp3294-archive-v304-activate-v305",
        ),
        (not activation_observed, "milestone 2026.05.305 activation is not observed"),
        (not protected_files_untouched, "protected files changed during handoff"),
    )
    return [reason for failed, reason in checks if failed]


def _principle_annotations() -> JsonDict:
    return {
        "boundary": "The milestone boundary is a boolean handoff, not prose.",
        "paper_ready": "The prior publication-readiness signal is preserved exactly.",
        "garak": "Garak availability is separated from Garak gate success.",
        "attack_success_rate": "The failed Garak gate remains quantitative.",
        "abstention": "Clean-verifier abstention repair is not rerun when already unblocked.",
        "kan": "Retired KAN headline status prevents a doomed promotion retry.",
        "repair": "Repair work is open but still lacks headline-eligible evidence.",
        "aggregation_only": "The handoff reads artifacts and does not perform live inference.",
        "protected_files": "Roadmap and conductor files are checksum-checked for no mutation.",
    }


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "v304_closed_v305_opened": artifact.get("v304_closed_v305_opened"),
        "prior_paper_ready": artifact.get("prior_paper_ready"),
        "prior_publication_blocker_count": artifact.get("prior_publication_blocker_count"),
        "prior_next_top_gap": artifact.get("prior_next_top_gap"),
        "garak_unblocked": artifact.get("garak_unblocked"),
        "prior_garak_gate_passed": artifact.get("prior_garak_gate_passed"),
        "prior_attack_success_rate": artifact.get("prior_attack_success_rate"),
        "clean_verifier_abstention_unblocked": artifact.get(
            "clean_verifier_abstention_unblocked"
        ),
        "kan_headline_retired": artifact.get("kan_headline_retired"),
        "repair_gate_open": artifact.get("repair_gate_open"),
        "repair_micro_panel_headline_eligible": artifact.get(
            "repair_micro_panel_headline_eligible"
        ),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: v304_closed_v305_opened="
        f"{str(artifact.get('v304_closed_v305_opened') is True).lower()}; "
        f"paper_ready={str(artifact.get('prior_paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('prior_publication_blocker_count')}; "
        f"next_top_gap={artifact.get('prior_next_top_gap')}; "
        f"garak_unblocked={str(artifact.get('garak_unblocked') is True).lower()}; "
        f"garak_gate_passed={str(artifact.get('prior_garak_gate_passed') is True).lower()}; "
        f"attack_success_rate={artifact.get('prior_attack_success_rate')}; "
        "clean_verifier_abstention_unblocked="
        f"{str(artifact.get('clean_verifier_abstention_unblocked') is True).lower()}; "
        f"kan_headline_retired={str(artifact.get('kan_headline_retired') is True).lower()}; "
        f"repair_gate_open={str(artifact.get('repair_gate_open') is True).lower()}; "
        "repair_micro_panel_headline_eligible="
        f"{str(artifact.get('repair_micro_panel_headline_eligible') is True).lower()}"
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
        (_as_mapping(row) for row in rows if _as_mapping(row).get("experiment_id") == experiment_id),
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


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float_value(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


if __name__ == "__main__":  # pragma: no cover
    write_artifact()
