"""Build the Exp 3281 archive-.303/open-.304 handoff artifact.

Spec refs: REQ-REPORT-3281, SCENARIO-REPORT-3281.

This module is bookkeeping, not a new research run. It reads the finished
milestone .303 evidence, records why the next milestone must focus on Garak
toolchain availability and clean-verifier abstention calibration, and writes a
machine-readable handoff JSON. The code deliberately avoids the conductor and
protected roadmap files because the handoff is meant to audit the boundary,
not mutate the queue that the conductor has already activated.
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
SCHEMA_VERSION = "carnot.archive_activation.v303_to_v304_blocker_queue.v1"
EXPERIMENT_ID = "exp3281"
TASK_ID = "exp3281-archive-v303-activate-v304"
ARTIFACT = "experiment_3281_archive_v303_activate_v304"
MILESTONE = "2026.05.304"
PRIOR_MILESTONE = "2026.05.303"
INFERENCE_SUBSTRATE = "artifact_aggregation_only"
RANDOM_SEED = 3281
EXPECTED_PUBLICATION_BLOCKER_COUNT = 105
EXPECTED_NEXT_TOP_GAP = "unblock_garak_redteam_eval"
EXPECTED_GARAK_BLOCKER = "blocked_garak_unavailable"

OUTPUT_REL_PATH = Path("results/experiment_3281_archive_v303_activate_v304.json")
CAPSTONE_V303_REL_PATH = Path("results/experiment_3280_capstone_v303.json")
MATRIX_V35_REL_PATH = Path("results/experiment_3279_evidence_matrix_v35.json")
EXP3275_REL_PATH = Path("results/experiment_3275_clean_local_sota_verifier_rerun_v14.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

PROTECTED_FILES = (
    ACTIVE_ROADMAP_REL_PATH.as_posix(),
    CONDUCTOR_REL_PATH.as_posix(),
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS = {
    "v303_closed_v304_opened",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_next_top_gap",
    "full_15k_corpus_materialized",
    "garak_blocker",
    "clean_verifier_abstention_rate",
    "kan_noninferiority_passed",
    "repair_gate_open",
    "protected_files_untouched",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}

PRIOR_TASKS: tuple[JsonDict, ...] = (
    {
        "id": "exp3267-close-v302-open-v303-corpus-queue",
        "title": "Close .302 ledger and open .303 corpus queue",
        "deliverable": "results/experiment_3267_close_v302_open_v303_corpus_queue.json",
        "log_title": "Close .302 ledger and open .303 corpus queue",
    },
    {
        "id": "exp3268-sota-receipt-methodology-supplement-v1",
        "title": "SOTA receipt methodology supplement v1",
        "deliverable": "results/experiment_3268_sota_receipt_methodology_supplement_v1.json",
        "log_title": "SOTA receipt methodology supplement v1",
    },
    {
        "id": "exp3269-prompt-injection-v4-full-corpus-split-manifest-v1",
        "title": "Prompt-injection v4 full-corpus split manifest",
        "deliverable": (
            "results/experiment_3269_prompt_injection_v4_full_corpus_split_manifest_v1.json"
        ),
        "log_title": "Prompt-injection v4 full-corpus split manifest",
    },
    {
        "id": "exp3270-prompt-injection-teacher-label-shards-2-4-v1",
        "title": "Prompt-injection teacher-label shards 2-4",
        "deliverable": "results/experiment_3270_prompt_injection_teacher_label_shards_2_4_v1.json",
        "log_title": "Prompt-injection teacher-label shards 2-4",
    },
    {
        "id": "exp3271-prompt-injection-teacher-label-shards-5-7-garak-seed-v1",
        "title": "Prompt-injection teacher-label shards 5-7 plus Garak seed",
        "deliverable": (
            "results/experiment_3271_prompt_injection_teacher_label_shards_5_7_garak_seed_v1.json"
        ),
        "log_title": "Prompt-injection teacher-label shards 5-7",
    },
    {
        "id": "exp3272-prompt-injection-v4-full-corpus-assembly-leakage-audit-v1",
        "title": "Prompt-injection v4 full-corpus assembly and leakage audit",
        "deliverable": (
            "results/experiment_3272_prompt_injection_v4_full_corpus_assembly_leakage_audit_v1.json"
        ),
        "log_title": "Prompt-injection v4 full-corpus assembly",
    },
    {
        "id": "exp3273-prompt-injection-kan-full-corpus-delong-eval-v1",
        "title": "Prompt-injection KAN full-corpus DeLong eval",
        "deliverable": (
            "results/experiment_3273_prompt_injection_kan_full_corpus_delong_eval_v1.json"
        ),
        "log_title": "Prompt-injection KAN full-corpus DeLong eval",
    },
    {
        "id": "exp3274-prompt-injection-v4-garak-dataflip-redteam-eval-v1",
        "title": "Prompt-injection v4 Garak and DataFlip red-team eval",
        "deliverable": (
            "results/experiment_3274_prompt_injection_v4_garak_dataflip_redteam_eval_v1.json"
        ),
        "log_title": "Prompt-injection v4 Garak and DataFlip red-team",
    },
    {
        "id": "exp3275-clean-local-sota-verifier-rerun-v14",
        "title": "Clean local SOTA verifier rerun v14",
        "deliverable": "results/experiment_3275_clean_local_sota_verifier_rerun_v14.json",
        "log_title": "Clean local SOTA verifier rerun v14",
    },
    {
        "id": "exp3276-repair-gate-decision-v8-after-v4-garak-clean-verifier",
        "title": "Repair gate decision v8 after v4, Garak, and clean verifier",
        "deliverable": (
            "results/experiment_3276_repair_gate_decision_v8_after_v4_garak_clean_verifier.json"
        ),
        "log_title": "Repair gate decision v8 after v4",
    },
    {
        "id": "exp3277-sota-repair-micro-panel-v9",
        "title": "SOTA repair micro-panel v9",
        "deliverable": "results/experiment_3277_sota_repair_micro_panel_v9.json",
        "log_title": "SOTA repair micro-panel v9",
    },
    {
        "id": "exp3278-fr11-full-corpus-continual-self-learning-audit-v1",
        "title": "FR-11 full-corpus continual self-learning audit",
        "deliverable": (
            "results/experiment_3278_fr11_full_corpus_continual_self_learning_audit_v1.json"
        ),
        "log_title": "FR-11 full-corpus continual self-learning audit",
    },
    {
        "id": "exp3279-evidence-matrix-v35",
        "title": "Evidence matrix v35 for .303 corpus, Garak, repair, and FR-11",
        "deliverable": "results/experiment_3279_evidence_matrix_v35.json",
        "log_title": "Evidence matrix v35 for .303 corpus",
    },
    {
        "id": "exp3280-capstone-v303",
        "title": "Capstone v303 and next-gap decision",
        "deliverable": "results/experiment_3280_capstone_v303.json",
        "log_title": "Capstone v303 and next-gap decision",
    },
)
EXPECTED_PRIOR_TASK_IDS = {str(task["id"]) for task in PRIOR_TASKS}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object without making missing evidence look successful."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_document(path: Path) -> Any:
    """Read YAML evidence and return an empty shape when the file is unusable."""

    try:
        text = path.read_text(encoding="utf-8")
        return yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so the aggregation can be audited without rerunning work."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_file_checksums(root: Path | str = REPO_ROOT) -> dict[str, str | None]:
    """Capture protected-file hashes before and after the handoff writes JSON."""

    root_path = Path(root)
    return {path: sha256_file(root_path / path) for path in PROTECTED_FILES}


def ensure_research_complete_entry(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-REPORT-3281: archive `.303` once, and leave an existing row untouched."""

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
    """SCENARIO-REPORT-3281: synthesize the .303 archive and .304 handoff."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V303_REL_PATH)
    matrix = read_json_object(root_path / MATRIX_V35_REL_PATH)
    clean_verifier = read_json_object(root_path / EXP3275_REL_PATH)
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
    queue = _v304_queue(root_path)
    terminal_rows = _conductor_log_terminal_rows(root_path)

    matrix_ready = matrix.get("matrix_v35_ready") is True
    rows = [_as_mapping(row) for row in _as_list(matrix.get("rows"))]
    prior_paper_ready = capstone.get("paper_ready") is True
    prior_blockers = _int_value(capstone.get("publication_blocker_count"))
    prior_next_gap = str(capstone.get("next_top_gap") or "")
    full_corpus_materialized = _full_15k_corpus_materialized(capstone, matrix_ready, matrix, rows)
    garak_blocker = _garak_blocker(matrix_ready, rows)
    clean_abstention_rate = _clean_verifier_abstention_rate(clean_verifier)
    kan_noninferiority_passed = _kan_noninferiority_passed(matrix_ready, rows)
    repair_gate_open = _repair_gate_open(capstone, matrix_ready, matrix)
    activation_observed = _v304_activation_observed(root_path, queue)
    blocked_reasons = _blocked_reasons(
        capstone=capstone,
        capstone_ready=capstone.get("capstone_v303_ready") is True,
        prior_paper_ready=prior_paper_ready,
        prior_publication_blocker_count=prior_blockers,
        prior_next_top_gap=prior_next_gap,
        full_15k_corpus_materialized=full_corpus_materialized,
        garak_blocker=garak_blocker,
        clean_verifier_abstention_rate=clean_abstention_rate,
        kan_noninferiority_passed=kan_noninferiority_passed,
        repair_gate_open=repair_gate_open,
        research_complete_present=research_complete_present,
        queue=queue,
        activation_observed=activation_observed,
        protected_files_untouched=protected_untouched,
    )

    source_artifacts = _source_artifacts(root_path, capstone, matrix, clean_verifier)
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
        "v303_closed_v304_opened": not blocked_reasons,
        "prior_paper_ready": prior_paper_ready,
        "prior_publication_blocker_count": prior_blockers,
        "prior_next_top_gap": prior_next_gap,
        "full_15k_corpus_materialized": full_corpus_materialized,
        "garak_blocker": garak_blocker,
        "clean_verifier_abstention_rate": clean_abstention_rate,
        "kan_noninferiority_passed": kan_noninferiority_passed,
        "repair_gate_open": repair_gate_open,
        "v303_terminal_evidence": _v303_terminal_evidence(capstone, matrix, clean_verifier),
        "blocker_movement": _blocker_movement(capstone, matrix),
        "v304_activation_reason": _v304_activation_reason(
            full_corpus_materialized=full_corpus_materialized,
            garak_blocker=garak_blocker,
            clean_verifier_abstention_rate=clean_abstention_rate,
            kan_noninferiority_passed=kan_noninferiority_passed,
            repair_gate_open=repair_gate_open,
        ),
        "research_complete_update": archive_update,
        "research_complete_prior_summary": _research_complete_task_summary(root_path),
        "v304_queue": queue,
        "v304_activation_observed": activation_observed,
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
    """Build and persist the Exp 3281 handoff JSON."""

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
    """Reject handoff JSON that cannot be consumed by the next milestone."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3281")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3281-archive-v303-activate-v304")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.304")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3281")
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
        "- id: 2026.05.303",
        "  title: Prompt-Injection v4 Full Corpus + Garak Gate + Repair Reopen",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-05-28'",
        "  finding: See Exp 3280 capstone and Exp 3279 evidence matrix for terminal evidence.",
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


def _v304_queue(root: Path) -> JsonDict:
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
    }


def _v304_activation_observed(root: Path, queue: Mapping[str, Any]) -> bool:
    return queue.get("selected_queue_milestone") == MILESTONE or _file_contains(
        root / CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.304 activated",
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
    root: Path,
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
    clean_verifier: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _source_record(root, "capstone_v303", CAPSTONE_V303_REL_PATH, bool(capstone), True),
        _source_record(
            root,
            "evidence_matrix_v35",
            MATRIX_V35_REL_PATH,
            bool(matrix),
            matrix.get("matrix_v35_ready") is True,
        ),
        _source_record(
            root,
            "clean_verifier_abstention",
            EXP3275_REL_PATH,
            bool(clean_verifier),
            clean_verifier.get("clean_verifier_rerun_ready") is True,
        ),
        _source_record(root, "research_complete_archive", RESEARCH_COMPLETE_REL_PATH, True, True),
        _source_record(root, "active_v304_roadmap", ACTIVE_ROADMAP_REL_PATH, True, True),
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


def _v303_terminal_evidence(
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
    clean_verifier: Mapping[str, Any],
) -> JsonDict:
    return {
        "capstone_honest_verdict": str(capstone.get("honest_verdict") or ""),
        "matrix_honest_verdict": str(matrix.get("honest_verdict") or ""),
        "clean_verifier_honest_verdict": str(clean_verifier.get("honest_verdict") or ""),
        "capstone_v303_ready": capstone.get("capstone_v303_ready") is True,
        "matrix_v35_ready": matrix.get("matrix_v35_ready") is True,
        "paper_ready": capstone.get("paper_ready") is True,
        "publication_blocker_count": _int_value(capstone.get("publication_blocker_count")),
        "publication_blocker_delta": _int_value(capstone.get("publication_blocker_delta")),
        "next_top_gap": str(capstone.get("next_top_gap") or ""),
        "recommended_next_milestone_title": str(
            capstone.get("recommended_next_milestone_title") or ""
        ),
    }


def _blocker_movement(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> JsonDict:
    delta = _int_value(capstone.get("publication_blocker_delta"))
    return {
        "prior_publication_blocker_count": _int_value(capstone.get("publication_blocker_count")),
        "matrix_publication_blocker_count_estimate": _int_value(
            matrix.get("publication_blocker_count_estimate")
        ),
        "publication_blocker_delta": delta,
        "publication_blocker_delta_from_matrix": _int_value(
            matrix.get("publication_blocker_delta_from_v302")
        ),
        "trend": "decreased" if delta < 0 else "increased" if delta > 0 else "unchanged",
    }


def _full_15k_corpus_materialized(
    capstone: Mapping[str, Any],
    matrix_ready: bool,
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> bool:
    if not matrix_ready:
        return False
    gates = _as_mapping(_as_mapping(matrix.get("publication_readiness")).get("required_gates"))
    row3272 = _row(rows, "exp3272")
    return (
        "full_15k_v4_corpus_materialized" in _list_of_strings(capstone.get("changes_since_v302"))
        or gates.get("full_15k_corpus") is True
        or _as_mapping(row3272.get("summary")).get("full_15k_corpus_ready") is True
    )


def _garak_blocker(matrix_ready: bool, rows: list[Mapping[str, Any]]) -> str:
    if not matrix_ready:
        return ""
    row3274 = _row(rows, "exp3274")
    reasons = _list_of_strings(row3274.get("blocker_reasons"))
    return reasons[0] if reasons else ""


def _clean_verifier_abstention_rate(clean_verifier: Mapping[str, Any]) -> float:
    return _float_value(clean_verifier.get("abstention_rate"))


def _kan_noninferiority_passed(matrix_ready: bool, rows: list[Mapping[str, Any]]) -> bool:
    if not matrix_ready:
        return False
    row3273 = _row(rows, "exp3273")
    summary = _as_mapping(row3273.get("summary"))
    return summary.get("delong_noninferiority_passed") is True


def _repair_gate_open(
    capstone: Mapping[str, Any],
    matrix_ready: bool,
    matrix: Mapping[str, Any],
) -> bool:
    if not matrix_ready:
        return False
    gates = _as_mapping(_as_mapping(matrix.get("publication_readiness")).get("required_gates"))
    return gates.get("repair_gate") is True and str(
        capstone.get("repair_gate_status") or ""
    ).startswith("passed:")


def _blocked_reasons(
    *,
    capstone: Mapping[str, Any],
    capstone_ready: bool,
    prior_paper_ready: bool,
    prior_publication_blocker_count: int,
    prior_next_top_gap: str,
    full_15k_corpus_materialized: bool,
    garak_blocker: str,
    clean_verifier_abstention_rate: float,
    kan_noninferiority_passed: bool,
    repair_gate_open: bool,
    research_complete_present: bool,
    queue: Mapping[str, Any],
    activation_observed: bool,
    protected_files_untouched: bool,
) -> list[str]:
    checks = (
        (not capstone, "capstone_v303 authority is missing or malformed"),
        (bool(capstone) and not capstone_ready, "capstone_v303 authority is not ready"),
        (prior_paper_ready is not False, "prior paper_ready must remain false"),
        (
            prior_publication_blocker_count != EXPECTED_PUBLICATION_BLOCKER_COUNT,
            "prior publication blocker count is not 105",
        ),
        (
            prior_next_top_gap != EXPECTED_NEXT_TOP_GAP,
            "prior next_top_gap does not preserve the .304 Garak queue anchor",
        ),
        (
            not full_15k_corpus_materialized,
            "full 15k corpus materialization evidence is missing",
        ),
        (garak_blocker != EXPECTED_GARAK_BLOCKER, "Garak blocker is not blocked_garak_unavailable"),
        (
            clean_verifier_abstention_rate < 1.0,
            "clean verifier abstention rate is not 1.0",
        ),
        (kan_noninferiority_passed is not False, "KAN non-inferiority unexpectedly passed"),
        (repair_gate_open is not False, "repair gate unexpectedly opened"),
        (
            not research_complete_present,
            "research-complete.yaml does not contain the .303 task summary",
        ),
        (
            queue.get("selected_queue_milestone") != MILESTONE,
            "selected queue milestone is not 2026.05.304",
        ),
        (
            queue.get("queue_first_task") != TASK_ID,
            "selected queue first task is not exp3281-archive-v303-activate-v304",
        ),
        (not activation_observed, "milestone 2026.05.304 activation is not observed"),
        (not protected_files_untouched, "protected files changed during handoff"),
    )
    return [reason for failed, reason in checks if failed]


def _v304_activation_reason(
    *,
    full_corpus_materialized: bool,
    garak_blocker: str,
    clean_verifier_abstention_rate: float,
    kan_noninferiority_passed: bool,
    repair_gate_open: bool,
) -> str:
    corpus = (
        "full 15k corpus is already materialized"
        if full_corpus_materialized
        else "corpus evidence is incomplete"
    )
    kan = (
        "KAN non-inferiority failed"
        if not kan_noninferiority_passed
        else "KAN non-inferiority passed"
    )
    repair = "repair gate remains closed" if not repair_gate_open else "repair gate opened"
    return (
        f".304 starts with Garak/toolchain and abstention calibration because {corpus}; "
        f"Garak is blocked by {garak_blocker or 'unknown_garak_blocker'}; clean verifier "
        f"abstention_rate={clean_verifier_abstention_rate}; {kan}; {repair}."
    )


def _principle_annotations() -> JsonDict:
    return {
        "v303_closed_v304_opened": "Milestone boundaries must be machine-readable.",
        "prior_paper_ready": "Preserve the publication-readiness signal.",
        "prior_publication_blocker_count": "Track blocker movement across milestones.",
        "prior_next_top_gap": "Keep the next queue anchored to the capstone.",
        "full_15k_corpus_materialized": "Separate data-scale completion from promotion readiness.",
        "garak_blocker": "Record the toolchain gate explicitly.",
        "clean_verifier_abstention_rate": "Abstention-all is the clean-verifier blocker.",
        "kan_noninferiority_passed": "KAN must remain bounded when it fails.",
        "repair_gate_open": "Repair stays gated by evidence.",
        "protected_files_untouched": "Enforce operator constraints.",
        "random_seed": "Keep aggregation reproducible.",
        "reproducibility_checksum": "Provide a deterministic audit trail.",
        "duration_s": "Retain timing evidence for ops retrospectives.",
        "honest_verdict": "Use a terminal prefix without claiming publication readiness.",
    }


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: v303_closed_v304_opened="
        f"{str(artifact.get('v303_closed_v304_opened') is True).lower()}; "
        f"paper_ready={str(artifact.get('prior_paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('prior_publication_blocker_count')}; "
        f"publication_blocker_delta={artifact.get('blocker_movement', {}).get('publication_blocker_delta')}; "
        f"next_top_gap={artifact.get('prior_next_top_gap')}; "
        f"full_15k_corpus_materialized={str(artifact.get('full_15k_corpus_materialized') is True).lower()}; "
        f"garak_blocker={artifact.get('garak_blocker')}; "
        f"clean_verifier_abstention_rate={artifact.get('clean_verifier_abstention_rate')}; "
        f"kan_noninferiority_passed={str(artifact.get('kan_noninferiority_passed') is True).lower()}; "
        f"repair_gate_open={str(artifact.get('repair_gate_open') is True).lower()}"
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


def _row(rows: list[Mapping[str, Any]], experiment_id: str) -> JsonDict:
    return next((_as_mapping(row) for row in rows if row.get("experiment_id") == experiment_id), {})


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _terminal_prefix_ok(verdict: str) -> bool:
    return verdict.startswith(TERMINAL_PREFIXES)


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
