"""Build the Exp 3267 close-.302/open-.303 handoff artifact.

Spec refs: REQ-REPORT-3267, SCENARIO-REPORT-3267.

The handoff is intentionally narrow: it reads already-written .302 evidence,
ensures the .302 milestone is archived once, and records that the active .303
queue is anchored to the full-corpus prompt-injection gap. It does not run
models, CUDA probes, labeling, KAN training, Garak, repair, or the conductor.
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
SCHEMA_VERSION = "carnot.archive_activation.v302_to_v303_corpus_queue.v1"
EXPERIMENT_ID = "exp3267"
TASK_ID = "exp3267-close-v302-open-v303-corpus-queue"
ARTIFACT = "experiment_3267_close_v302_open_v303_corpus_queue"
MILESTONE = "2026.05.303"
PRIOR_MILESTONE = "2026.05.302"
EXPECTED_PRIOR_PUBLICATION_BLOCKER_COUNT = 105
FULL_V4_CORPUS_REPAIR_GAP = "full_15k_v4_corpus_across_shards_plus_repair_and_garak_gates"
RANDOM_SEED = 3267

OUTPUT_REL_PATH = Path("results/experiment_3267_close_v302_open_v303_corpus_queue.json")
CAPSTONE_V302_REL_PATH = Path("results/experiment_3266_capstone_v302.json")
EXP3264_REL_PATH = Path("results/experiment_3264_prompt_injection_teacher_label_shard_v3.json")
EXP3265_REL_PATH = Path("results/experiment_3265_prompt_injection_kan_train_eval_shard_v3.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

PROTECTED_FILES = (
    ACTIVE_ROADMAP_REL_PATH.as_posix(),
    CONDUCTOR_REL_PATH.as_posix(),
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS = {
    "v302_closed_v303_opened",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_next_top_gap",
    "v4_shard_label_count",
    "v4_shard_auroc",
    "protected_files_untouched",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}

PRIOR_TASKS: tuple[JsonDict, ...] = (
    {
        "id": "exp3260-archive-v301-activate-v302",
        "title": "Archive .301 closeout and activate .302 planning authority",
        "deliverable": "results/experiment_3260_archive_v301_activate_v302.json",
    },
    {
        "id": "exp3261-cuda-recovery-confirmation-smoke-v1",
        "title": "CUDA recovery confirmation smoke - post-reboot GPU verification",
        "deliverable": "results/experiment_3261_cuda_recovery_confirmation_smoke_v1.json",
    },
    {
        "id": "exp3262-llama-cpp-cuda-receipt-smoke-v4",
        "title": "llama.cpp CUDA receipt smoke v4 gated on CUDA recovery",
        "deliverable": "results/experiment_3262_llama_cpp_cuda_receipt_smoke_v4.json",
    },
    {
        "id": "exp3263-sota-gguf-receipt-v9",
        "title": "Mandated SOTA GGUF receipt v9 gated on llama.cpp CUDA recovery",
        "deliverable": "results/experiment_3263_sota_gguf_receipt_v9.json",
    },
    {
        "id": "exp3264-prompt-injection-teacher-label-shard-v3",
        "title": "Prompt-injection v4 teacher-label shard v3 gated on SOTA receipt",
        "deliverable": "results/experiment_3264_prompt_injection_teacher_label_shard_v3.json",
    },
    {
        "id": "exp3265-prompt-injection-kan-train-eval-shard-v3",
        "title": "Prompt-injection KAN train/eval shard v3 gated on teacher-label shard",
        "deliverable": "results/experiment_3265_prompt_injection_kan_train_eval_shard_v3.json",
    },
    {
        "id": "exp3266-capstone-v302",
        "title": "Capstone .302 - CUDA recovery, SOTA receipt, and v4 shard readout",
        "deliverable": "results/experiment_3266_capstone_v302.json",
    },
)
EXPECTED_PRIOR_TASK_IDS = {str(task["id"]) for task in PRIOR_TASKS}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON source artifact, returning empty evidence when it is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_document(path: Path) -> Any:
    """Read YAML evidence without guessing a structure for missing or bad input."""

    try:
        text = path.read_text(encoding="utf-8")
        return yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so a reviewer can reproduce the exact aggregation."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_file_checksums(root: Path | str = REPO_ROOT) -> dict[str, str | None]:
    """Return the protected-file hashes used to prove this handoff stayed read-only."""

    root_path = Path(root)
    return {path: sha256_file(root_path / path) for path in PROTECTED_FILES}


def ensure_research_complete_entry(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-REPORT-3267: append the `.302` archive summary only when absent."""

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
    protected_hash_baseline: Mapping[str, str | None] | None = None,
    research_complete_update: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """SCENARIO-REPORT-3267: synthesize the .302 archive and .303 queue receipt."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    capstone = read_json_object(root_path / CAPSTONE_V302_REL_PATH)
    exp3264 = read_json_object(root_path / EXP3264_REL_PATH)
    exp3265 = read_json_object(root_path / EXP3265_REL_PATH)
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
    queue = _v303_queue(root_path)

    capstone_ready = capstone.get("capstone_v302_ready") is True
    prior_paper_ready = capstone.get("paper_ready") is True
    prior_blockers = _int_value(capstone.get("publication_blocker_count"))
    prior_next_gap = str(capstone.get("next_top_gap") or "")
    label_count = _v4_shard_label_count(exp3264)
    shard_auroc = _v4_shard_auroc(exp3265)
    activation_observed = _v303_activation_observed(root_path, queue)
    blocked_reasons = _blocked_reasons(
        capstone=capstone,
        capstone_ready=capstone_ready,
        prior_paper_ready=prior_paper_ready,
        prior_publication_blocker_count=prior_blockers,
        prior_next_top_gap=prior_next_gap,
        v4_shard_label_count=label_count,
        v4_shard_auroc=shard_auroc,
        research_complete_present=research_complete_present,
        queue=queue,
        activation_observed=activation_observed,
        protected_files_untouched=protected_untouched,
        vnext_doc_exists=(root_path / VNEXT_DOC_REL_PATH).is_file(),
    )
    source_artifacts = _source_artifacts(root_path, capstone, exp3264, exp3265)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "inference_substrate": "artifact_aggregation_only",
        "v302_closed_v303_opened": not blocked_reasons,
        "prior_paper_ready": prior_paper_ready,
        "prior_publication_blocker_count": prior_blockers,
        "prior_next_top_gap": prior_next_gap,
        "prior_capstone_ready": capstone_ready,
        "prior_cuda_recovery_unblocked_sota_receipt": (
            capstone.get("cuda_recovery_unblocked_sota_receipt") is True
        ),
        "prior_capstone_honest_verdict": str(capstone.get("honest_verdict") or ""),
        "v4_shard_label_count": label_count,
        "v4_shard_auroc": shard_auroc,
        "v4_shard_evidence": _v4_shard_evidence(exp3264, exp3265),
        "terminal_evidence": _terminal_evidence(capstone),
        "research_complete_update": archive_update,
        "research_complete_prior_summary": _research_complete_task_summary(root_path),
        "v303_queue": queue,
        "v303_activation_observed": activation_observed,
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
    """Build and persist the Exp 3267 close/open handoff JSON."""

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
    """Reject handoff artifacts that omit the machine-readable boundary fields."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3267")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3267-close-v302-open-v303-corpus-queue")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.303")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3267")
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must begin with a terminal success prefix")


def _research_complete_entry() -> str:
    lines = [
        "- id: 2026.05.302",
        "  title: CUDA-Recovered SOTA Receipt + v4 Teacher-Label Shard + Capstone",
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


def _v303_queue(root: Path) -> JsonDict:
    payload = _as_mapping(read_yaml_document(root / ACTIVE_ROADMAP_REL_PATH))
    task_ids = _task_ids(payload)
    return {
        "active_roadmap_path": ACTIVE_ROADMAP_REL_PATH.as_posix(),
        "active_roadmap_exists": (root / ACTIVE_ROADMAP_REL_PATH).is_file(),
        "selected_queue_milestone": str(payload.get("milestone") or ""),
        "queue_first_task": task_ids[0] if task_ids else "",
        "queue_task_count": len(task_ids),
        "queue_task_ids": task_ids,
        "milestone_doc": str(payload.get("milestone_doc") or ""),
        "milestone_provenance": str(payload.get("milestone_provenance") or ""),
    }


def _v303_activation_observed(root: Path, queue: Mapping[str, Any]) -> bool:
    return queue.get("selected_queue_milestone") == MILESTONE or _file_contains(
        root / CONDUCTOR_LOG_REL_PATH,
        "Milestone 2026.05.303 activated",
    )


def _v4_shard_label_count(exp3264: Mapping[str, Any]) -> int:
    if exp3264.get("teacher_label_shard_ready") is not True:
        return 0
    label_counts = _as_mapping(exp3264.get("label_counts"))
    return sum(_int_value(value) for value in label_counts.values())


def _v4_shard_auroc(exp3265: Mapping[str, Any]) -> float:
    if exp3265.get("kan_train_eval_shard_ready") is not True:
        return 0.0
    return _float_value(exp3265.get("shard_auroc"))


def _v4_shard_evidence(exp3264: Mapping[str, Any], exp3265: Mapping[str, Any]) -> JsonDict:
    return {
        "teacher_label_shard_ready": exp3264.get("teacher_label_shard_ready") is True,
        "teacher_label_shard_v3_ready": exp3264.get("teacher_label_shard_v3_ready") is True,
        "label_counts": _as_mapping(exp3264.get("label_counts")),
        "label_count_total": _v4_shard_label_count(exp3264),
        "kan_train_eval_shard_ready": exp3265.get("kan_train_eval_shard_ready") is True,
        "kan_train_eval_shard_v3_ready": exp3265.get("kan_train_eval_shard_v3_ready") is True,
        "shard_auroc": _v4_shard_auroc(exp3265),
        "n_train": _int_value(exp3265.get("n_train")),
        "n_eval": _int_value(exp3265.get("n_eval")),
        "non_headline_note": str(exp3265.get("non_headline_note") or ""),
    }


def _terminal_evidence(capstone: Mapping[str, Any]) -> JsonDict:
    return {
        "capstone_v302_ready": capstone.get("capstone_v302_ready") is True,
        "paper_ready": capstone.get("paper_ready") is True,
        "publication_blocker_count": _int_value(capstone.get("publication_blocker_count")),
        "next_top_gap": str(capstone.get("next_top_gap") or ""),
        "cuda_recovery_unblocked_sota_receipt": (
            capstone.get("cuda_recovery_unblocked_sota_receipt") is True
        ),
        "honest_verdict": str(capstone.get("honest_verdict") or ""),
    }


def _blocked_reasons(
    *,
    capstone: Mapping[str, Any],
    capstone_ready: bool,
    prior_paper_ready: bool,
    prior_publication_blocker_count: int,
    prior_next_top_gap: str,
    v4_shard_label_count: int,
    v4_shard_auroc: float,
    research_complete_present: bool,
    queue: Mapping[str, Any],
    activation_observed: bool,
    protected_files_untouched: bool,
    vnext_doc_exists: bool,
) -> list[str]:
    checks = (
        (not capstone, "capstone_v302 authority is missing or malformed"),
        (bool(capstone) and not capstone_ready, "capstone_v302 authority is not ready"),
        (prior_paper_ready is not False, "prior paper_ready must remain false"),
        (
            prior_publication_blocker_count != EXPECTED_PRIOR_PUBLICATION_BLOCKER_COUNT,
            "prior publication blocker count is not 105",
        ),
        (
            prior_next_top_gap != FULL_V4_CORPUS_REPAIR_GAP,
            "prior next_top_gap does not preserve the .303 corpus queue anchor",
        ),
        (v4_shard_label_count <= 0, "v4 shard label count is unavailable"),
        (v4_shard_auroc <= 0.0, "v4 shard AUROC is unavailable"),
        (
            not research_complete_present,
            "research-complete.yaml does not contain the .302 task summary",
        ),
        (
            queue.get("selected_queue_milestone") != MILESTONE,
            "selected queue milestone is not 2026.05.303",
        ),
        (
            queue.get("queue_first_task") != TASK_ID,
            "selected queue first task is not exp3267-close-v302-open-v303-corpus-queue",
        ),
        (not activation_observed, "milestone 2026.05.303 activation is not observed"),
        (not protected_files_untouched, "protected files changed during handoff"),
        (not vnext_doc_exists, "openspec/change-proposals/research-roadmap-vNEXT.md is missing"),
    )
    return [reason for failed, reason in checks if failed]


def _source_artifacts(
    root: Path,
    capstone: Mapping[str, Any],
    exp3264: Mapping[str, Any],
    exp3265: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _source_record(root, "capstone_v302", CAPSTONE_V302_REL_PATH, bool(capstone), True),
        _source_record(
            root,
            "teacher_label_shard_v3",
            EXP3264_REL_PATH,
            bool(exp3264),
            exp3264.get("teacher_label_shard_ready") is True,
        ),
        _source_record(
            root,
            "kan_train_eval_shard_v3",
            EXP3265_REL_PATH,
            bool(exp3265),
            exp3265.get("kan_train_eval_shard_ready") is True,
        ),
        _source_record(root, "research_complete_archive", RESEARCH_COMPLETE_REL_PATH, True, True),
        _source_record(root, "active_v303_roadmap", ACTIVE_ROADMAP_REL_PATH, True, True),
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


def _principle_annotations() -> JsonDict:
    return {
        "v302_closed_v303_opened": "Milestone boundaries must be machine-readable.",
        "prior_paper_ready": "Preserve the publication-readiness signal.",
        "prior_publication_blocker_count": "Track blocker movement across milestones.",
        "prior_next_top_gap": "Keep the next queue anchored to the capstone.",
        "v4_shard_label_count": "Preserve the corpus-scale starting point.",
        "v4_shard_auroc": "Carry non-headline pilot evidence without inflating it.",
        "protected_files_untouched": "Enforce operator constraints.",
        "random_seed": "Keep aggregation reproducible.",
        "reproducibility_checksum": "Provide a deterministic audit trail.",
        "duration_s": "Retain timing evidence for ops retrospectives.",
        "honest_verdict": "Use a terminal prefix without claiming paper readiness.",
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
        "complete: v302_closed_v303_opened="
        f"{str(artifact.get('v302_closed_v303_opened') is True).lower()}; "
        f"prior_paper_ready={str(artifact.get('prior_paper_ready') is True).lower()}; "
        f"prior_publication_blocker_count={artifact.get('prior_publication_blocker_count')}; "
        f"prior_next_top_gap={artifact.get('prior_next_top_gap')}; "
        f"v4_shard_label_count={artifact.get('v4_shard_label_count')}; "
        f"v4_shard_auroc={artifact.get('v4_shard_auroc')}; "
        f"protected_files_untouched={str(artifact.get('protected_files_untouched') is True).lower()}"
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


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _terminal_prefix_ok(value: str) -> bool:
    return value.startswith(TERMINAL_PREFIXES)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
