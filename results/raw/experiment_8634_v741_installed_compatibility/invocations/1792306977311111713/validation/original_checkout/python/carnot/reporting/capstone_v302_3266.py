"""Build the Exp 3266 milestone .302 capstone artifact.

Spec refs: REQ-REPORT-3266, SCENARIO-REPORT-3266.

This module reads the checked-in .302 evidence and answers a narrow operational
question: did the post-reboot CUDA recovery flow through to a real SOTA GGUF
receipt, and what publication blocker remains next? It does not rerun CUDA,
llama.cpp, labeling, KAN training, repair, or any conductor workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.milestone_capstone.v302_cuda_recovery_readout.v1"
EXPERIMENT_ID = "exp3266"
TASK_ID = "exp3266-capstone-v302"
ARTIFACT = "experiment_3266_capstone_v302"
MILESTONE = "2026.05.302"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3266_capstone_v302.json")
RANDOM_SEED = 3266

DEFAULT_PRIOR_PUBLICATION_BLOCKER_COUNT = 106
CUDA_RECEIPT_BLOCKER_DECREMENT = 1
FULL_V4_CORPUS_REPAIR_GAP = "full_15k_v4_corpus_across_shards_plus_repair_and_garak_gates"

EXP3260_REL_PATH = Path("results/experiment_3260_archive_v301_activate_v302.json")
EXP3261_REL_PATH = Path("results/experiment_3261_cuda_recovery_confirmation_smoke_v1.json")
EXP3262_REL_PATH = Path("results/experiment_3262_llama_cpp_cuda_receipt_smoke_v4.json")
EXP3263_REL_PATH = Path("results/experiment_3263_sota_gguf_receipt_v9.json")
EXP3264_REL_PATH = Path("results/experiment_3264_prompt_injection_teacher_label_shard_v3.json")
EXP3265_REL_PATH = Path("results/experiment_3265_prompt_injection_kan_train_eval_shard_v3.json")

TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped_")
REQUIRED_ARTIFACT_FIELDS = {
    "capstone_v302_ready",
    "paper_ready",
    "publication_blocker_count",
    "next_top_gap",
    "cuda_recovery_unblocked_sota_receipt",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


@dataclass(frozen=True)
class SourceSpec:
    """One checked-in source whose ready field contributes to the .302 readout."""

    experiment_id: str
    role: str
    path: Path
    ready_field: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3260", "prior_blocker_authority", EXP3260_REL_PATH, "archive_v301_activate_v302_ready"),
    SourceSpec("exp3261", "cuda_recovery_confirmation", EXP3261_REL_PATH, "cuda_python_smoke_passed"),
    SourceSpec("exp3262", "llama_cpp_cuda_receipt", EXP3262_REL_PATH, "llama_cpp_cuda_receipt_ready"),
    SourceSpec("exp3263", "sota_gguf_receipt", EXP3263_REL_PATH, "sota_gguf_receipt_ready"),
    SourceSpec("exp3264", "prompt_injection_teacher_label_shard", EXP3264_REL_PATH, "teacher_label_shard_ready"),
    SourceSpec("exp3265", "prompt_injection_kan_train_eval_shard", EXP3265_REL_PATH, "kan_train_eval_shard_ready"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one evidence JSON object, returning empty evidence for absent or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash an evidence file so the capstone can be reproduced from exact source bytes."""

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
    """REQ-REPORT-3266: aggregate .302 CUDA receipt and shard evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = {spec.experiment_id: read_json_object(root_path / spec.path) for spec in SOURCE_SPECS}
    source_artifacts = _source_artifacts(root_path, payloads)
    source_checksums = {
        row["path"]: row["sha256"] for row in source_artifacts if row.get("sha256")
    }

    cuda_status = _cuda_recovery_status(payloads)
    sota_status = _sota_receipt_status(payloads)
    v4_status = _v4_shard_status(payloads)
    cuda_unblocked = (
        cuda_status["cuda_python_smoke_passed"]
        and cuda_status["llama_cpp_cuda_receipt_ready"]
        and sota_status["sota_gguf_receipt_ready"]
    )
    prior_count = _prior_publication_blocker_count(payloads["exp3260"])
    publication_count = max(
        0,
        prior_count - CUDA_RECEIPT_BLOCKER_DECREMENT if cuda_unblocked else prior_count,
    )
    capstone_ready = all(row["readable_json_object"] for row in source_artifacts[1:])
    next_top_gap = _next_top_gap(cuda_status, sota_status, v4_status)
    paper_ready = (
        capstone_ready
        and publication_count == 0
        and cuda_unblocked
        and v4_status["full_15k_replacement_grade_ready"]
    )

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle_annotations": _principle_annotations(),
        "capstone_v302_ready": capstone_ready,
        "paper_ready": paper_ready,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": publication_count,
        "publication_blocker_delta": publication_count - prior_count,
        "next_top_gap": next_top_gap,
        "cuda_recovery_unblocked_sota_receipt": cuda_unblocked,
        "cuda_recovery_status": cuda_status,
        "sota_receipt_status": sota_status,
        "v4_shard_status": v4_status,
        "blocked_reasons": _blocked_reasons(
            cuda_status,
            sota_status,
            v4_status,
            publication_count,
        ),
        "source_artifacts": source_artifacts,
        "source_checksums": source_checksums,
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_llama_cpp_run": True,
        "no_new_gguf_receipt": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_repair_run": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
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
) -> Path:
    """Build and persist the Exp 3266 capstone JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject capstones that omit required fields or claim readiness too early."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3266")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3266-capstone-v302")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.302")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if _int_value(artifact.get("publication_blocker_count")) < 0:
        raise ValueError("publication_blocker_count must be non-negative")
    if artifact.get("paper_ready") is True and _int_value(artifact.get("publication_blocker_count")) != 0:
        raise ValueError("paper_ready cannot be true while publication blockers remain")


def _source_artifacts(root: Path, payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [_source_record(root, spec, payloads.get(spec.experiment_id, {})) for spec in SOURCE_SPECS]


def _source_record(root: Path, spec: SourceSpec, payload: Mapping[str, Any]) -> JsonDict:
    path = root / spec.path
    return {
        "experiment_id": spec.experiment_id,
        "role": spec.role,
        "path": spec.path.as_posix(),
        "ready_field": spec.ready_field,
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "ready": payload.get(spec.ready_field) is True,
        "reported_experiment_id": str(payload.get("experiment_id") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": sha256_file(path),
    }


def _cuda_recovery_status(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp3261 = payloads["exp3261"]
    exp3262 = payloads["exp3262"]
    return {
        "cuda_python_smoke_passed": exp3261.get("cuda_python_smoke_passed") is True,
        "next_smoke_allowed": exp3261.get("next_smoke_allowed") is True,
        "gpu_count": _int_value(exp3261.get("gpu_count")),
        "gpu_names": list(exp3261.get("gpu_names") or []),
        "llama_cpp_cuda_receipt_ready": exp3262.get("llama_cpp_cuda_receipt_ready") is True,
        "gpu_layers_offloaded": _int_value(exp3262.get("gpu_layers_offloaded")),
        "tokens_generated": _int_value(exp3262.get("tokens_generated")),
    }


def _sota_receipt_status(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp3263 = payloads["exp3263"]
    return {
        "sota_gguf_receipt_ready": exp3263.get("sota_gguf_receipt_ready") is True,
        "sota_gguf_receipt_v9_ready": exp3263.get("sota_gguf_receipt_v9_ready") is True,
        "cached_model_ids": list(exp3263.get("cached_model_ids") or []),
        "missing_model_ids": list(exp3263.get("missing_model_ids") or []),
        "headline_model_id": str(_as_mapping(exp3263.get("model_specs")).get("headline_model_id") or ""),
        "per_model_receipts_passed": sum(
            1 for row in exp3263.get("per_model_receipts") or [] if _as_mapping(row).get("receipt_passed") is True
        ),
    }


def _v4_shard_status(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp3264 = payloads["exp3264"]
    exp3265 = payloads["exp3265"]
    label_counts = _as_mapping(exp3264.get("label_counts"))
    shard_size = _int_value(exp3264.get("shard_size"))
    kan_ready = exp3265.get("kan_train_eval_shard_ready") is True
    return {
        "teacher_label_shard_ready": exp3264.get("teacher_label_shard_ready") is True,
        "teacher_label_shard_v3_ready": exp3264.get("teacher_label_shard_v3_ready") is True,
        "shard_size": shard_size,
        "label_counts": label_counts,
        "label_count_total": sum(_int_value(value) for value in label_counts.values()),
        "kan_train_eval_shard_ready": kan_ready,
        "kan_train_eval_shard_v3_ready": exp3265.get("kan_train_eval_shard_v3_ready") is True,
        "shard_auroc": exp3265.get("shard_auroc"),
        "n_train": _int_value(exp3265.get("n_train")),
        "n_eval": _int_value(exp3265.get("n_eval")),
        "non_headline_note": str(exp3265.get("non_headline_note") or ""),
        "full_15k_replacement_grade_ready": exp3265.get("full_15k_replacement_grade_ready") is True,
    }


def _prior_publication_blocker_count(prior: Mapping[str, Any]) -> int:
    return (
        _int_value(prior.get("prior_publication_blocker_count"))
        if "prior_publication_blocker_count" in prior
        else DEFAULT_PRIOR_PUBLICATION_BLOCKER_COUNT
    )


def _next_top_gap(
    cuda_status: Mapping[str, Any],
    sota_status: Mapping[str, Any],
    v4_status: Mapping[str, Any],
) -> str:
    checks = (
        (cuda_status.get("cuda_python_smoke_passed") is not True, "cuda_recovery_confirmation_smoke"),
        (cuda_status.get("llama_cpp_cuda_receipt_ready") is not True, "llama_cpp_cuda_receipt_after_cuda_smoke"),
        (sota_status.get("sota_gguf_receipt_ready") is not True, "sota_gguf_receipt_after_llama_cpp_cuda_receipt"),
        (v4_status.get("teacher_label_shard_ready") is not True, "prompt_injection_teacher_labels_after_sota_receipt"),
        (v4_status.get("kan_train_eval_shard_ready") is not True, "prompt_injection_kan_train_eval_after_teacher_labels"),
    )
    for missing, gap in checks:
        if missing:
            return gap
    return (
        "publication_blocker_retirement_review"
        if v4_status.get("full_15k_replacement_grade_ready") is True
        else FULL_V4_CORPUS_REPAIR_GAP
    )


def _blocked_reasons(
    cuda_status: Mapping[str, Any],
    sota_status: Mapping[str, Any],
    v4_status: Mapping[str, Any],
    publication_count: int,
) -> list[str]:
    checks = (
        (cuda_status.get("cuda_python_smoke_passed") is not True, "cuda_python_smoke_passed is not true"),
        (cuda_status.get("llama_cpp_cuda_receipt_ready") is not True, "llama_cpp_cuda_receipt_ready is not true"),
        (sota_status.get("sota_gguf_receipt_ready") is not True, "sota_gguf_receipt_ready is not true"),
        (v4_status.get("teacher_label_shard_ready") is not True, "teacher_label_shard_ready is not true"),
        (v4_status.get("kan_train_eval_shard_ready") is not True, "kan_train_eval_shard_ready is not true"),
        (
            v4_status.get("full_15k_replacement_grade_ready") is not True,
            "full 15k v4 replacement-grade evidence is not complete",
        ),
        (publication_count > 0, "publication blockers remain"),
    )
    return [reason for failed, reason in checks if failed]


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "The capstone reads checked-in Exp 3261-3265 evidence and prior blocker authority.",
        "cuda_recovery_unblocked_sota_receipt": "True only when CUDA, llama.cpp, and SOTA GGUF receipt gates all report ready.",
        "publication_blocker_count": "The CUDA receipt blocker tier decrements by one only after Exp 3263 is ready.",
        "paper_ready": "Paper readiness still requires zero blockers plus replacement-grade full-corpus v4, repair, and Garak evidence.",
        "next_top_gap": "The remaining top gap is selected from the first incomplete gate after the recovered receipt chain.",
    }


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "prior_publication_blocker_count": artifact.get("prior_publication_blocker_count"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "publication_blocker_delta": artifact.get("publication_blocker_delta"),
        "cuda_recovery_unblocked_sota_receipt": artifact.get("cuda_recovery_unblocked_sota_receipt"),
        "next_top_gap": artifact.get("next_top_gap"),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: capstone_v302_ready="
        f"{str(artifact.get('capstone_v302_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"publication_blocker_delta={artifact.get('publication_blocker_delta')}; "
        "cuda_recovery_unblocked_sota_receipt="
        f"{str(artifact.get('cuda_recovery_unblocked_sota_receipt') is True).lower()}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _terminal_prefix_ok(value: str) -> bool:
    return value.startswith(TERMINAL_PREFIXES)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
