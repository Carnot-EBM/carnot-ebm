"""Build the Exp 3245 milestone .300 capstone artifact.

Spec refs: REQ-REPORT-3245, SCENARIO-REPORT-3245.

This capstone is deliberately boring: it reads the matrix v33 closeout and
nearby milestone artifacts, then records the publication-readiness decision
without running any model, repair, verifier, solver, hardware, conductor, or
push workflow. The important safety property is that missing or gate-blocked
evidence remains visible instead of being converted into a success claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
MILESTONE = "2026.05.300"
SCHEMA_VERSION = "carnot.milestone_capstone.v300_matrix_v33_closeout.v1"
EXPERIMENT_ID = "exp3245"
TASK_ID = "exp3245-capstone-v300"
ARTIFACT = "experiment_3245_capstone_v300"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3245_capstone_v300.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3245_capstone_v300.py"

MATRIX_V33_REL_PATH = Path("results/experiment_3244_cross_corpus_matrix_v33.json")
CAPSTONE_V299_REL_PATH = Path("results/experiment_3223_capstone_v299.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
EXP3236_REL_PATH = Path("results/experiment_3236_isolated_cuda_python_smoke_v1.json")
EXP3237_REL_PATH = Path("results/experiment_3237_llama_cpp_cuda_receipt_smoke_v2.json")
EXP3238_REL_PATH = Path("results/experiment_3238_sota_gguf_receipt_v7.json")
EXP3239_REL_PATH = Path("results/experiment_3239_prompt_injection_kan_v4_resource_manifest_v1.json")
EXP3240_REL_PATH = Path("results/experiment_3240_prompt_injection_kan_teacher_label_shard_v1.json")
EXP3241_REL_PATH = Path("results/experiment_3241_prompt_injection_kan_train_eval_shard_v1.json")
EXP3242_REL_PATH = Path(
    "results/experiment_3242_dccd_exact_row_structured_proposal_preflight_v1.json"
)
EXP3243_REL_PATH = Path("results/experiment_3243_fr11_failure_memory_controller_v1.json")

REQUIRED_ARTIFACT_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "capstone_v300_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v299",
    "local_sota_receipt_state",
    "prompt_injection_v4_state",
    "fr11_failure_memory_state",
    "next_top_gap",
    "protected_files_untouched",
    "honest_verdict",
}


def read_json_object(path: Path) -> JsonDict:
    """Read JSON evidence as an object and fail closed on absent/bad inputs."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash evidence files so the capstone can be reproduced byte-for-byte."""

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
    """REQ-REPORT-3245: aggregate matrix v33 into the terminal .300 capstone."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V33_REL_PATH)
    capstone_v299 = read_json_object(root_path / CAPSTONE_V299_REL_PATH)
    source_artifacts = _source_artifacts(root_path, matrix, capstone_v299)
    local_sota_state = _local_sota_receipt_state(_as_mapping(matrix.get("runtime_receipt_state")))
    prompt_state = _prompt_injection_v4_state(_as_mapping(matrix.get("prompt_injection_v4_state")))
    proposal_state = _structured_proposal_preflight_state(
        _as_mapping(matrix.get("structured_proposal_state"))
    )
    fr11_state = _fr11_failure_memory_state(_as_mapping(matrix.get("fr11_failure_memory_state")))
    publication_blocker_count = _int_value(matrix.get("publication_blocker_count"))
    prior_blocker_count = _int_value(capstone_v299.get("publication_blocker_count"))
    blocker_delta = publication_blocker_count - prior_blocker_count
    invariant_violations = _invariant_violations(matrix, capstone_v299)
    capstone_ready = not invariant_violations
    next_top_gap = str(
        matrix.get("next_top_gap")
        or local_sota_state.get("next_action")
        or capstone_v299.get("next_top_gap")
        or "repair_selected_python_torch_cuda_before_exp3237"
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
        "capstone_v300_ready": capstone_ready,
        "paper_ready": _paper_ready(capstone_ready, matrix, publication_blocker_count),
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v299": blocker_delta,
        "prior_capstone_v299_summary": _prior_summary(capstone_v299),
        "matrix_v33_summary": _matrix_summary(matrix),
        "local_sota_receipt_state": local_sota_state,
        "prompt_injection_v4_state": prompt_state,
        "structured_proposal_preflight_state": proposal_state,
        "fr11_failure_memory_state": fr11_state,
        "next_top_gap": next_top_gap,
        "next_top_gap_rationale": _next_gap_rationale(
            next_top_gap, local_sota_state, prompt_state, proposal_state, fr11_state
        ),
        "operator_safe_notes": _operator_safe_notes(),
        "protected_files_untouched": {
            ROADMAP_REL_PATH.as_posix(): True,
            CONDUCTOR_REL_PATH.as_posix(): True,
        },
        "protected_file_checksums": {
            ROADMAP_REL_PATH.as_posix(): sha256_file(root_path / ROADMAP_REL_PATH),
            CONDUCTOR_REL_PATH.as_posix(): sha256_file(root_path / CONDUCTOR_REL_PATH),
        },
        "source_artifacts": source_artifacts,
        "source_checksums": {
            row["path"]: row["sha256"]
            for row in source_artifacts
            if row.get("readable_json_object") is True
        },
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
        "scripts_research_conductor_modified": False,
        "research_roadmap_modified": False,
        "publication_submission_claimed": False,
        "paper_publication_claimed": False,
        "invariant_violations": invariant_violations,
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
    """Build and persist the Exp 3245 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject schema gaps and publication-readiness overclaims."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3245")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3245-capstone-v300")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.300")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must begin with complete:")
    if _int_value(artifact.get("publication_blocker_count")) < 0:
        raise ValueError("publication_blocker_count must be non-negative")
    protected = _as_mapping(artifact.get("protected_files_untouched"))
    if (
        protected.get(ROADMAP_REL_PATH.as_posix()) is not True
        or protected.get(CONDUCTOR_REL_PATH.as_posix()) is not True
    ):
        raise ValueError("protected_files_untouched must include roadmap and conductor")


def _source_artifacts(
    root: Path, matrix: Mapping[str, Any], capstone_v299: Mapping[str, Any]
) -> list[JsonDict]:
    sources = [
        ("matrix_v33", MATRIX_V33_REL_PATH, matrix),
        ("capstone_v299", CAPSTONE_V299_REL_PATH, capstone_v299),
    ]
    for role, rel_path in (
        ("isolated_cuda_python_smoke", EXP3236_REL_PATH),
        ("llama_cpp_cuda_receipt_smoke", EXP3237_REL_PATH),
        ("mandated_sota_gguf_receipt", EXP3238_REL_PATH),
        ("prompt_injection_manifest", EXP3239_REL_PATH),
        ("prompt_injection_teacher_label_shard", EXP3240_REL_PATH),
        ("prompt_injection_train_eval_shard", EXP3241_REL_PATH),
        ("dccd_structured_proposal_preflight", EXP3242_REL_PATH),
        ("fr11_failure_memory_controller", EXP3243_REL_PATH),
        ("conductor_log", CONDUCTOR_LOG_REL_PATH),
    ):
        sources.append((role, rel_path, read_json_object(root / rel_path)))
    return [_source_record(root, role, rel_path, payload) for role, rel_path, payload in sources]


def _source_record(root: Path, role: str, rel_path: Path, payload: Mapping[str, Any]) -> JsonDict:
    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "experiment_id": _source_experiment_id(payload, role),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": sha256_file(path),
    }


def _source_experiment_id(payload: Mapping[str, Any], fallback: str) -> str:
    if payload.get("experiment_id"):
        return str(payload["experiment_id"])
    experiment = payload.get("experiment")
    return (
        f"exp{experiment}"
        if isinstance(experiment, int) and not isinstance(experiment, bool)
        else fallback
    )


def _local_sota_receipt_state(runtime: Mapping[str, Any]) -> JsonDict:
    status = (
        "complete"
        if runtime.get("receipt_chain_ready") is True
        else "gate_blocked"
        if runtime.get("cuda_python_smoke_passed") is True
        else "blocked"
    )
    return {
        "status": status,
        "completed": status == "complete",
        "state": str(runtime.get("state") or "missing_runtime_receipt_state"),
        "cuda_driver_visible": runtime.get("cuda_driver_visible") is True,
        "selected_python_torch_cuda_available": (
            runtime.get("selected_python_torch_cuda_available") is True
        ),
        "cuda_python_smoke_passed": runtime.get("cuda_python_smoke_passed") is True,
        "llama_cpp_cuda_receipt_ready": runtime.get("llama_cpp_cuda_receipt_ready") is True,
        "sota_gguf_receipt_ready": runtime.get("sota_gguf_receipt_ready") is True,
        "receipt_chain_ready": runtime.get("receipt_chain_ready") is True,
        "blocking_artifacts": list(runtime.get("blocking_artifacts") or []),
        "next_action": str(
            runtime.get("next_action") or "repair_selected_python_torch_cuda_before_exp3237"
        ),
        "operator_safe_note": (
            "Do not rerun full GGUF receipt until selected Python CUDA, llama.cpp CUDA, "
            "and SOTA receipt gates are green."
        ),
    }


def _prompt_injection_v4_state(prompt: Mapping[str, Any]) -> JsonDict:
    status = (
        "complete"
        if prompt.get("publication_evidence_ready") is True
        else "gate_blocked"
        if prompt.get("manifest_ready") is True
        else "blocked"
    )
    return {
        "status": status,
        "completed": status == "complete",
        "state": str(prompt.get("state") or "missing_prompt_injection_v4_state"),
        "manifest_ready": prompt.get("manifest_ready") is True,
        "teacher_label_plan_ready": prompt.get("teacher_label_plan_ready") is True,
        "teacher_label_shard_status": str(prompt.get("teacher_label_shard_status") or ""),
        "train_eval_shard_status": str(prompt.get("train_eval_shard_status") or ""),
        "publication_evidence_ready": prompt.get("publication_evidence_ready") is True,
        "blocking_artifacts": list(prompt.get("blocking_artifacts") or []),
        "operator_safe_note": (
            "The manifest is useful, but v4 labels and KAN train/eval metrics remain "
            "nonexistent until gated shard artifacts complete."
        ),
    }


def _structured_proposal_preflight_state(proposal: Mapping[str, Any]) -> JsonDict:
    ready = proposal.get("structured_proposal_preflight_ready") is True
    status = (
        "complete"
        if ready
        else "gate_blocked"
        if proposal.get("artifact_status") == "missing"
        or "gate_blocked" in str(proposal.get("state") or "")
        else "blocked"
    )
    return {
        "status": status,
        "completed": status == "complete",
        "state": str(proposal.get("state") or "missing_structured_proposal_state"),
        "artifact_status": str(proposal.get("artifact_status") or ""),
        "structured_proposal_preflight_ready": ready,
        "repair_acceptance_claimed": proposal.get("repair_acceptance_claimed") is True,
        "blocking_artifacts": list(proposal.get("blocking_artifacts") or []),
        "operator_safe_note": (
            "Structured proposal preflight is not repair evidence while exp3242 is missing "
            "or gate-blocked."
        ),
    }


def _fr11_failure_memory_state(fr11: Mapping[str, Any]) -> JsonDict:
    complete = (
        fr11.get("fr11_controller_update_ready") is True
        and fr11.get("model_weight_update_claimed") is not True
        and fr11.get("controller_memory_updates_are_not_training") is True
    )
    return {
        "status": "complete" if complete else "blocked",
        "completed": complete,
        "state": str(fr11.get("state") or "missing_fr11_failure_memory_state"),
        "fr11_controller_update_ready": fr11.get("fr11_controller_update_ready") is True,
        "failure_trace_count": _int_value(fr11.get("failure_trace_count")),
        "heldout_replay_count": _int_value(fr11.get("heldout_replay_count")),
        "heldout_replay_delta": _int_value(fr11.get("heldout_replay_delta")),
        "doomed_rerun_avoidance_count": _int_value(fr11.get("doomed_rerun_avoidance_count")),
        "model_weight_update_claimed": fr11.get("model_weight_update_claimed") is True,
        "controller_memory_updates_are_not_training": (
            fr11.get("controller_memory_updates_are_not_training") is True
        ),
        "operator_safe_note": (
            "FR-11 is complete for controller failure memory only; it is not a model-weight "
            "learning or publication-readiness claim."
        ),
    }


def _next_gap_rationale(
    next_top_gap: str,
    local_sota: Mapping[str, Any],
    prompt: Mapping[str, Any],
    proposal: Mapping[str, Any],
    fr11: Mapping[str, Any],
) -> str:
    rationales = {
        "local_sota": (
            "selected Python CUDA smoke blocks exp3237, the SOTA GGUF receipt, "
            "prompt-injection shards, and structured proposal preflight"
        ),
        "prompt": "prompt-injection v4 still lacks completed teacher-label and train/eval shard evidence",
        "proposal": "DCCD structured proposal preflight is still missing or gate-blocked",
        "fr11": "FR-11 failure memory would be next only if controller memory were incomplete",
        "fallback": "publication blocker reconciliation remains after the visible gated artifacts clear",
    }
    key = (
        "local_sota"
        if local_sota.get("status") != "complete"
        else "prompt"
        if prompt.get("status") != "complete"
        else "proposal"
        if proposal.get("status") != "complete"
        else "fr11"
        if fr11.get("status") != "complete"
        else "fallback"
    )
    return rationales[key] if next_top_gap else rationales["fallback"]


def _paper_ready(capstone_ready: bool, matrix: Mapping[str, Any], blocker_count: int) -> bool:
    return capstone_ready and matrix.get("paper_ready") is True and blocker_count == 0


def _invariant_violations(matrix: Mapping[str, Any], capstone_v299: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    if matrix.get("cross_corpus_matrix_v33_ready") is not True:
        violations.append("matrix v33 is missing or not ready")
    if capstone_v299.get("capstone_v299_ready") is not True:
        violations.append("capstone v299 is missing or not ready")
    return violations


def _prior_summary(capstone_v299: Mapping[str, Any]) -> JsonDict:
    return {
        "path": CAPSTONE_V299_REL_PATH.as_posix(),
        "ready": capstone_v299.get("capstone_v299_ready") is True,
        "paper_ready": capstone_v299.get("paper_ready") is True,
        "publication_blocker_count": _int_value(capstone_v299.get("publication_blocker_count")),
        "next_top_gap": str(capstone_v299.get("next_top_gap") or ""),
        "v4_outcome": str(capstone_v299.get("v4_outcome") or ""),
    }


def _matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "path": MATRIX_V33_REL_PATH.as_posix(),
        "ready": matrix.get("cross_corpus_matrix_v33_ready") is True,
        "paper_ready": matrix.get("paper_ready") is True,
        "publication_blocker_count": _int_value(matrix.get("publication_blocker_count")),
        "publication_blocker_delta_from_v299": _int_value(
            matrix.get("publication_blocker_delta_from_v299")
        ),
        "next_top_gap": str(matrix.get("next_top_gap") or ""),
    }


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "All capstone fields are derived from checked-in artifacts.",
        "missing_is_not_success": "Missing and gate-blocked rows remain publication blockers.",
        "paper_ready_rule": "paper_ready requires matrix authority, capstone readiness, and zero blockers.",
        "operator_boundary": "This task does not push or edit protected conductor/roadmap files.",
    }


def _operator_safe_notes() -> list[str]:
    return [
        "Do NOT push.",
        "Do NOT modify scripts/research_conductor.py.",
        "Do not claim public submission or paper publication from this capstone.",
        "Do not rerun full GGUF, teacher labels, KAN training, DeLong, or Garak until gates pass.",
        "FR-11 failure memory is controller-only and does not update model weights.",
    ]


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "blocker_delta_from_v299": artifact.get("blocker_delta_from_v299"),
        "paper_ready": artifact.get("paper_ready"),
        "next_top_gap": artifact.get("next_top_gap"),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: capstone_v300_ready="
        f"{str(artifact.get('capstone_v300_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v299={artifact.get('blocker_delta_from_v299')}; "
        f"local_sota_receipt_status={_as_mapping(artifact.get('local_sota_receipt_state')).get('status')}; "
        f"prompt_injection_v4_status={_as_mapping(artifact.get('prompt_injection_v4_state')).get('status')}; "
        f"fr11_failure_memory_status={_as_mapping(artifact.get('fr11_failure_memory_state')).get('status')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )
