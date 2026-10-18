"""Build the Exp 3258 milestone .301 capstone artifact.

Spec refs: REQ-REPORT-3258, SCENARIO-REPORT-3258.

This capstone reads the matrix v34 authority and the available `.301`
artifacts, then states whether the milestone actually reduced publication
blockers. It deliberately does not run CUDA, llama.cpp, teacher labeling, KAN
training, repair, verifier, solver, hardware, conductor, or push workflows.
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
SCHEMA_VERSION = "carnot.milestone_capstone.v301_matrix_v34_closeout.v1"
EXPERIMENT_ID = "exp3258"
TASK_ID = "exp3258-capstone-v301"
ARTIFACT = "experiment_3258_capstone_v301"
MILESTONE = "2026.05.301"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3258_capstone_v301.json")

V300_PUBLICATION_BLOCKER_BASELINE = 106
MATRIX_V34_REL_PATH = Path("results/experiment_3257_cross_corpus_matrix_v34.json")
CAPSTONE_V300_REL_PATH = Path("results/experiment_3245_capstone_v300.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
ROADMAP_REL_PATH = Path("research-roadmap.yaml")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

EXP3246_REL_PATH = Path("results/experiment_3246_archive_v300_activate_v301.json")
EXP3247_REL_PATH = Path("results/experiment_3247_selected_python_cuda_root_cause_surgery_v1.json")
EXP3248_REL_PATH = Path("results/experiment_3248_isolated_cuda_selected_python_smoke_v2.json")
EXP3249_REL_PATH = Path("results/experiment_3249_llama_cpp_cuda_receipt_smoke_v3.json")
EXP3250_REL_PATH = Path("results/experiment_3250_sota_gguf_receipt_v8.json")
EXP3251_REL_PATH = Path("results/experiment_3251_prompt_injection_v4_constraint_tax_manifest_v2.json")
EXP3252_REL_PATH = Path("results/experiment_3252_prompt_injection_teacher_label_shard_v2.json")
EXP3253_REL_PATH = Path("results/experiment_3253_prompt_injection_kan_train_eval_shard_v2.json")
EXP3254_REL_PATH = Path("results/experiment_3254_dccd_severa_structured_proposal_preflight_v2.json")
EXP3255_REL_PATH = Path("results/experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.json")
EXP3256_REL_PATH = Path("results/experiment_3256_pdit_potts_multistate_sampler_diagnostic_v1.json")

STATUSES = ("complete", "blocked", "gate_blocked", "missing")
REQUIRED_ARTIFACT_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "capstone_v301_ready",
    "paper_ready",
    "publication_blocker_count",
    "blocker_delta_from_v300",
    "local_sota_receipt_status",
    "prompt_injection_v4_status",
    "dccd_severa_preflight_status",
    "fr11_lifelong_retention_status",
    "pdit_potts_status",
    "next_top_gap",
    "recommended_next_milestone_theme",
    "protected_files_untouched",
    "honest_verdict",
}


@dataclass(frozen=True)
class SourceSpec:
    """One expected `.301` source artifact the capstone should inventory."""

    experiment_id: str
    task_id: str
    path: Path
    role: str
    ready_field: str


EXPECTED_DOT301_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp3246",
        "exp3246-archive-v300-activate-v301",
        EXP3246_REL_PATH,
        "archive_v300_activate_v301",
        "archive_v300_activate_v301_ready",
    ),
    SourceSpec(
        "exp3247",
        "exp3247-selected-python-cuda-root-cause-surgery-v1",
        EXP3247_REL_PATH,
        "selected_python_cuda_root_cause_surgery",
        "next_smoke_allowed",
    ),
    SourceSpec(
        "exp3248",
        "exp3248-isolated-cuda-selected-python-smoke-v2",
        EXP3248_REL_PATH,
        "isolated_selected_python_cuda_smoke",
        "cuda_python_smoke_passed",
    ),
    SourceSpec(
        "exp3249",
        "exp3249-llama-cpp-cuda-receipt-smoke-v3",
        EXP3249_REL_PATH,
        "llama_cpp_cuda_receipt_smoke",
        "llama_cpp_cuda_receipt_ready",
    ),
    SourceSpec(
        "exp3250",
        "exp3250-sota-gguf-receipt-v8",
        EXP3250_REL_PATH,
        "sota_gguf_receipt_v8",
        "sota_gguf_receipt_ready",
    ),
    SourceSpec(
        "exp3251",
        "exp3251-prompt-injection-v4-constraint-tax-manifest-v2",
        EXP3251_REL_PATH,
        "prompt_injection_constraint_tax_manifest",
        "v4_manifest_v2_ready",
    ),
    SourceSpec(
        "exp3252",
        "exp3252-prompt-injection-teacher-label-shard-v2",
        EXP3252_REL_PATH,
        "prompt_injection_teacher_label_shard",
        "teacher_label_shard_ready",
    ),
    SourceSpec(
        "exp3253",
        "exp3253-prompt-injection-kan-train-eval-shard-v2",
        EXP3253_REL_PATH,
        "prompt_injection_kan_train_eval_shard",
        "train_eval_completed",
    ),
    SourceSpec(
        "exp3254",
        "exp3254-dccd-severa-structured-proposal-preflight-v2",
        EXP3254_REL_PATH,
        "dccd_severa_structured_proposal_preflight",
        "structured_proposal_preflight_ready",
    ),
    SourceSpec(
        "exp3255",
        "exp3255-fr11-lifelong-failure-memory-retention-audit-v1",
        EXP3255_REL_PATH,
        "fr11_lifelong_failure_memory_retention_audit",
        "lifelong_eval_ready",
    ),
    SourceSpec(
        "exp3256",
        "exp3256-pdit-potts-multistate-sampler-diagnostic-v1",
        EXP3256_REL_PATH,
        "pdit_potts_multistate_sampler_diagnostic",
        "pdit_potts_mapping_ready",
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read JSON evidence as an object and treat absent or malformed input as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash source evidence so the capstone can be reproduced from exact bytes."""

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
    """REQ-REPORT-3258: aggregate `.301` evidence into the terminal capstone."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V34_REL_PATH)
    capstone_v300 = read_json_object(root_path / CAPSTONE_V300_REL_PATH)
    conductor_log = _read_text(root_path / CONDUCTOR_LOG_REL_PATH)
    matrix_rows = _matrix_rows_by_experiment(matrix)
    source_artifacts = _source_artifacts(root_path, matrix, capstone_v300, matrix_rows)

    local_sota = _local_sota_receipt_status(_as_mapping(matrix.get("runtime_receipt_status")))
    prompt = _prompt_injection_v4_status(_as_mapping(matrix.get("prompt_injection_status")))
    dccd = _dccd_severa_preflight_status(_as_mapping(prompt.get("repair_proposal_preflight")))
    fr11 = _fr11_lifelong_retention_status(_as_mapping(matrix.get("fr11_lifelong_status")))
    pdit = _pdit_potts_status(_as_mapping(matrix.get("pdit_potts_status")))
    prior_count = _v300_publication_blocker_count(capstone_v300)
    publication_count = _publication_blocker_count(matrix, prior_count)
    capstone_ready = not _invariant_violations(matrix, capstone_v300)
    required_evidence = _required_evidence_exists(local_sota, prompt, dccd, fr11, pdit)
    next_top_gap = _next_top_gap(matrix, local_sota, prompt, dccd, fr11, pdit, conductor_log)

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
        "capstone_v301_ready": capstone_ready,
        "paper_ready": (
            capstone_ready
            and matrix.get("paper_ready") is True
            and publication_count == 0
            and required_evidence
        ),
        "publication_blocker_count": publication_count,
        "blocker_delta_from_v300": publication_count - prior_count,
        "prior_capstone_v300_summary": _prior_summary(capstone_v300),
        "matrix_v34_summary": _matrix_summary(matrix),
        "local_sota_receipt_status": local_sota,
        "prompt_injection_v4_status": prompt,
        "dccd_severa_preflight_status": dccd,
        "fr11_lifelong_retention_status": fr11,
        "pdit_potts_status": pdit,
        "next_top_gap": next_top_gap,
        "recommended_next_milestone_theme": _recommended_next_milestone_theme(next_top_gap),
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
            if row.get("present") is True and row.get("sha256")
        },
        "required_evidence_exists": required_evidence,
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
        "research_roadmap_modified": False,
        "scripts_research_conductor_modified": False,
        "invariant_violations": _invariant_violations(matrix, capstone_v300),
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
    """Build and persist the Exp 3258 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when schema fields or paper-readiness honesty constraints drift."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3258")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3258-capstone-v301")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.301")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must begin with complete:")
    if _int_value(artifact.get("publication_blocker_count")) < 0:
        raise ValueError("publication_blocker_count must be non-negative")
    if artifact.get("paper_ready") is True and _int_value(artifact.get("publication_blocker_count")) != 0:
        raise ValueError("paper_ready cannot be true while publication blockers remain")
    protected = _as_mapping(artifact.get("protected_files_untouched"))
    if (
        protected.get(ROADMAP_REL_PATH.as_posix()) is not True
        or protected.get(CONDUCTOR_REL_PATH.as_posix()) is not True
    ):
        raise ValueError("protected_files_untouched must include roadmap and conductor")


def _source_artifacts(
    root: Path,
    matrix: Mapping[str, Any],
    capstone_v300: Mapping[str, Any],
    matrix_rows: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    sources = [
        _source_record(
            root,
            "exp3257",
            "exp3257-cross-corpus-matrix-v34",
            MATRIX_V34_REL_PATH,
            "matrix_v34",
            "matrix_v34_ready",
            matrix,
            {"status": "complete" if matrix.get("matrix_v34_ready") is True else "blocked"},
        ),
        _source_record(
            root,
            "exp3245",
            "exp3245-capstone-v300",
            CAPSTONE_V300_REL_PATH,
            "capstone_v300",
            "capstone_v300_ready",
            capstone_v300,
            {"status": "complete" if capstone_v300.get("capstone_v300_ready") is True else "blocked"},
        ),
    ]
    for spec in EXPECTED_DOT301_SOURCES:
        sources.append(
            _source_record(
                root,
                spec.experiment_id,
                spec.task_id,
                spec.path,
                spec.role,
                spec.ready_field,
                read_json_object(root / spec.path),
                matrix_rows.get(spec.experiment_id, {}),
            )
        )
    return sources


def _source_record(
    root: Path,
    experiment_id: str,
    task_id: str,
    rel_path: Path,
    role: str,
    ready_field: str,
    payload: Mapping[str, Any],
    matrix_row: Mapping[str, Any],
) -> JsonDict:
    path = root / rel_path
    present = path.is_file()
    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "path": rel_path.as_posix(),
        "role": role,
        "ready_field": ready_field,
        "present": present,
        "readable_json_object": bool(payload),
        "status": _source_status(payload, matrix_row, ready_field, present),
        "reported_experiment_id": str(payload.get("experiment_id") or f"exp{payload.get('experiment')}" if payload else ""),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": sha256_file(path),
    }


def _source_status(
    payload: Mapping[str, Any], matrix_row: Mapping[str, Any], ready_field: str, present: bool
) -> str:
    row_status = _normal_status(str(matrix_row.get("status") or ""))
    if row_status != "missing":
        return row_status
    if not present:
        return "missing"
    if _is_gate_blocked(payload):
        return "gate_blocked"
    return _status_from_ready(payload.get(ready_field) is True, "")


def _local_sota_receipt_status(runtime: Mapping[str, Any]) -> JsonDict:
    selected = _as_mapping(runtime.get("selected_python_cuda"))
    llama = _as_mapping(runtime.get("llama_cpp_cuda"))
    sota = _as_mapping(runtime.get("sota_gguf_receipt"))
    receipt_ready = runtime.get("receipt_chain_ready") is True
    return {
        "status": "complete" if receipt_ready else "blocked",
        "advanced": receipt_ready,
        "selected_python_cuda": {
            "artifact_status": _normal_status(str(selected.get("artifact_status") or "")),
            "root_cause_class": str(selected.get("root_cause_class") or ""),
            "next_smoke_allowed": selected.get("next_smoke_allowed") is True,
            "selected_python_cuda_repaired_candidate": (
                selected.get("selected_python_cuda_repaired_candidate") is True
            ),
            "cuda_python_smoke_passed": selected.get("cuda_python_smoke_passed") is True,
            "state": str(selected.get("state") or ""),
        },
        "llama_cpp_cuda": {
            "artifact_status": _normal_status(str(llama.get("artifact_status") or "")),
            "llama_cpp_cuda_receipt_ready": llama.get("llama_cpp_cuda_receipt_ready") is True,
        },
        "sota_gguf_receipt": {
            "artifact_status": _normal_status(str(sota.get("artifact_status") or "")),
            "sota_gguf_receipt_ready": sota.get("sota_gguf_receipt_ready") is True,
            "mandatory_model_receipt_count": _int_value(sota.get("mandatory_model_receipt_count")),
            "models_used": list(sota.get("models_used") or []),
        },
        "clean_rerun_allowed": runtime.get("clean_rerun_allowed") is True,
        "receipt_chain_ready": receipt_ready,
        "next_action": str(runtime.get("next_action") or "repair_selected_python_cuda_runtime"),
    }


def _prompt_injection_v4_status(prompt: Mapping[str, Any]) -> JsonDict:
    manifest = _as_mapping(prompt.get("manifest_v2"))
    tax = _as_mapping(prompt.get("constraint_tax_diagnostic"))
    labels = _as_mapping(prompt.get("teacher_labels"))
    kan = _as_mapping(prompt.get("kan_shard"))
    repair = _as_mapping(prompt.get("repair_proposal_preflight"))
    publication_ready = prompt.get("publication_evidence_ready") is True
    plan_ready = manifest.get("constraint_tax_control_plan_ready") is True
    return {
        "status": "complete" if publication_ready else "gate_blocked" if plan_ready else "blocked",
        "constraint_tax_plan_advanced": plan_ready,
        "manifest_v2": {
            "artifact_status": _normal_status(str(manifest.get("artifact_status") or "")),
            "v4_manifest_v2_ready": manifest.get("v4_manifest_v2_ready") is True,
            "constraint_tax_control_plan_ready": plan_ready,
            "garak_config_ready": manifest.get("garak_config_ready") is True,
            "no_llm_invoked": manifest.get("no_llm_invoked") is True,
        },
        "constraint_tax_diagnostic": {
            "status": str(tax.get("status") or "missing"),
            "constraint_tax_delta_accuracy_or_parse": tax.get("constraint_tax_delta_accuracy_or_parse"),
        },
        "teacher_labels": {
            "artifact_status": _normal_status(str(labels.get("artifact_status") or "")),
            "teacher_label_shard_ready": labels.get("teacher_label_shard_ready") is True,
            "completed_free_reasoning_count": _int_value(labels.get("completed_free_reasoning_count")),
            "completed_schema_constrained_count": _int_value(
                labels.get("completed_schema_constrained_count")
            ),
        },
        "kan_shard": {
            "artifact_status": _normal_status(str(kan.get("artifact_status") or "")),
            "train_eval_completed": kan.get("train_eval_completed") is True,
            "headline_claim_allowed": kan.get("headline_claim_allowed") is True,
        },
        "repair_proposal_preflight": repair,
        "publication_evidence_ready": publication_ready,
    }


def _dccd_severa_preflight_status(repair: Mapping[str, Any]) -> JsonDict:
    ready = repair.get("structured_proposal_preflight_ready") is True
    artifact_status = _normal_status(str(repair.get("artifact_status") or ""))
    return {
        "status": _status_from_ready(ready, artifact_status),
        "artifact_status": artifact_status,
        "structured_proposal_preflight_ready": ready,
        "repair_acceptance_claimed": repair.get("repair_acceptance_claimed") is True,
    }


def _fr11_lifelong_retention_status(fr11: Mapping[str, Any]) -> JsonDict:
    complete = fr11.get("lifelong_eval_ready") is True and fr11.get("model_weight_update_claimed") is not True
    return {
        "status": "complete" if complete else "blocked",
        "advanced": complete,
        "artifact_status": _normal_status(str(fr11.get("artifact_status") or "")),
        "continuous_self_learning_task": fr11.get("continuous_self_learning_task") is True,
        "fr11_controller_update_ready": fr11.get("fr11_controller_update_ready") is True,
        "lifelong_eval_ready": fr11.get("lifelong_eval_ready") is True,
        "failure_trace_count": _int_value(fr11.get("failure_trace_count")),
        "heldout_replay_count": _int_value(fr11.get("heldout_replay_count")),
        "retention_score": fr11.get("retention_score"),
        "adaptation_score": fr11.get("adaptation_score"),
        "forgetting_score": fr11.get("forgetting_score"),
        "negative_control_regression_count": _int_value(fr11.get("negative_control_regression_count")),
        "doomed_rerun_avoidance_count": _int_value(fr11.get("doomed_rerun_avoidance_count")),
        "model_weight_update_claimed": fr11.get("model_weight_update_claimed") is True,
        "no_new_llm_invoked": fr11.get("no_new_llm_invoked") is True,
    }


def _pdit_potts_status(pdit: Mapping[str, Any]) -> JsonDict:
    complete = (
        pdit.get("pdit_potts_mapping_ready") is True
        and pdit.get("exact_fallback_preserved") is True
        and pdit.get("hardware_speedup_claim_allowed") is not True
    )
    return {
        "status": "complete" if complete else "blocked",
        "advanced": complete,
        "artifact_status": _normal_status(str(pdit.get("artifact_status") or "")),
        "pdit_potts_mapping_ready": pdit.get("pdit_potts_mapping_ready") is True,
        "candidate_verifier_row_type_count": _int_value(
            pdit.get("candidate_verifier_row_type_count")
        ),
        "q_state_energy_mapping_count": _int_value(pdit.get("q_state_energy_mapping_count")),
        "exact_fallback_preserved": pdit.get("exact_fallback_preserved") is True,
        "hardware_speedup_claim_allowed": pdit.get("hardware_speedup_claim_allowed") is True,
        "retired_pimi_scope_reopened": pdit.get("retired_pimi_scope_reopened") is True,
        "thrml_scaling_sweep_reopened": pdit.get("thrml_scaling_sweep_reopened") is True,
    }


def _required_evidence_exists(
    local_sota: Mapping[str, Any],
    prompt: Mapping[str, Any],
    dccd: Mapping[str, Any],
    fr11: Mapping[str, Any],
    pdit: Mapping[str, Any],
) -> bool:
    return (
        local_sota.get("receipt_chain_ready") is True
        and prompt.get("publication_evidence_ready") is True
        and dccd.get("structured_proposal_preflight_ready") is True
        and fr11.get("status") == "complete"
        and pdit.get("status") == "complete"
    )


def _next_top_gap(
    matrix: Mapping[str, Any],
    local_sota: Mapping[str, Any],
    prompt: Mapping[str, Any],
    dccd: Mapping[str, Any],
    fr11: Mapping[str, Any],
    pdit: Mapping[str, Any],
    conductor_log: str,
) -> str:
    observed = str(matrix.get("next_top_gap") or "")
    if local_sota.get("status") != "complete":
        return observed or str(local_sota.get("next_action") or "repair_selected_python_cuda_runtime")
    if prompt.get("publication_evidence_ready") is not True:
        return "prompt_injection_teacher_labels_and_kan_shard_after_sota_receipt"
    if dccd.get("status") != "complete":
        return "dccd_severa_preflight_after_clean_sota_receipt"
    if fr11.get("status") != "complete":
        return "fr11_lifelong_retention_audit"
    if pdit.get("status") != "complete":
        return "pdit_potts_diagnostic_mapping"
    return observed or ("publication_blocker_retirement_review" if conductor_log or not conductor_log else "")


def _recommended_next_milestone_theme(next_top_gap: str) -> str:
    gap = next_top_gap.lower()
    if "cuda" in gap or "exp3248" in gap:
        return "Repair selected-Python CUDA runtime before reopening runtime receipts."
    if "prompt_injection" in gap or "kan" in gap:
        return "Recover prompt-injection teacher labels and KAN shard after clean SOTA receipts."
    if "dccd" in gap or "severa" in gap:
        return "Run DCCD/SEVerA preflight only after clean SOTA receipt evidence exists."
    if "fr11" in gap:
        return "Continue FR-11 retention audits without model-weight update claims."
    if "pdit" in gap or "potts" in gap:
        return "Extend p-dit/Potts diagnostics while preserving exact fallback authority."
    return "Review publication blockers after all observed gates clear."


def _publication_blocker_count(matrix: Mapping[str, Any], prior_count: int) -> int:
    count = _int_value(matrix.get("publication_blocker_count"))
    if matrix.get("matrix_v34_ready") is True or count > 0:
        return count
    return prior_count


def _v300_publication_blocker_count(capstone_v300: Mapping[str, Any]) -> int:
    count = _int_value(capstone_v300.get("publication_blocker_count"))
    return count if count > 0 else V300_PUBLICATION_BLOCKER_BASELINE


def _invariant_violations(matrix: Mapping[str, Any], capstone_v300: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    if matrix.get("matrix_v34_ready") is not True:
        violations.append("matrix v34 is missing or not ready")
    if capstone_v300.get("capstone_v300_ready") is not True:
        violations.append("capstone v300 is missing or not ready")
    return violations


def _matrix_rows_by_experiment(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for key in ("artifacts_found", "artifacts_missing", "gate_blocked_artifacts"):
        for row in matrix.get(key) or []:
            row_map = _as_mapping(row)
            experiment_id = str(row_map.get("experiment_id") or "")
            if experiment_id:
                rows[experiment_id] = row_map
    return rows


def _prior_summary(capstone_v300: Mapping[str, Any]) -> JsonDict:
    return {
        "path": CAPSTONE_V300_REL_PATH.as_posix(),
        "ready": capstone_v300.get("capstone_v300_ready") is True,
        "paper_ready": capstone_v300.get("paper_ready") is True,
        "publication_blocker_count": _v300_publication_blocker_count(capstone_v300),
        "honest_verdict": str(capstone_v300.get("honest_verdict") or ""),
    }


def _matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "path": MATRIX_V34_REL_PATH.as_posix(),
        "ready": matrix.get("matrix_v34_ready") is True,
        "paper_ready": matrix.get("paper_ready") is True,
        "publication_blocker_count": _int_value(matrix.get("publication_blocker_count")),
        "next_top_gap": str(matrix.get("next_top_gap") or ""),
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "The capstone reads matrix v34 and checked-in .301 artifacts only.",
        "baseline_comparison": "Blocker delta is measured against the .300 baseline of 106.",
        "missing_is_not_success": "Missing and gate-blocked evidence remains a publication blocker.",
        "paper_ready_rule": "paper_ready requires zero blockers and all required evidence.",
        "operator_boundary": "This task does not push or edit roadmap/conductor files.",
    }


def _operator_safe_notes() -> list[str]:
    return [
        "Do NOT push.",
        "Do NOT modify scripts/research_conductor.py.",
        "Do NOT modify research-roadmap.yaml.",
        "Do not claim paper readiness while publication blockers remain.",
        "Do not rerun GGUF receipts, teacher labels, KAN training, or DCCD/SEVerA until gates pass.",
    ]


def _is_gate_blocked(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema") == "blocked_gate_check_v1"
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
        or str(payload.get("honest_verdict") or "").startswith("blocked_gate_check")
    )


def _status_from_ready(ready: bool, artifact_status: str) -> str:
    if ready:
        return "complete"
    normalized = _normal_status(artifact_status)
    return normalized if normalized != "complete" else "blocked"


def _normal_status(status: str) -> str:
    normalized = status.strip().lower().replace("-", "_")
    return normalized if normalized in STATUSES else "missing"


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "blocker_delta_from_v300": artifact.get("blocker_delta_from_v300"),
        "paper_ready": artifact.get("paper_ready"),
        "next_top_gap": artifact.get("next_top_gap"),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: capstone_v301_ready="
        f"{str(artifact.get('capstone_v301_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v300={artifact.get('blocker_delta_from_v300')}; "
        f"local_sota_receipt_status={_as_mapping(artifact.get('local_sota_receipt_status')).get('status')}; "
        f"prompt_injection_v4_status={_as_mapping(artifact.get('prompt_injection_v4_status')).get('status')}; "
        f"dccd_severa_preflight_status={_as_mapping(artifact.get('dccd_severa_preflight_status')).get('status')}; "
        f"fr11_lifelong_retention_status={_as_mapping(artifact.get('fr11_lifelong_retention_status')).get('status')}; "
        f"pdit_potts_status={_as_mapping(artifact.get('pdit_potts_status')).get('status')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )
