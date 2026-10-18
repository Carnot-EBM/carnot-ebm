"""Build the Exp 3257 cross-corpus matrix v34 artifact.

Spec refs: REQ-REPORT-3257, SCENARIO-REPORT-3257.

This module is intentionally a ledger, not an experiment runner. It reads the
`.301` artifact files and conductor gate lines that already exist, then records
which runtime receipts and downstream evidence are complete, blocked, missing,
or gate-blocked. That separation matters because a missing downstream JSON is
not evidence that the downstream task succeeded.
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
SCHEMA_VERSION = "carnot.cross_corpus_matrix.v34_301_artifact_aggregation.v1"
EXPERIMENT_ID = "exp3257"
TASK_ID = "exp3257-cross-corpus-matrix-v34"
ARTIFACT = "experiment_3257_cross_corpus_matrix_v34"
MILESTONE = "2026.05.301"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3257_cross_corpus_matrix_v34.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
PRIOR_PUBLICATION_BLOCKER_COUNT = 106

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
    "matrix_v34_ready",
    "artifacts_expected",
    "artifacts_found",
    "artifacts_missing",
    "gate_blocked_artifacts",
    "runtime_receipt_status",
    "prompt_injection_status",
    "fr11_lifelong_status",
    "pdit_potts_status",
    "publication_blocker_count",
    "paper_ready",
    "next_top_gap",
    "honest_verdict",
}


@dataclass(frozen=True)
class SourceSpec:
    """One planned `.301` artifact that matrix v34 must account for."""

    experiment_id: str
    task_id: str
    path: Path
    role: str
    ready_field: str
    gate_phrase: str = ""


EXPECTED_SOURCES: tuple[SourceSpec, ...] = (
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
        "Isolated selected-Python CUDA smoke v2 gated on root-cause surgery",
    ),
    SourceSpec(
        "exp3249",
        "exp3249-llama-cpp-cuda-receipt-smoke-v3",
        EXP3249_REL_PATH,
        "llama_cpp_cuda_receipt_smoke",
        "llama_cpp_cuda_receipt_ready",
        "llama.cpp CUDA receipt smoke v3 gated on selected-",
    ),
    SourceSpec(
        "exp3250",
        "exp3250-sota-gguf-receipt-v8",
        EXP3250_REL_PATH,
        "sota_gguf_receipt_v8",
        "sota_gguf_receipt_ready",
        "Mandated SOTA GGUF receipt v8 gated on llama.cpp CUDA",
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
        "Prompt-injection teacher-label shard v2 gated on S",
    ),
    SourceSpec(
        "exp3253",
        "exp3253-prompt-injection-kan-train-eval-shard-v2",
        EXP3253_REL_PATH,
        "prompt_injection_kan_train_eval_shard",
        "train_eval_completed",
        "Prompt-injection KAN shard train/eval v2 with constraint-tax guardrail",
    ),
    SourceSpec(
        "exp3254",
        "exp3254-dccd-severa-structured-proposal-preflight-v2",
        EXP3254_REL_PATH,
        "dccd_severa_structured_proposal_preflight",
        "structured_proposal_preflight_ready",
        "DCCD/SEVerA structured proposal preflight v2 gated",
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
    """Read a JSON object and treat missing, malformed, or array input as absent evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash an artifact file so matrix rows can be tied to exact source bytes."""

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
    """REQ-REPORT-3257: aggregate matrix v34 from checked-in `.301` evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    conductor_log = _read_text(root_path / CONDUCTOR_LOG_REL_PATH)
    rows = [_source_row(root_path, spec, conductor_log) for spec in EXPECTED_SOURCES]
    payloads = {row["experiment_id"]: _as_mapping(row.get("payload")) for row in rows}
    public_rows = [_public_row(row) for row in rows]

    runtime_status = _runtime_receipt_status(public_rows, payloads)
    prompt_status = _prompt_injection_status(public_rows, payloads)
    fr11_status = _fr11_lifelong_status(public_rows, payloads)
    pdit_status = _pdit_potts_status(public_rows, payloads)
    required_evidence = (
        _required_evidence_exists(runtime_status, prompt_status)
        and fr11_status["lifelong_eval_ready"] is True
        and pdit_status["pdit_potts_mapping_ready"] is True
    )
    prior_count = _prior_publication_blocker_count(payloads.get("exp3246", {}))
    publication_blocker_count = 0 if required_evidence else prior_count
    paper_ready = required_evidence and publication_blocker_count == 0
    invariant_violations = _invariant_violations(payloads)

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
        "matrix_v34_ready": not invariant_violations,
        "artifacts_expected": [_expected_record(spec) for spec in EXPECTED_SOURCES],
        "artifacts_found": [row for row in public_rows if row["present"]],
        "artifacts_missing": [row for row in public_rows if not row["present"]],
        "gate_blocked_artifacts": [
            row for row in public_rows if _as_mapping(row.get("gate_evidence")).get("status") == "gate_blocked"
        ],
        "artifact_status_counts": _status_counts(public_rows),
        "runtime_receipt_status": runtime_status,
        "prompt_injection_status": prompt_status,
        "fr11_lifelong_status": fr11_status,
        "pdit_potts_status": pdit_status,
        "publication_blockers": _publication_blockers(public_rows),
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": publication_blocker_count,
        "paper_ready": paper_ready,
        "next_top_gap": _next_top_gap(runtime_status, prompt_status, fr11_status, pdit_status),
        "source_checksums": {
            row["path"]: row["sha256"]
            for row in public_rows
            if row["present"] and row.get("sha256")
        },
        "protected_files_untouched": {"scripts/research_conductor.py": True},
        "no_new_model_execution": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_repair_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "scripts_research_conductor_modified": False,
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
    """Build and persist the Exp 3257 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the matrix schema or paper-ready honesty rule is violated."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3257")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3257-cross-corpus-matrix-v34")
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


def _source_row(root: Path, spec: SourceSpec, conductor_log: str) -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path)
    gate = _gate_evidence(spec, conductor_log, payload)
    status = _status_for_source(spec, payload)
    if status == "missing" and gate["status"] == "gate_blocked":
        status = "gate_blocked"
    return {
        "experiment_id": spec.experiment_id,
        "task_id": spec.task_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "ready_field": spec.ready_field,
        "present": path.is_file(),
        "payload": payload,
        "status": status,
        "gate_evidence": gate,
        "sha256": sha256_file(path),
    }


def _public_row(row: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(row.get("payload"))
    return {
        "experiment_id": str(row.get("experiment_id") or ""),
        "task_id": str(row.get("task_id") or ""),
        "path": str(row.get("path") or ""),
        "role": str(row.get("role") or ""),
        "ready_field": str(row.get("ready_field") or ""),
        "present": row.get("present") is True,
        "status": _normal_status(str(row.get("status") or "missing")),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "reported_experiment_id": str(payload.get("experiment_id") or f"exp{payload.get('experiment')}"),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "summary": _payload_summary(payload),
        "gate_evidence": _as_mapping(row.get("gate_evidence")),
        "sha256": row.get("sha256"),
    }


def _expected_record(spec: SourceSpec) -> JsonDict:
    return {
        "experiment_id": spec.experiment_id,
        "task_id": spec.task_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "ready_field": spec.ready_field,
    }


def _status_for_source(spec: SourceSpec, payload: Mapping[str, Any]) -> str:
    if not payload:
        return "missing"
    if _is_gate_blocked(payload):
        return "gate_blocked"
    if payload.get(spec.ready_field) is True:
        return "complete"
    return "blocked"


def _runtime_receipt_status(
    rows: list[Mapping[str, Any]], payloads: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    root_cause = payloads.get("exp3247", {})
    smoke = payloads.get("exp3248", {})
    llama = payloads.get("exp3249", {})
    sota = payloads.get("exp3250", {})
    smoke_passed = smoke.get("cuda_python_smoke_passed") is True
    llama_ready = llama.get("llama_cpp_cuda_receipt_ready") is True
    sota_ready = sota.get("sota_gguf_receipt_ready") is True
    clean_rerun_allowed = sota.get("clean_rerun_allowed") is True
    selected_state = (
        "blocked_root_cause_surgery_next_smoke_not_allowed"
        if root_cause.get("next_smoke_allowed") is not True
        else "selected_python_cuda_smoke_passed"
        if smoke_passed
        else "blocked_selected_python_cuda_smoke"
    )
    return {
        "selected_python_cuda": {
            "artifact_status": _row_status(rows, "exp3248"),
            "root_cause_class": str(root_cause.get("cuda_root_cause_class") or ""),
            "next_smoke_allowed": root_cause.get("next_smoke_allowed") is True,
            "selected_python_cuda_repaired_candidate": (
                root_cause.get("selected_python_cuda_repaired_candidate") is True
            ),
            "cuda_python_smoke_passed": smoke_passed,
            "state": selected_state,
        },
        "llama_cpp_cuda": {
            "artifact_status": _row_status(rows, "exp3249"),
            "llama_cpp_cuda_receipt_ready": llama_ready,
        },
        "sota_gguf_receipt": {
            "artifact_status": _row_status(rows, "exp3250"),
            "sota_gguf_receipt_ready": sota_ready,
            "mandatory_model_receipt_count": _int_value(sota.get("mandatory_model_receipt_count")),
            "models_used": list(sota.get("models_used") or []),
        },
        "clean_rerun_allowed": clean_rerun_allowed,
        "receipt_chain_ready": smoke_passed and llama_ready and sota_ready and clean_rerun_allowed,
        "next_action": str(
            root_cause.get("recommended_next_task")
            or payloads.get("exp3246", {}).get("next_top_gap")
            or "repair_selected_python_torch_cuda_before_exp3248"
        ),
    }


def _prompt_injection_status(
    rows: list[Mapping[str, Any]], payloads: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    manifest = payloads.get("exp3251", {})
    labels = payloads.get("exp3252", {})
    kan = payloads.get("exp3253", {})
    repair = payloads.get("exp3254", {})
    manifest_ready = manifest.get("v4_manifest_v2_ready") is True
    control_plan_ready = manifest.get("constraint_tax_control_plan_ready") is True
    teacher_ready = labels.get("teacher_label_shard_ready") is True
    train_eval_completed = kan.get("train_eval_completed") is True or kan.get("train_eval_shard_ready") is True
    repair_ready = repair.get("structured_proposal_preflight_ready") is True
    tax_status = (
        "measured"
        if teacher_ready and "constraint_tax_delta_accuracy_or_parse" in labels
        else "plan_ready_no_measurement"
        if control_plan_ready
        else "missing_constraint_tax_plan"
    )
    return {
        "manifest_v2": {
            "artifact_status": _row_status(rows, "exp3251"),
            "v4_manifest_v2_ready": manifest_ready,
            "constraint_tax_control_plan_ready": control_plan_ready,
            "garak_config_ready": manifest.get("garak_config_ready") is True,
            "no_llm_invoked": manifest.get("no_llm_invoked") is True,
        },
        "constraint_tax_diagnostic": {
            "status": tax_status,
            "constraint_tax_delta_accuracy_or_parse": labels.get(
                "constraint_tax_delta_accuracy_or_parse"
            ),
        },
        "teacher_labels": {
            "artifact_status": _row_status(rows, "exp3252"),
            "teacher_label_shard_ready": teacher_ready,
            "completed_free_reasoning_count": _int_value(labels.get("completed_free_reasoning_count")),
            "completed_schema_constrained_count": _int_value(labels.get("completed_schema_constrained_count")),
        },
        "kan_shard": {
            "artifact_status": _row_status(rows, "exp3253"),
            "train_eval_completed": train_eval_completed,
            "headline_claim_allowed": kan.get("headline_claim_allowed") is True,
        },
        "repair_proposal_preflight": {
            "artifact_status": _row_status(rows, "exp3254"),
            "structured_proposal_preflight_ready": repair_ready,
            "repair_acceptance_claimed": repair.get("repair_acceptance_claimed") is True,
        },
        "publication_evidence_ready": (
            manifest_ready and teacher_ready and train_eval_completed and repair_ready
        ),
    }


def _fr11_lifelong_status(
    rows: list[Mapping[str, Any]], payloads: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    fr11 = payloads.get("exp3255", {})
    return {
        "artifact_status": _row_status(rows, "exp3255"),
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


def _pdit_potts_status(
    rows: list[Mapping[str, Any]], payloads: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    pdit = payloads.get("exp3256", {})
    return {
        "artifact_status": _row_status(rows, "exp3256"),
        "pdit_potts_mapping_ready": pdit.get("pdit_potts_mapping_ready") is True,
        "candidate_verifier_row_type_count": len(pdit.get("candidate_verifier_row_types") or []),
        "q_state_energy_mapping_count": len(pdit.get("q_state_energy_mapping") or []),
        "exact_fallback_preserved": pdit.get("exact_fallback_preserved") is True,
        "hardware_speedup_claim_allowed": pdit.get("hardware_speedup_claim_allowed") is True,
        "retired_pimi_scope_reopened": pdit.get("retired_pimi_scope_reopened") is True,
        "thrml_scaling_sweep_reopened": pdit.get("thrml_scaling_sweep_reopened") is True,
    }


def _required_evidence_exists(
    runtime_status: Mapping[str, Any], prompt_status: Mapping[str, Any]
) -> bool:
    return (
        runtime_status.get("receipt_chain_ready") is True
        and _as_mapping(prompt_status.get("teacher_labels")).get("teacher_label_shard_ready") is True
        and _as_mapping(prompt_status.get("kan_shard")).get("train_eval_completed") is True
        and _as_mapping(prompt_status.get("repair_proposal_preflight")).get(
            "structured_proposal_preflight_ready"
        )
        is True
    )


def _next_top_gap(
    runtime_status: Mapping[str, Any],
    prompt_status: Mapping[str, Any],
    fr11_status: Mapping[str, Any],
    pdit_status: Mapping[str, Any],
) -> str:
    if runtime_status.get("receipt_chain_ready") is not True:
        return str(runtime_status.get("next_action") or "repair_selected_python_cuda_runtime")
    if prompt_status.get("publication_evidence_ready") is not True:
        return "prompt_injection_teacher_labels_and_kan_shard_after_sota_receipt"
    if _as_mapping(prompt_status.get("repair_proposal_preflight")).get(
        "structured_proposal_preflight_ready"
    ) is not True:
        return "dccd_severa_preflight_after_clean_sota_receipt"
    if fr11_status.get("lifelong_eval_ready") is not True:
        return "fr11_lifelong_retention_audit"
    if pdit_status.get("pdit_potts_mapping_ready") is not True:
        return "pdit_potts_diagnostic_mapping"
    return "publication_blocker_retirement_review"


def _publication_blockers(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row.get("experiment_id") or ""),
            "task_id": str(row.get("task_id") or ""),
            "status": _normal_status(str(row.get("status") or "missing")),
            "role": str(row.get("role") or ""),
            "present": row.get("present") is True,
        }
        for row in rows
        if _normal_status(str(row.get("status") or "missing")) != "complete"
    ]


def _prior_publication_blocker_count(archive: Mapping[str, Any]) -> int:
    count = _int_value(archive.get("prior_publication_blocker_count"))
    return count if count > 0 else PRIOR_PUBLICATION_BLOCKER_COUNT


def _invariant_violations(payloads: Mapping[str, Mapping[str, Any]]) -> list[str]:
    violations: list[str] = []
    if payloads.get("exp3246", {}).get("archive_v300_activate_v301_ready") is not True:
        violations.append("exp3246 archive/activation artifact is missing or not ready")
    return violations


def _payload_summary(payload: Mapping[str, Any]) -> JsonDict:
    keys = (
        "archive_v300_activate_v301_ready",
        "next_smoke_allowed",
        "cuda_python_smoke_passed",
        "llama_cpp_cuda_receipt_ready",
        "sota_gguf_receipt_ready",
        "clean_rerun_allowed",
        "v4_manifest_v2_ready",
        "constraint_tax_control_plan_ready",
        "teacher_label_shard_ready",
        "train_eval_completed",
        "structured_proposal_preflight_ready",
        "lifelong_eval_ready",
        "pdit_potts_mapping_ready",
        "status",
        "blocked_at_layer",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _gate_evidence(
    spec: SourceSpec, conductor_log: str, payload: Mapping[str, Any] | None = None
) -> JsonDict:
    payload_map = _as_mapping(payload)
    if _is_gate_blocked(payload_map):
        return {
            "status": "gate_blocked",
            "source": "artifact",
            "summary": str(payload_map.get("gate_check_summary") or ""),
        }
    if not spec.gate_phrase or not conductor_log:
        return {"status": "absent"}
    for line in conductor_log.splitlines():
        if spec.gate_phrase in line and "GATE_BLOCK" in line:
            return {
                "status": "gate_blocked",
                "source": CONDUCTOR_LOG_REL_PATH.as_posix(),
                "line": line.strip(),
            }
    return {"status": "absent"}


def _is_gate_blocked(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema") == "blocked_gate_check_v1"
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
        or str(payload.get("honest_verdict") or "").startswith("blocked_gate_check")
    )


def _row_status(rows: list[Mapping[str, Any]], experiment_id: str) -> str:
    for row in rows:
        if row.get("experiment_id") == experiment_id:
            return _normal_status(str(row.get("status") or "missing"))
    return "missing"


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[_normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "Matrix v34 reads checked-in artifacts and conductor gate lines only.",
        "missing_is_not_success": "Absent downstream JSONs stay in artifacts_missing.",
        "gate_blocks_are_evidence": "Conductor pre-gates are represented without fabricating payload fields.",
        "paper_ready_rule": "paper_ready requires zero blockers and all required receipts/downstream evidence.",
        "no_model_weight_learning_claim": "FR-11 retention is controller-memory evidence only.",
    }


def _normal_status(status: str) -> str:
    normalized = status.strip().lower().replace("-", "_")
    return normalized if normalized in STATUSES else "missing"


def _bool_value(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


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
        "artifacts_missing": artifact.get("artifacts_missing"),
        "gate_blocked_artifacts": artifact.get("gate_blocked_artifacts"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "paper_ready": artifact.get("paper_ready"),
        "next_top_gap": artifact.get("next_top_gap"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v34_ready="
        f"{str(artifact.get('matrix_v34_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"artifacts_found={len(artifact.get('artifacts_found') or [])}; "
        f"artifacts_missing={len(artifact.get('artifacts_missing') or [])}; "
        f"gate_blocked_artifacts={len(artifact.get('gate_blocked_artifacts') or [])}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )
