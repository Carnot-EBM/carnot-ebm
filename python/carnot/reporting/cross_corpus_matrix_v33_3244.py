"""Build the Exp 3244 cross-corpus matrix v33 artifact.

Spec refs: REQ-REPORT-3244, SCENARIO-REPORT-3244.

Matrix v33 is an evidence ledger for milestone `.300`.  It reads the checked-in
runtime, prompt-injection, DCCD proposal, and FR-11 artifacts and records what
is complete, gated, blocked, or missing.  It does not run models, hardware,
teacher labeling, repair, or the conductor because the matrix is only allowed
to aggregate evidence that already exists.
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
MILESTONE = "2026.05.300"
SCHEMA_VERSION = "carnot.cross_corpus_matrix.v33_300_artifact_aggregation.v1"
EXPERIMENT_ID = "exp3244"
TASK_ID = "exp3244-cross-corpus-matrix-v33"
ARTIFACT = "experiment_3244_cross_corpus_matrix_v33"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3244_cross_corpus_matrix_v33.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3244_cross_corpus_matrix_v33.py"

PREVIOUS_MATRIX_REL_PATH = Path("results/experiment_3231_cross_corpus_matrix_v32.json")
CAPSTONE_V299_REL_PATH = Path("results/experiment_3223_capstone_v299.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")

EXP3233_REL_PATH = Path("results/experiment_3233_archive_v299_activate_v300.json")
EXP3234_REL_PATH = Path("results/experiment_3234_cli_backend_failure_root_cause_ledger_v1.json")
EXP3235_REL_PATH = Path("results/experiment_3235_cuda_driver_boundary_operator_package_v1.json")
EXP3236_REL_PATH = Path("results/experiment_3236_isolated_cuda_python_smoke_v1.json")
EXP3237_REL_PATH = Path("results/experiment_3237_llama_cpp_cuda_receipt_smoke_v2.json")
EXP3238_REL_PATH = Path("results/experiment_3238_sota_gguf_receipt_v7.json")
EXP3239_REL_PATH = Path("results/experiment_3239_prompt_injection_kan_v4_resource_manifest_v1.json")
EXP3240_REL_PATH = Path("results/experiment_3240_prompt_injection_kan_teacher_label_shard_v1.json")
EXP3241_REL_PATH = Path("results/experiment_3241_prompt_injection_kan_train_eval_shard_v1.json")
EXP3242_REL_PATH = Path("results/experiment_3242_dccd_exact_row_structured_proposal_preflight_v1.json")
EXP3243_REL_PATH = Path("results/experiment_3243_fr11_failure_memory_controller_v1.json")

STATUSES = ("complete", "blocked", "gate_blocked", "missing", "partial")
PUBLICATION_BLOCKING_STATUSES = {"blocked", "gate_blocked", "missing"}
REQUIRED_ARTIFACT_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "cross_corpus_matrix_v33_ready",
    "artifact_inventory",
    "runtime_receipt_state",
    "prompt_injection_v4_state",
    "structured_proposal_state",
    "fr11_failure_memory_state",
    "paper_ready",
    "publication_blocker_count",
    "next_top_gap",
    "honest_verdict",
}


@dataclass(frozen=True)
class SourceSpec:
    """One planned `.300` artifact that matrix v33 must account for."""

    experiment_id: str
    task_id: str
    path: Path
    role: str
    ready_field: str
    gate_title: str = ""


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp3233",
        "exp3233-archive-v299-activate-v300",
        EXP3233_REL_PATH,
        "archive_v299_activate_v300",
        "archive_v299_activate_v300_ready",
    ),
    SourceSpec(
        "exp3234",
        "exp3234-cli-backend-failure-root-cause-ledger-v1",
        EXP3234_REL_PATH,
        "prompt_injection_cli_failure_ledger",
        "split_run_plan_ready",
    ),
    SourceSpec(
        "exp3235",
        "exp3235-cuda-driver-boundary-operator-package-v1",
        EXP3235_REL_PATH,
        "cuda_driver_boundary_package",
        "cuda_boundary_package_ready",
    ),
    SourceSpec(
        "exp3236",
        "exp3236-isolated-cuda-python-smoke-v1",
        EXP3236_REL_PATH,
        "isolated_cuda_python_smoke",
        "cuda_python_smoke_passed",
    ),
    SourceSpec(
        "exp3237",
        "exp3237-llama-cpp-cuda-receipt-smoke-v2",
        EXP3237_REL_PATH,
        "llama_cpp_cuda_receipt_smoke",
        "llama_cpp_cuda_receipt_ready",
        "llama.cpp CUDA receipt smoke v2 gated",
    ),
    SourceSpec(
        "exp3238",
        "exp3238-sota-gguf-receipt-v7",
        EXP3238_REL_PATH,
        "mandated_sota_gguf_receipt_v7",
        "sota_gguf_receipt_ready",
        "Mandated local SOTA GGUF receipt v7 gated",
    ),
    SourceSpec(
        "exp3239",
        "exp3239-prompt-injection-kan-v4-resource-manifest-v1",
        EXP3239_REL_PATH,
        "prompt_injection_v4_resource_manifest",
        "v4_manifest_ready",
    ),
    SourceSpec(
        "exp3240",
        "exp3240-prompt-injection-kan-teacher-label-shard-v1",
        EXP3240_REL_PATH,
        "prompt_injection_v4_teacher_label_shard",
        "teacher_label_shard_ready",
        "Prompt-injection KAN v4 teacher-label shard gated",
    ),
    SourceSpec(
        "exp3241",
        "exp3241-prompt-injection-kan-train-eval-shard-v1",
        EXP3241_REL_PATH,
        "prompt_injection_v4_train_eval_shard",
        "train_eval_shard_ready",
        "Prompt-injection KAN v4 shard train/eval",
    ),
    SourceSpec(
        "exp3242",
        "exp3242-dccd-exact-row-structured-proposal-preflight-v1",
        EXP3242_REL_PATH,
        "dccd_exact_row_structured_proposal_preflight",
        "structured_proposal_preflight_ready",
        "DCCD exact-row structured proposal preflight gated",
    ),
    SourceSpec(
        "exp3243",
        "exp3243-fr11-failure-memory-controller-v1",
        EXP3243_REL_PATH,
        "fr11_failure_memory_controller",
        "fr11_controller_update_ready",
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read JSON evidence and fail closed on absent or malformed artifacts."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash source bytes so matrix rows can be tied back to exact inputs."""

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
    """REQ-REPORT-3244: aggregate matrix v33 from checked-in `.300` evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    previous_matrix = read_json_object(root_path / PREVIOUS_MATRIX_REL_PATH)
    capstone_v299 = read_json_object(root_path / CAPSTONE_V299_REL_PATH)
    conductor_log = _read_text(root_path / CONDUCTOR_LOG_REL_PATH)
    rows = [_source_row(root_path, spec, conductor_log) for spec in SOURCE_SPECS]
    payloads = {row["experiment_id"]: _as_mapping(row.get("payload")) for row in rows}
    public_rows = [_public_row(row) for row in rows]
    inventory = _artifact_inventory(public_rows)
    runtime_state = _runtime_receipt_state(public_rows, payloads)
    prompt_state = _prompt_injection_v4_state(public_rows, payloads)
    proposal_state = _structured_proposal_state(public_rows, payloads)
    fr11_state = _fr11_failure_memory_state(payloads)
    blockers = _publication_blockers(public_rows)
    prior_count = _prior_publication_blocker_count(previous_matrix, capstone_v299)
    publication_blocker_count = prior_count + len(blockers)
    invariant_violations = _invariant_violations(previous_matrix, capstone_v299, public_rows)
    matrix_ready = not invariant_violations
    paper_ready = bool(
        matrix_ready
        and previous_matrix.get("paper_ready") is True
        and capstone_v299.get("paper_ready") is True
        and publication_blocker_count == 0
        and runtime_state["receipt_chain_ready"] is True
        and prompt_state["publication_evidence_ready"] is True
        and proposal_state["structured_proposal_preflight_ready"] is True
        and fr11_state["fr11_controller_update_ready"] is True
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
        "cross_corpus_matrix_v33_ready": matrix_ready,
        "prior_authorities": _prior_authorities(previous_matrix, capstone_v299),
        "artifact_inventory": inventory,
        "matrix_rows": [
            _matrix_row("cuda_receipt_chain", runtime_state),
            _matrix_row("prompt_injection_split_run", prompt_state),
            _matrix_row("dccd_structured_proposal_preflight", proposal_state),
            _matrix_row("fr11_failure_memory", fr11_state),
        ],
        "runtime_receipt_state": runtime_state,
        "prompt_injection_v4_state": prompt_state,
        "structured_proposal_state": proposal_state,
        "fr11_failure_memory_state": fr11_state,
        "publication_blockers": blockers,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": publication_blocker_count,
        "publication_blocker_delta_from_v299": publication_blocker_count
        - _int_value(capstone_v299.get("publication_blocker_count")),
        "paper_ready": paper_ready,
        "next_top_gap": _next_top_gap(runtime_state, prompt_state, proposal_state, fr11_state),
        "source_checksums": {
            row["path"]: row["sha256"]
            for row in public_rows
            if row.get("readable_json_object") is True
        },
        "protected_files_untouched": {CONDUCTOR_REL_PATH.as_posix(): True},
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
    """Build and persist the Exp 3244 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the matrix violates the required schema or honest verdict rule."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3244")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3244-cross-corpus-matrix-v33")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.300")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must begin with complete:")
    if _int_value(artifact.get("publication_blocker_count")) < 0:
        raise ValueError("publication_blocker_count must be non-negative")


def _source_row(root: Path, spec: SourceSpec, conductor_log: str) -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path)
    status, rationale = _classify(spec, payload)
    return {
        "experiment_id": spec.experiment_id,
        "task_id": spec.task_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "ready_field": spec.ready_field,
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "payload": payload,
        "status": status,
        "status_rationale": rationale,
        "sha256": sha256_file(path),
        "gated_skip_evidence": _gate_evidence(spec, conductor_log, payload),
    }


def _classify(spec: SourceSpec, payload: Mapping[str, Any]) -> tuple[str, str]:
    if not payload:
        return "missing", "planned `.300` artifact is absent or malformed"
    if _is_gate_blocked(payload):
        return "gate_blocked", "artifact was blocked by a conductor pre-gate"
    if _normal_status(str(payload.get("status") or "")) == "partial":
        return "partial", "artifact reports partial evidence only"
    if payload.get(spec.ready_field) is True:
        return "complete", f"{spec.ready_field}=true"
    return "blocked", f"{spec.ready_field} is not true"


def _public_row(row: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(row.get("payload"))
    gate = _as_mapping(row.get("gated_skip_evidence"))
    status = _normal_status(str(row.get("status") or "missing"))
    public = {
        "experiment_id": str(row.get("experiment_id") or ""),
        "task_id": str(row.get("task_id") or ""),
        "path": str(row.get("path") or ""),
        "role": str(row.get("role") or ""),
        "ready_field": str(row.get("ready_field") or ""),
        "present": row.get("present") is True,
        "readable_json_object": row.get("readable_json_object") is True,
        "status": status,
        "status_rationale": str(row.get("status_rationale") or ""),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "reported_experiment_id": _reported_experiment_id(
            payload, str(row.get("experiment_id") or "")
        ),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": row.get("sha256"),
        "gated_skip_evidence": gate,
        "summary": _payload_summary(payload),
    }
    public["publication_blocker"] = _row_blocks_publication(public)
    return public


def _artifact_inventory(rows: list[Mapping[str, Any]]) -> JsonDict:
    return {
        "planned_artifacts": rows,
        "complete_artifacts": [_inventory_record(row) for row in rows if row["status"] == "complete"],
        "blocked_artifacts": [_inventory_record(row) for row in rows if row["status"] == "blocked"],
        "gate_blocked_artifacts": [
            _inventory_record(row)
            for row in rows
            if row["status"] == "gate_blocked"
            or _as_mapping(row.get("gated_skip_evidence")).get("status") == "gate_blocked"
        ],
        "missing_artifacts": [_inventory_record(row) for row in rows if row["status"] == "missing"],
        "partial_artifacts": [_inventory_record(row) for row in rows if row["status"] == "partial"],
        "status_counts": _status_counts(rows),
    }


def _inventory_record(row: Mapping[str, Any]) -> JsonDict:
    record = {
        "experiment_id": str(row.get("experiment_id") or ""),
        "task_id": str(row.get("task_id") or ""),
        "path": str(row.get("path") or ""),
        "role": str(row.get("role") or ""),
        "status": _normal_status(str(row.get("status") or "missing")),
        "ready_field": str(row.get("ready_field") or ""),
        "status_rationale": str(row.get("status_rationale") or ""),
    }
    gate = _as_mapping(row.get("gated_skip_evidence"))
    if gate.get("status") == "gate_blocked":
        record["gated_skip_evidence"] = gate
    return record


def _runtime_receipt_state(
    rows: list[Mapping[str, Any]], payloads: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    exp3235 = payloads.get("exp3235", {})
    exp3236 = payloads.get("exp3236", {})
    cuda_smoke = exp3236.get("cuda_python_smoke_passed") is True
    llama_ready = _row_status(rows, "exp3237") == "complete"
    sota_ready = _row_status(rows, "exp3238") == "complete"
    if cuda_smoke and llama_ready and sota_ready:
        state = "complete_runtime_receipt_chain_ready"
    elif not cuda_smoke:
        state = "blocked_selected_python_cuda_smoke_failed"
    elif not llama_ready:
        state = "gate_blocked_llama_cpp_cuda_receipt_missing"
    else:
        state = "gate_blocked_sota_gguf_receipt_missing"
    next_action = str(
        exp3236.get("recommended_next_task")
        or exp3235.get("recommended_next_task")
        or "repair_selected_python_torch_cuda_before_exp3237"
    )
    return {
        "state": state,
        "cuda_boundary_package_ready": exp3235.get("cuda_boundary_package_ready") is True,
        "full_gguf_rerun_allowed_now": exp3235.get("full_gguf_rerun_allowed_now") is True,
        "cuda_driver_visible": exp3236.get("cuda_driver_visible") is True,
        "selected_python_torch_cuda_available": (
            exp3236.get("selected_python_torch_cuda_available") is True
        ),
        "selected_python_device_count": _int_value(exp3236.get("selected_python_device_count")),
        "cuda_bindings_device_count": _int_value(exp3236.get("cuda_bindings_device_count")),
        "cuda_python_smoke_passed": cuda_smoke,
        "llama_cpp_cuda_receipt_ready": llama_ready,
        "sota_gguf_receipt_ready": sota_ready,
        "receipt_chain_ready": cuda_smoke and llama_ready and sota_ready,
        "blocking_artifacts": _blocking_ids(rows, ("exp3236", "exp3237", "exp3238")),
        "next_action": next_action,
    }


def _prompt_injection_v4_state(
    rows: list[Mapping[str, Any]], payloads: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    manifest = payloads.get("exp3239", {})
    manifest_ready = manifest.get("v4_manifest_ready") is True
    teacher_status = _row_status(rows, "exp3240")
    train_status = _row_status(rows, "exp3241")
    publication_ready = manifest_ready and teacher_status == "complete" and train_status == "complete"
    if publication_ready:
        state = "complete_prompt_injection_v4_split_run_ready"
    elif manifest_ready:
        state = "blocked_after_manifest_teacher_label_shard_gate_blocked"
    else:
        state = "blocked_prompt_injection_v4_manifest_missing"
    return {
        "state": state,
        "manifest_ready": manifest_ready,
        "teacher_label_plan_ready": manifest.get("teacher_label_plan_ready") is True,
        "delong_plan_ready": manifest.get("delong_plan_ready") is True,
        "garak_config_ready": manifest.get("garak_config_ready") is True,
        "teacher_label_shard_status": teacher_status,
        "train_eval_shard_status": train_status,
        "publication_evidence_ready": publication_ready,
        "no_llm_invoked_by_manifest": manifest.get("no_llm_invoked") is True,
        "no_teacher_labels_claimed_by_manifest": manifest.get("no_new_teacher_labeling") is True,
        "no_kan_training_claimed_by_manifest": manifest.get("no_kan_training") is True,
        "blocking_artifacts": _blocking_ids(rows, ("exp3240", "exp3241")),
    }


def _structured_proposal_state(
    rows: list[Mapping[str, Any]], payloads: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    proposal = payloads.get("exp3242", {})
    status = _row_status(rows, "exp3242")
    ready = proposal.get("structured_proposal_preflight_ready") is True and status == "complete"
    if ready:
        state = "complete_dccd_structured_proposal_preflight_ready"
    elif status == "missing" and _gate_status(rows, "exp3242") == "gate_blocked":
        state = "missing_gate_blocked_on_exp3238_clean_rerun_allowed"
    elif status == "gate_blocked":
        state = "gate_blocked_on_exp3238_clean_rerun_allowed"
    else:
        state = "blocked_dccd_structured_proposal_preflight"
    return {
        "state": state,
        "artifact_status": status,
        "upstream_clean_rerun_allowed": proposal.get("upstream_clean_rerun_allowed") is True,
        "structured_proposal_preflight_ready": ready,
        "repair_acceptance_claimed": proposal.get("repair_acceptance_claimed") is True,
        "blocking_artifacts": _blocking_ids(rows, ("exp3242",)),
    }


def _fr11_failure_memory_state(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    fr11 = payloads.get("exp3243", {})
    ready = fr11.get("fr11_controller_update_ready") is True
    no_training = (
        fr11.get("model_weight_update_claimed") is False
        and fr11.get("controller_memory_updates_are_not_training") is True
    )
    state = (
        "ready_controller_memory_update_no_model_weight_training"
        if ready and no_training
        else "blocked_fr11_failure_memory_or_training_boundary"
    )
    return {
        "state": state,
        "fr11_controller_update_ready": ready,
        "failure_trace_count": _int_value(fr11.get("failure_trace_count")),
        "heldout_replay_count": _int_value(fr11.get("heldout_replay_count")),
        "heldout_replay_delta": _int_value(fr11.get("heldout_replay_delta")),
        "nonforgetting_delta": _int_value(fr11.get("nonforgetting_delta")),
        "stale_premise_rejection_count": _int_value(fr11.get("stale_premise_rejection_count")),
        "doomed_rerun_avoidance_count": _int_value(fr11.get("doomed_rerun_avoidance_count")),
        "model_weight_update_claimed": fr11.get("model_weight_update_claimed") is True,
        "controller_memory_updates_are_not_training": (
            fr11.get("controller_memory_updates_are_not_training") is True
        ),
    }


def _publication_blockers(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [_inventory_record(row) for row in rows if _row_blocks_publication(row)]


def _row_blocks_publication(row: Mapping[str, Any]) -> bool:
    return _normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES


def _prior_authorities(matrix: Mapping[str, Any], capstone: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_v32": {
            "path": PREVIOUS_MATRIX_REL_PATH.as_posix(),
            "ready": matrix.get("cross_corpus_matrix_v32_ready") is True,
            "paper_ready": matrix.get("paper_ready") is True,
            "publication_blocker_count": _int_value(matrix.get("publication_blocker_count")),
            "next_top_gap": str(matrix.get("next_top_gap") or ""),
        },
        "capstone_v299": {
            "path": CAPSTONE_V299_REL_PATH.as_posix(),
            "ready": capstone.get("capstone_v299_ready") is True,
            "paper_ready": capstone.get("paper_ready") is True,
            "publication_blocker_count": _int_value(capstone.get("publication_blocker_count")),
            "v4_outcome": str(capstone.get("v4_outcome") or ""),
            "next_top_gap": str(capstone.get("next_top_gap") or ""),
        },
    }


def _prior_publication_blocker_count(
    matrix: Mapping[str, Any], capstone: Mapping[str, Any]
) -> int:
    return max(
        _int_value(matrix.get("publication_blocker_count")),
        _int_value(capstone.get("publication_blocker_count")),
    )


def _invariant_violations(
    matrix: Mapping[str, Any], capstone: Mapping[str, Any], rows: list[Mapping[str, Any]]
) -> list[str]:
    violations: list[str] = []
    if matrix.get("cross_corpus_matrix_v32_ready") is not True:
        violations.append("prior matrix v32 is missing or not ready")
    if capstone.get("capstone_v299_ready") is not True:
        violations.append("prior capstone v299 is missing or not ready")
    if len(rows) != len(SOURCE_SPECS):
        violations.append("artifact inventory does not cover every planned .300 source")
    return violations


def _next_top_gap(
    runtime_state: Mapping[str, Any],
    prompt_state: Mapping[str, Any],
    proposal_state: Mapping[str, Any],
    fr11_state: Mapping[str, Any],
) -> str:
    if runtime_state.get("receipt_chain_ready") is not True:
        return str(runtime_state.get("next_action") or "repair_selected_python_torch_cuda_before_exp3237")
    if prompt_state.get("publication_evidence_ready") is not True:
        return "prompt_injection_v4_teacher_label_and_train_eval_shards"
    if proposal_state.get("structured_proposal_preflight_ready") is not True:
        return "dccd_structured_proposal_preflight_after_clean_sota_receipt"
    if fr11_state.get("fr11_controller_update_ready") is not True:
        return "fr11_failure_memory_controller_ready_replay"
    return "publication_blocker_retirement_review"


def _matrix_row(row_id: str, state: Mapping[str, Any]) -> JsonDict:
    return {
        "row_id": row_id,
        "state": str(state.get("state") or ""),
        "ready": (
            state.get("receipt_chain_ready") is True
            or state.get("publication_evidence_ready") is True
            or state.get("structured_proposal_preflight_ready") is True
            or state.get("fr11_controller_update_ready") is True
        ),
        "blocking_artifacts": list(state.get("blocking_artifacts") or []),
    }


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "All claims are derived from checked-in artifacts and logs.",
        "missing_is_not_success": "Missing and gated artifacts stay visible as blockers.",
        "paper_ready_rule": "paper_ready requires zero blockers and prior paper-ready authority.",
        "no_training_boundary": "FR-11 controller memory is not treated as model-weight training.",
    }


def _payload_summary(payload: Mapping[str, Any]) -> JsonDict:
    keys = (
        "archive_v299_activate_v300_ready",
        "split_run_plan_ready",
        "cuda_boundary_package_ready",
        "full_gguf_rerun_allowed_now",
        "cuda_python_smoke_passed",
        "llama_cpp_cuda_receipt_ready",
        "sota_gguf_receipt_ready",
        "v4_manifest_ready",
        "teacher_label_shard_ready",
        "train_eval_shard_ready",
        "structured_proposal_preflight_ready",
        "repair_acceptance_claimed",
        "fr11_controller_update_ready",
        "model_weight_update_claimed",
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
    if not spec.gate_title or not conductor_log:
        return {"status": "absent"}
    for line in conductor_log.splitlines():
        if spec.gate_title in line and "GATE_BLOCK" in line:
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


def _blocking_ids(rows: list[Mapping[str, Any]], experiment_ids: tuple[str, ...]) -> list[str]:
    ids: list[str] = []
    for experiment_id in experiment_ids:
        status = _row_status(rows, experiment_id)
        if status in PUBLICATION_BLOCKING_STATUSES:
            ids.append(experiment_id)
    return ids


def _gate_status(rows: list[Mapping[str, Any]], experiment_id: str) -> str:
    for row in rows:
        if row.get("experiment_id") == experiment_id:
            return str(_as_mapping(row.get("gated_skip_evidence")).get("status") or "absent")
    return "absent"


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


def _reported_experiment_id(payload: Mapping[str, Any], fallback: str) -> str:
    if payload.get("experiment_id"):
        return str(payload["experiment_id"])
    experiment = payload.get("experiment")
    if isinstance(experiment, int) and not isinstance(experiment, bool):
        return f"exp{experiment}"
    return fallback


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
        "artifact_inventory": artifact.get("artifact_inventory"),
        "publication_blocker_count": artifact.get("publication_blocker_count"),
        "paper_ready": artifact.get("paper_ready"),
        "next_top_gap": artifact.get("next_top_gap"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    inventory = _as_mapping(artifact.get("artifact_inventory"))
    counts = _as_mapping(inventory.get("status_counts"))
    return (
        "complete: cross_corpus_matrix_v33_ready="
        f"{str(artifact.get('cross_corpus_matrix_v33_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"complete_artifacts={_int_value(counts.get('complete'))}; "
        f"gate_blocked_artifacts={len(inventory.get('gate_blocked_artifacts') or [])}; "
        f"missing_artifacts={_int_value(counts.get('missing'))}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )
