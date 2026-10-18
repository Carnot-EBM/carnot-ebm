"""Build the Exp 3217 cross-corpus matrix v31 artifact.

Spec refs: REQ-REPORT-3217, SCENARIO-REPORT-3217.

Matrix v31 reconciles the `.297` milestone artifacts against matrix v30.  It is
claim accounting, not experiment execution: it reads existing JSON/log evidence,
counts blockers, and keeps hardware and self-learning claims inside the
boundaries proved by checked-in transcripts.
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
RUN_DATE = "20260527"
MILESTONE = "2026.05.297"
SCHEMA_VERSION = "carnot.cross_corpus_matrix.v31_297_artifact_aggregation.v1"
EXPERIMENT_ID = "exp3217"
ARTIFACT = "experiment_3217_cross_corpus_matrix_v31"
MATRIX_VERSION = "v31"
PREVIOUS_MATRIX_REL_PATH = Path("results/experiment_3203_cross_corpus_matrix_v30.json")
OUTPUT_REL_PATH = Path("results/experiment_3217_cross_corpus_matrix_v31.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3217_cross_corpus_matrix_v31.py"
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
NEXT_TOP_GAP = "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"

EXP3205_REL_PATH = Path("results/experiment_3205_archive_v296_activate_v297.json")
EXP3206_REL_PATH = Path("results/experiment_3206_cuda_env_forensics_ledger_v1.json")
EXP3207_REL_PATH = Path("results/experiment_3207_llama_cpp_cuda_rebuild_clean_subprocess_v1.json")
EXP3208_REL_PATH = Path("results/experiment_3208_full_local_sota_receipt_v5.json")
EXP3209_REL_PATH = Path("results/experiment_3209_clean_live_sota_verifier_rerun_v12.json")
EXP3210_REL_PATH = Path(
    "results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json"
)
EXP3211_REL_PATH = Path("results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json")
EXP3212_REL_PATH = Path("results/experiment_3212_structured_repair_proposal_preflight_v1.json")
EXP3213_REL_PATH = Path("results/experiment_3213_repair_gate_decision_v6.json")
EXP3214_REL_PATH = Path("results/experiment_3214_multi_turn_repair_ladder_v7.json")
EXP3215_REL_PATH = Path("results/experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.json")
EXP3216_REL_PATH = Path("results/experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.json")

STATUSES = ("clean", "blocked", "gated_skipped", "diagnostic_only", "retired", "missing")
PUBLICATION_BLOCKING_STATUSES = {"blocked", "gated_skipped", "missing"}


@dataclass(frozen=True)
class SourceSpec:
    """One `.297` artifact that v31 must account for, even when absent."""

    experiment_id: str
    path: Path
    role: str
    source_field: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3205", EXP3205_REL_PATH, "archive_v296_activate_v297", "activation_ready"),
    SourceSpec("exp3206", EXP3206_REL_PATH, "cuda_environment_forensics", "cuda_init_clean"),
    SourceSpec("exp3207", EXP3207_REL_PATH, "llama_cpp_cuda_rebuild_gate", "cuda_receipt_ready"),
    SourceSpec("exp3208", EXP3208_REL_PATH, "full_local_sota_receipt_v5", "clean_rerun_allowed"),
    SourceSpec("exp3209", EXP3209_REL_PATH, "clean_live_sota_verifier_v12", "clean_verifier_ready"),
    SourceSpec("exp3210", EXP3210_REL_PATH, "context_shortcut_fixtures", "ready_for_clean_verifier"),
    SourceSpec("exp3211", EXP3211_REL_PATH, "constraintbench_fixture_pilot", "ready_for_clean_verifier"),
    SourceSpec("exp3212", EXP3212_REL_PATH, "structured_repair_preflight", "ready_for_repair_gate"),
    SourceSpec("exp3213", EXP3213_REL_PATH, "repair_gate_decision_v6", "repair_gate_state"),
    SourceSpec("exp3214", EXP3214_REL_PATH, "multi_turn_repair_ladder_v7", "repair_ladder_complete"),
    SourceSpec("exp3215", EXP3215_REL_PATH, "fr11_trace_replay_controller_v2", "promotion_allowed"),
    SourceSpec(
        "exp3216",
        EXP3216_REL_PATH,
        "fr11_grounded_continuation_queue",
        "controller_memory_promotion_allowed",
    ),
)

GATE_SKIP_TITLES = {
    "exp3209": "Clean live SOTA verifier rerun v12",
    "exp3212": "Structured repair proposal preflight",
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object from disk and fail closed on absent or malformed files."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum for loaded evidence."""

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
    """REQ-REPORT-3217: aggregate v31 from checked-in `.297` evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    previous_matrix = read_json_object(root_path / PREVIOUS_MATRIX_REL_PATH)
    conductor_log = _read_text(root_path / CONDUCTOR_LOG_REL_PATH)
    sources = [_source_payload(root_path, spec, conductor_log) for spec in SOURCE_SPECS]
    rows = [_public_source(row) for row in sources]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    status_counts = _status_counts(rows)
    prior_count = _prior_publication_blocker_count(previous_matrix)
    new_blockers = _new_publication_blockers(rows)
    publication_blocker_count = (
        prior_count + len(new_blockers) if prior_count is not None else len(new_blockers)
    )
    blocker_delta = (
        publication_blocker_count - prior_count if prior_count is not None else None
    )
    missing_artifacts = _missing_artifacts(rows)
    gated_skipped_artifacts = [
        _blocker_record(row) for row in rows if row["status"] == "gated_skipped"
    ]
    local_sota_receipt_status = _local_sota_receipt_status(payloads, rows)
    clean_verifier_status = _clean_verifier_status(rows)
    repair_gate_status = _repair_gate_status(payloads, rows)
    repair_ladder_status = _repair_ladder_status(rows)
    repair_status = _repair_status(repair_gate_status, repair_ladder_status)
    context_fixture_status = _context_fixture_status(payloads, rows)
    constraintbench_fixture_status = _constraintbench_fixture_status(payloads, rows)
    fr11_boundaries = _fr11_claim_boundaries(payloads)
    fr11_self_learning_status = _fr11_self_learning_status(payloads, rows, fr11_boundaries)
    hardware_boundaries = _hardware_claim_boundaries(payloads, previous_matrix)
    hardware_sampler_status = _hardware_sampler_status(hardware_boundaries)
    required_blockers = _required_evidence_blocked_or_missing(
        local_sota_receipt_status,
        clean_verifier_status,
        repair_status,
        fr11_self_learning_status,
        hardware_sampler_status,
    )
    invariant_violations = _invariant_violations(
        previous_matrix,
        rows,
        status_counts,
        publication_blocker_count,
        prior_count,
        new_blockers,
        hardware_boundaries,
        fr11_boundaries,
    )
    ready = not invariant_violations
    paper_ready = bool(
        ready
        and not required_blockers
        and publication_blocker_count == 0
        and previous_matrix.get("paper_ready") is True
    )
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_version": MATRIX_VERSION,
        "cross_corpus_matrix_v31_ready": ready,
        "previous_matrix_artifact": PREVIOUS_MATRIX_REL_PATH.as_posix(),
        "previous_matrix_summary": _previous_matrix_summary(previous_matrix),
        "upstream_artifacts": rows,
        "source_checksums": {
            str(row["path"]): row["sha256"]
            for row in rows
            if row.get("readable_json_object") is True
        },
        "missing_artifacts": missing_artifacts,
        "gated_skipped_artifacts": gated_skipped_artifacts,
        "status_counts": status_counts,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v30": blocker_delta,
        "new_publication_blockers": new_blockers,
        "publication_blocker_accounting": {
            "previous_matrix": PREVIOUS_MATRIX_REL_PATH.as_posix(),
            "previous_publication_blocker_count": prior_count,
            "new_blocking_artifact_count": len(new_blockers),
            "publication_blocker_count": publication_blocker_count,
        },
        "local_sota_receipt_status": local_sota_receipt_status,
        "clean_verifier_status": clean_verifier_status,
        "repair_status": repair_status,
        "repair_gate_status": repair_gate_status,
        "repair_ladder_status": repair_ladder_status,
        "context_fixture_status": context_fixture_status,
        "constraintbench_fixture_status": constraintbench_fixture_status,
        "fr11_self_learning_status": fr11_self_learning_status,
        "fr11_claim_boundaries": fr11_boundaries,
        "hardware_sampler_status": hardware_sampler_status,
        "hardware_claim_boundaries": hardware_boundaries,
        "required_evidence_blocked_or_missing": required_blockers,
        "paper_ready": paper_ready,
        "next_top_gap": _next_top_gap(
            local_sota_receipt_status,
            clean_verifier_status,
            repair_status,
            fr11_self_learning_status,
            hardware_sampler_status,
        ),
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "ops_docs_updated": False,
        "invariant_violations": invariant_violations,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3217 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_payload(root: Path, spec: SourceSpec, conductor_log: str = "") -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path)
    status, rationale = _classification_status(spec.experiment_id, payload)
    gate_evidence = _gate_skip_evidence(spec.experiment_id, conductor_log)
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "source_field": spec.source_field,
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "payload": payload,
        "status": status,
        "status_rationale": rationale,
        "sha256": sha256_file(path),
        "gated_skip_evidence": gate_evidence,
    }


def _classification_status(experiment_id: str, payload: Mapping[str, Any]) -> tuple[str, str]:
    if not payload:
        return "missing", "expected `.297` artifact is absent or malformed"
    if experiment_id == "exp3205":
        if payload.get("activation_ready") is True:
            return "clean", "archive activation confirmed"
        return "blocked", "archive activation is not ready"
    if experiment_id == "exp3206":
        if payload.get("cuda_env_diagnosed") is True and payload.get("cuda_init_clean") is True:
            return "clean", "selected Python CUDA initialization is clean"
        return "blocked", "selected Python CUDA initialization remains blocked"
    if experiment_id == "exp3207":
        if payload.get("cuda_receipt_ready") is True:
            return "clean", "CUDA receipt gate is ready"
        return "blocked", "CUDA receipt gate is blocked"
    if experiment_id == "exp3208":
        if _is_gate_skip(payload):
            return "gated_skipped", "full local SOTA receipt was skipped by conductor gate"
        if payload.get("clean_rerun_allowed") is True or payload.get("full_local_sota_receipt_ready") is True:
            return "clean", "full local SOTA receipt allows a clean rerun"
        return "blocked", "full local SOTA receipt is not clean"
    if experiment_id == "exp3209":
        if _is_gate_skip(payload):
            return "gated_skipped", "clean verifier was skipped by conductor gate"
        if payload.get("clean_verifier_ready") is True or payload.get("headline_claim_allowed") is True:
            return "clean", "clean live SOTA verifier evidence is ready"
        return "blocked", "clean live SOTA verifier evidence is not ready"
    if experiment_id == "exp3210":
        if _int_value(payload.get("fixture_count")) <= 0:
            return "blocked", "context-shortcut fixture count is absent"
        if payload.get("ready_for_clean_verifier") is True:
            return "clean", "context-shortcut fixtures are ready for clean verifier use"
        return "diagnostic_only", "context-shortcut fixtures exist but are not verifier-ready"
    if experiment_id == "exp3211":
        if _int_value(payload.get("fixture_count")) <= 0:
            return "blocked", "ConstraintBench fixture count is absent"
        if payload.get("ready_for_clean_verifier") is True:
            return "clean", "ConstraintBench pilot fixtures are ready for clean verifier use"
        return "diagnostic_only", "ConstraintBench pilot exists but is not verifier-ready"
    if experiment_id == "exp3212":
        if _is_gate_skip(payload):
            return "gated_skipped", "structured repair preflight was skipped by conductor gate"
        if (
            payload.get("ready_for_repair_gate") is True
            and payload.get("repair_correctness_claimed") is False
        ):
            return "clean", "structured repair preflight is ready without correctness overclaim"
        return "blocked", "structured repair preflight is not ready"
    if experiment_id == "exp3213":
        if str(payload.get("repair_gate_state") or "").startswith("unblocked"):
            return "clean", "repair gate is unblocked"
        return "blocked", "repair gate remains blocked"
    if experiment_id == "exp3214":
        if _is_gate_skip(payload):
            return "gated_skipped", "repair ladder was skipped by conductor gate"
        if payload.get("repair_ladder_complete") is True or payload.get("repair_ladder_ready") is True:
            return "clean", "repair ladder completed"
        return "blocked", "repair ladder is not complete"
    if experiment_id == "exp3215":
        if _model_weight_update_claimed(payload):
            return "blocked", "FR-11 trace replay claims a model-weight update"
        if payload.get("promotion_allowed") is True and _int_value(
            payload.get("negative_control_regression_count")
        ) == 0:
            return "clean", "FR-11 trace replay controller promotion is allowed"
        if _int_value(payload.get("negative_control_regression_count")) == 0:
            return "diagnostic_only", "FR-11 trace replay is model-free but not promoted"
        return "blocked", "FR-11 trace replay reports regressions"
    if experiment_id == "exp3216":
        if _model_weight_update_claimed(payload):
            return "blocked", "FR-11 continuation queue claims a model-weight update"
        if payload.get("controller_memory_promotion_allowed") is True:
            return "clean", "FR-11 continuation queue allows controller-memory promotion"
        if (
            payload.get("nonforgetting_queue_defined") is True
            or _as_mapping(payload.get("inference_substrate")).get("nonforgetting_queue_report_only")
            is True
        ):
            return "diagnostic_only", "FR-11 continuation queue is audit-only"
        return "blocked", "FR-11 continuation queue is not defined"
    return "blocked", "unknown expected source classification"


def _public_source(source: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(source.get("payload"))
    status = _normal_status(str(source.get("status") or "missing"))
    return {
        "experiment_id": str(source.get("experiment_id") or ""),
        "path": str(source.get("path") or ""),
        "role": str(source.get("role") or ""),
        "source_field": str(source.get("source_field") or ""),
        "present": source.get("present") is True,
        "readable_json_object": source.get("readable_json_object") is True,
        "status": status,
        "status_rationale": str(source.get("status_rationale") or ""),
        "publication_blocker": status in PUBLICATION_BLOCKING_STATUSES,
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "reported_experiment_id": _experiment_id(payload, str(source.get("experiment_id") or "")),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": source.get("sha256"),
        "gated_skip_evidence": _as_mapping(source.get("gated_skip_evidence")),
    }


def _previous_matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "cross_corpus_matrix_v30_ready": matrix.get("cross_corpus_matrix_v30_ready") is True,
        "paper_ready": matrix.get("paper_ready") is True,
        "publication_blocker_count": _prior_publication_blocker_count(matrix),
        "next_top_gap": str(matrix.get("next_top_gap") or ""),
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _missing_artifacts(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    missing: list[JsonDict] = []
    for row in rows:
        if row["status"] != "missing":
            continue
        record = _blocker_record(row)
        record["reason"] = "missing_or_malformed_expected_dot297_artifact"
        record["gated_skip_evidence"] = _as_mapping(row.get("gated_skip_evidence"))
        missing.append(record)
    return missing


def _new_publication_blockers(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _blocker_record(row)
        for row in rows
        if _normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES
    ]


def _blocker_record(row: Mapping[str, Any]) -> JsonDict:
    return {
        "experiment_id": str(row.get("experiment_id") or ""),
        "path": str(row.get("path") or ""),
        "role": str(row.get("role") or ""),
        "status": _normal_status(str(row.get("status") or "missing")),
        "source_field": str(row.get("source_field") or ""),
        "status_rationale": str(row.get("status_rationale") or ""),
    }


def _local_sota_receipt_status(
    payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]
) -> str:
    rebuild = payloads["exp3207"]
    receipt_status = _row_status(rows, "exp3208")
    if rebuild.get("cuda_receipt_ready") is True and receipt_status == "clean":
        return "passed_cuda_receipt_ready_full_local_sota_clean_rerun_allowed"
    if rebuild.get("cuda_receipt_ready") is False and receipt_status == "gated_skipped":
        return "blocked_selected_python_torch_cuda_cuda_receipt_ready_false_full_receipt_gated_skipped"
    if _row_status(rows, "exp3208") == "missing":
        return "missing_full_local_sota_receipt_v5"
    return "blocked_no_full_local_sota_receipt"


def _clean_verifier_status(rows: list[Mapping[str, Any]]) -> str:
    status = _row_status(rows, "exp3209")
    row = _row(rows, "exp3209")
    if status == "clean":
        return "clean_live_verifier_v12_ready"
    if status == "gated_skipped":
        return "gated_skipped_clean_verifier_v12_after_full_receipt_gate"
    if status == "missing" and _as_mapping(row.get("gated_skip_evidence")).get("status") == "gated_skipped":
        return "gated_skipped_missing_clean_verifier_v12_after_full_receipt_gate"
    if status == "missing":
        return "missing_clean_verifier_v12"
    return "blocked_clean_verifier_v12"


def _repair_gate_status(payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]) -> str:
    gate = payloads["exp3213"]
    status = _row_status(rows, "exp3213")
    if status == "clean":
        return "unblocked"
    if status == "missing":
        return "missing"
    return str(gate.get("repair_gate_state") or status)


def _repair_ladder_status(rows: list[Mapping[str, Any]]) -> str:
    return _row_status(rows, "exp3214")


def _repair_status(repair_gate_status: str, repair_ladder_status: str) -> str:
    if repair_gate_status == "unblocked" and repair_ladder_status == "clean":
        return "repair_ready"
    if repair_gate_status == "blocked" and repair_ladder_status == "gated_skipped":
        return "repair_gate_blocked_v6_ladder_gated_skipped_v7"
    if repair_gate_status == "missing" or repair_ladder_status == "missing":
        return "missing_repair_gate_or_ladder"
    return "blocked_repair"


def _context_fixture_status(payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]) -> str:
    context = payloads["exp3210"]
    if _row_status(rows, "exp3210") == "missing":
        return "missing_context_shortcut_fixtures"
    count = _int_value(context.get("fixture_count"))
    if context.get("ready_for_clean_verifier") is True and count > 0:
        return f"available_ready_for_clean_verifier_fixture_count_{count}"
    if count > 0:
        return f"available_not_clean_verifier_ready_fixture_count_{count}"
    return "blocked_context_shortcut_fixtures"


def _constraintbench_fixture_status(
    payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]
) -> str:
    constraint = payloads["exp3211"]
    if _row_status(rows, "exp3211") == "missing":
        return "missing_constraintbench_fixture_pilot"
    count = _int_value(constraint.get("fixture_count"))
    verdict = str(constraint.get("honest_verdict") or "")
    if constraint.get("ready_for_clean_verifier") is True and count > 0:
        if "no full ConstraintBench coverage claimed" in verdict:
            return f"available_exact_pilot_fixture_count_{count}_no_full_coverage_claimed"
        return f"available_exact_pilot_fixture_count_{count}"
    if count > 0:
        return f"available_not_clean_verifier_ready_fixture_count_{count}"
    return "blocked_constraintbench_fixture_pilot"


def _fr11_claim_boundaries(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    controller = payloads["exp3215"]
    queue = payloads["exp3216"]
    return {
        "controller_memory_promotion_allowed": controller.get("promotion_allowed") is True,
        "queue_promotion_allowed": queue.get("controller_memory_promotion_allowed") is True,
        "model_weight_update_claimed": (
            _model_weight_update_claimed(controller) or _model_weight_update_claimed(queue)
        ),
    }


def _fr11_self_learning_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
    boundaries: Mapping[str, Any],
) -> str:
    if boundaries.get("model_weight_update_claimed") is True:
        return "blocked_fr11_model_weight_update_claimed"
    controller_status = _row_status(rows, "exp3215")
    queue_status = _row_status(rows, "exp3216")
    if controller_status == "clean" and queue_status == "clean":
        return "controller_trace_replay_and_queue_promoted_no_model_weight_update_claimed"
    if controller_status == "clean" and queue_status == "diagnostic_only":
        return (
            "controller_trace_replay_promoted_nonforgetting_queue_audit_only_"
            "no_model_weight_update_claimed"
        )
    if controller_status == "clean" and queue_status == "missing":
        return "controller_trace_replay_promoted_queue_missing_no_model_weight_update_claimed"
    if not payloads["exp3215"] and not payloads["exp3216"]:
        return "missing_fr11_self_learning_artifacts"
    return "blocked_fr11_self_learning"


def _hardware_claim_boundaries(
    payloads: Mapping[str, Mapping[str, Any]], previous_matrix: Mapping[str, Any]
) -> JsonDict:
    payload_text = json.dumps(payloads, sort_keys=True).lower()
    transcript = any(
        _boolish(payload.get("authenticated_hardware_transcript_present"))
        for payload in payloads.values()
    )
    prior_hardware_status = str(previous_matrix.get("hardware_sampler_status") or "")
    speedup = any(
        _boolish(payload.get("speedup_claim_allowed")) or _boolish(payload.get("hardware_claim_made"))
        for payload in payloads.values()
    )
    speedup = speedup or prior_hardware_status == "authenticated_hardware_speedup_claim_allowed"
    tsu_or_kona = (
        "tsu_or_kona_claim_allowed" in payload_text
        or "kona_execution_claimed" in payload_text
        or "tsu_z1_xtr0_kona_execution_claimed" in payload_text
    )
    return {
        "authenticated_hardware_transcript_present": transcript,
        "speedup_claim_allowed": bool(speedup and transcript),
        "tsu_or_kona_claim_allowed": bool(tsu_or_kona and transcript),
    }


def _hardware_sampler_status(boundaries: Mapping[str, Any]) -> str:
    if boundaries.get("authenticated_hardware_transcript_present") is True and (
        boundaries.get("speedup_claim_allowed") is True
        or boundaries.get("tsu_or_kona_claim_allowed") is True
    ):
        return "authenticated_hardware_claim_allowed"
    return "no_authenticated_hardware_transcript_no_speedup_tsu_kona_claim"


def _required_evidence_blocked_or_missing(
    local_sota_receipt_status: str,
    clean_verifier_status: str,
    repair_status: str,
    fr11_self_learning_status: str,
    hardware_sampler_status: str,
) -> list[str]:
    blocked: list[str] = []
    if local_sota_receipt_status != "passed_cuda_receipt_ready_full_local_sota_clean_rerun_allowed":
        blocked.append("local_sota_receipt")
    if clean_verifier_status != "clean_live_verifier_v12_ready":
        blocked.append("clean_verifier")
    if repair_status != "repair_ready":
        blocked.append("repair")
    if not fr11_self_learning_status.startswith("controller_trace_replay"):
        blocked.append("fr11_self_learning")
    if hardware_sampler_status != "authenticated_hardware_claim_allowed":
        blocked.append("hardware_sampler")
    return blocked


def _next_top_gap(
    local_sota_receipt_status: str,
    clean_verifier_status: str,
    repair_status: str,
    fr11_self_learning_status: str,
    hardware_sampler_status: str,
) -> str:
    if local_sota_receipt_status != "passed_cuda_receipt_ready_full_local_sota_clean_rerun_allowed":
        return NEXT_TOP_GAP
    if clean_verifier_status != "clean_live_verifier_v12_ready":
        return "clean_live_verifier_v12_gate_clearance"
    if repair_status != "repair_ready":
        return "repair_gate_v6_unblock_and_ladder_v7_execution"
    if not fr11_self_learning_status.startswith("controller_trace_replay"):
        return "fr11_controller_memory_nonforgetting_promotion"
    if hardware_sampler_status != "authenticated_hardware_claim_allowed":
        return "authenticated_hardware_transcript_or_explicit_no_speedup_boundary"
    return "publication_blocker_retirement_review"


def _invariant_violations(
    previous_matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
    publication_blocker_count: int,
    prior_count: int | None,
    new_blockers: list[Mapping[str, Any]],
    hardware_boundaries: Mapping[str, Any],
    fr11_boundaries: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    if previous_matrix.get("cross_corpus_matrix_v30_ready") is not True:
        violations.append("previous matrix v30 is missing or not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v31 statuses")
    if sum(status_counts.values()) != len(rows):
        violations.append("status_counts do not sum to upstream artifact rows")
    if prior_count is not None and publication_blocker_count != prior_count + len(new_blockers):
        violations.append("publication_blocker_count does not reconcile with v30 delta")
    if (
        hardware_boundaries.get("speedup_claim_allowed") is True
        and hardware_boundaries.get("authenticated_hardware_transcript_present") is not True
    ):
        violations.append("hardware speedup claim lacks authenticated transcript")
    if fr11_boundaries.get("model_weight_update_claimed") is True:
        violations.append("FR-11 model-weight update claim is outside matrix v31 boundary")
    return violations


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_dot297_artifacts",
        "source": "matrix_v30_and_dot297_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("cross_corpus_matrix_v31_ready") is not True:
        return (
            "blocked_matrix_v31_preconditions: "
            + "; ".join(str(item) for item in artifact.get("invariant_violations", []))
        )
    return (
        "complete: cross_corpus_matrix_v31_ready=true; "
        f"previous_matrix_artifact={artifact['previous_matrix_artifact']}; "
        f"paper_ready={str(artifact['paper_ready']).lower()}; "
        f"publication_blocker_count={artifact['publication_blocker_count']}; "
        f"blocker_delta_from_v30={artifact['blocker_delta_from_v30']}; "
        f"next_top_gap={artifact['next_top_gap']}"
    )


def _gate_skip_evidence(experiment_id: str, conductor_log: str) -> JsonDict:
    title = GATE_SKIP_TITLES.get(experiment_id, "")
    if not title or not conductor_log:
        return {"status": "absent"}
    for line in conductor_log.splitlines():
        if title in line and "GATE_BLOCK" in line:
            return {
                "status": "gated_skipped",
                "source": CONDUCTOR_LOG_REL_PATH.as_posix(),
                "line": line.strip(),
            }
    return {"status": "absent"}


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[_normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _normal_status(status: str) -> str:
    normalized = status.strip().lower().replace("-", "_")
    if normalized == "gated_skip":
        return "gated_skipped"
    return normalized if normalized in STATUSES else "missing"


def _row_status(rows: list[Mapping[str, Any]], experiment_id: str) -> str:
    return _normal_status(str(_row(rows, experiment_id).get("status") or "missing"))


def _row(rows: list[Mapping[str, Any]], experiment_id: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("experiment_id") == experiment_id:
            return row
    return {}


def _is_gate_skip(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema") == "blocked_gate_check_v1"
        or payload.get("gated_skip") is True
        or str(payload.get("honest_verdict") or "").startswith("blocked_gate_check")
    )


def _prior_publication_blocker_count(matrix: Mapping[str, Any]) -> int | None:
    value = matrix.get("publication_blocker_count")
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _model_weight_update_claimed(payload: Mapping[str, Any]) -> bool:
    if payload.get("model_weight_update_claimed") is True:
        return True
    if payload.get("model_weight_update_performed") is True:
        return True
    substrate = _as_mapping(payload.get("inference_substrate"))
    return any(
        substrate.get(key) is True
        for key in (
            "base_model_weights_updated",
            "hidden_state_mutation_claimed",
            "model_weight_learning",
            "model_weight_mutation",
            "model_weight_training",
            "kan_model_weight_training",
        )
    )


def _experiment_id(payload: Mapping[str, Any], fallback: str) -> str:
    value = payload.get("experiment_id")
    if value:
        return str(value)
    experiment = payload.get("experiment")
    if isinstance(experiment, int) and not isinstance(experiment, bool):
        return f"exp{experiment}"
    return fallback


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return False


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
    return max(0.0, round(end - started_s, 6))
