"""Build the Exp 3292 evidence matrix v36 artifact.

Spec refs: REQ-REPORT-3292, SCENARIO-REPORT-3292.

This module is a ledger. It does not rerun Garak, a verifier, repair, KAN, or
FR-11 work. It reads the checked-in `.304` artifacts, carries forward the
unresolved `.303` blockers that those artifacts did not retire, and writes a
machine-readable capstone input. The important safety property is that a
complete matrix is not the same thing as publication readiness: blocked gates,
methodology flags, and bounded sidecar evidence remain visible.
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
SCHEMA_VERSION = "carnot.evidence_matrix.v36_dot304_artifact_aggregation.v1"
EXPERIMENT_ID = "exp3292"
TASK_ID = "exp3292-evidence-matrix-v36"
ARTIFACT = "experiment_3292_evidence_matrix_v36"
MILESTONE = "2026.05.304"
PRIOR_MILESTONE = "2026.05.303"
INFERENCE_SUBSTRATE = "artifact_aggregation_only"
OUTPUT_REL_PATH = Path("results/experiment_3292_evidence_matrix_v36.json")
RANDOM_SEED = 3292

PRIOR_MATRIX_REL_PATH = Path("results/experiment_3279_evidence_matrix_v35.json")
EXP3281_REL_PATH = Path("results/experiment_3281_archive_v303_activate_v304.json")
EXP3282_REL_PATH = Path("results/experiment_3282_garak_install_and_probe_manifest_v1.json")
EXP3283_REL_PATH = Path(
    "results/experiment_3283_prompt_injection_corrigendum_duration_audit_v1.json"
)
EXP3284_REL_PATH = Path("results/experiment_3284_garak_local_smoke_sota_gguf_v1.json")
EXP3285_REL_PATH = Path("results/experiment_3285_full_garak_dataflip_redteam_eval_v2.json")
EXP3286_REL_PATH = Path("results/experiment_3286_clean_verifier_abstention_root_cause_v1.json")
EXP3287_REL_PATH = Path("results/experiment_3287_abstention_calibrated_clean_verifier_v15.json")
EXP3288_REL_PATH = Path("results/experiment_3288_kan_sidecar_failure_autopsy_boundary_v1.json")
EXP3289_REL_PATH = Path("results/experiment_3289_repair_gate_decision_v9_after_garak_abstention.json")
EXP3290_REL_PATH = Path("results/experiment_3290_gated_sota_repair_micro_panel_v10.json")
EXP3291_REL_PATH = Path("results/experiment_3291_fr11_garak_abstention_memory_replay_v1.json")

STATUSES = ("clean", "blocked", "flagged", "sidecar-only", "missing", "paper-blocking")
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
REQUIRED_ARTIFACT_FIELDS = {
    "matrix_v36_ready",
    "artifact_count_scanned",
    "clean_evidence_count",
    "blocked_evidence_count",
    "flagged_evidence_count",
    "sidecar_only_count",
    "missing_evidence_count",
    "paper_blocker_count",
    "top_gaps",
    "gate_summary",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}
SUMMARY_KEYS = (
    "v303_closed_v304_opened",
    "prior_paper_ready",
    "prior_publication_blocker_count",
    "prior_next_top_gap",
    "full_15k_corpus_materialized",
    "garak_blocker",
    "clean_verifier_abstention_rate",
    "kan_noninferiority_passed",
    "garak_install_probe_manifest_ready",
    "garak_runner_ready",
    "garak_available",
    "garak_version",
    "promptinject_probe_count",
    "corrigendum_ready",
    "garak_local_smoke_v1_ready",
    "garak_smoke_ready",
    "garak_probe_count",
    "attack_success_rate",
    "garak_dataflip_redteam_eval_v2_ready",
    "garak_redteam_eval_ready",
    "garak_gate_passed",
    "dataflip_gate_passed",
    "abstention_root_cause_audit_ready",
    "abstention_root_cause_identified",
    "dominant_root_cause",
    "prior_abstention_rate",
    "abstention_calibrated_clean_verifier_v15_ready",
    "clean_verifier_rerun_ready",
    "repair_gate_input_clean_enough",
    "false_accept_rate",
    "false_reject_rate",
    "abstention_rate",
    "coverage_rate",
    "kan_failure_autopsy_ready",
    "kan_boundary_decision_ready",
    "prior_full_corpus_auroc",
    "prior_delong_noninferiority_passed",
    "kan_boundary_decision",
    "repair_gate_decision_v9_ready",
    "repair_gate_open",
    "sota_repair_micro_panel_v10_ready",
    "repair_panel_ran",
    "panel_case_count",
    "verified_success_count",
    "false_accept_count",
    "abstention_count",
    "repair_success_rate",
    "headline_claim_allowed",
    "fr11_garak_abstention_memory_replay_ready",
    "controller_memory_only",
    "foundation_weight_updates_performed",
    "raw_episodes_preserved",
    "retention_score",
    "adaptation_score",
    "forgetting_rate",
    "negative_transfer_rate",
    "heldout_trace_count",
)


@dataclass(frozen=True)
class SourceSpec:
    """One planned `.304` artifact row that matrix v36 must account for."""

    experiment_id: str
    task_id: str
    path: Path
    role: str
    ready_field: str


EXPECTED_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp3281",
        "exp3281-archive-v303-activate-v304",
        EXP3281_REL_PATH,
        "v303_archive_v304_handoff",
        "v303_closed_v304_opened",
    ),
    SourceSpec(
        "exp3282",
        "exp3282-garak-install-and-probe-manifest-v1",
        EXP3282_REL_PATH,
        "garak_toolchain_manifest",
        "garak_install_probe_manifest_ready",
    ),
    SourceSpec(
        "exp3283",
        "exp3283-prompt-injection-corrigendum-duration-audit-v1",
        EXP3283_REL_PATH,
        "prompt_injection_corrigendum",
        "corrigendum_ready",
    ),
    SourceSpec(
        "exp3284",
        "exp3284-garak-local-smoke-sota-gguf-v1",
        EXP3284_REL_PATH,
        "garak_local_smoke",
        "garak_local_smoke_v1_ready",
    ),
    SourceSpec(
        "exp3285",
        "exp3285-full-garak-dataflip-redteam-eval-v2",
        EXP3285_REL_PATH,
        "full_garak_dataflip_redteam",
        "garak_dataflip_redteam_eval_v2_ready",
    ),
    SourceSpec(
        "exp3286",
        "exp3286-clean-verifier-abstention-root-cause-v1",
        EXP3286_REL_PATH,
        "clean_verifier_abstention_root_cause",
        "abstention_root_cause_audit_ready",
    ),
    SourceSpec(
        "exp3287",
        "exp3287-abstention-calibrated-clean-verifier-v15",
        EXP3287_REL_PATH,
        "abstention_calibrated_clean_verifier",
        "abstention_calibrated_clean_verifier_v15_ready",
    ),
    SourceSpec(
        "exp3288",
        "exp3288-kan-sidecar-failure-autopsy-boundary-v1",
        EXP3288_REL_PATH,
        "kan_sidecar_failure_boundary",
        "kan_failure_autopsy_ready",
    ),
    SourceSpec(
        "exp3289",
        "exp3289-repair-gate-decision-v9-after-garak-abstention",
        EXP3289_REL_PATH,
        "repair_gate_decision_v9",
        "repair_gate_decision_v9_ready",
    ),
    SourceSpec(
        "exp3290",
        "exp3290-gated-sota-repair-micro-panel-v10",
        EXP3290_REL_PATH,
        "gated_sota_repair_micro_panel",
        "sota_repair_micro_panel_v10_ready",
    ),
    SourceSpec(
        "exp3291",
        "exp3291-fr11-garak-abstention-memory-replay-v1",
        EXP3291_REL_PATH,
        "fr11_garak_abstention_memory_replay",
        "fr11_garak_abstention_memory_replay_ready",
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and treat absent, malformed, or array input as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash exact artifact bytes so the capstone can verify what was aggregated."""

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
    """REQ-REPORT-3292: aggregate matrix v36 from checked-in `.304` evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    prior_matrix = read_json_object(root_path / PRIOR_MATRIX_REL_PATH)
    rows = [_source_row(root_path, spec) for spec in EXPECTED_SOURCES]
    payloads = {row["experiment_id"]: _as_mapping(row.get("payload")) for row in rows}
    public_rows = [_public_row(row) for row in rows]
    primary_counts = _status_counts(public_rows)
    prior_resolution = _prior_blocker_resolution(prior_matrix, payloads)
    carried_forward = [
        row for row in prior_resolution if row["resolution_status"] == "unresolved"
    ]
    gate_summary = _gate_summary(public_rows, payloads, prior_matrix, carried_forward)
    paper_blockers = _paper_blocker_records(public_rows, carried_forward)
    top_gaps = _top_gaps(public_rows, carried_forward)
    invariant_violations = _invariant_violations(prior_matrix, payloads)

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
        "principle_annotations": _principle_annotations(),
        "matrix_v36_ready": not invariant_violations,
        "rows": public_rows,
        "evidence_rows": public_rows,
        "primary_status_counts": primary_counts,
        "artifact_count_scanned": sum(1 for row in public_rows if row["present"]),
        "expected_artifact_count": len(EXPECTED_SOURCES),
        "clean_evidence_count": primary_counts["clean"],
        "blocked_evidence_count": primary_counts["blocked"],
        "flagged_evidence_count": sum(1 for row in public_rows if row["quality_flags"]),
        "sidecar_only_count": primary_counts["sidecar-only"],
        "missing_evidence_count": primary_counts["missing"],
        "paper_blocker_count": len(paper_blockers),
        "paper_blockers": paper_blockers,
        "paper_ready": len(paper_blockers) == 0 and not carried_forward,
        "top_gaps": top_gaps,
        "gate_summary": gate_summary,
        "prior_matrix": _prior_matrix_record(root_path, prior_matrix),
        "prior_blocker_resolution": prior_resolution,
        "carried_forward_blockers": carried_forward,
        "artifacts_expected": [_expected_record(spec) for spec in EXPECTED_SOURCES],
        "artifacts_missing": [row for row in public_rows if not row["present"]],
        "loaded_artifact_paths": [row["path"] for row in public_rows if row["present"]],
        "source_checksums": {
            row["path"]: row["sha256"] for row in public_rows if row.get("sha256")
        },
        "protected_files_untouched": {
            "scripts/research_conductor.py": True,
            "ops/status.md": True,
            "ops/changelog.md": True,
            "_bmad/traceability.md": True,
        },
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
        "scripts_research_conductor_modified": False,
        "invariant_violations": invariant_violations,
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
    """Build and persist the Exp 3292 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when v36 omits capstone fields or overclaims publication readiness."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3292")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3292-evidence-matrix-v36")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.304")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be artifact_aggregation_only")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if _int_value(artifact.get("paper_blocker_count")) < 0:
        raise ValueError("paper_blocker_count must be non-negative")
    if artifact.get("paper_ready") is True and _int_value(artifact.get("paper_blocker_count")) != 0:
        raise ValueError("paper_ready cannot be true while paper blockers remain")


def _source_row(root: Path, spec: SourceSpec) -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path)
    present = path.is_file()
    return {
        "experiment_id": spec.experiment_id,
        "task_id": spec.task_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "ready_field": spec.ready_field,
        "present": present,
        "payload": payload,
        "status": _status_for_source(spec, payload, present),
        "sha256": sha256_file(path),
    }


def _public_row(row: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(row.get("payload"))
    ready_field = str(row.get("ready_field") or "")
    return {
        "experiment_id": str(row.get("experiment_id") or ""),
        "task_id": str(payload.get("task_id") or row.get("task_id") or ""),
        "path": str(row.get("path") or ""),
        "role": str(row.get("role") or ""),
        "ready_field": ready_field,
        "present": row.get("present") is True,
        "status": _normal_status(str(row.get("status") or "missing")),
        "ready": payload.get(ready_field) is True,
        "reported_experiment_id": str(payload.get("experiment_id") or payload.get("experiment") or ""),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "blocker_reasons": _blocker_reasons(row, payload),
        "quality_flags": _quality_flags(payload),
        "bounded_claims": _bounded_claims(payload),
        "paper_blocking": _row_paper_blocking(row, payload),
        "summary": _row_summary(payload),
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


def _status_for_source(spec: SourceSpec, payload: Mapping[str, Any], present: bool) -> str:
    if not present or not payload:
        return "missing"
    if _is_gate_blocked(payload) or _has_blockers(spec, payload):
        return "blocked"
    if _quality_flags(payload):
        return "flagged"
    if _is_sidecar_only(payload):
        return "sidecar-only"
    if _is_paper_blocking(payload):
        return "paper-blocking"
    return "clean"


def _is_gate_blocked(payload: Mapping[str, Any]) -> bool:
    verdict = str(payload.get("honest_verdict") or "")
    return (
        payload.get("schema") == "blocked_gate_check_v1"
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
        or verdict.startswith("blocked_gate_check")
    )


def _has_blockers(spec: SourceSpec, payload: Mapping[str, Any]) -> bool:
    return bool(_explicit_blockers(payload)) or payload.get(spec.ready_field) is False


def _explicit_blockers(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    blockers += _list_of_strings(payload.get("blocked_reasons"))
    blockers += _list_of_strings(payload.get("gate_reasons"))
    blockers += _list_of_strings(payload.get("install_blockers"))
    blocked_reason = str(payload.get("blocked_reason") or "").strip()
    gate_summary = str(payload.get("gate_check_summary") or "").strip()
    if blocked_reason:
        blockers.append(blocked_reason)
    if gate_summary:
        blockers.append(gate_summary)
    return blockers


def _blocker_reasons(row: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    if row.get("present") is not True:
        return [f"artifact_missing: {row.get('path')}"]
    reasons = _explicit_blockers(payload)
    reasons += [
        str(gate.get("reason"))
        for gate in _as_list(payload.get("gates_evaluated"))
        if _as_mapping(gate).get("passed") is False and gate.get("reason") is not None
    ]
    for gate_field in ("garak_gate_passed", "dataflip_gate_passed"):
        if payload.get(gate_field) is False:
            reasons.append(f"{gate_field}=false")
    ready_field = str(row.get("ready_field") or "")
    if not reasons and ready_field and payload.get(ready_field) is False:
        reasons.append(f"{ready_field}=false")
    return reasons


def _quality_flags(payload: Mapping[str, Any]) -> list[JsonDict]:
    flags = [_as_mapping(item) for item in _as_list(payload.get("corrigendum_pending"))]
    flags += [_as_mapping(item) for item in _as_list(payload.get("duration_flags"))]
    flags += [_as_mapping(item) for item in _as_list(payload.get("tautology_flags"))]
    if payload.get("flagged_adversarial") is True and not flags:
        flags.append({"kind": "flagged_adversarial", "detail": "flagged_adversarial=true"})
    return [
        {
            "kind": str(flag.get("kind") or "flagged_adversarial"),
            "detail": str(flag.get("detail") or flag.get("severity") or ""),
        }
        for flag in flags
    ]


def _bounded_claims(payload: Mapping[str, Any]) -> list[str]:
    claims: list[str] = []
    if payload.get("sidecar_only") is True:
        claims.append("sidecar_only=true")
    if payload.get("kan_boundary_decision") == "retire_from_prompt_injection_headline":
        claims.append("kan_boundary_decision=retire_from_prompt_injection_headline")
    if payload.get("prior_delong_noninferiority_passed") is False:
        claims.append("prior_delong_noninferiority_passed=false")
    if payload.get("headline_claim_allowed") is False:
        claims.append("headline_claim_allowed=false")
    if payload.get("controller_memory_only") is True:
        claims.append("controller_memory_only=true")
    if payload.get("foundation_weight_updates_performed") is False:
        claims.append("foundation_weight_updates_performed=false")
    return claims


def _is_sidecar_only(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("sidecar_only") is True
        or payload.get("kan_boundary_decision") == "retire_from_prompt_injection_headline"
    )


def _is_paper_blocking(payload: Mapping[str, Any]) -> bool:
    paper_claims = _as_mapping(_as_mapping(payload.get("downstream_usage_rules")).get("paper_claims"))
    return (
        payload.get("prior_paper_ready") is False
        or payload.get("garak_gate_passed") is False
        or payload.get("dataflip_gate_passed") is False
        or payload.get("headline_claim_allowed") is False
        or paper_claims.get("headline_performance_metrics_allowed") is False
    )


def _row_paper_blocking(row: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    status = _normal_status(str(row.get("status") or "missing"))
    return status != "clean" or _is_paper_blocking(payload)


def _row_summary(payload: Mapping[str, Any]) -> JsonDict:
    return {key: payload.get(key) for key in SUMMARY_KEYS if key in payload}


def _prior_matrix_record(root: Path, prior_matrix: Mapping[str, Any]) -> JsonDict:
    path = root / PRIOR_MATRIX_REL_PATH
    return {
        "path": PRIOR_MATRIX_REL_PATH.as_posix(),
        "present": path.is_file(),
        "matrix_v35_ready": prior_matrix.get("matrix_v35_ready") is True,
        "paper_ready": prior_matrix.get("paper_ready") is True,
        "publication_blocker_count_estimate": _int_value(
            prior_matrix.get("publication_blocker_count_estimate")
        ),
        "sha256": sha256_file(path),
    }


def _prior_blocker_resolution(
    prior_matrix: Mapping[str, Any], payloads: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    return [
        _resolve_prior_blocker(_as_mapping(row), payloads)
        for row in _as_list(prior_matrix.get("rows"))
        if _normal_status(str(_as_mapping(row).get("status") or "missing")) != "clean"
    ]


def _resolve_prior_blocker(
    prior_row: Mapping[str, Any], payloads: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    prior_id = str(prior_row.get("experiment_id") or "")
    prior_status = _normal_status(str(prior_row.get("status") or "missing"))
    if prior_id in {"exp3270", "exp3271", "exp3272"}:
        return _prior_resolution(
            prior_row,
            "unresolved" if _corrigendum_carries_flag(payloads.get("exp3283", {}), prior_id) else "resolved",
            "exp3283",
            ".303 methodology flag remains in Exp 3283 corrigendum",
        )
    if prior_id == "exp3273":
        decision = str(payloads.get("exp3288", {}).get("kan_boundary_decision") or "")
        status = "bounded" if payloads.get("exp3288", {}).get("kan_boundary_decision_ready") else "unresolved"
        return _prior_resolution(prior_row, status, "exp3288", decision or "KAN boundary missing")
    if prior_id == "exp3274":
        garak = payloads.get("exp3285", {})
        status = "replaced" if garak.get("garak_redteam_eval_ready") is True else "unresolved"
        reason = (
            "Garak availability blocker replaced by garak_gate_passed=false"
            if garak.get("garak_gate_passed") is False
            else "Garak red-team gate reran"
        )
        return _prior_resolution(prior_row, status, "exp3285", reason)
    if prior_id == "exp3275":
        clean = payloads.get("exp3287", {})
        status = (
            "resolved"
            if clean.get("clean_verifier_rerun_ready") is True
            and clean.get("repair_gate_input_clean_enough") is True
            else "unresolved"
        )
        return _prior_resolution(prior_row, status, "exp3287", "clean verifier abstention calibrated")
    if prior_id == "exp3276":
        status = "resolved" if payloads.get("exp3289", {}).get("repair_gate_open") is True else "unresolved"
        return _prior_resolution(prior_row, status, "exp3289", "repair gate v9 evaluated")
    if prior_id == "exp3277":
        status = (
            "resolved"
            if payloads.get("exp3290", {}).get("sota_repair_micro_panel_v10_ready") is True
            else "unresolved"
        )
        return _prior_resolution(prior_row, status, "exp3290", "repair micro-panel v10 produced")
    return _prior_resolution(prior_row, "unresolved", "", f"no .304 resolver for {prior_id}")


def _prior_resolution(
    prior_row: Mapping[str, Any],
    status: str,
    current_id: str,
    reason: str,
) -> JsonDict:
    return {
        "prior_experiment_id": str(prior_row.get("experiment_id") or ""),
        "prior_status": _normal_status(str(prior_row.get("status") or "missing")),
        "prior_blocker_reasons": _list_of_strings(prior_row.get("blocker_reasons")),
        "prior_quality_flags": [
            {
                "kind": str(_as_mapping(flag).get("kind") or "flagged_adversarial"),
                "detail": str(_as_mapping(flag).get("detail") or ""),
            }
            for flag in _as_list(prior_row.get("quality_flags"))
        ],
        "prior_bounded_claims": _list_of_strings(prior_row.get("bounded_claims")),
        "resolution_status": status,
        "current_source_experiment_id": current_id,
        "reason": reason,
    }


def _corrigendum_carries_flag(corrigendum: Mapping[str, Any], prior_id: str) -> bool:
    flags = _as_list(corrigendum.get("duration_flags")) + _as_list(corrigendum.get("tautology_flags"))
    return any(str(_as_mapping(flag).get("experiment_id") or "") == prior_id for flag in flags)


def _paper_blocker_records(
    rows: list[Mapping[str, Any]], carried_forward: list[Mapping[str, Any]]
) -> list[JsonDict]:
    current = [
        {
            "source_experiment_id": str(row.get("experiment_id") or ""),
            "status": str(row.get("status") or ""),
            "reason": _first_reason(row),
        }
        for row in rows
        if row.get("paper_blocking") is True
    ]
    return current + [
        {
            "source_experiment_id": str(row.get("prior_experiment_id") or ""),
            "status": "carried-forward",
            "reason": str(row.get("reason") or ""),
        }
        for row in carried_forward
        if str(row.get("prior_experiment_id") or "") not in {"exp3270", "exp3271", "exp3272"}
    ]


def _top_gaps(
    rows: list[Mapping[str, Any]], carried_forward: list[Mapping[str, Any]]
) -> list[JsonDict]:
    by_id = {str(row.get("experiment_id")): row for row in rows}
    candidates = [
        _gap("pass_garak_redteam_gate", by_id.get("exp3285")),
        _gap("repair_panel_duration_and_scope_boundary", by_id.get("exp3290")),
        _carried_gap(carried_forward),
        _gap("keep_kan_retired_from_headline_or_rebuild", by_id.get("exp3288")),
        _gap("retire_remaining_prior_publication_blockers", by_id.get("exp3281")),
    ]
    compact = [candidate for candidate in candidates if candidate["source_experiment_id"]]
    fallback = {
        "gap": "ready_for_capstone",
        "source_experiment_id": "",
        "status": "clean",
        "reason": "all matrix rows are clean",
    }
    return [dict(candidate, rank=index) for index, candidate in enumerate(compact or [fallback], 1)]


def _gap(name: str, row: Mapping[str, Any] | None) -> JsonDict:
    row_map = _as_mapping(row)
    return {
        "gap": name if row_map.get("paper_blocking") else "",
        "source_experiment_id": str(row_map.get("experiment_id") or "")
        if row_map.get("paper_blocking")
        else "",
        "status": str(row_map.get("status") or ""),
        "reason": _first_reason(row_map),
    }


def _carried_gap(carried_forward: list[Mapping[str, Any]]) -> JsonDict:
    first = _as_mapping(carried_forward[0]) if carried_forward else {}
    return {
        "gap": "resolve_dot303_methodology_flags" if first else "",
        "source_experiment_id": str(first.get("prior_experiment_id") or ""),
        "status": "carried-forward" if first else "",
        "reason": str(first.get("reason") or ""),
    }


def _first_reason(row: Mapping[str, Any]) -> str:
    blockers = _list_of_strings(row.get("blocker_reasons"))
    flags = [str(_as_mapping(flag).get("kind") or "") for flag in _as_list(row.get("quality_flags"))]
    bounded = _list_of_strings(row.get("bounded_claims"))
    return next((item for item in blockers + flags + bounded if item), str(row.get("status") or ""))


def _gate_summary(
    rows: list[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    prior_matrix: Mapping[str, Any],
    carried_forward: list[Mapping[str, Any]],
) -> JsonDict:
    return {
        "prior_dot303": {
            "source_experiment_id": "exp3279",
            "matrix_v35_ready": prior_matrix.get("matrix_v35_ready") is True,
            "paper_ready": prior_matrix.get("paper_ready") is True,
            "carried_forward_count": len(carried_forward),
            "status": "paper-blocking" if carried_forward else "clean",
        },
        "garak_toolchain": _gate_status(rows, payloads, "exp3282")
        | {
            "garak_runner_ready": payloads.get("exp3282", {}).get("garak_runner_ready") is True,
            "garak_available": payloads.get("exp3282", {}).get("garak_available") is True,
        },
        "garak_redteam": _gate_status(rows, payloads, "exp3285")
        | {
            "garak_redteam_eval_ready": payloads.get("exp3285", {}).get("garak_redteam_eval_ready")
            is True,
            "garak_gate_passed": payloads.get("exp3285", {}).get("garak_gate_passed") is True,
            "dataflip_gate_passed": payloads.get("exp3285", {}).get("dataflip_gate_passed") is True,
        },
        "clean_verifier": _gate_status(rows, payloads, "exp3287")
        | {
            "root_cause_source_experiment_id": "exp3286",
            "clean_verifier_rerun_ready": payloads.get("exp3287", {}).get(
                "clean_verifier_rerun_ready"
            )
            is True,
            "repair_gate_input_clean_enough": payloads.get("exp3287", {}).get(
                "repair_gate_input_clean_enough"
            )
            is True,
            "abstention_rate": payloads.get("exp3287", {}).get("abstention_rate"),
        },
        "kan_boundary": _gate_status(rows, payloads, "exp3288")
        | {
            "kan_boundary_decision": payloads.get("exp3288", {}).get("kan_boundary_decision"),
            "kan_boundary_decision_ready": payloads.get("exp3288", {}).get(
                "kan_boundary_decision_ready"
            )
            is True,
        },
        "repair_gate": _gate_status(rows, payloads, "exp3289")
        | {"repair_gate_open": payloads.get("exp3289", {}).get("repair_gate_open") is True},
        "repair_panel": _gate_status(rows, payloads, "exp3290")
        | {
            "repair_panel_ran": payloads.get("exp3290", {}).get("repair_panel_ran") is True,
            "headline_claim_allowed": payloads.get("exp3290", {}).get("headline_claim_allowed")
            is True,
        },
        "fr11": _gate_status(rows, payloads, "exp3291")
        | {
            "controller_memory_only": payloads.get("exp3291", {}).get("controller_memory_only")
            is True,
            "foundation_weight_updates_performed": payloads.get("exp3291", {}).get(
                "foundation_weight_updates_performed"
            )
            is True,
        },
    }


def _gate_status(
    rows: list[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    experiment_id: str,
) -> JsonDict:
    row = _row_by_id(rows, experiment_id)
    return {
        "source_experiment_id": experiment_id,
        "present": row.get("present") is True,
        "status": str(row.get("status") or "missing"),
        "ready": row.get("ready") is True,
        "blocker_reasons": _list_of_strings(row.get("blocker_reasons")),
        "honest_verdict": str(payloads.get(experiment_id, {}).get("honest_verdict") or ""),
    }


def _row_by_id(rows: list[Mapping[str, Any]], experiment_id: str) -> Mapping[str, Any]:
    return next((row for row in rows if row.get("experiment_id") == experiment_id), {})


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[_normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _invariant_violations(
    prior_matrix: Mapping[str, Any], payloads: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    violations: list[str] = []
    if prior_matrix.get("matrix_v35_ready") is not True:
        violations.append("prior matrix v35 is missing or not ready")
    if payloads.get("exp3281", {}).get("v303_closed_v304_opened") is not True:
        violations.append("exp3281 .304 handoff artifact is missing or not ready")
    return violations


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "Matrix v36 reads checked-in .304 artifacts and v35 only.",
        "missing_is_not_failed": "Absent artifacts stay missing instead of becoming failed gates.",
        "blocked_gates_visible": "Failed gates preserve exact blockers and false gate booleans.",
        "flags_visible": "Methodology flags remain counted even when downstream gates open.",
        "sidecar_bounded": "KAN and controller-memory-only evidence stays explicitly bounded.",
        "paper_ready_rule": "Publication readiness requires zero current blockers and no carried .303 blockers.",
    }


def _normal_status(status: str) -> str:
    normalized = status.strip().lower().replace("_", "-")
    return normalized if normalized in STATUSES else "missing"


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _list_of_strings(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "rows": artifact.get("rows"),
        "paper_blocker_count": artifact.get("paper_blocker_count"),
        "top_gaps": artifact.get("top_gaps"),
        "gate_summary": artifact.get("gate_summary"),
        "carried_forward_blockers": artifact.get("carried_forward_blockers"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    top_gap = _as_mapping(_as_list(artifact.get("top_gaps"))[0]).get("gap")
    return (
        "complete: matrix_v36_ready="
        f"{str(artifact.get('matrix_v36_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"artifact_count_scanned={artifact.get('artifact_count_scanned')}; "
        f"clean={artifact.get('clean_evidence_count')}; "
        f"blocked={artifact.get('blocked_evidence_count')}; "
        f"flagged={artifact.get('flagged_evidence_count')}; "
        f"sidecar_only={artifact.get('sidecar_only_count')}; "
        f"missing={artifact.get('missing_evidence_count')}; "
        f"paper_blockers={artifact.get('paper_blocker_count')}; "
        f"next_top_gap={top_gap}"
    )
