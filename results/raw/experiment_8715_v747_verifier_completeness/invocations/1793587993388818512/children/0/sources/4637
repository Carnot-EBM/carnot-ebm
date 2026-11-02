"""Build the Exp 3305 evidence matrix v37 artifact.

Spec refs: REQ-REPORT-3305, SCENARIO-REPORT-3305.

This module is intentionally a ledger, not a runner. It reads the checked-in
`.305` artifacts, records which evidence is live, gated, historical, sidecar
bounded, or repair-headline scoped, and writes the machine-readable matrix that
the next capstone can consume without rerunning Garak, repair, FR-11, or model
inference.
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
RUN_DATE = "20260529"
SCHEMA_VERSION = "carnot.evidence_matrix.v37_dot305_artifact_aggregation.v1"
EXPERIMENT_ID = "exp3305"
TASK_ID = "exp3305-evidence-matrix-v37"
ARTIFACT = "experiment_3305_evidence_matrix_v37"
MILESTONE = "2026.05.305"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3305_evidence_matrix_v37.json")
RANDOM_SEED = 3305

EXP3293_REL_PATH = Path("results/experiment_3293_capstone_v304.json")
EXP3294_REL_PATH = Path("results/experiment_3294_archive_v304_activate_v305.json")
EXP3295_REL_PATH = Path("results/experiment_3295_garak_failure_mode_autopsy_v1.json")
EXP3296_REL_PATH = Path("results/experiment_3296_substrate_corrigendum_kan_no_retry_v1.json")
EXP3297_REL_PATH = Path("results/experiment_3297_prefix_closed_garak_guard_v1.json")
EXP3298_REL_PATH = Path("results/experiment_3298_redteam_energy_telemetry_router_v1.json")
EXP3299_REL_PATH = Path("results/experiment_3299_garak_defense_ablation_v1.json")
EXP3300_REL_PATH = Path("results/experiment_3300_full_garak_dataflip_gate_rerun_v3.json")
EXP3301_REL_PATH = Path("results/experiment_3301_exact_repair_panel_manifest_v11.json")
EXP3302_REL_PATH = Path("results/experiment_3302_headline_sota_repair_panel_v11.json")
EXP3303_REL_PATH = Path("results/experiment_3303_repair_headline_evidence_audit_v1.json")
EXP3304_REL_PATH = Path("results/experiment_3304_fr11_redteam_repair_memory_replay_v2.json")

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
CRITICAL_SEVERITIES = {"critical", "error", "fail", "failed", "blocker"}
REQUIRED_ARTIFACT_FIELDS = {
    "matrix_v37_ready",
    "artifact_count_scanned",
    "artifacts_missing",
    "clean_evidence_count",
    "blocked_evidence_count",
    "flagged_evidence_count",
    "sidecar_only_evidence_count",
    "garak_gate_passed",
    "repair_headline_claim_allowed",
    "fr11_replay_safe",
    "paper_ready",
    "paper_blocker_count",
    "top_gap",
    "cited_upstream_artifacts",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}
SUMMARY_KEYS = (
    "capstone_v304_ready",
    "paper_ready",
    "publication_blocker_count",
    "next_top_gap",
    "v304_closed_v305_opened",
    "prior_garak_gate_passed",
    "garak_failure_autopsy_ready",
    "prior_garak_gate_passed",
    "prior_attack_success_rate",
    "headline_claim_made",
    "substrate_corrigendum_ready",
    "kan_no_retry_ledger_ready",
    "kan_prompt_injection_headline_retired",
    "prefix_guard_policy_ready",
    "live_benchmark_claim",
    "guard_kind",
    "redteam_telemetry_policy_ready",
    "live_probe_count",
    "garak_defense_ablation_ready",
    "selected_defense_config_ready",
    "garak_redteam_eval_v3_ready",
    "garak_gate_passed",
    "dataflip_gate_passed",
    "attack_success_rate",
    "error_count",
    "repair_panel_manifest_ready",
    "panel_case_count",
    "headline_repair_panel_ready",
    "repair_panel_ran",
    "headline_claim_allowed",
    "provenance_clean",
    "verified_success_count",
    "false_accept_count",
    "repair_headline_evidence_audit_ready",
    "headline_claim_allowed_after_audit",
    "source_headline_claim_allowed",
    "source_provenance_clean",
    "substrate_consistency_passed",
    "fr11_redteam_repair_memory_replay_ready",
    "continuous_self_learning_task",
    "controller_memory_only",
    "foundation_weight_updates_performed",
    "consolidation_gate_passed",
    "retention_score",
    "negative_transfer_rate",
)


@dataclass(frozen=True)
class SourceSpec:
    """One expected `.305` upstream artifact and how v37 should read it."""

    experiment_id: str
    task_id: str
    path: Path
    role: str
    ready_field: str
    evidence_kind: str
    optional: bool = False


EXPECTED_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp3293",
        "exp3293-capstone-v304",
        EXP3293_REL_PATH,
        "v304_capstone_historical_closeout",
        "capstone_v304_ready",
        "historical-corrigendum",
    ),
    SourceSpec(
        "exp3294",
        "exp3294-archive-v304-activate-v305",
        EXP3294_REL_PATH,
        "v304_archive_v305_handoff",
        "v304_closed_v305_opened",
        "historical-corrigendum",
    ),
    SourceSpec(
        "exp3295",
        "exp3295-garak-failure-mode-autopsy-v1",
        EXP3295_REL_PATH,
        "garak_failure_autopsy",
        "garak_failure_autopsy_ready",
        "historical-corrigendum",
    ),
    SourceSpec(
        "exp3296",
        "exp3296-substrate-corrigendum-kan-no-retry-v1",
        EXP3296_REL_PATH,
        "substrate_corrigendum_kan_no_retry",
        "substrate_corrigendum_ready",
        "sidecar-only",
    ),
    SourceSpec(
        "exp3297",
        "exp3297-prefix-closed-garak-guard-v1",
        EXP3297_REL_PATH,
        "cached_prefix_guard",
        "prefix_guard_policy_ready",
        "gated-skipped",
    ),
    SourceSpec(
        "exp3298",
        "exp3298-redteam-energy-telemetry-router-v1",
        EXP3298_REL_PATH,
        "live_redteam_telemetry_router",
        "redteam_telemetry_policy_ready",
        "clean-live",
    ),
    SourceSpec(
        "exp3299",
        "exp3299-garak-defense-ablation-v1",
        EXP3299_REL_PATH,
        "live_garak_defense_ablation",
        "garak_defense_ablation_ready",
        "clean-live",
    ),
    SourceSpec(
        "exp3300",
        "exp3300-full-garak-dataflip-gate-rerun-v3",
        EXP3300_REL_PATH,
        "full_garak_dataflip_gate_rerun",
        "garak_redteam_eval_v3_ready",
        "clean-live",
    ),
    SourceSpec(
        "exp3301",
        "exp3301-exact-repair-panel-manifest-v11",
        EXP3301_REL_PATH,
        "exact_repair_panel_manifest",
        "repair_panel_manifest_ready",
        "gated-skipped",
    ),
    SourceSpec(
        "exp3302",
        "exp3302-headline-sota-repair-panel-v11",
        EXP3302_REL_PATH,
        "headline_sota_repair_panel",
        "headline_repair_panel_ready",
        "headline-repair",
    ),
    SourceSpec(
        "exp3303",
        "exp3303-repair-headline-evidence-audit-v1",
        EXP3303_REL_PATH,
        "repair_headline_evidence_audit",
        "repair_headline_evidence_audit_ready",
        "headline-repair",
    ),
    SourceSpec(
        "exp3304",
        "exp3304-fr11-redteam-repair-memory-replay-v2",
        EXP3304_REL_PATH,
        "fr11_controller_memory_replay",
        "fr11_redteam_repair_memory_replay_ready",
        "sidecar-only",
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
    """Hash source bytes so the matrix can be traced back to exact artifacts."""

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
    """REQ-REPORT-3305: aggregate v37 claim eligibility from `.305` artifacts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    rows = [_source_row(root_path, spec) for spec in EXPECTED_SOURCES]
    payloads = {row["experiment_id"]: _as_mapping(row.get("payload")) for row in rows}
    public_rows = [_public_row(row) for row in rows]
    paper_blockers = _paper_blocker_records(public_rows, payloads)
    garak_gate_passed = _garak_gate_passed(payloads)
    repair_claim_allowed = _repair_headline_claim_allowed(payloads)
    repair_audit_required = _repair_headline_audit_required(payloads)
    fr11_safe = _fr11_replay_safe(payloads)
    historical_bounded = _historical_flagged_evidence_bounded(public_rows)
    top_gap = _top_gap(
        public_rows,
        paper_blockers,
        garak_gate_passed=garak_gate_passed,
        repair_claim_allowed=repair_claim_allowed,
        repair_audit_required=repair_audit_required,
        fr11_safe=fr11_safe,
        historical_bounded=historical_bounded,
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
        "matrix_v37_ready": True,
        "expected_artifact_count": len(EXPECTED_SOURCES),
        "artifact_count_scanned": sum(1 for row in public_rows if row["readable_json_object"]),
        "artifacts_expected": [_expected_record(spec) for spec in EXPECTED_SOURCES],
        "artifacts_missing": [
            row["path"] for row in public_rows if row["present"] is not True
        ],
        "rows": public_rows,
        "evidence_rows": public_rows,
        "claim_class_counts": _class_counts(public_rows),
        "clean_evidence_count": sum(
            1 for row in public_rows if row["evidence_class"] == "clean-live"
        ),
        "blocked_evidence_count": sum(
            1 for row in public_rows if row["evidence_class"] == "blocked"
        ),
        "flagged_evidence_count": sum(1 for row in public_rows if row["quality_flags"]),
        "sidecar_only_evidence_count": sum(
            1 for row in public_rows if row["evidence_class"] == "sidecar-only"
        ),
        "gated_skipped_evidence_count": sum(
            1 for row in public_rows if row["evidence_class"] == "gated-skipped"
        ),
        "historical_corrigendum_count": sum(
            1 for row in public_rows if row["evidence_kind"] == "historical-corrigendum"
        ),
        "headline_repair_evidence_count": sum(
            1 for row in public_rows if row["evidence_kind"] == "headline-repair"
        ),
        "garak_gate_passed": garak_gate_passed,
        "dataflip_gate_passed": payloads.get("exp3300", {}).get("dataflip_gate_passed")
        is True,
        "repair_headline_audit_required": repair_audit_required,
        "repair_headline_claim_allowed": repair_claim_allowed,
        "fr11_replay_safe": fr11_safe,
        "historical_flagged_evidence_bounded": historical_bounded,
        "paper_ready": (
            garak_gate_passed
            and (repair_claim_allowed or not repair_audit_required)
            and fr11_safe
            and historical_bounded
            and not paper_blockers
        ),
        "paper_blocker_count": len(paper_blockers),
        "paper_blockers": paper_blockers,
        "top_gap": top_gap,
        "next_gap_recommendation": _next_gap_recommendation(top_gap),
        "gate_summary": _gate_summary(public_rows, payloads),
        "source_checksums": {
            row["path"]: row["sha256"] for row in public_rows if row.get("sha256")
        },
        "cited_upstream_artifacts": [spec.path.as_posix() for spec in EXPECTED_SOURCES],
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_repair_run": True,
        "no_new_verifier_run": True,
        "no_new_fr11_weight_update": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "scripts_research_conductor_modified": False,
        "ops_status_modified": False,
        "ops_changelog_modified": False,
        "traceability_modified": False,
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
    """Build and persist the Exp 3305 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject v37 matrices that omit capstone fields or overclaim readiness."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3305")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3305-evidence-matrix-v37")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.305")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not str(artifact.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if _int_value(artifact.get("paper_blocker_count")) < 0:
        raise ValueError("paper_blocker_count must be non-negative")
    if artifact.get("paper_ready") is True and _int_value(artifact.get("paper_blocker_count")) != 0:
        raise ValueError("paper_ready cannot be true while paper blockers remain")


def _source_row(root: Path, spec: SourceSpec) -> JsonDict:
    path = root / spec.path
    present = path.is_file()
    payload = read_json_object(path)
    return {
        "experiment_id": spec.experiment_id,
        "task_id": spec.task_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "ready_field": spec.ready_field,
        "evidence_kind": spec.evidence_kind,
        "optional": spec.optional,
        "present": present,
        "readable_json_object": present and bool(payload),
        "payload": payload,
        "sha256": sha256_file(path),
    }


def _public_row(row: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(row.get("payload"))
    ready_field = str(row.get("ready_field") or "")
    blockers = _blocker_reasons(row, payload)
    flags = _quality_flags(payload)
    return {
        "experiment_id": str(row.get("experiment_id") or ""),
        "task_id": str(payload.get("task_id") or row.get("task_id") or ""),
        "path": str(row.get("path") or ""),
        "role": str(row.get("role") or ""),
        "ready_field": ready_field,
        "present": row.get("present") is True,
        "readable_json_object": row.get("readable_json_object") is True,
        "ready": payload.get(ready_field) is True,
        "evidence_kind": str(row.get("evidence_kind") or ""),
        "evidence_class": _evidence_class(row, payload, blockers),
        "reported_experiment_id": str(payload.get("experiment_id") or payload.get("experiment") or ""),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "inference_substrate": str(payload.get("inference_substrate") or ""),
        "blocker_reasons": blockers,
        "quality_flags": flags,
        "critical_quality_flags": [flag for flag in flags if _flag_is_critical(flag)],
        "claim_boundaries": _claim_boundaries(row, payload),
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
        "evidence_kind": spec.evidence_kind,
        "optional": spec.optional,
    }


def _evidence_class(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    blockers: list[str],
) -> str:
    if row.get("present") is not True or row.get("readable_json_object") is not True:
        return "missing"
    if blockers:
        return "blocked"
    kind = str(row.get("evidence_kind") or "")
    if kind == "sidecar-only":
        return "sidecar-only"
    if kind == "gated-skipped":
        return "gated-skipped"
    if kind == "clean-live" and _is_live_substrate(str(payload.get("inference_substrate") or "")):
        return "clean-live"
    if kind == "headline-repair" and payload.get("headline_claim_allowed") is True:
        return "headline-repair"
    if kind == "historical-corrigendum":
        return "historical-corrigendum"
    return kind or "gated-skipped"


def _blocker_reasons(row: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    if row.get("present") is not True:
        return [f"artifact_missing: {row.get('path')}"]
    if row.get("readable_json_object") is not True:
        return [f"artifact_unreadable_or_not_json_object: {row.get('path')}"]
    reasons = _explicit_blockers(payload)
    ready_field = str(row.get("ready_field") or "")
    if ready_field and payload.get(ready_field) is False:
        reasons.append(f"{ready_field}=false")
    if str(row.get("experiment_id") or "") == "exp3300":
        if payload.get("garak_gate_passed") is False:
            reasons.append("garak_gate_passed=false")
        if payload.get("dataflip_gate_passed") is False:
            reasons.append("dataflip_gate_passed=false")
        if _int_value(payload.get("error_count")) > 0:
            reasons.append("error_count>0")
    if str(row.get("experiment_id") or "") == "exp3302":
        if payload.get("headline_claim_allowed") is False:
            reasons.append("headline_claim_allowed=false")
        if payload.get("provenance_clean") is False:
            reasons.append("provenance_clean=false")
    if str(row.get("experiment_id") or "") == "exp3303":
        if payload.get("headline_claim_allowed_after_audit") is False:
            reasons.append("headline_claim_allowed_after_audit=false")
        if payload.get("source_headline_claim_allowed") is False:
            reasons.append("source_headline_claim_allowed=false")
        if payload.get("source_provenance_clean") is False:
            reasons.append("source_provenance_clean=false")
        if payload.get("substrate_consistency_passed") is False:
            reasons.append("substrate_consistency_passed=false")
    if str(row.get("experiment_id") or "") == "exp3304" and not _fr11_payload_safe(payload):
        reasons.append("fr11_replay_safe=false")
    return _dedupe(reasons)


def _explicit_blockers(payload: Mapping[str, Any]) -> list[str]:
    reasons = _list_of_strings(payload.get("blocked_reasons"))
    reasons += _list_of_strings(payload.get("gate_reasons"))
    for key in ("blocked_reason", "gate_check_summary", "runner_error"):
        value = str(payload.get(key) or "").strip()
        if value:
            reasons.append(value)
    return reasons


def _quality_flags(payload: Mapping[str, Any]) -> list[JsonDict]:
    flags = [_normalize_flag(flag) for flag in _as_list(payload.get("corrigendum_pending"))]
    flags += [_normalize_flag(flag) for flag in _as_list(payload.get("duration_flags"))]
    flags += [_normalize_flag(flag) for flag in _as_list(payload.get("quality_flags"))]
    flags += [_normalize_flag(flag) for flag in _as_list(payload.get("adversarial_verify_flags"))]
    if payload.get("flagged_adversarial") is True and not flags:
        flags.append(
            {
                "kind": "flagged_adversarial",
                "severity": "unknown",
                "detail": "flagged_adversarial=true",
            }
        )
    return flags


def _normalize_flag(value: Any) -> JsonDict:
    flag = _as_mapping(value)
    return {
        "kind": str(flag.get("kind") or "flagged_adversarial"),
        "severity": str(flag.get("severity") or "unknown"),
        "detail": str(flag.get("detail") or flag.get("message") or ""),
    }


def _claim_boundaries(row: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    boundaries: list[str] = []
    if str(row.get("evidence_kind") or "") == "historical-corrigendum":
        boundaries.append("historical_or_aggregation_context_only")
    if str(row.get("evidence_kind") or "") == "sidecar-only":
        boundaries.append("bounded_sidecar_not_headline")
    if payload.get("kan_prompt_injection_headline_retired") is True:
        boundaries.append("kan_prompt_injection_headline_retired=true")
    if payload.get("live_benchmark_claim") is False:
        boundaries.append("live_benchmark_claim=false")
    if payload.get("controller_memory_only") is True:
        boundaries.append("controller_memory_only=true")
    if payload.get("foundation_weight_updates_performed") is False:
        boundaries.append("foundation_weight_updates_performed=false")
    if payload.get("headline_claim_allowed") is False:
        boundaries.append("headline_claim_allowed=false")
    if payload.get("headline_claim_allowed_after_audit") is False:
        boundaries.append("headline_claim_allowed_after_audit=false")
    return _dedupe(boundaries)


def _row_summary(payload: Mapping[str, Any]) -> JsonDict:
    return {key: payload.get(key) for key in SUMMARY_KEYS if key in payload}


def _paper_blocker_records(
    rows: list[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    records: list[JsonDict] = []
    for row in rows:
        if row.get("evidence_class") == "missing":
            records.append(_paper_blocker(row, "missing_or_unreadable_artifact"))
        if str(row.get("experiment_id") or "") == "exp3300":
            if payloads.get("exp3300", {}).get("garak_gate_passed") is not True:
                records.append(_paper_blocker(row, "garak_gate_not_passed"))
            if payloads.get("exp3300", {}).get("dataflip_gate_passed") is not True:
                records.append(_paper_blocker(row, "dataflip_gate_not_passed"))
        if str(row.get("experiment_id") or "") == "exp3302":
            if payloads.get("exp3302", {}).get("headline_claim_allowed") is not True:
                records.append(_paper_blocker(row, "repair_headline_source_not_allowed"))
            if payloads.get("exp3302", {}).get("provenance_clean") is not True:
                records.append(_paper_blocker(row, "repair_source_provenance_not_clean"))
        if str(row.get("experiment_id") or "") == "exp3303":
            if _repair_headline_audit_required(payloads) and not _repair_headline_claim_allowed(payloads):
                records.append(_paper_blocker(row, "repair_headline_audit_not_allowed"))
            if payloads.get("exp3303", {}).get("substrate_consistency_passed") is False:
                records.append(_paper_blocker(row, "repair_audit_substrate_inconsistent"))
        if _current_critical_flags(row):
            records.append(_paper_blocker(row, "current_critical_quality_flags"))
    if not _fr11_replay_safe(payloads):
        row = _row_by_id(rows, "exp3304")
        records.append(_paper_blocker(row, "fr11_replay_not_safe"))
    if not _historical_flagged_evidence_bounded(rows):
        row = next((item for item in rows if _historical_unbounded(item)), {})
        records.append(_paper_blocker(row, "historical_flagged_evidence_unbounded"))
    return _unique_blockers(records)


def _paper_blocker(row: Mapping[str, Any], reason: str) -> JsonDict:
    return {
        "source_experiment_id": str(row.get("experiment_id") or ""),
        "path": str(row.get("path") or ""),
        "evidence_kind": str(row.get("evidence_kind") or ""),
        "evidence_class": str(row.get("evidence_class") or ""),
        "reason": reason,
    }


def _unique_blockers(records: list[Mapping[str, Any]]) -> list[JsonDict]:
    seen: set[tuple[str, str]] = set()
    unique: list[JsonDict] = []
    for record in records:
        key = (str(record.get("source_experiment_id") or ""), str(record.get("reason") or ""))
        if key not in seen:
            unique.append(dict(record))
            seen.add(key)
    return unique


def _current_critical_flags(row: Mapping[str, Any]) -> bool:
    if str(row.get("evidence_kind") or "") in {"historical-corrigendum", "sidecar-only", "gated-skipped"}:
        return False
    return any(_flag_is_critical(flag) for flag in _as_list(row.get("quality_flags")))


def _flag_is_critical(flag: Mapping[str, Any]) -> bool:
    return str(flag.get("severity") or "").strip().lower() in CRITICAL_SEVERITIES


def _garak_gate_passed(payloads: Mapping[str, Mapping[str, Any]]) -> bool:
    exp3300 = payloads.get("exp3300", {})
    return (
        exp3300.get("garak_redteam_eval_v3_ready") is True
        and exp3300.get("garak_gate_passed") is True
    )


def _repair_headline_audit_required(payloads: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(payloads.get("exp3302") or payloads.get("exp3303"))


def _repair_headline_claim_allowed(payloads: Mapping[str, Mapping[str, Any]]) -> bool:
    audit = payloads.get("exp3303", {})
    if audit:
        return (
            audit.get("repair_headline_evidence_audit_ready") is True
            and audit.get("headline_claim_allowed_after_audit") is True
        )
    source = payloads.get("exp3302", {})
    return source.get("headline_claim_allowed") is True


def _fr11_replay_safe(payloads: Mapping[str, Mapping[str, Any]]) -> bool:
    return _fr11_payload_safe(payloads.get("exp3304", {}))


def _fr11_payload_safe(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("fr11_redteam_repair_memory_replay_ready") is True
        and payload.get("controller_memory_only") is True
        and payload.get("foundation_weight_updates_performed") is False
        and payload.get("consolidation_gate_passed") is True
        and not str(payload.get("blocked_reason") or "").strip()
    )


def _historical_flagged_evidence_bounded(rows: list[Mapping[str, Any]]) -> bool:
    return not any(_historical_unbounded(row) for row in rows)


def _historical_unbounded(row: Mapping[str, Any]) -> bool:
    if str(row.get("evidence_kind") or "") != "historical-corrigendum":
        return False
    if not row.get("quality_flags"):
        return False
    boundaries = set(_list_of_strings(row.get("claim_boundaries")))
    return "historical_or_aggregation_context_only" not in boundaries


def _top_gap(
    rows: list[Mapping[str, Any]],
    paper_blockers: list[Mapping[str, Any]],
    *,
    garak_gate_passed: bool,
    repair_claim_allowed: bool,
    repair_audit_required: bool,
    fr11_safe: bool,
    historical_bounded: bool,
) -> str:
    if any(row.get("evidence_class") == "missing" for row in rows):
        return "restore_missing_v305_artifacts"
    if not garak_gate_passed:
        return "pass_garak_redteam_gate"
    exp3300_blockers = {
        str(record.get("reason") or "")
        for record in paper_blockers
        if record.get("source_experiment_id") == "exp3300"
    }
    if exp3300_blockers:
        return "clear_garak_dataflip_and_quality_flags"
    if repair_audit_required and not repair_claim_allowed:
        return "clear_repair_headline_evidence_audit"
    if not fr11_safe:
        return "repair_fr11_controller_memory_replay_safety"
    if not historical_bounded:
        return "bound_historical_flagged_evidence"
    return "ready_for_v305_capstone"


def _next_gap_recommendation(top_gap: str) -> str:
    recommendations = {
        "restore_missing_v305_artifacts": "Restore or rerun missing `.305` artifacts before capstone synthesis.",
        "pass_garak_redteam_gate": "Rerun the defended Garak gate until the attack-success gate passes with clean provenance.",
        "clear_garak_dataflip_and_quality_flags": "Resolve Exp 3300 DataFlip failure and critical quality flags before publication readiness.",
        "clear_repair_headline_evidence_audit": "Clear the Exp 3303 repair audit before promoting repair evidence.",
        "repair_fr11_controller_memory_replay_safety": "Repair FR-11 replay safety without foundation-weight updates.",
        "bound_historical_flagged_evidence": "Add corrigendum boundaries for historical flagged evidence.",
        "ready_for_v305_capstone": "Proceed to the v305 capstone with zero matrix blockers.",
    }
    return recommendations.get(top_gap, "Review unresolved matrix blockers before capstone synthesis.")


def _gate_summary(
    rows: list[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    return {
        "garak_gate": _gate_record(rows, "exp3300")
        | {
            "garak_gate_passed": _garak_gate_passed(payloads),
            "dataflip_gate_passed": payloads.get("exp3300", {}).get("dataflip_gate_passed")
            is True,
        },
        "repair_headline": _gate_record(rows, "exp3303")
        | {
            "audit_required": _repair_headline_audit_required(payloads),
            "repair_headline_claim_allowed": _repair_headline_claim_allowed(payloads),
            "source_headline_claim_allowed": payloads.get("exp3302", {}).get(
                "headline_claim_allowed"
            )
            is True,
        },
        "fr11_replay": _gate_record(rows, "exp3304")
        | {
            "fr11_replay_safe": _fr11_replay_safe(payloads),
            "controller_memory_only": payloads.get("exp3304", {}).get("controller_memory_only")
            is True,
            "foundation_weight_updates_performed": payloads.get("exp3304", {}).get(
                "foundation_weight_updates_performed"
            )
            is True,
        },
        "historical_boundaries": {
            "historical_flagged_evidence_bounded": _historical_flagged_evidence_bounded(rows),
            "flagged_historical_rows": [
                row["experiment_id"]
                for row in rows
                if row.get("evidence_kind") == "historical-corrigendum"
                and row.get("quality_flags")
            ],
        },
    }


def _gate_record(rows: list[Mapping[str, Any]], experiment_id: str) -> JsonDict:
    row = _row_by_id(rows, experiment_id)
    return {
        "source_experiment_id": experiment_id,
        "present": row.get("present") is True,
        "readable_json_object": row.get("readable_json_object") is True,
        "evidence_class": str(row.get("evidence_class") or "missing"),
        "ready": row.get("ready") is True,
        "blocker_reasons": _list_of_strings(row.get("blocker_reasons")),
        "quality_flags": _as_list(row.get("quality_flags")),
    }


def _row_by_id(rows: list[Mapping[str, Any]], experiment_id: str) -> Mapping[str, Any]:
    return next((row for row in rows if row.get("experiment_id") == experiment_id), {})


def _class_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("evidence_class") or "missing")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _principle_annotations() -> JsonDict:
    return {
        "aggregation_only": "Matrix v37 reads checked-in `.305` artifacts only.",
        "missing_is_blocking": "Missing upstream artifacts stay explicit and block paper readiness.",
        "gated_skipped_visible": "Gated or deterministic support tasks stay present as rows.",
        "quality_flags_visible": "Critical adversarial flags are counted even when a gate boolean passed.",
        "sidecar_bounded": "KAN and controller-memory-only rows cannot become headline evidence.",
        "paper_ready_rule": "Readiness requires Garak pass, repair-audit allowance, safe FR-11, and zero current blockers.",
    }


def _is_live_substrate(substrate: str) -> bool:
    normalized = substrate.lower()
    return "llama_cpp" in normalized or "live" in normalized or "gpu" in normalized


def _bool_is_false(payload: Mapping[str, Any], key: str) -> bool:
    return payload.get(key) is False


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _list_of_strings(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "rows": artifact.get("rows"),
        "paper_blockers": artifact.get("paper_blockers"),
        "top_gap": artifact.get("top_gap"),
        "gate_summary": artifact.get("gate_summary"),
        "cited_upstream_artifacts": artifact.get("cited_upstream_artifacts"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v37_ready="
        f"{str(artifact.get('matrix_v37_ready') is True).lower()}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"artifact_count_scanned={artifact.get('artifact_count_scanned')}; "
        f"clean={artifact.get('clean_evidence_count')}; "
        f"blocked={artifact.get('blocked_evidence_count')}; "
        f"flagged={artifact.get('flagged_evidence_count')}; "
        f"sidecar_only={artifact.get('sidecar_only_evidence_count')}; "
        f"garak_gate_passed={str(artifact.get('garak_gate_passed') is True).lower()}; "
        f"repair_headline_claim_allowed="
        f"{str(artifact.get('repair_headline_claim_allowed') is True).lower()}; "
        f"fr11_replay_safe={str(artifact.get('fr11_replay_safe') is True).lower()}; "
        f"paper_blockers={artifact.get('paper_blocker_count')}; "
        f"top_gap={artifact.get('top_gap')}"
    )
