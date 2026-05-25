"""Build the Exp 3042 repair promotion reconciliation artifact.

Spec refs: REQ-REPORT-3042, SCENARIO-REPORT-3042.

This module is a ledger reconciliation step for matrix v19. It does not run a
model, verifier, solver, or board workflow. It reads the already-written repair
and flag-hygiene artifacts, removes only the aggregation false positives that
Exp 3041 explicitly cleared, and keeps the remaining repair blockers attached
to a single matrix-row decision.
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
RUN_DATE = "20260525"
MILESTONE = "2026.05.285"
SCHEMA = "carnot.repair_promotion.reconciliation.v3"
ARTIFACT = "experiment_3042_repair_promotion_reconciliation_v3"
OUTPUT_REL_PATH = Path("results/experiment_3042_repair_promotion_reconciliation_v3.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3042_repair_promotion_reconciliation_v3.py"

EXP3028_REL_PATH = Path("results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json")
EXP3029_REL_PATH = Path("results/experiment_3029_repair_promotion_boundary_audit_v2.json")
MATRIX_V18_REL_PATH = Path("results/experiment_3038_cross_corpus_matrix_v18.json")
CAPSTONE_V284_REL_PATH = Path("results/experiment_3039_capstone_v284.json")
EXP3041_REL_PATH = Path("results/experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1.json")

DELTA_FIELDS = (
    "pass_at_1_delta",
    "pass_at_k_delta",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "false_accept_delta",
)
SUMMARY_FIELDS = (
    "n_tasks",
    "n_live_transcripts",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "false_accept_delta",
    "intent_drift_count",
    "candidate_intent_drift_count",
)
BLOCKER_LIST_KEYS = (
    "true_blocker_rows",
    "missing_metadata_rows",
    "unresolved_bound_rows",
)
REPAIR_EXPERIMENT_IDS = {"exp3016", "exp3028", "exp3029"}
DECISIONS = {"clean_candidate", "bounded", "blocked", "retired"}


@dataclass(frozen=True)
class SourceSpec:
    """A required checked-in artifact consumed by the reconciliation."""

    experiment_id: str
    path: Path
    role: str
    required: bool = True


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3028", EXP3028_REL_PATH, "clean_repair_evidence"),
    SourceSpec("exp3029", EXP3029_REL_PATH, "prior_repair_boundary"),
    SourceSpec("exp3038", MATRIX_V18_REL_PATH, "matrix_v18_context"),
    SourceSpec("exp3039", CAPSTONE_V284_REL_PATH, "capstone_v284_context"),
    SourceSpec("exp3041", EXP3041_REL_PATH, "flag_hygiene_authority"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while treating absence, arrays, and malformed JSON as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for a present source artifact."""

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
    """REQ-REPORT-3042: reconcile the matrix v19 repair-row decision."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    loaded = _load_sources(root_path)
    source_artifacts = [
        _source_artifact(root_path, spec, loaded[spec.experiment_id]) for spec in SOURCE_SPECS
    ]
    required_errors = _required_source_errors(loaded)
    payloads = {exp_id: row["payload"] for exp_id, row in loaded.items()}
    exp3028 = payloads.get("exp3028", {})
    exp3041 = payloads.get("exp3041", {})
    duration_s = _duration(start, now_s)

    aggregation_false_positives_removed = _aggregation_false_positives_removed(exp3041)
    exp3028_checks = _exp3028_evidence_checks(exp3028)
    exp3028_blockers = _exp3028_evidence_blockers(exp3028, exp3028_checks)
    flag_hygiene_blockers = _repair_relevant_flag_hygiene_blockers(exp3041)

    remaining_blockers = _unique_rows(
        _source_error_blockers(required_errors) + exp3028_blockers + flag_hygiene_blockers
    )
    repair_claim_status = _repair_claim_status(required_errors, exp3028_checks, remaining_blockers)
    repair_reconciliation_ready = not required_errors and repair_claim_status in DECISIONS
    repair_promotion_candidate = repair_claim_status == "clean_candidate"

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "repair_reconciliation_ready": repair_reconciliation_ready,
        "repair_promotion_candidate": repair_promotion_candidate,
        "repair_claim_status": repair_claim_status,
        "accepted_source_artifacts": source_artifacts,
        "remaining_blockers": remaining_blockers,
        "aggregation_false_positives_removed": aggregation_false_positives_removed,
        "repair_delta_summary": _repair_delta_summary(exp3028),
        "exp3028_evidence_checks": exp3028_checks,
        "prior_repair_status": _prior_repair_status(payloads),
        "source_checksums": _source_checksums(source_artifacts),
        "missing_source_artifacts": [
            row["path"] for row in source_artifacts if row.get("present") is not True
        ],
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_historical_artifact_rewrite": True,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(
            repair_claim_status,
            repair_reconciliation_ready,
            len(remaining_blockers),
            len(aggregation_false_positives_removed),
            required_errors,
        ),
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3042 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_sources(root: Path) -> dict[str, JsonDict]:
    loaded: dict[str, JsonDict] = {}
    for spec in SOURCE_SPECS:
        path = root / spec.path
        loaded[spec.experiment_id] = {
            "payload": read_json_object(path),
            "present": path.is_file(),
        }
    return loaded


def _source_artifact(root: Path, spec: SourceSpec, loaded: Mapping[str, Any]) -> JsonDict:
    path = root / spec.path
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "present": bool(loaded.get("present")),
        "readable_json_object": bool(loaded.get("payload")),
        "required": spec.required,
        "sha256": sha256_file(path),
    }


def _required_source_errors(loaded: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    errors: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        if spec.required and not loaded[spec.experiment_id].get("payload"):
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "path": spec.path.as_posix(),
                    "reason": "missing_or_malformed_artifact",
                }
            )
    return errors


def _source_error_blockers(errors: list[JsonDict]) -> list[JsonDict]:
    return [
        _blocker(
            row_id=f"{error['experiment_id']}:required_source_missing",
            classification="source_gap",
            source_artifact=str(error["path"]),
            source_field="artifact",
            rationale="Required Exp 3042 source artifact is absent or malformed.",
            evidence=error,
            experiment_id=str(error["experiment_id"]),
        )
        for error in errors
    ]


def _exp3028_evidence_checks(exp3028: Mapping[str, Any]) -> JsonDict:
    n_tasks = _int_or_none(exp3028.get("n_tasks"))
    n_live_transcripts = _int_or_none(exp3028.get("n_live_transcripts"))
    model_specs = _as_list(exp3028.get("model_specs"))
    deltas = {field: _float_or_none(exp3028.get(field)) for field in DELTA_FIELDS}
    precondition_checks = _as_list(exp3028.get("precondition_checks"))
    substrate = _as_mapping(exp3028.get("inference_substrate"))
    checksum = str(exp3028.get("reproducibility_checksum") or "")
    return {
        "clean_repair_rerun_ready": exp3028.get("clean_repair_rerun_ready") is True,
        "repair_controller_clean": exp3028.get("repair_controller_clean") is True,
        "clean_repair_claim_promotable_candidate": (
            exp3028.get("clean_repair_claim_promotable_candidate") is True
        ),
        "model_specs_present": bool(model_specs),
        "model_specs_have_identity": all(_model_spec_has_identity(row) for row in model_specs),
        "transcript_counts_present": n_tasks is not None and n_live_transcripts is not None,
        "transcript_counts_sufficient": n_tasks is not None
        and n_tasks > 0
        and n_live_transcripts is not None
        and n_live_transcripts >= n_tasks,
        "deltas_present": all(value is not None for value in deltas.values()),
        "positive_pass_at_1_delta": (deltas["pass_at_1_delta"] or 0.0) > 0.0,
        "nonnegative_pass_at_k_delta": (deltas["pass_at_k_delta"] or -1.0) >= 0.0,
        "syntax_schema_nonregression": (
            (deltas["syntax_failure_rate_delta"] or 0.0) <= 0.0
            and (deltas["schema_failure_rate_delta"] or 0.0) <= 0.0
        ),
        "false_accept_nonregression": (deltas["false_accept_delta"] or 0.0) <= 0.0,
        "tautology_gate_clean": exp3028.get("tautology_gate_clean") is True,
        "intent_drift_clean": (_int_or_none(exp3028.get("intent_drift_count")) or 0) == 0,
        "legacy_smoke_only_absent": exp3028.get("legacy_smoke_only_used") is False,
        "reproducibility_checksum_present": bool(checksum),
        "acceptance_controller_evidence_present": _acceptance_controller_evidence_present(
            precondition_checks,
            substrate,
        ),
    }


def _exp3028_evidence_blockers(
    exp3028: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    if not checks.get("clean_repair_rerun_ready"):
        blockers.append(_exp3028_blocker("clean_repair_rerun_not_ready", "clean_repair_rerun_ready"))
    if not checks.get("repair_controller_clean"):
        blockers.append(_exp3028_blocker("repair_controller_not_clean", "repair_controller_clean"))
    if not checks.get("clean_repair_claim_promotable_candidate"):
        blockers.append(
            _exp3028_blocker(
                "clean_repair_claim_candidate_missing",
                "clean_repair_claim_promotable_candidate",
            )
        )
    if not checks.get("model_specs_present") or not checks.get("model_specs_have_identity"):
        blockers.append(_exp3028_blocker("model_specs_missing", "model_specs"))
    if not checks.get("transcript_counts_present") or not checks.get("transcript_counts_sufficient"):
        blockers.append(_exp3028_blocker("transcript_count_gap", "n_tasks/n_live_transcripts"))
    if not checks.get("deltas_present"):
        blockers.append(_exp3028_blocker("delta_fields_missing", ",".join(DELTA_FIELDS)))
    if not checks.get("positive_pass_at_1_delta"):
        blockers.append(_exp3028_blocker("pass_at_1_delta_not_positive", "pass_at_1_delta"))
    if not checks.get("nonnegative_pass_at_k_delta"):
        blockers.append(_exp3028_blocker("pass_at_k_delta_negative", "pass_at_k_delta"))
    if not checks.get("syntax_schema_nonregression"):
        blockers.append(
            _exp3028_blocker(
                "syntax_schema_regression",
                "syntax_failure_rate_delta/schema_failure_rate_delta",
                evidence={
                    "syntax_failure_rate_delta": exp3028.get("syntax_failure_rate_delta"),
                    "schema_failure_rate_delta": exp3028.get("schema_failure_rate_delta"),
                },
            )
        )
    if not checks.get("false_accept_nonregression"):
        blockers.append(
            _exp3028_blocker(
                "false_accept_regression",
                "false_accept_delta",
                evidence={"false_accept_delta": exp3028.get("false_accept_delta")},
            )
        )
    if not checks.get("tautology_gate_clean"):
        blockers.append(_exp3028_blocker("tautology_gate_not_clean", "tautology_gate_clean"))
    if not checks.get("intent_drift_clean"):
        blockers.append(
            _exp3028_blocker(
                "intent_drift",
                "intent_drift_count",
                evidence={"intent_drift_count": exp3028.get("intent_drift_count")},
            )
        )
    if not checks.get("legacy_smoke_only_absent"):
        blockers.append(_exp3028_blocker("legacy_smoke_only_used", "legacy_smoke_only_used"))
    if not checks.get("reproducibility_checksum_present"):
        blockers.append(_exp3028_blocker("reproducibility_checksum_missing", "reproducibility_checksum"))
    if not checks.get("acceptance_controller_evidence_present"):
        blockers.append(
            _exp3028_blocker(
                "acceptance_controller_evidence_missing",
                "precondition_checks/inference_substrate",
            )
        )
    return blockers


def _exp3028_blocker(
    suffix: str,
    source_field: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> JsonDict:
    return _blocker(
        row_id=f"exp3028:{suffix}",
        classification="source_gap",
        source_artifact=EXP3028_REL_PATH.as_posix(),
        source_field=source_field,
        rationale="Exp 3028 clean repair evidence is incomplete or regressed.",
        evidence=dict(evidence or {}),
        experiment_id="exp3028",
    )


def _acceptance_controller_evidence_present(
    precondition_checks: list[Any],
    substrate: Mapping[str, Any],
) -> bool:
    resources = {
        str(_as_mapping(row).get("resource") or "")
        for row in precondition_checks
        if _as_mapping(row).get("available") is True
    }
    return (
        "exp3015_acceptance_controller" in resources
        and substrate.get("live_repair_generation_run") is False
        and substrate.get("model_load_attempted") is False
    )


def _model_spec_has_identity(row: Any) -> bool:
    model = _as_mapping(row)
    return bool(model.get("hf_id")) and bool(model.get("checksum")) and bool(model.get("model_path"))


def _aggregation_false_positives_removed(exp3041: Mapping[str, Any]) -> list[JsonDict]:
    rows = _as_list(exp3041.get("aggregation_false_positive_rows"))
    return [
        _compact_row(row)
        for row in rows
        if _as_mapping(row).get("classification") == "aggregation_false_positive"
    ]


def _repair_relevant_flag_hygiene_blockers(exp3041: Mapping[str, Any]) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    for key in BLOCKER_LIST_KEYS:
        for row in _as_list(exp3041.get(key)):
            row_map = _as_mapping(row)
            if row_map.get("blocking") is True and _repair_relevant_blocker(row_map):
                blockers.append(_compact_row(row_map))
    return blockers


def _repair_relevant_blocker(row: Mapping[str, Any]) -> bool:
    experiment_id = str(row.get("experiment_id") or "")
    if experiment_id in REPAIR_EXPERIMENT_IDS:
        return True
    if str(row.get("classification") or "") in {"hardware_blocked", "gate_skipped"}:
        return False
    haystack = " ".join(
        [
            str(row.get("row_id") or ""),
            str(row.get("source_field") or ""),
            str(row.get("rationale") or ""),
        ]
    ).lower()
    return any(token in haystack for token in ("repair", "false_accept", "syntax", "schema", "intent"))


def _compact_row(row: Mapping[str, Any]) -> JsonDict:
    compact: JsonDict = {
        "row_id": str(row.get("row_id") or ""),
        "classification": str(row.get("classification") or ""),
        "blocking": bool(row.get("blocking")),
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "rationale": str(row.get("rationale") or ""),
    }
    for key in ("experiment_id", "matrix_status", "flag_kinds", "nested_source_artifact", "evidence"):
        if key in row:
            compact[key] = row[key]
    return compact


def _blocker(
    *,
    row_id: str,
    classification: str,
    source_artifact: str,
    source_field: str,
    rationale: str,
    evidence: Any,
    experiment_id: str,
) -> JsonDict:
    return {
        "row_id": row_id,
        "classification": classification,
        "blocking": True,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "rationale": rationale,
        "evidence": evidence,
        "experiment_id": experiment_id,
    }


def _unique_rows(rows: list[JsonDict]) -> list[JsonDict]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[JsonDict] = []
    for row in rows:
        key = (
            str(row.get("row_id") or ""),
            str(row.get("classification") or ""),
            str(row.get("source_artifact") or ""),
            str(row.get("source_field") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _repair_claim_status(
    required_errors: list[JsonDict],
    exp3028_checks: Mapping[str, bool],
    remaining_blockers: list[JsonDict],
) -> str:
    if required_errors:
        return "blocked"
    if not all(bool(value) for value in exp3028_checks.values()):
        return "blocked"
    if remaining_blockers:
        return "bounded"
    return "clean_candidate"


def _repair_delta_summary(exp3028: Mapping[str, Any]) -> JsonDict:
    summary: JsonDict = {}
    for field in SUMMARY_FIELDS:
        value = exp3028.get(field)
        if field in {
            "n_tasks",
            "n_live_transcripts",
            "intent_drift_count",
            "candidate_intent_drift_count",
        }:
            summary[field] = _int_or_none(value)
        else:
            summary[field] = _float_or_none(value)
    return summary


def _prior_repair_status(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    matrix_rows = _matrix_rows(payloads.get("exp3038", {}))
    exp3028_row = _matrix_row(matrix_rows, "exp3028")
    exp3029_row = _matrix_row(matrix_rows, "exp3029")
    capstone = payloads.get("exp3039", {})
    return {
        "exp3029_repair_claim_status": str(payloads.get("exp3029", {}).get("repair_claim_status") or ""),
        "matrix_v18_exp3028_status": str(exp3028_row.get("status") or ""),
        "matrix_v18_exp3029_status": str(exp3029_row.get("status") or ""),
        "capstone_repair_claim_status": str(capstone.get("repair_claim_status") or ""),
        "capstone_paper_ready": capstone.get("paper_ready") is True,
    }


def _matrix_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    return [dict(row) for row in _as_list(matrix.get("matrix_rows")) if isinstance(row, Mapping)]


def _matrix_row(rows: list[Mapping[str, Any]], experiment_id: str) -> JsonDict:
    for row in rows:
        if str(row.get("experiment_id") or "") == experiment_id:
            return dict(row)
    return {}


def _source_checksums(source_artifacts: list[Mapping[str, Any]]) -> JsonDict:
    return {str(row["path"]): row.get("sha256") for row in source_artifacts}


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }


def _honest_verdict(
    status: str,
    ready: bool,
    remaining_blockers: int,
    false_positives_removed: int,
    required_errors: list[JsonDict],
) -> str:
    if required_errors:
        return "blocked_required_source_missing: " + ",".join(
            str(error["experiment_id"]) for error in required_errors
        )
    return (
        f"complete: repair_claim_status={status}; "
        f"repair_reconciliation_ready={str(ready).lower()}; "
        f"repair_promotion_candidate={str(status == 'clean_candidate').lower()}; "
        f"remaining_blockers={remaining_blockers}; "
        f"aggregation_false_positives_removed={false_positives_removed}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CAPSTONE_V284_REL_PATH",
    "EXP3028_REL_PATH",
    "EXP3029_REL_PATH",
    "EXP3041_REL_PATH",
    "MATRIX_V18_REL_PATH",
    "OUTPUT_REL_PATH",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "build_artifact",
    "read_json_object",
    "sha256_file",
    "write_artifact",
]
