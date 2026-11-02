"""Build the Exp 3094 milestone .288 capstone artifact.

Spec refs: REQ-REPORT-3094, SCENARIO-REPORT-3094.

The .288 capstone is a closeout ledger over matrix v22. It reads the checked-in
matrix and every input artifact that matrix names, then reports what moved and
what is still blocked. It deliberately does not rerun models, verifiers,
solvers, repair loops, hardware tools, or the conductor because a capstone is
an authority summary over evidence that already exists.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.288"
NEXT_MILESTONE = "2026.05.289"
SCHEMA = "carnot.milestone_capstone.v288_matrix_v22_aggregation.v1"
ARTIFACT = "experiment_3094_capstone_v288"
OUTPUT_REL_PATH = Path("results/experiment_3094_capstone_v288.json")
MATRIX_V22_REL_PATH = Path("results/experiment_3093_cross_corpus_matrix_v22.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3094_capstone_v288.py"

STATUSES = (
    "clean",
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "missing",
    "retired",
    "projection_only",
)
PUBLICATION_BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "missing",
    "projection_only",
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and fail closed when evidence is absent or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a file checksum so capstone citations are independently checkable."""

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
    """REQ-REPORT-3094: close .288 from matrix v22 and named inputs only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V22_REL_PATH)
    input_paths = _capstone_input_paths(matrix)
    source_artifacts = _source_artifacts(root_path, input_paths)
    rows = _matrix_rows(matrix)
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(matrix, rows)
    publication_blocker_count = _publication_blocker_count(matrix, publication_blockers)
    missing_inputs = _missing_capstone_input_artifacts(source_artifacts)
    required_source_errors = _required_source_errors(source_artifacts)
    invariant_violations = _invariant_violations(
        matrix,
        rows,
        status_counts,
        publication_blockers,
        publication_blocker_count,
        required_source_errors,
    )
    capstone_ready = not invariant_violations
    headline_gaps = [
        dict(gap)
        for gap in _as_list(matrix.get("headline_model_spec_gaps"))
        if isinstance(gap, Mapping)
    ]
    paper_checks = _paper_ready_checks(capstone_ready, publication_blocker_count, headline_gaps, missing_inputs)
    paper_ready = all(check["passed"] is True for check in paper_checks)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": publication_blocker_count,
        "verifier_gain_status": _verifier_gain_status(rows),
        "repair_claim_status": _repair_claim_status(rows),
        "fr11_self_learning_status": _fr11_self_learning_status(rows),
        "ebt_arm_status": _ebt_arm_status(rows),
        "gatemate_status": _gatemate_status(rows),
        "ssqa_status": _ssqa_status(rows),
        "matrix_v22_summary": _matrix_v22_summary(matrix, rows, status_counts),
        "row_status_counts": status_counts,
        "publication_blockers": publication_blockers,
        "prd_gap_summary": _prd_gap_summary(rows, publication_blockers),
        "status_movement_from_v21": _status_movement_from_v21(matrix),
        "paper_ready_checks": paper_checks,
        "headline_model_spec_gaps": headline_gaps,
        "missing_capstone_input_artifacts": missing_inputs,
        "source_artifacts_loaded": _source_artifacts_loaded(source_artifacts),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "required_source_errors": required_source_errors,
        "invariant_violations": invariant_violations,
        "next_milestone_recommendation": _next_milestone_recommendation(rows, publication_blockers),
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_live_repair_rerun": True,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
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
    """Build and persist the Exp 3094 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _capstone_input_paths(matrix: Mapping[str, Any]) -> list[str]:
    paths = [
        str(path)
        for path in _as_list(matrix.get("capstone_input_artifacts"))
        if str(path)
    ]
    if MATRIX_V22_REL_PATH.as_posix() not in paths:
        paths.insert(0, MATRIX_V22_REL_PATH.as_posix())
    return list(dict.fromkeys(paths))


def _source_artifacts(root: Path, input_paths: list[str]) -> list[JsonDict]:
    return [_source_artifact_row(root, Path(path)) for path in input_paths]


def _source_artifact_row(root: Path, rel_path: Path) -> JsonDict:
    path = root / rel_path
    payload = read_json_object(path) if rel_path.suffix == ".json" else {}
    return {
        "experiment_id": _experiment_id(rel_path, payload),
        "path": rel_path.as_posix(),
        "role": _source_role(rel_path),
        "required": rel_path == MATRIX_V22_REL_PATH,
        "source_type": "json" if rel_path.suffix == ".json" else "text",
        "present": path.is_file(),
        "readable_json_object": bool(payload) if rel_path.suffix == ".json" else path.is_file(),
        "sha256": sha256_file(path),
    }


def _experiment_id(path: Path, payload: Mapping[str, Any]) -> str:
    artifact = str(payload.get("artifact") or "")
    if artifact:
        return artifact
    stem = path.stem
    if stem.startswith("experiment_"):
        return stem
    return f"source:{path.as_posix()}"


def _source_role(path: Path) -> str:
    if path == MATRIX_V22_REL_PATH:
        return "matrix_v22_authority"
    if "capstone" in path.name:
        return "capstone_input_context"
    if "matrix" in path.name:
        return "matrix_input_context"
    return "capstone_input_artifact"


def _matrix_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    return [_claim_entry(row) for row in _as_list(matrix.get("rows")) if isinstance(row, Mapping)]


def _claim_entry(row: Mapping[str, Any]) -> JsonDict:
    status = normal_status(str(row.get("status") or "missing"))
    return {
        "row_id": str(row.get("row_id") or ""),
        "status": status,
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "evidence_class": str(row.get("evidence_class") or ""),
        "blocker_class": str(row.get("blocker_class") or blocker_class(status)),
        "claim_scope": str(row.get("claim_scope") or ""),
        "summary": _as_mapping(row.get("summary")),
    }


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _publication_blockers(matrix: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    raw_blockers = _as_list(matrix.get("publication_blockers"))
    source = raw_blockers or [
        row
        for row in rows
        if normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES
    ]
    return [_publication_blocker(row) for row in source if isinstance(row, Mapping)]


def _publication_blocker(row: Mapping[str, Any]) -> JsonDict:
    status = normal_status(str(row.get("status") or "missing"))
    return {
        "row_id": str(row.get("row_id") or ""),
        "status": status,
        "blocker_class": str(row.get("blocker_class") or blocker_class(status)),
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "claim_scope": str(row.get("claim_scope") or ""),
    }


def _publication_blocker_count(
    matrix: Mapping[str, Any],
    publication_blockers: list[Mapping[str, Any]],
) -> int:
    return _int_or_none(matrix.get("publication_blocker_count")) or len(publication_blockers)


def _required_source_errors(source_artifacts: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "path": str(row["path"]),
            "reason": "missing_or_malformed_required_artifact",
        }
        for row in source_artifacts
        if row.get("required") is True and row.get("readable_json_object") is not True
    ]


def _missing_capstone_input_artifacts(source_artifacts: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "path": str(row["path"]),
            "reason": "named by matrix v22 capstone_input_artifacts but not readable",
        }
        for row in source_artifacts
        if row.get("required") is not True and row.get("readable_json_object") is not True
    ]


def _source_artifacts_loaded(source_artifacts: list[Mapping[str, Any]]) -> JsonDict:
    return {
        "named_by_matrix_v22": len(source_artifacts),
        "present": sum(row.get("present") is True for row in source_artifacts),
        "readable_json_object": sum(row.get("readable_json_object") is True for row in source_artifacts),
        "missing_or_malformed": sum(row.get("readable_json_object") is not True for row in source_artifacts),
    }


def _invariant_violations(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
    publication_blockers: list[Mapping[str, Any]],
    publication_blocker_count: int,
    required_source_errors: list[Mapping[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if required_source_errors:
        violations.append("required source artifacts missing or malformed")
    if matrix and matrix.get("matrix_v22_ready") is not True:
        violations.append("matrix v22 authority is not ready")
    if matrix and (_int_or_none(matrix.get("rows_total")) or 0) != len(rows):
        violations.append("matrix v22 rows_total does not match rows")
    if matrix and _matrix_status_counts(matrix) != dict(status_counts):
        violations.append("matrix v22 status_counts do not match rows")
    if matrix and publication_blocker_count != len(publication_blockers):
        violations.append("publication_blocker_count does not match publication_blockers")
    return violations


def _matrix_status_counts(matrix: Mapping[str, Any]) -> dict[str, int]:
    raw = _as_mapping(matrix.get("status_counts"))
    return {status: _int_or_none(raw.get(status)) or 0 for status in STATUSES}


def _paper_ready_checks(
    capstone_ready: bool,
    publication_blocker_count: int,
    headline_gaps: list[Mapping[str, Any]],
    missing_inputs: list[Mapping[str, Any]],
) -> list[JsonDict]:
    return [
        {
            "check": "capstone_ready",
            "passed": capstone_ready,
            "reason": "matrix v22 authority loaded and row/blocker counts reconcile",
        },
        {
            "check": "publication_blocker_count_zero",
            "passed": publication_blocker_count == 0,
            "reason": f"publication_blocker_count={publication_blocker_count}",
        },
        {
            "check": "headline_model_spec_gaps_clear",
            "passed": not headline_gaps,
            "reason": f"headline_model_spec_gaps={len(headline_gaps)}",
        },
        {
            "check": "headline_missing_inputs_clear",
            "passed": not missing_inputs,
            "reason": f"missing_capstone_input_artifacts={len(missing_inputs)}",
        },
    ]


def _matrix_v22_summary(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
) -> JsonDict:
    return {
        "matrix_v22_ready": matrix.get("matrix_v22_ready") is True,
        "rows_total": _int_or_none(matrix.get("rows_total")) or len(rows),
        "row_count_observed": len(rows),
        "status_counts": dict(status_counts),
        "publication_blocker_count": _int_or_none(matrix.get("publication_blocker_count")) or 0,
        "blocker_delta_from_v21": _int_or_none(matrix.get("blocker_delta_from_v21")) or 0,
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _status_movement_from_v21(matrix: Mapping[str, Any]) -> JsonDict:
    reconciliation = _as_mapping(matrix.get("blocker_reconciliation_from_ledger"))
    decreases = [
        _as_mapping(row)
        for row in _as_list(reconciliation.get("decreases"))
        if isinstance(row, Mapping)
    ]
    fr11_rows: list[str] = []
    for decrease in decreases:
        if decrease.get("ledger_category") == "fr11_budget":
            fr11_rows.extend(str(row_id) for row_id in _as_list(decrease.get("row_ids")))
    return {
        "publication_blocker_count_before": _int_or_none(
            reconciliation.get("publication_blocker_count_before")
        )
        or 0,
        "publication_blocker_count_after": _int_or_none(
            reconciliation.get("publication_blocker_count_after")
        )
        or _int_or_none(matrix.get("publication_blocker_count"))
        or 0,
        "blocker_delta_from_v21": _int_or_none(matrix.get("blocker_delta_from_v21")) or 0,
        "blocker_decreases": decreases,
        "blocker_increases": [
            _as_mapping(row)
            for row in _as_list(reconciliation.get("increases"))
            if isinstance(row, Mapping)
        ],
        "fr11_controller_only_rows_moved": sorted(fr11_rows),
    }


def _prd_gap_summary(
    rows: list[Mapping[str, Any]],
    publication_blockers: list[Mapping[str, Any]],
) -> JsonDict:
    return {
        "verifier_repair": _gap_summary(
            rows,
            publication_blockers,
            ("verifier", "local_sota", "abstention", "calibration", "repair", "feedback"),
            "Verifier/repair remains a paper blocker until exact verifier evidence and the repair micro-panel clear.",
        ),
        "fr11": _gap_summary(
            rows,
            publication_blockers,
            ("fr11", "controller_only", "self_learning", "kancl"),
            "FR-11 is controller-only; no base model weight update or completeness claim is inferred.",
        ),
        "ebt_arm_bridge": _gap_summary(
            rows,
            publication_blockers,
            ("ebt", "arm", "adapter", "sidecar", "future_adapter"),
            "Sidecar/schema evidence is not model-integration evidence.",
        ),
        "hardware_evidence": _gap_summary(
            rows,
            publication_blockers,
            ("gatemate", "ssqa", "readback", "hardware", "host_visible"),
            "Operator evidence ingestion is not host-visible hardware speedup or readback evidence.",
        ),
        "publication_readiness": _gap_summary(
            rows,
            publication_blockers,
            ("paper_readiness", "capstone"),
            "Publication remains blocked until the matrix blocker count reaches zero.",
        ),
    }


def _gap_summary(
    rows: list[Mapping[str, Any]],
    publication_blockers: list[Mapping[str, Any]],
    tokens: tuple[str, ...],
    claim_boundary: str,
) -> JsonDict:
    row_by_id = {str(row.get("row_id") or ""): row for row in rows}
    gap_rows = _topic_rows(rows, *tokens)
    gap_blockers = [
        blocker
        for blocker in publication_blockers
        if _row_matches(_as_mapping(row_by_id.get(str(blocker.get("row_id") or ""))) or blocker, tokens)
    ]
    return {
        "row_ids": [str(row.get("row_id") or "") for row in gap_rows],
        "statuses_present": sorted({normal_status(str(row.get("status") or "missing")) for row in gap_rows}),
        "publication_blocker_count": len(gap_blockers),
        "publication_blocker_row_ids": [str(row.get("row_id") or "") for row in gap_blockers],
        "claim_boundary": claim_boundary,
    }


def _verifier_gain_status(rows: list[Mapping[str, Any]]) -> str:
    statuses = _topic_statuses(rows, "verifier", "local_sota", "abstention", "calibration")
    if statuses <= {"clean", "retired"} and statuses:
        return "clean_verifier_gain_exact_grounded"
    if statuses & {"flagged", "gated_skipped", "blocked", "missing"}:
        return "flagged_or_gated_verifier_gain_recovery_incomplete"
    return "bounded_verifier_gain_recovery_not_promoted"


def _repair_claim_status(rows: list[Mapping[str, Any]]) -> str:
    statuses = _topic_statuses(rows, "repair", "feedback")
    if statuses <= {"clean", "retired"} and statuses:
        return "clean_or_retired"
    if statuses & {"flagged", "bounded", "blocked", "gated_skipped", "missing"}:
        return "bounded_flagged_gated_missing_verifier_gated"
    return "missing_repair_evidence"


def _fr11_self_learning_status(rows: list[Mapping[str, Any]]) -> str:
    statuses = _topic_statuses(rows, "fr11", "controller_only", "self_learning", "kancl")
    if statuses <= {"clean", "retired"} and statuses:
        return "clean_controller_only_zero_mistake_budget"
    return "flagged_controller_only_budget_exceeded"


def _ebt_arm_status(rows: list[Mapping[str, Any]]) -> str:
    statuses = _topic_statuses(rows, "ebt", "arm", "adapter", "sidecar", "future_adapter")
    if statuses <= {"clean", "retired"} and statuses:
        return "clean_adapter_implementation_evidence"
    if "projection_only" in statuses:
        return "projection_only_sidecar_schema_no_model_integration"
    return "bounded_or_blocked_no_model_integration"


def _gatemate_status(rows: list[Mapping[str, Any]]) -> str:
    statuses = _topic_statuses(rows, "gatemate")
    if statuses <= {"clean", "retired"} and statuses:
        return "clean_host_visible_output_ready"
    return "blocked_no_rerun_operator_actions_required"


def _ssqa_status(rows: list[Mapping[str, Any]]) -> str:
    statuses = _topic_statuses(rows, "ssqa", "readback")
    if statuses <= {"clean", "retired"} and statuses:
        return "clean_host_visible_readback_ready"
    return "gated_skipped_host_visible_smoke_missing"


def _topic_rows(rows: list[Mapping[str, Any]], *tokens: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if _row_matches(row, tokens)]


def _row_matches(row: Mapping[str, Any], tokens: tuple[str, ...]) -> bool:
    return any(token in _row_text(row) for token in tokens)


def _topic_statuses(rows: list[Mapping[str, Any]], *tokens: str) -> set[str]:
    return {normal_status(str(row.get("status") or "missing")) for row in _topic_rows(rows, *tokens)}


def _row_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("row_id") or ""),
            str(row.get("evidence_class") or ""),
            str(row.get("claim_scope") or ""),
        ]
    ).lower()


def _next_milestone_recommendation(
    rows: list[Mapping[str, Any]],
    publication_blockers: list[Mapping[str, Any]],
) -> str:
    gap_counts = _prd_gap_summary(rows, publication_blockers)
    verifier_repair_count = gap_counts["verifier_repair"]["publication_blocker_count"]
    hardware_count = gap_counts["hardware_evidence"]["publication_blocker_count"]
    return (
        f"{NEXT_MILESTONE}: clear verifier/repair first "
        f"({verifier_repair_count} blocker rows: verifier gain, formal feedback, "
        "and missing repair micro-panel), because repair promotion remains gated "
        "by verifier evidence; then collect operator-owned GateMate/SSQA "
        f"host-visible evidence ({hardware_count} blocker rows) without claiming "
        "speedup; keep EBT/ARM projection-only until live model integration has "
        "tests; preserve FR-11 as controller-only unless a later matrix proves "
        "broader model-learning boundaries."
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_artifacts",
        "source": "matrix_v22_and_named_capstone_input_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("capstone_ready") is not True:
        return (
            "blocked_capstone_v288_preconditions: "
            f"invariant_violations={_as_list(artifact.get('invariant_violations'))}"
        )
    return (
        "complete: "
        "capstone_ready=true; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        "matrix_v22_ready=true; "
        f"publication_blocker_count={artifact['publication_blocker_count']}; "
        f"verifier_gain_status={artifact['verifier_gain_status']}; "
        f"repair_claim_status={artifact['repair_claim_status']}; "
        f"fr11_self_learning_status={artifact['fr11_self_learning_status']}; "
        f"ebt_arm_status={artifact['ebt_arm_status']}; "
        f"gatemate_status={artifact['gatemate_status']}; "
        f"ssqa_status={artifact['ssqa_status']}"
    )


def normal_status(status: str) -> str:
    """Normalize legacy matrix labels into the v22 status vocabulary."""

    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str) -> str:
    """Map a normalized status to the publication blocker class."""

    return {
        "clean": "none",
        "flagged": "adversarial_or_methodology_flag",
        "bounded": "bounded_claim",
        "blocked": "required_blocker",
        "gated_skipped": "structured_gate_skip",
        "missing": "missing_artifact",
        "retired": "retired_claim",
        "projection_only": "projection_only",
    }[normal_status(status)]


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "MATRIX_V22_REL_PATH",
    "OUTPUT_REL_PATH",
    "PUBLICATION_BLOCKING_STATUSES",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "STATUSES",
    "blocker_class",
    "build_artifact",
    "normal_status",
    "read_json_object",
    "sha256_file",
    "write_artifact",
]
