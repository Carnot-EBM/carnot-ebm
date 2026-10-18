"""Build the Exp 3080 milestone .287 capstone artifact.

Spec refs: REQ-REPORT-3080, SCENARIO-REPORT-3080.

The .287 capstone is an authority artifact over matrix v21. It summarizes
what the already-written evidence allows, and it deliberately does not rerun
models, solvers, repair loops, synthesis, hardware, or the conductor. That
separation keeps paper readiness tied to checked-in rows instead of letting a
closeout task accidentally create or promote new evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.287"
NEXT_MILESTONE = "2026.05.288"
SCHEMA = "carnot.milestone_capstone.v287_matrix_v21_aggregation.v1"
ARTIFACT = "experiment_3080_capstone_v287"
OUTPUT_REL_PATH = Path("results/experiment_3080_capstone_v287.json")
MATRIX_V21_REL_PATH = Path("results/experiment_3079_cross_corpus_matrix_v21.json")
CAPSTONE_V286_REL_PATH = Path("results/experiment_3066_capstone_v286.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3080_capstone_v287.py"

STATUSES = (
    "clean",
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
    "retired",
)
COUNT_FIELDS = {
    "clean": "clean_rows",
    "flagged": "flagged_rows",
    "bounded": "bounded_rows",
    "blocked": "blocked_rows",
    "gated_skipped": "gated_skipped_rows",
    "projection_only": "projection_only_rows",
    "missing": "missing_rows",
    "retired": "retired_rows",
}
PUBLICATION_BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
}
REQUIRED_ROW_KEYS = {
    "row_id",
    "status",
    "source_artifact",
    "source_field",
    "evidence_class",
    "blocker_class",
    "claim_scope",
    "summary",
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and fail closed when evidence is absent or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a source digest so the capstone can cite immutable inputs."""

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
    """REQ-REPORT-3080: close milestone .287 from matrix v21 only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V21_REL_PATH)
    capstone_v286 = read_json_object(root_path / CAPSTONE_V286_REL_PATH)
    rows = _matrix_rows(matrix)
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(matrix, rows)
    source_artifacts = _source_artifacts(root_path, matrix, rows)
    required_errors = _required_source_errors(source_artifacts)
    matrix_summary = _matrix_v21_summary(
        matrix,
        rows,
        status_counts,
        publication_blockers,
        required_errors,
        capstone_v286,
    )
    capstone_ready = _capstone_ready(matrix, matrix_summary, required_errors)
    paper_checks = _paper_ready_checks(
        capstone_ready=capstone_ready,
        rows=rows,
        publication_blockers=publication_blockers,
        matrix_summary=matrix_summary,
    )
    paper_ready = all(check["passed"] is True for check in paper_checks)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "verifier_gain_status": _verifier_gain_status(rows),
        "repair_claim_status": _repair_claim_status(rows),
        "fr11_self_learning_status": _fr11_self_learning_status(rows),
        "ebt_arm_status": _ebt_arm_status(rows),
        "gatemate_status": _gatemate_status(rows),
        "ssqa_status": _ssqa_status(rows),
        "matrix_v21_summary": matrix_summary,
        "row_status_counts": status_counts,
        "publication_blocker_count": len(publication_blockers),
        "publication_blockers": publication_blockers,
        "paper_ready_checks": paper_checks,
        "status_rollup": _status_rollup(rows),
        "next_milestone_recommendation": _next_milestone_recommendation(
            len(publication_blockers)
        ),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "missing_source_artifacts": [
            row["path"] for row in source_artifacts if row.get("present") is not True
        ],
        "required_source_errors": required_errors,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_live_repair_rerun": True,
        "no_historical_artifact_rewrite": True,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "blocked_required_matrix_v21_missing",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact, bool(matrix))
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3080 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


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


def _publication_blockers(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> list[JsonDict]:
    blockers = _as_list(matrix.get("publication_blockers")) or [
        row
        for row in rows
        if normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES
    ]
    return [_normal_publication_blocker(row) for row in blockers if isinstance(row, Mapping)]


def _normal_publication_blocker(row: Mapping[str, Any]) -> JsonDict:
    status = normal_status(str(row.get("status") or "missing"))
    return {
        "row_id": str(row.get("row_id") or ""),
        "status": status,
        "blocker_class": str(row.get("blocker_class") or blocker_class(status)),
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "claim_scope": str(row.get("claim_scope") or ""),
    }


def _source_artifacts(
    root: Path,
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> list[JsonDict]:
    sources: dict[str, JsonDict] = {
        MATRIX_V21_REL_PATH.as_posix(): {
            "experiment_id": "exp3079",
            "path": MATRIX_V21_REL_PATH.as_posix(),
            "role": "matrix_v21_authority",
            "required": True,
        },
        CAPSTONE_V286_REL_PATH.as_posix(): {
            "experiment_id": "exp3066",
            "path": CAPSTONE_V286_REL_PATH.as_posix(),
            "role": "capstone_v286_context",
            "required": True,
        },
    }
    for raw_source in _as_list(matrix.get("source_artifacts")):
        source = _as_mapping(raw_source)
        path = str(source.get("path") or "")
        if path:
            sources.setdefault(
                path,
                {
                    "experiment_id": str(source.get("experiment_id") or f"source:{path}"),
                    "path": path,
                    "role": str(source.get("role") or "matrix_v21_source"),
                    "required": False,
                },
            )
    for row in rows:
        path = str(row.get("source_artifact") or "")
        if path:
            sources.setdefault(
                path,
                {
                    "experiment_id": f"row_source:{path}",
                    "path": path,
                    "role": "row_source_citation",
                    "required": False,
                },
            )
    return [_source_artifact_row(root, source) for source in sources.values()]


def _source_artifact_row(root: Path, source: Mapping[str, Any]) -> JsonDict:
    path = Path(str(source.get("path") or ""))
    artifact_path = root / path
    source_type = "json" if path.suffix == ".json" else "text"
    payload = read_json_object(artifact_path) if source_type == "json" else {}
    return {
        "experiment_id": str(source.get("experiment_id") or "unknown"),
        "path": path.as_posix(),
        "role": str(source.get("role") or ""),
        "required": source.get("required") is True,
        "source_type": source_type,
        "present": artifact_path.is_file(),
        "readable_json_object": bool(payload),
        "sha256": sha256_file(artifact_path),
    }


def _required_source_errors(source_artifacts: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row.get("experiment_id")),
            "reason": "missing_or_malformed_required_artifact",
        }
        for row in source_artifacts
        if row.get("required") is True and row.get("readable_json_object") is not True
    ]


def _matrix_v21_summary(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
    publication_blockers: list[Mapping[str, Any]],
    required_errors: list[Mapping[str, Any]],
    capstone_v286: Mapping[str, Any],
) -> JsonDict:
    count_matches = all(
        _int_or_none(matrix.get(COUNT_FIELDS[status])) == status_counts[status]
        for status in STATUSES
    )
    rows_total = _int_or_none(matrix.get("rows_total")) or len(rows)
    blocker_count = _int_or_none(matrix.get("publication_blocker_count"))
    blocker_count_matches = blocker_count == len(publication_blockers)
    return {
        "matrix_v21_ready": matrix.get("matrix_v21_ready") is True,
        "rows_total": rows_total,
        "row_count_observed": len(rows),
        "rows_total_matches": rows_total == len(rows),
        "status_counts": dict(status_counts),
        "integer_status_counts_present": all(
            _int_or_none(matrix.get(field)) is not None for field in COUNT_FIELDS.values()
        ),
        "counts_match_rows": rows_total == len(rows) and count_matches,
        "rows_machine_readable": _rows_machine_readable(rows),
        "publication_blocker_count": blocker_count,
        "publication_blocker_count_observed": len(publication_blockers),
        "publication_blocker_count_matches": blocker_count_matches,
        "required_source_artifacts_readable": not required_errors,
        "required_source_error_count": len(required_errors),
        "capstone_v286_ready": capstone_v286.get("capstone_ready") is True,
        "capstone_v286_paper_ready": capstone_v286.get("paper_ready") is True,
        "status_by_row": {
            str(row.get("row_id")): normal_status(str(row.get("status") or "missing"))
            for row in rows
            if row.get("row_id")
        },
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _rows_machine_readable(rows: list[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        REQUIRED_ROW_KEYS <= set(row)
        and normal_status(str(row.get("status") or "missing")) in STATUSES
        and bool(row.get("source_artifact"))
        and bool(row.get("source_field"))
        for row in rows
    )


def _capstone_ready(
    matrix: Mapping[str, Any],
    matrix_summary: Mapping[str, Any],
    required_errors: list[Mapping[str, Any]],
) -> bool:
    return (
        bool(matrix)
        and not required_errors
        and matrix.get("matrix_v21_ready") is True
        and matrix_summary.get("rows_total_matches") is True
        and matrix_summary.get("integer_status_counts_present") is True
        and matrix_summary.get("counts_match_rows") is True
        and matrix_summary.get("rows_machine_readable") is True
        and matrix_summary.get("publication_blocker_count_matches") is True
    )


def _paper_ready_checks(
    *,
    capstone_ready: bool,
    rows: list[Mapping[str, Any]],
    publication_blockers: list[Mapping[str, Any]],
    matrix_summary: Mapping[str, Any],
) -> list[JsonDict]:
    blocking_rows = [
        str(row.get("row_id") or "")
        for row in rows
        if normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES
    ]
    projection_rows = [
        str(row.get("row_id") or "")
        for row in rows
        if normal_status(str(row.get("status") or "missing")) == "projection_only"
    ]
    return [
        {
            "check": "capstone_ready",
            "passed": capstone_ready,
            "reason": "matrix v21 exists, row counts reconcile, and authority sources read",
        },
        {
            "check": "matrix_has_no_publication_blockers",
            "passed": len(publication_blockers) == 0,
            "reason": f"publication_blocker_count={len(publication_blockers)}",
        },
        {
            "check": "no_required_claim_has_blocking_status",
            "passed": not blocking_rows,
            "reason": f"blocking_required_rows={blocking_rows}",
        },
        {
            "check": "no_projection_only_publication_claim",
            "passed": not projection_rows,
            "reason": f"projection_only_rows={projection_rows}",
        },
        {
            "check": "matrix_v21_declared_ready",
            "passed": matrix_summary.get("matrix_v21_ready") is True,
            "reason": str(matrix_summary.get("honest_verdict") or ""),
        },
    ]


def _status_rollup(rows: list[Mapping[str, Any]]) -> JsonDict:
    return {
        "verifier_gain_rows": _topic_row_ids(rows, "verifier", "solver", "first_token", "verge"),
        "repair_rows": _topic_row_ids(rows, "repair"),
        "fr11_rows": _topic_row_ids(rows, "fr11"),
        "ebt_arm_rows": _topic_row_ids(rows, "ebt", "arm"),
        "gatemate_rows": _topic_row_ids(rows, "gatemate"),
        "ssqa_rows": _topic_row_ids(rows, "ssqa"),
    }


def _verifier_gain_status(rows: list[Mapping[str, Any]]) -> str:
    topic = _topic_rows(rows, "verifier", "solver", "first_token", "verge")
    statuses = _topic_statuses(topic)
    if not topic:
        return "missing_verifier_gain_evidence"
    if statuses <= {"clean", "retired"}:
        return "clean_verifier_gain_recovered"
    if statuses & {"flagged", "gated_skipped", "blocked"}:
        return "flagged_or_gated_verifier_gain_recovery_incomplete"
    return "bounded_verifier_gain_recovery_not_promoted"


def _repair_claim_status(rows: list[Mapping[str, Any]]) -> str:
    topic = _topic_rows(rows, "repair")
    statuses = _topic_statuses(topic)
    if not topic:
        return "missing_repair_evidence"
    if statuses <= {"clean", "retired"}:
        return "clean_or_retired"
    if statuses & {"blocked", "gated_skipped"}:
        return "bounded_and_gated_skipped"
    if statuses & {"flagged", "bounded"}:
        return "bounded"
    return "missing_repair_evidence"


def _fr11_self_learning_status(rows: list[Mapping[str, Any]]) -> str:
    topic = _topic_rows(rows, "fr11")
    statuses = _topic_statuses(topic)
    if not topic:
        return "missing_fr11_evidence"
    if statuses <= {"clean", "retired"}:
        return "clean_controller_only"
    if "flagged" in statuses:
        return "flagged_controller_only_budget_exceeded"
    if "bounded" in statuses:
        return "bounded_controller_only"
    return "blocked_controller_only"


def _ebt_arm_status(rows: list[Mapping[str, Any]]) -> str:
    topic = _topic_rows(rows, "ebt", "arm")
    statuses = _topic_statuses(topic)
    if not topic:
        return "missing_ebt_arm_evidence"
    if statuses <= {"clean", "retired"}:
        return "clean_adapter_implementation_evidence"
    if "projection_only" in statuses:
        return "projection_only_feasible_no_implementation"
    return "bounded_or_blocked_no_implementation"


def _gatemate_status(rows: list[Mapping[str, Any]]) -> str:
    topic = _topic_rows(rows, "gatemate")
    statuses = _topic_statuses(topic)
    if not topic:
        return "missing_gatemate_evidence"
    if statuses <= {"clean", "retired"}:
        return "clean_host_visible_output_ready"
    if statuses & {"blocked", "missing", "gated_skipped"}:
        return "blocked_no_rerun_operator_actions_required"
    return "bounded_gatemate_claim"


def _ssqa_status(rows: list[Mapping[str, Any]]) -> str:
    topic = _topic_rows(rows, "ssqa")
    statuses = _topic_statuses(topic)
    if not topic:
        return "missing_ssqa_evidence"
    if statuses <= {"clean", "retired"}:
        return "clean_host_visible_readback_ready"
    if statuses & {"gated_skipped", "missing", "blocked"}:
        return "gated_skipped_host_visible_smoke_missing"
    return "bounded_ssqa_claim"


def _topic_rows(rows: list[Mapping[str, Any]], *tokens: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if any(token in _row_text(row) for token in tokens)]


def _topic_row_ids(rows: list[Mapping[str, Any]], *tokens: str) -> list[str]:
    return [str(row.get("row_id") or "") for row in _topic_rows(rows, *tokens)]


def _topic_statuses(rows: list[Mapping[str, Any]]) -> set[str]:
    return {normal_status(str(row.get("status") or "missing")) for row in rows}


def _row_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("row_id") or ""),
            str(row.get("evidence_class") or ""),
            str(row.get("claim_scope") or ""),
        ]
    ).lower()


def _next_milestone_recommendation(publication_blocker_count: int) -> str:
    return (
        f"{NEXT_MILESTONE}: reduce publication_blocker_count from "
        f"{publication_blocker_count} by first clearing verifier-gain recovery "
        "(fix or retire Exp3070 adversarial flags, raise abstention_precision "
        "to the gate, rerun Exp3072), then run Exp3075 repair micro-panel only "
        "after that gate passes; in parallel commit the GateMate output "
        "contract and host-visible smoke transcript so SSQA can leave "
        "gate-skipped status; keep FR-11 controller-only until the "
        "completeness budget is zero and keep EBT/ARM-EBM projection-only "
        "until an adapter implementation has tests."
    )


def _honest_verdict(artifact: Mapping[str, Any], matrix_present: bool) -> str:
    if not matrix_present:
        return "blocked_required_matrix_v21_missing"
    summary = _as_mapping(artifact.get("matrix_v21_summary"))
    if artifact.get("capstone_ready") is not True:
        return (
            "blocked_capstone_preconditions: "
            f"required_source_errors={len(_as_list(artifact.get('required_source_errors')))}; "
            f"counts_match_rows={summary.get('counts_match_rows')}; "
            "publication_blocker_count_matches="
            f"{summary.get('publication_blocker_count_matches')}"
        )
    return (
        "complete: "
        f"capstone_ready={str(artifact['capstone_ready']).lower()}; "
        f"paper_ready={str(artifact['paper_ready']).lower()}; "
        f"matrix_v21_ready={str(summary.get('matrix_v21_ready')).lower()}; "
        f"publication_blocker_count={artifact['publication_blocker_count']}; "
        f"verifier_gain_status={artifact['verifier_gain_status']}; "
        f"repair_claim_status={artifact['repair_claim_status']}; "
        f"fr11_self_learning_status={artifact['fr11_self_learning_status']}; "
        f"gatemate_status={artifact['gatemate_status']}; "
        f"ssqa_status={artifact['ssqa_status']}"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "matrix_v21_and_checked_in_results",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def normal_status(status: str) -> str:
    """Normalize legacy labels into the eight matrix v21 row classes."""

    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str) -> str:
    """Map one normalized row class to its publication-boundary reason."""

    return {
        "clean": "none",
        "flagged": "adversarial_or_methodology_flag",
        "bounded": "bounded_claim",
        "blocked": "required_blocker",
        "gated_skipped": "structured_gate_skip",
        "projection_only": "projection_only",
        "missing": "missing_artifact",
        "retired": "retired_claim",
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
    "CAPSTONE_V286_REL_PATH",
    "CONDUCTOR_LOG_REL_PATH",
    "MATRIX_V21_REL_PATH",
    "OUTPUT_REL_PATH",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "blocker_class",
    "build_artifact",
    "normal_status",
    "read_json_object",
    "sha256_file",
    "write_artifact",
]
