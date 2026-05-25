"""Build the Exp 3053 milestone .285 capstone artifact.

Spec refs: REQ-REPORT-3053, SCENARIO-REPORT-3053.

This module closes milestone .285 by reading the matrix v19 aggregation and the
source artifacts named by that matrix. It deliberately performs no model,
solver, verifier, synthesis, or hardware work; it only decides which claims are
safe to promote, which are bounded, and which remain blocked.
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
MILESTONE = "2026.05.285"
NEXT_MILESTONE = "2026.05.286"
SCHEMA = "carnot.milestone_capstone.v285_matrix_v19_aggregation.v1"
ARTIFACT = "experiment_3053_capstone_v285"
OUTPUT_REL_PATH = Path("results/experiment_3053_capstone_v285.json")
MATRIX_V19_REL_PATH = Path("results/experiment_3052_cross_corpus_matrix_v19.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3053_capstone_v285.py"

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
    "clean": "clean_count",
    "flagged": "flagged_count",
    "bounded": "bounded_count",
    "blocked": "blocked_count",
    "gated_skipped": "gated_skipped_count",
    "projection_only": "projection_only_count",
    "missing": "missing_count",
    "retired": "retired_count",
}
PUBLICATION_BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
}
BLOCKED_CLAIM_STATUSES = {"flagged", "blocked", "projection_only", "missing"}
FR11_PRD_ALIGNED_STATUSES = {
    "controller_only_solver_feedback_and_locality_ready",
    "controller_only_solver_feedback_ready",
    "controller_only_locality_ready",
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while treating absent, malformed, or array JSON as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 digest for a present artifact."""

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
    """REQ-REPORT-3053: synthesize .285 closure from matrix v19 and sources."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V19_REL_PATH)
    rows = _matrix_rows(matrix)
    source_artifacts = _source_artifacts(root_path, matrix)
    required_errors = _required_source_errors(source_artifacts)
    matrix_summary = _matrix_v19_summary(matrix, rows, required_errors)
    rows_by_id = {str(row.get("row_id")): row for row in rows if row.get("row_id")}
    repair_status = str(matrix.get("repair_claim_status") or "missing")
    fr11_status = str(matrix.get("fr11_self_learning_status") or "missing")
    gatemate_status = str(matrix.get("gatemate_status") or "missing")
    ssqa_status = str(matrix.get("ssqa_status") or "missing")
    capstone_ready = _capstone_ready(matrix, matrix_summary, required_errors)
    paper_checks = _paper_ready_checks(
        capstone_ready=capstone_ready,
        matrix_summary=matrix_summary,
        repair_status=repair_status,
        fr11_status=fr11_status,
        gatemate_status=gatemate_status,
        ssqa_status=ssqa_status,
        rows_by_id=rows_by_id,
    )
    paper_ready = all(bool(check["passed"]) for check in paper_checks)
    promoted_claims = _claims_with_status(rows, {"clean"})
    bounded_claims = _claims_with_status(rows, {"bounded"})
    blocked_claims = _claims_with_status(rows, BLOCKED_CLAIM_STATUSES)
    gated_skipped_claims = _claims_with_status(rows, {"gated_skipped"})
    retired_claims = _claims_with_status(rows, {"retired"})

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "repair_claim_status": repair_status,
        "fr11_self_learning_status": fr11_status,
        "gatemate_status": gatemate_status,
        "ssqa_status": ssqa_status,
        "matrix_v19_summary": matrix_summary,
        "what_285_proved": _what_285_proved(matrix_summary, promoted_claims, bounded_claims),
        "promoted_claims": promoted_claims,
        "bounded_claims": bounded_claims,
        "blocked_claims": blocked_claims,
        "gated_skipped_claims": gated_skipped_claims,
        "gate_skipped_claims": gated_skipped_claims,
        "flagged_claims": _claims_with_status(rows, {"flagged"}),
        "missing_claims": _claims_with_status(rows, {"missing"}),
        "projection_only_claims": _claims_with_status(rows, {"projection_only"}),
        "retired_claims": retired_claims,
        "paper_ready_checks": paper_checks,
        "next_milestone_recommendation": _next_milestone_recommendation(
            repair_status,
            fr11_status,
            gatemate_status,
            ssqa_status,
            matrix_summary,
            retired_claims,
        ),
        "source_artifacts": source_artifacts,
        "source_checksums": {str(row["path"]): row.get("sha256") for row in source_artifacts},
        "missing_source_artifacts": [
            str(row["path"]) for row in source_artifacts if row.get("present") is not True
        ],
        "required_source_errors": required_errors,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_historical_artifact_rewrite": True,
        "publication_action_allowed": paper_ready,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "blocked_required_matrix_v19_missing",
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
    """Build and persist the Exp 3053 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _matrix_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    return [dict(row) for row in _as_list(matrix.get("rows")) if isinstance(row, Mapping)]


def _source_artifacts(root: Path, matrix: Mapping[str, Any]) -> list[JsonDict]:
    rows = [
        _source_artifact_row(
            root,
            {
                "experiment_id": "exp3052",
                "path": MATRIX_V19_REL_PATH.as_posix(),
                "role": "matrix_v19_authority",
                "required": True,
            },
        )
    ]
    for source in _as_list(matrix.get("source_artifacts")):
        if isinstance(source, Mapping):
            rows.append(_source_artifact_row(root, source))
    return rows


def _source_artifact_row(root: Path, source: Mapping[str, Any]) -> JsonDict:
    path = Path(str(source.get("path") or ""))
    artifact_path = root / path
    payload = read_json_object(artifact_path)
    return {
        "experiment_id": str(source.get("experiment_id") or "unknown"),
        "path": path.as_posix(),
        "role": str(source.get("role") or ""),
        "required": source.get("required") is True,
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


def _matrix_v19_summary(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    required_errors: list[Mapping[str, Any]],
) -> JsonDict:
    row_counts = _status_counts(rows)
    top_counts = {status: _int_or_none(matrix.get(COUNT_FIELDS[status])) or 0 for status in STATUSES}
    status_by_row = {
        str(row.get("row_id")): _normal_status(str(row.get("status") or "missing"))
        for row in rows
        if row.get("row_id")
    }
    return {
        "matrix_v19_ready": matrix.get("matrix_v19_ready") is True,
        "rows_total": _int_or_none(matrix.get("rows_total")) or len(rows),
        "row_count_observed": len(rows),
        "clean": top_counts["clean"],
        "flagged": top_counts["flagged"],
        "bounded": top_counts["bounded"],
        "blocked": top_counts["blocked"],
        "gated_skipped": top_counts["gated_skipped"],
        "projection_only": top_counts["projection_only"],
        "missing": top_counts["missing"],
        "retired": top_counts["retired"],
        "counts_from_rows": row_counts,
        "counts_match_rows": row_counts == top_counts,
        "all_rows_classified": bool(rows) and all(status in STATUSES for status in status_by_row.values()),
        "status_by_row": status_by_row,
        "required_source_artifacts_readable": not required_errors,
        "required_source_error_count": len(required_errors),
        "nonclean_publication_blockers": sum(row_counts[status] for status in PUBLICATION_BLOCKING_STATUSES),
        "retired_count": row_counts["retired"],
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[_normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _capstone_ready(
    matrix: Mapping[str, Any],
    matrix_summary: Mapping[str, Any],
    required_errors: list[Mapping[str, Any]],
) -> bool:
    return (
        bool(matrix)
        and not required_errors
        and matrix.get("matrix_v19_ready") is True
        and matrix_summary.get("rows_total") == matrix_summary.get("row_count_observed")
        and matrix_summary.get("counts_match_rows") is True
        and matrix_summary.get("all_rows_classified") is True
    )


def _paper_ready_checks(
    *,
    capstone_ready: bool,
    matrix_summary: Mapping[str, Any],
    repair_status: str,
    fr11_status: str,
    gatemate_status: str,
    ssqa_status: str,
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    repair_row = _as_mapping(rows_by_id.get("repair:headline_status"))
    fr11_scope_ok = _fr11_prd_scope_ok(fr11_status, rows_by_id)
    return [
        {
            "check": "capstone_ready",
            "passed": capstone_ready,
            "reason": "matrix v19 exists, rows classify, counts reconcile, and required sources read",
        },
        {
            "check": "repair_promotable",
            "passed": repair_status == "clean_candidate"
            and _normal_status(str(repair_row.get("status") or "missing")) == "clean",
            "reason": f"repair_claim_status={repair_status}",
        },
        {
            "check": "fr11_prd_scope",
            "passed": fr11_scope_ok,
            "reason": f"fr11_self_learning_status={fr11_status}",
        },
        {
            "check": "gatemate_host_visible_transcript",
            "passed": gatemate_status == "host_visible_transcript_ready"
            and _row_status(rows_by_id, "gatemate:host_visible_smoke") == "clean",
            "reason": f"gatemate_status={gatemate_status}",
        },
        {
            "check": "ssqa_readback_eligible",
            "passed": ssqa_status not in {"missing", "blocked", "gated_skipped_host_visible_smoke_missing"}
            and _row_status(rows_by_id, "ssqa:readback_gate") == "clean",
            "reason": f"ssqa_status={ssqa_status}",
        },
        {
            "check": "matrix_has_no_publication_blockers",
            "passed": _int_or_none(matrix_summary.get("nonclean_publication_blockers")) == 0,
            "reason": (
                "nonclean_publication_blockers="
                f"{matrix_summary.get('nonclean_publication_blockers')}"
            ),
        },
    ]


def _fr11_prd_scope_ok(fr11_status: str, rows_by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    if fr11_status not in FR11_PRD_ALIGNED_STATUSES:
        return False
    for row_id in ("fr11:solver_feedback", "fr11:kan_locality"):
        if str(_as_mapping(rows_by_id.get(row_id)).get("blocker_class") or "") == "model_weight_scope_violation":
            return False
    return True


def _row_status(rows_by_id: Mapping[str, Mapping[str, Any]], row_id: str) -> str:
    return _normal_status(str(_as_mapping(rows_by_id.get(row_id)).get("status") or "missing"))


def _claims_with_status(rows: list[Mapping[str, Any]], statuses: set[str]) -> list[JsonDict]:
    return [
        _claim_entry(row)
        for row in rows
        if _normal_status(str(row.get("status") or "missing")) in statuses
    ]


def _claim_entry(row: Mapping[str, Any]) -> JsonDict:
    return {
        "row_id": str(row.get("row_id") or ""),
        "status": _normal_status(str(row.get("status") or "missing")),
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "evidence_class": str(row.get("evidence_class") or ""),
        "blocker_class": str(row.get("blocker_class") or ""),
        "claim_scope": str(row.get("claim_scope") or ""),
        "summary": _as_mapping(row.get("summary")),
    }


def _what_285_proved(
    matrix_summary: Mapping[str, Any],
    promoted_claims: list[Mapping[str, Any]],
    bounded_claims: list[Mapping[str, Any]],
) -> JsonDict:
    return {
        "matrix_v19_accounting": dict(matrix_summary),
        "promoted_claim_count": len(promoted_claims),
        "bounded_claim_count": len(bounded_claims),
        "capstone_scope": "synthesis_only_from_matrix_v19_and_checked_in_source_artifacts",
    }


def _next_milestone_recommendation(
    repair_status: str,
    fr11_status: str,
    gatemate_status: str,
    ssqa_status: str,
    matrix_summary: Mapping[str, Any],
    retired_claims: list[Mapping[str, Any]],
) -> str:
    return (
        f"{NEXT_MILESTONE}: retire unsupported repair headline wording "
        f"({len(retired_claims)} retired row(s)); rerun repair promotion only after "
        f"repair_status={repair_status} clears adversarial/methodology blockers; gate GateMate "
        f"and SSQA until gatemate_status={gatemate_status} becomes host-visible transcript "
        f"evidence and ssqa_status={ssqa_status} consumes that smoke gate; carry FR-11 as "
        f"{fr11_status} without model-weight overclaim; drive matrix publication blockers "
        f"from {matrix_summary.get('nonclean_publication_blockers')} to 0."
    )


def _honest_verdict(artifact: Mapping[str, Any], matrix_present: bool) -> str:
    if not matrix_present:
        return "blocked_required_matrix_v19_missing"
    if artifact["capstone_ready"] is not True:
        return (
            "blocked_capstone_preconditions: "
            f"required_source_errors={len(_as_list(artifact.get('required_source_errors')))}; "
            f"counts_match_rows={artifact['matrix_v19_summary'].get('counts_match_rows')}; "
            f"all_rows_classified={artifact['matrix_v19_summary'].get('all_rows_classified')}"
        )
    return (
        "complete: "
        f"capstone_ready={str(artifact['capstone_ready']).lower()}; "
        f"paper_ready={str(artifact['paper_ready']).lower()}; "
        f"repair_claim_status={artifact['repair_claim_status']}; "
        f"fr11_self_learning_status={artifact['fr11_self_learning_status']}; "
        f"gatemate_status={artifact['gatemate_status']}; "
        f"ssqa_status={artifact['ssqa_status']}; "
        f"next={NEXT_MILESTONE}_retire_gate_rerun_blockers"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }


def _normal_status(status: str) -> str:
    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    return normalized if normalized in STATUSES else "missing"


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
    "MATRIX_V19_REL_PATH",
    "OUTPUT_REL_PATH",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "build_artifact",
    "read_json_object",
    "sha256_file",
    "write_artifact",
]
