"""Build the Exp 3066 milestone .286 capstone artifact.

Spec refs: REQ-REPORT-3066, SCENARIO-REPORT-3066.

This module closes milestone .286 by treating matrix v20 as the authority. It
does not rerun models, solvers, repair loops, synthesis, or hardware. That
boundary matters because this capstone is supposed to decide what the existing
evidence allows, not create new evidence while summarizing it.
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
MILESTONE = "2026.05.286"
NEXT_MILESTONE = "2026.05.287"
SCHEMA = "carnot.milestone_capstone.v286_matrix_v20_aggregation.v1"
ARTIFACT = "experiment_3066_capstone_v286"
OUTPUT_REL_PATH = Path("results/experiment_3066_capstone_v286.json")
MATRIX_V20_REL_PATH = Path("results/experiment_3065_cross_corpus_matrix_v20.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3066_capstone_v286.py"

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
CLASS_FIELDS = {
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
    """Read a JSON object and fail closed when the path is absent or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for a present source file."""

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
    """REQ-REPORT-3066: synthesize .286 closure from matrix v20 and sources."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V20_REL_PATH)
    row_classes = _matrix_row_classes(matrix)
    rows = [row for field in CLASS_FIELDS.values() for row in row_classes[field]]
    source_artifacts = _source_artifacts(root_path, matrix)
    required_errors = _required_source_errors(source_artifacts)
    publication_blockers = _publication_blockers(matrix, rows)
    promoted_claims = [_claim_entry(row) for row in row_classes["clean_rows"]]
    source_coverage = _promoted_claim_source_coverage(root_path, promoted_claims)
    matrix_summary = _matrix_v20_summary(
        matrix, row_classes, rows, publication_blockers, source_coverage, required_errors
    )
    capstone_ready = _capstone_ready(matrix, matrix_summary, required_errors)
    statuses = _capstone_statuses(matrix, rows)
    paper_checks = _paper_ready_checks(
        capstone_ready=capstone_ready,
        matrix_summary=matrix_summary,
        source_coverage=source_coverage,
        rows=rows,
    )
    paper_ready = all(check["passed"] is True for check in paper_checks)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "capstone_ready": capstone_ready,
        "paper_ready": paper_ready,
        "repair_claim_status": statuses["repair_claim_status"],
        "solver_grounding_status": statuses["solver_grounding_status"],
        "fr11_self_learning_status": statuses["fr11_self_learning_status"],
        "kan_pwa_status": statuses["kan_pwa_status"],
        "gatemate_status": statuses["gatemate_status"],
        "ssqa_status": statuses["ssqa_status"],
        "matrix_v20_summary": matrix_summary,
        "promoted_claims": promoted_claims,
        "blocked_claims": [
            _claim_entry(row)
            for row in rows
            if normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES
        ],
        "retired_claims": [_claim_entry(row) for row in row_classes["retired_rows"]],
        "publication_blockers": publication_blockers,
        "paper_ready_checks": paper_checks,
        "promoted_claim_source_coverage": source_coverage,
        "next_milestone_recommendation": _next_milestone_recommendation(
            len(publication_blockers)
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
        "no_live_repair_rerun": True,
        "no_historical_artifact_rewrite": True,
        "publication_action_allowed": paper_ready,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "blocked_required_matrix_v20_missing",
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
    """Build and persist the Exp 3066 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _matrix_row_classes(matrix: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    return {
        field: [
            _claim_entry(row)
            for row in _as_list(matrix.get(field))
            if isinstance(row, Mapping)
        ]
        for field in CLASS_FIELDS.values()
    }


def _source_artifacts(root: Path, matrix: Mapping[str, Any]) -> list[JsonDict]:
    rows = [
        _source_artifact_row(
            root,
            {
                "experiment_id": "exp3065",
                "path": MATRIX_V20_REL_PATH.as_posix(),
                "role": "matrix_v20_authority",
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


def _matrix_v20_summary(
    matrix: Mapping[str, Any],
    row_classes: Mapping[str, list[Mapping[str, Any]]],
    rows: list[Mapping[str, Any]],
    publication_blockers: list[Mapping[str, Any]],
    source_coverage: Mapping[str, Any],
    required_errors: list[Mapping[str, Any]],
) -> JsonDict:
    counts = _status_counts(rows)
    rows_total = _int_or_none(matrix.get("rows_total")) or len(rows)
    publication_blocker_count = _int_or_none(matrix.get("publication_blocker_count")) or 0
    return {
        "matrix_v20_ready": matrix.get("matrix_v20_ready") is True,
        "rows_total": rows_total,
        "row_count_observed": len(rows),
        "row_class_lists_present": _row_class_lists_present(matrix),
        "rows_machine_readable": _rows_machine_readable(rows),
        "counts_from_rows": counts,
        "counts_match_rows": rows_total == len(rows),
        "publication_blocker_count": publication_blocker_count,
        "publication_blocker_count_observed": len(publication_blockers),
        "publication_blocker_count_matches": publication_blocker_count == len(publication_blockers),
        "required_source_artifacts_readable": not required_errors,
        "required_source_error_count": len(required_errors),
        "promoted_claim_count": len(row_classes["clean_rows"]),
        "all_promoted_claims_have_sources": source_coverage.get(
            "all_promoted_claims_have_sources"
        )
        is True,
        "status_by_row": {
            str(row.get("row_id")): normal_status(str(row.get("status") or "missing"))
            for row in rows
            if row.get("row_id")
        },
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _row_class_lists_present(matrix: Mapping[str, Any]) -> bool:
    return all(isinstance(matrix.get(field), list) for field in CLASS_FIELDS.values())


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
        and matrix.get("matrix_v20_ready") is True
        and matrix_summary.get("row_class_lists_present") is True
        and matrix_summary.get("rows_machine_readable") is True
        and matrix_summary.get("counts_match_rows") is True
        and matrix_summary.get("publication_blocker_count_matches") is True
    )


def _paper_ready_checks(
    *,
    capstone_ready: bool,
    matrix_summary: Mapping[str, Any],
    source_coverage: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> list[JsonDict]:
    return [
        {
            "check": "capstone_ready",
            "passed": capstone_ready,
            "reason": "matrix v20 exists, rows classify, counts reconcile, and sources read",
        },
        {
            "check": "matrix_has_no_publication_blockers",
            "passed": matrix_summary.get("publication_blocker_count_observed") == 0,
            "reason": (
                "publication_blocker_count="
                f"{matrix_summary.get('publication_blocker_count_observed')}"
            ),
        },
        {
            "check": "promoted_claims_have_source_artifacts",
            "passed": source_coverage.get("all_promoted_claims_have_sources") is True,
            "reason": (
                "missing_promoted_claim_sources="
                f"{source_coverage.get('missing_promoted_claim_sources')}"
            ),
        },
        {
            "check": "fr11_model_weight_boundary",
            "passed": _fr11_model_weight_boundary(rows),
            "reason": "model-weight learning requires explicit train-and-verify source evidence",
        },
        {
            "check": "hardware_host_visible_output",
            "passed": _hardware_host_visible_output(rows),
            "reason": "GateMate/SSQA/hardware claims require host-visible output evidence",
        },
    ]


def _promoted_claim_source_coverage(root: Path, claims: list[Mapping[str, Any]]) -> JsonDict:
    missing: list[JsonDict] = []
    for claim in claims:
        source_artifact = str(claim.get("source_artifact") or "")
        source_field = str(claim.get("source_field") or "")
        if not source_artifact or not source_field or not read_json_object(root / source_artifact):
            missing.append(
                {
                    "row_id": str(claim.get("row_id") or ""),
                    "source_artifact": source_artifact,
                    "source_field": source_field,
                }
            )
    return {
        "promoted_claim_count": len(claims),
        "missing_promoted_claim_sources": missing,
        "all_promoted_claims_have_sources": not missing,
    }


def _fr11_model_weight_boundary(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        text = _row_text(row)
        summary = _as_mapping(row.get("summary"))
        if "fr11" in text and (
            summary.get("model_weight_training") is True
            or summary.get("model_weight_mutation") is True
        ):
            if not (
                summary.get("model_weight_training_verified") is True
                or summary.get("model_weight_learning_verified") is True
            ):
                return False
    return True


def _hardware_host_visible_output(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        if normal_status(str(row.get("status") or "missing")) != "clean":
            continue
        text = _row_text(row)
        summary = _as_mapping(row.get("summary"))
        if any(token in text for token in ("gatemate", "ssqa", "hardware_speedup")):
            if not (
                summary.get("host_visible_output_evidence") is True
                or summary.get("host_visible_smoke_present") is True
                or summary.get("readback_transcript_present") is True
            ):
                return False
    return True


def _capstone_statuses(matrix: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> JsonDict:
    summaries = _as_mapping(matrix.get("status_summaries"))
    fr11_status = _status_from_summary(summaries, "fr11", _status_from_rows(rows, "fr11"))
    gatemate_status = _status_from_summary(
        summaries, "gatemate", _status_from_rows(rows, "gatemate")
    )
    if not _fr11_model_weight_boundary(rows):
        fr11_status = "blocked_model_weight_learning_unverified"
    if not _hardware_host_visible_output(rows):
        gatemate_status = "blocked_host_visible_output_missing"
    return {
        "repair_claim_status": _status_from_summary(
            summaries, "repair", _status_from_rows(rows, "repair")
        ),
        "solver_grounding_status": _status_from_summary(
            summaries,
            "solver_grounded_verification",
            _status_from_rows(rows, "solver"),
        ),
        "fr11_self_learning_status": fr11_status,
        "kan_pwa_status": _status_from_summary(
            summaries, "kan_pwa", _status_from_rows(rows, "kan")
        ),
        "gatemate_status": gatemate_status,
        "ssqa_status": _status_from_summary(summaries, "ssqa", _status_from_rows(rows, "ssqa")),
    }


def _status_from_summary(
    summaries: Mapping[str, Any],
    key: str,
    fallback: str,
) -> str:
    return str(_as_mapping(summaries.get(key)).get("status") or fallback)


def _status_from_rows(rows: list[Mapping[str, Any]], token: str) -> str:
    statuses = [
        normal_status(str(row.get("status") or "missing"))
        for row in rows
        if token in _row_text(row)
    ]
    return "_and_".join(sorted(set(statuses))) if statuses else "missing"


def _row_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("row_id") or ""),
            str(row.get("evidence_class") or ""),
            str(row.get("claim_scope") or ""),
        ]
    ).lower()


def _claim_entry(row: Mapping[str, Any]) -> JsonDict:
    return {
        "row_id": str(row.get("row_id") or ""),
        "status": normal_status(str(row.get("status") or "missing")),
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "evidence_class": str(row.get("evidence_class") or ""),
        "blocker_class": str(row.get("blocker_class") or ""),
        "claim_scope": str(row.get("claim_scope") or ""),
        "summary": _as_mapping(row.get("summary")),
    }


def _next_milestone_recommendation(publication_blocker_count: int) -> str:
    return (
        f"{NEXT_MILESTONE}: keep repair headline wording retired or bounded until repair "
        "disqualifiers, adversarial flags, and the gated repair rerun clear; repair "
        "solver-grounded verification by producing positive local verifier gain and "
        "non-flagged SMT guidance over solver-only authority; carry FR-11 only as "
        "controller-side self-learning unless a source artifact explicitly trains and "
        "verifies model weights; keep KAN/PWA bounded to controller-anchor locality; "
        "unblock GateMate with host-visible output-contract and smoke transcript "
        "evidence before any SSQA, readback, or hardware-speedup claim; drive "
        f"publication_blocker_count from {publication_blocker_count} to 0."
    )


def _honest_verdict(artifact: Mapping[str, Any], matrix_present: bool) -> str:
    if not matrix_present:
        return "blocked_required_matrix_v20_missing"
    if artifact["capstone_ready"] is not True:
        return (
            "blocked_capstone_preconditions: "
            f"required_source_errors={len(_as_list(artifact.get('required_source_errors')))}; "
            f"counts_match_rows={artifact['matrix_v20_summary'].get('counts_match_rows')}; "
            "publication_blocker_count_matches="
            f"{artifact['matrix_v20_summary'].get('publication_blocker_count_matches')}"
        )
    return (
        "complete: "
        f"capstone_ready={str(artifact['capstone_ready']).lower()}; "
        f"paper_ready={str(artifact['paper_ready']).lower()}; "
        f"publication_blocker_count={len(_as_list(artifact.get('publication_blockers')))}; "
        f"repair_claim_status={artifact['repair_claim_status']}; "
        f"solver_grounding_status={artifact['solver_grounding_status']}; "
        f"fr11_self_learning_status={artifact['fr11_self_learning_status']}; "
        f"gatemate_status={artifact['gatemate_status']}; "
        f"ssqa_status={artifact['ssqa_status']}"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def normal_status(status: str) -> str:
    """Normalize legacy labels into the eight matrix v20 row classes."""

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
    "MATRIX_V20_REL_PATH",
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
