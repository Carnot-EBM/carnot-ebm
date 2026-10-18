"""Build the Exp 3082 publication blocker reduction ledger.

Spec refs: REQ-REPORT-3082, SCENARIO-REPORT-3082.

The ledger is a pure accounting step between matrix v21 and matrix v22. It
does not create new evidence. Instead, it records which existing blockers can
be reduced by the .288 research tasks and which rows must stay outside
conductor claims until an operator supplies external hardware evidence.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.288"
SOURCE_MILESTONE = "2026.05.287"
SCHEMA = "carnot.publication_blocker_reduction_ledger.v1"
ARTIFACT = "experiment_3082_publication_blocker_reduction_ledger_v1"
OUTPUT_REL_PATH = Path("results/experiment_3082_publication_blocker_reduction_ledger_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3082_publication_blocker_reduction_ledger.py"

MATRIX_V21_REL_PATH = Path("results/experiment_3079_cross_corpus_matrix_v21.json")
CAPSTONE_V287_REL_PATH = Path("results/experiment_3080_capstone_v287.json")

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
PUBLICATION_BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
}
CATEGORY_NAMES = (
    "verifier_gain",
    "repair_gate",
    "fr11_budget",
    "hardware_evidence",
    "adapter_projection",
    "missing_artifact",
    "bounded_status",
    "retired_status",
    "documentation_hygiene",
)
REDUCIBLE_CATEGORIES = {
    "verifier_gain",
    "repair_gate",
    "fr11_budget",
    "adapter_projection",
    "missing_artifact",
    "bounded_status",
    "documentation_hygiene",
}
HARDWARE_TOKENS = (
    "gatemate",
    "ssqa",
    "host_visible",
    "readback",
    "hardware",
    "board",
    "flash",
    "pinout",
    "smoke",
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON authority artifact and fail closed on absence or malformed data."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for an existing source file."""

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
    """REQ-REPORT-3082: categorize matrix v21 blockers for matrix v22."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V21_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V287_REL_PATH)
    rows = _matrix_rows(matrix)
    row_by_id = {str(row.get("row_id") or ""): row for row in rows}
    publication_blockers = _publication_blockers(matrix, rows, row_by_id)
    categories = _blocker_categories(publication_blockers, rows)
    coverage = _blocker_coverage(publication_blockers, categories)
    source_artifacts = [
        _source_artifact(root_path, "matrix_v21_authority", MATRIX_V21_REL_PATH),
        _source_artifact(root_path, "capstone_v287_authority", CAPSTONE_V287_REL_PATH),
    ]
    before_count = _int_or_none(matrix.get("publication_blocker_count")) or len(
        publication_blockers
    )
    capstone_count = _int_or_none(capstone.get("publication_blocker_count"))
    blocked_reasons = _blocked_reasons(
        matrix=matrix,
        capstone=capstone,
        before_count=before_count,
        capstone_count=capstone_count,
        coverage=coverage,
    )
    ready = not blocked_reasons

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "source_milestone": SOURCE_MILESTONE,
        "blocker_ledger_ready": ready,
        "publication_blocker_count_before": before_count,
        "capstone_publication_blocker_count": capstone_count,
        "blocker_categories": categories,
        "blocker_category_counts": {
            category: len(entries) for category, entries in categories.items()
        },
        "reducible_in_v288": _reducible_in_v288(categories),
        "operator_evidence_required": _operator_evidence_required(categories),
        "retire_or_promote_criteria": _retire_or_promote_criteria(),
        "blocker_coverage": coverage,
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "required_source_errors": _required_source_errors(source_artifacts),
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
        "blocked_reasons": blocked_reasons,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact, matrix_present=bool(matrix))
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3082 deliverable JSON."""

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


def _publication_blockers(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    row_by_id: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    blockers = _as_list(matrix.get("publication_blockers"))
    if not blockers:
        blockers = [
            row
            for row in rows
            if normal_status(str(row.get("status") or "")) in PUBLICATION_BLOCKING_STATUSES
        ]
    result: list[JsonDict] = []
    for blocker in blockers:
        if isinstance(blocker, Mapping):
            row_id = str(blocker.get("row_id") or "")
            result.append(_claim_entry(row_by_id.get(row_id) or blocker))
    return result


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


def _blocker_categories(
    publication_blockers: list[Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> dict[str, list[JsonDict]]:
    categories: dict[str, list[JsonDict]] = {category: [] for category in CATEGORY_NAMES}
    for row in publication_blockers:
        category = _category_for_row(row)
        if category == "retired_status":
            category = "documentation_hygiene"
        categories[category].append(_ledger_entry(row, category))
    for row in rows:
        if normal_status(str(row.get("status") or "")) == "retired":
            category = _category_for_row(row)
            categories[category].append(_ledger_entry(row, category))
    return categories


def _ledger_entry(row: Mapping[str, Any], category: str) -> JsonDict:
    status = normal_status(str(row.get("status") or "missing"))
    requires_operator = category == "hardware_evidence"
    return {
        "row_id": str(row.get("row_id") or ""),
        "status": status,
        "category": category,
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "evidence_class": str(row.get("evidence_class") or ""),
        "blocker_class": str(row.get("blocker_class") or blocker_class(status)),
        "claim_scope": str(row.get("claim_scope") or ""),
        "requires_external_operator_evidence": requires_operator,
        "reducible_in_v288": category in REDUCIBLE_CATEGORIES and status != "retired",
        "v288_action": _v288_action(category),
    }


def _category_for_row(row: Mapping[str, Any]) -> str:
    status = normal_status(str(row.get("status") or "missing"))
    text = _row_text(row)
    if status == "retired":
        return "retired_status"
    if "source_artifact_accounting" in text or "artifact_alias" in text:
        return "documentation_hygiene"
    if status == "bounded" and str(row.get("claim_scope") or "") == "paper_readiness":
        return "bounded_status"
    if any(token in text for token in HARDWARE_TOKENS):
        return "hardware_evidence"
    if status == "projection_only" or "future_adapter" in text or "ebt" in text or "arm" in text:
        return "adapter_projection"
    if "fr11" in text or "controller_only" in text or "kan" in text or "self_learning" in text:
        return "fr11_budget"
    if "repair" in text or "dccd" in text or "grammar" in text or "llguidance" in text:
        return "repair_gate"
    if "verifier" in text or "solver" in text or "abstention" in text or "verge" in text:
        return "verifier_gain"
    if status == "missing":
        return "missing_artifact"
    if status == "bounded":
        return "bounded_status"
    return "documentation_hygiene"


def _row_text(row: Mapping[str, Any]) -> str:
    summary = row.get("summary")
    try:
        summary_text = json.dumps(summary, sort_keys=True)
    except TypeError:
        summary_text = str(summary)
    return " ".join(
        [
            str(row.get("row_id") or ""),
            str(row.get("status") or ""),
            str(row.get("source_artifact") or ""),
            str(row.get("source_field") or ""),
            str(row.get("evidence_class") or ""),
            str(row.get("claim_scope") or ""),
            summary_text,
        ]
    ).lower()


def _reducible_in_v288(categories: Mapping[str, list[Mapping[str, Any]]]) -> list[JsonDict]:
    return [
        dict(row)
        for category in CATEGORY_NAMES
        if category in REDUCIBLE_CATEGORIES
        for row in categories.get(category, [])
        if row.get("reducible_in_v288") is True
    ]


def _operator_evidence_required(
    categories: Mapping[str, list[Mapping[str, Any]]],
) -> list[JsonDict]:
    return [
        dict(row)
        for row in categories.get("hardware_evidence", [])
        if row.get("status") not in {"clean", "retired"}
    ]


def _blocker_coverage(
    publication_blockers: list[Mapping[str, Any]],
    categories: Mapping[str, list[Mapping[str, Any]]],
) -> JsonDict:
    blocker_ids = [str(row.get("row_id") or "") for row in publication_blockers]
    covered_ids = [
        str(row.get("row_id") or "")
        for category, rows in categories.items()
        if category != "retired_status"
        for row in rows
        if str(row.get("row_id") or "") in blocker_ids
    ]
    covered_counts = Counter(covered_ids)
    blocker_set = set(blocker_ids)
    covered_set = set(covered_ids)
    return {
        "publication_blocker_ids": blocker_ids,
        "covered_publication_blocker_ids": covered_ids,
        "covered_publication_blocker_count": len(covered_ids),
        "uncategorized_publication_blocker_ids": sorted(blocker_set - covered_set),
        "duplicate_publication_blocker_ids": sorted(
            row_id for row_id, count in covered_counts.items() if count > 1
        ),
        "retired_row_count": len(categories.get("retired_status", [])),
    }


def _blocked_reasons(
    *,
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    before_count: int,
    capstone_count: int | None,
    coverage: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not matrix:
        reasons.append("matrix v21 authority missing or malformed")
    if not capstone:
        reasons.append("capstone .287 authority missing or malformed")
    if matrix and matrix.get("matrix_v21_ready") is not True:
        reasons.append("matrix v21 is not ready")
    if capstone and capstone.get("capstone_ready") is not True:
        reasons.append("capstone .287 is not ready")
    if capstone and capstone_count != before_count:
        reasons.append("matrix v21 and capstone .287 blocker counts disagree")
    if coverage.get("uncategorized_publication_blocker_ids"):
        reasons.append("one or more matrix v21 blockers were not categorized")
    if coverage.get("duplicate_publication_blocker_ids"):
        reasons.append("one or more matrix v21 blockers were categorized more than once")
    return reasons


def _source_artifact(root: Path, role: str, rel_path: Path) -> JsonDict:
    path = root / rel_path
    payload = read_json_object(path)
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "required": True,
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "sha256": sha256_file(path),
        "source_type": "json",
    }


def _required_source_errors(source_artifacts: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "path": str(row.get("path") or ""),
            "reason": "missing_or_malformed_required_artifact",
        }
        for row in source_artifacts
        if row.get("required") is True and row.get("readable_json_object") is not True
    ]


def _retire_or_promote_criteria() -> dict[str, str]:
    return {
        "verifier_gain": (
            "Retire if verifier-gain evidence remains flagged after .288; promote only "
            "when abstention/calibration gates pass and matrix v22 records clean gain."
        ),
        "repair_gate": (
            "Retire repair headline wording if the verifier-gain precondition fails; "
            "promote only after the gated repair micro-panel writes clean evidence."
        ),
        "fr11_budget": (
            "Keep controller-only until soundness mistakes are zero and completeness "
            "budget is zero; promote only with non-tautological held-out evidence."
        ),
        "hardware_evidence": (
            "Keep blocked until external operator evidence provides host-visible "
            "GateMate/SSQA smoke or readback artifacts; conductor claims alone do not promote."
        ),
        "adapter_projection": (
            "Keep projection-only until a checked-in adapter implementation and tests "
            "exist; otherwise retire from publication blocker accounting as future context."
        ),
        "missing_artifact": (
            "Reduce by writing the missing terminal artifact or by retiring the row with "
            "a matrix-v22 citation to the absent source."
        ),
        "bounded_status": (
            "Bounded rows exit only by a mechanical clean/retired status change in matrix "
            "v22; capstone prose cannot promote them."
        ),
        "retired_status": (
            "Keep out of publication_blocker_count while preserving row provenance and "
            "the retired claim wording."
        ),
        "documentation_hygiene": (
            "Resolve by alias, source-artifact, or documentation reconciliation only; "
            "must not clean an underlying research blocker."
        ),
    }


def _v288_action(category: str) -> str:
    return {
        "verifier_gain": "run .288 verifier autopsy, fixture, abstention, and calibration tasks",
        "repair_gate": "rerun repair only after verifier-gain gates pass",
        "fr11_budget": "repair soundness/completeness budget with exact non-tautological gates",
        "hardware_evidence": "wait for operator-supplied host-visible hardware evidence",
        "adapter_projection": "add checked-in adapter prototype and tests or keep projection-only",
        "missing_artifact": "write or explicitly retire the missing terminal artifact",
        "bounded_status": "convert to clean or retired with matrix-v22 mechanical evidence",
        "retired_status": "preserve retired provenance outside blocker counts",
        "documentation_hygiene": "reconcile artifact aliases without changing research status",
    }[category]


def _honest_verdict(artifact: Mapping[str, Any], *, matrix_present: bool) -> str:
    if not matrix_present:
        return "blocked_required_matrix_v21_missing"
    if artifact.get("blocker_ledger_ready") is not True:
        return (
            "blocked_ledger_preconditions: "
            f"blocked_reasons={_as_list(artifact.get('blocked_reasons'))}"
        )
    return (
        "complete: "
        "blocker_ledger_ready=true; "
        f"publication_blocker_count_before={artifact['publication_blocker_count_before']}; "
        f"reducible_in_v288={len(_as_list(artifact.get('reducible_in_v288')))}; "
        "operator_evidence_required="
        f"{len(_as_list(artifact.get('operator_evidence_required')))}"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "matrix_v21_and_capstone_v287",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def normal_status(status: str) -> str:
    """Normalize legacy matrix labels into the eight status classes."""

    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str) -> str:
    """Map a row status to the blocker class used by matrix artifacts."""

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
    "CAPSTONE_V287_REL_PATH",
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
