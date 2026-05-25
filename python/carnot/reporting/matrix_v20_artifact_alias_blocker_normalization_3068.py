"""Build the Exp 3068 matrix-v20 alias and blocker-normalization ledger.

Spec refs: REQ-REPORT-3068, SCENARIO-REPORT-3068.

This module is intentionally a bookkeeping pass. Matrix v21 needs to know that
one missing source path is a filename mismatch, but the research evidence under
that path is still a blocked gate result. The ledger therefore fixes only the
source pointer that a downstream matrix should use; it does not rerun repair or
promote any claim from blocked, bounded, flagged, or missing to clean.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.287"
SOURCE_MILESTONE = "2026.05.286"
SCHEMA = "carnot.matrix_v20_artifact_alias_blocker_normalization.v1"
ARTIFACT = "experiment_3068_matrix_v20_artifact_alias_blocker_normalization_v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3068_matrix_v20_artifact_alias_blocker_normalization_v1.json"
)
SCRIPT_REL_PATH = (
    REPO_ROOT / "scripts" / "experiment_3068_matrix_v20_artifact_alias_blocker_normalization.py"
)

MATRIX_V20_REL_PATH = Path("results/experiment_3065_cross_corpus_matrix_v20.json")
CAPSTONE_V286_REL_PATH = Path("results/experiment_3066_capstone_v286.json")
EXP3059_REQUESTED_REL_PATH = Path(
    "results/experiment_3059_gated_sota_repair_de_tautology_rerun_v1.json"
)
EXP3059_ACTUAL_REL_PATH = Path("results/experiment_3059_gated_sota_repair_de_tautology_rerun.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
OPS_STATUS_REL_PATH = Path("ops/status.md")
OPS_CHANGELOG_REL_PATH = Path("ops/changelog.md")

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
CLASS_FIELDS = (
    "clean_rows",
    "flagged_rows",
    "bounded_rows",
    "blocked_rows",
    "gated_skipped_rows",
    "projection_only_rows",
    "missing_rows",
    "retired_rows",
)
PUBLICATION_BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
}
RESEARCH_BLOCKING_STATUSES = {
    "flagged",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
}
EXP3054_TO_EXP3066_TASK_TITLES = (
    "Archive .285 and activate .286",
    "Repair headline retirement and blocker ledger",
    "Repair de-tautology protocol",
    "Local SOTA solution-verifier gain panel",
    "AquaForte-style LLM-guided SMT pilot",
    "Gated SOTA repair de-tautology rerun",
    "FR-11 solver self-model trace schema",
    "FR-11 delayed-regression solver self-model pilot",
    "KAN/PWA locality verification audit",
    "GateMate no-rerun operator-action ledger",
    "SSQA host-visible readback boundary ledger",
    "Cross-corpus matrix v20",
    "Capstone .286",
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while treating missing or malformed files as absent evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_text(path: Path) -> str:
    """Read UTF-8 text while treating missing files as empty source evidence."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def sha256_file(path: Path) -> str | None:
    """Return a checksum for source-ledger traceability without mutating the source file."""

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
    """REQ-REPORT-3068: produce a non-destructive ledger for matrix-v21 input."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V20_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V286_REL_PATH)
    actual_exp3059 = read_json_object(root_path / EXP3059_ACTUAL_REL_PATH)
    rows = _all_matrix_rows(matrix)
    publication_blockers = _publication_blockers(matrix, rows)
    artifact_aliases = _artifact_aliases(root_path, actual_exp3059)
    blocker_categories = _blocker_categories(rows, artifact_aliases)
    publication_blocker_count_before = _publication_blocker_count(matrix, publication_blockers)
    normalized_blocker_count_estimate = max(
        0,
        publication_blocker_count_before
        - len(blocker_categories["artifact_hygiene_blockers"]),
    )
    missing_artifacts_after_aliasing = _missing_artifacts_after_aliasing(
        root_path, matrix, rows, artifact_aliases
    )
    source_paths = _source_paths(matrix, rows, missing_artifacts_after_aliasing)
    source_artifacts = [_source_artifact(root_path, path) for path in source_paths]
    ready = _normalization_ready(matrix, capstone, actual_exp3059)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "source_milestone": SOURCE_MILESTONE,
        "matrix_v20_normalization_ready": ready,
        "artifact_aliases": artifact_aliases,
        "source_artifact_path_mismatches": artifact_aliases,
        "missing_artifacts_after_aliasing": missing_artifacts_after_aliasing,
        "blocker_categories": blocker_categories,
        "row_inventory": _row_inventory(rows),
        "exp3059_alias_status": _exp3059_alias_status(root_path, artifact_aliases),
        "publication_blocker_count_before": publication_blocker_count_before,
        "normalized_blocker_count_estimate": normalized_blocker_count_estimate,
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "actual_result_filenames": _actual_result_filenames(root_path, source_paths),
        "conductor_log_entries_3054_3066": _conductor_log_entries(root_path),
        "inference_substrate": _inference_substrate(),
        "no_live_model_inference": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_live_repair_rerun": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_historical_artifact_rewrite": True,
        "no_research_claim_cleaned_by_alias": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
        "duration_s": _duration(start, now_s),
        "honest_verdict": _honest_verdict(
            ready,
            publication_blocker_count_before,
            normalized_blocker_count_estimate,
            len(artifact_aliases),
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
    """Build and persist the Exp 3068 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _normalization_ready(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    actual_exp3059: Mapping[str, Any],
) -> bool:
    return (
        matrix.get("matrix_v20_ready") is True
        and capstone.get("capstone_ready") is True
        and bool(actual_exp3059)
    )


def _all_matrix_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    return [
        _ledger_row(row)
        for field in CLASS_FIELDS
        for row in _as_list(matrix.get(field))
        if isinstance(row, Mapping)
    ]


def _ledger_row(row: Mapping[str, Any]) -> JsonDict:
    status = normal_status(str(row.get("status") or "missing"))
    return {
        "row_id": str(row.get("row_id") or ""),
        "status": status,
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "evidence_class": str(row.get("evidence_class") or ""),
        "blocker_class": str(row.get("blocker_class") or blocker_class(status)),
        "claim_scope": str(row.get("claim_scope") or ""),
    }


def _publication_blockers(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> list[JsonDict]:
    blockers = _as_list(matrix.get("publication_blockers")) or [
        row
        for row in rows
        if normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES
    ]
    return [_ledger_row(row) for row in blockers if isinstance(row, Mapping)]


def _publication_blocker_count(
    matrix: Mapping[str, Any],
    publication_blockers: list[Mapping[str, Any]],
) -> int:
    count = _int_or_none(matrix.get("publication_blocker_count"))
    return count if count is not None else len(publication_blockers)


def _artifact_aliases(root: Path, actual_exp3059: Mapping[str, Any]) -> list[JsonDict]:
    requested_present = (root / EXP3059_REQUESTED_REL_PATH).is_file()
    actual_present = (root / EXP3059_ACTUAL_REL_PATH).is_file()
    if requested_present or not actual_present:
        return []
    return [
        {
            "alias_id": "exp3059_requested_v1_to_actual_gate_blocked",
            "experiment_id": "exp3059",
            "requested_path": EXP3059_REQUESTED_REL_PATH.as_posix(),
            "actual_path": EXP3059_ACTUAL_REL_PATH.as_posix(),
            "requested_present": False,
            "actual_present": True,
            "actual_status": str(actual_exp3059.get("status") or ""),
            "actual_honest_verdict": str(actual_exp3059.get("honest_verdict") or ""),
            "non_destructive": True,
            "claim_effect": "artifact_hygiene_only_research_status_stays_gated_skipped",
        }
    ]


def _exp3059_alias_status(root: Path, artifact_aliases: list[Mapping[str, Any]]) -> str:
    if artifact_aliases:
        return "actual_gate_blocked_artifact_present_alias_v21_to_actual_without_rewrite"
    if not (root / EXP3059_ACTUAL_REL_PATH).is_file():
        return "blocked_actual_gate_artifact_missing"
    return "requested_alias_file_present_no_alias_needed"


def _blocker_categories(
    rows: list[Mapping[str, Any]],
    artifact_aliases: list[Mapping[str, Any]],
) -> JsonDict:
    artifact_hygiene_ids = {
        "source:exp3059_requested_v1_alias"
        for alias in artifact_aliases
        if alias.get("actual_present") is True
    }
    artifact_hygiene = [
        _ledger_row(row) for row in rows if str(row.get("row_id")) in artifact_hygiene_ids
    ]
    research_blockers = [
        _ledger_row(row)
        for row in rows
        if normal_status(str(row.get("status") or "missing")) in RESEARCH_BLOCKING_STATUSES
        and str(row.get("row_id")) not in artifact_hygiene_ids
    ]
    bounded_rows = [
        _ledger_row(row)
        for row in rows
        if normal_status(str(row.get("status") or "missing")) == "bounded"
    ]
    retired_rows = [
        _ledger_row(row)
        for row in rows
        if normal_status(str(row.get("status") or "missing")) == "retired"
    ]
    true_missing = [
        _ledger_row(row)
        for row in research_blockers
        if normal_status(str(row.get("status") or "missing")) == "missing"
    ]
    blocked_rows = [
        _ledger_row(row)
        for row in rows
        if normal_status(str(row.get("status") or "missing")) == "blocked"
    ]
    projection_only_rows = [
        _ledger_row(row)
        for row in rows
        if normal_status(str(row.get("status") or "missing")) == "projection_only"
    ]
    return {
        "research_blockers": research_blockers,
        "artifact_hygiene_blockers": artifact_hygiene,
        "true_missing_evidence": true_missing,
        "honest_bounded_rows": bounded_rows,
        "retired_rows": retired_rows,
        "duplicate_rows": _duplicate_rows(rows),
        "blocked_rows": blocked_rows,
        "projection_only_rows": projection_only_rows,
        "missing_source_rows": [
            _ledger_row(row)
            for row in rows
            if normal_status(str(row.get("status") or "missing")) == "missing"
        ],
    }


def _duplicate_rows(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(
            [
                str(row.get("source_artifact") or ""),
                str(row.get("source_field") or ""),
                str(row.get("claim_scope") or ""),
            ]
        )
        groups[key].append(row)
    return [
        {
            "duplicate_key": key,
            "row_ids": [str(row.get("row_id") or "") for row in grouped],
            "statuses": [normal_status(str(row.get("status") or "missing")) for row in grouped],
        }
        for key, grouped in groups.items()
        if len(grouped) > 1
    ]


def _missing_artifacts_after_aliasing(
    root: Path,
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    artifact_aliases: list[Mapping[str, Any]],
) -> list[JsonDict]:
    alias_requested_paths = {
        str(alias.get("requested_path") or "")
        for alias in artifact_aliases
        if alias.get("actual_present") is True
    }
    missing_by_path: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if normal_status(str(row.get("status") or "missing")) != "missing":
            continue
        source_path = str(row.get("source_artifact") or "")
        if not source_path or source_path in alias_requested_paths:
            continue
        if not (root / source_path).is_file():
            missing_by_path[source_path].add(str(row.get("row_id") or ""))
    for source in _as_list(matrix.get("source_artifacts")):
        if not isinstance(source, Mapping):
            continue
        source_path = str(source.get("path") or "")
        if not source_path or source_path in alias_requested_paths:
            continue
        if not (root / source_path).is_file():
            missing_by_path[source_path].add(str(source.get("experiment_id") or "source_artifact"))
    if not (root / EXP3059_ACTUAL_REL_PATH).is_file():
        missing_by_path[EXP3059_ACTUAL_REL_PATH.as_posix()].add("exp3059_actual_gate_result")
    return [
        {
            "path": path,
            "row_ids": sorted(row_ids),
            "reason": "artifact_file_missing_after_aliasing",
        }
        for path, row_ids in sorted(missing_by_path.items())
    ]


def _source_paths(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    missing_artifacts_after_aliasing: list[Mapping[str, Any]],
) -> list[Path]:
    raw_paths = {
        MATRIX_V20_REL_PATH.as_posix(),
        CAPSTONE_V286_REL_PATH.as_posix(),
        EXP3059_ACTUAL_REL_PATH.as_posix(),
        CONDUCTOR_LOG_REL_PATH.as_posix(),
        OPS_STATUS_REL_PATH.as_posix(),
        OPS_CHANGELOG_REL_PATH.as_posix(),
    }
    raw_paths.update(str(row.get("source_artifact") or "") for row in rows)
    raw_paths.update(
        str(row.get("path") or "")
        for row in _as_list(matrix.get("source_artifacts"))
        if isinstance(row, Mapping)
    )
    raw_paths.update(str(row.get("path") or "") for row in missing_artifacts_after_aliasing)
    return [Path(path) for path in sorted(path for path in raw_paths if path)]


def _source_artifact(root: Path, rel_path: Path) -> JsonDict:
    path = root / rel_path
    return {
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "readable_json_object": bool(read_json_object(path)) if rel_path.suffix == ".json" else False,
        "sha256": sha256_file(path),
    }


def _actual_result_filenames(root: Path, source_paths: list[Path]) -> list[str]:
    return sorted(
        {
            path.name
            for path in source_paths
            if path.parent == Path("results")
            and path.name.startswith("experiment_")
            and (root / path).is_file()
        }
    )


def _conductor_log_entries(root: Path) -> list[JsonDict]:
    entries: list[JsonDict] = []
    for line in read_text(root / CONDUCTOR_LOG_REL_PATH).splitlines():
        if not any(title in line for title in EXP3054_TO_EXP3066_TASK_TITLES):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) >= 4:
            entries.append(
                {
                    "timestamp": parts[0],
                    "task": parts[1],
                    "status": parts[2],
                    "details": parts[3],
                    "raw": line,
                }
            )
    return entries


def _row_inventory(rows: list[Mapping[str, Any]]) -> JsonDict:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[normal_status(str(row.get("status") or "missing"))] += 1
    return {"rows_total": len(rows), "status_counts": counts}


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts_and_filenames",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


def _honest_verdict(
    ready: bool,
    publication_blocker_count_before: int,
    normalized_blocker_count_estimate: int,
    alias_count: int,
) -> str:
    if not ready:
        return (
            "blocked_matrix_v20_normalization_preconditions: "
            f"publication_blocker_count_before={publication_blocker_count_before}; "
            f"alias_count={alias_count}"
        )
    return (
        "complete: "
        "matrix_v20_normalization_ready=true; "
        f"publication_blocker_count_before={publication_blocker_count_before}; "
        f"normalized_blocker_count_estimate={normalized_blocker_count_estimate}; "
        f"artifact_alias_count={alias_count}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def normal_status(status: str) -> str:
    """Normalize legacy row labels into the matrix row-status vocabulary."""

    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str) -> str:
    """Map one normalized row status to the publication-boundary reason class."""

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
    "EXP3059_ACTUAL_REL_PATH",
    "EXP3059_REQUESTED_REL_PATH",
    "MATRIX_V20_REL_PATH",
    "OPS_CHANGELOG_REL_PATH",
    "OPS_STATUS_REL_PATH",
    "OUTPUT_REL_PATH",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "blocker_class",
    "build_artifact",
    "normal_status",
    "read_json_object",
    "read_text",
    "sha256_file",
    "write_artifact",
]
