"""Build the Exp 3096 publication blocker triage and retirement ledger.

Spec refs: REQ-REPORT-3096, SCENARIO-REPORT-3096.

This ledger is a matrix-v23 planning artifact. It does not create evidence or
try to make the paper look readier than matrix v22 says it is. Instead, it
turns the 36 matrix-v22 blockers into a mechanical queue: rows the .289
milestone can reduce in-repo, rows that need external operator hardware
evidence, rows that are projection-only, rows that are bounded until a later
matrix changes them, and rows that are already retired.
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
RUN_DATE = "20260526"
MILESTONE = "2026.05.289"
SOURCE_MILESTONE = "2026.05.288"
SCHEMA = "carnot.publication_blocker_triage_retirement_ledger.v2"
ARTIFACT = "experiment_3096_publication_blocker_triage_and_retirement_ledger_v2"
OUTPUT_REL_PATH = Path(
    "results/experiment_3096_publication_blocker_triage_and_retirement_ledger_v2.json"
)
SCRIPT_REL_PATH = REPO_ROOT / (
    "scripts/experiment_3096_publication_blocker_triage_and_retirement_ledger_v2.py"
)

MATRIX_V22_REL_PATH = Path("results/experiment_3093_cross_corpus_matrix_v22.json")
CAPSTONE_V288_REL_PATH = Path("results/experiment_3094_capstone_v288.json")
PRIOR_LEDGER_REL_PATH = Path("results/experiment_3082_publication_blocker_reduction_ledger_v1.json")

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
    "verifier_repair",
    "formal_feedback",
    "repair_missing_artifact",
    "fr11_boundary",
    "ebt_arm_projection",
    "hardware_evidence",
    "publication_readiness",
    "model_spec_gap",
    "missing_artifact",
    "bounded_status",
    "retired_status",
)
REDUCIBLE_CATEGORIES = {
    "verifier_repair",
    "formal_feedback",
    "repair_missing_artifact",
    "fr11_boundary",
    "ebt_arm_projection",
    "publication_readiness",
    "model_spec_gap",
    "missing_artifact",
    "bounded_status",
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
    """Return a source checksum so matrix-v23 consumers can audit citations."""

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
    """REQ-REPORT-3096: triage matrix-v22 blockers before matrix v23."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V22_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V288_REL_PATH)
    rows = _matrix_rows(matrix)
    row_by_id = {str(row.get("row_id") or ""): row for row in rows}
    publication_blockers = _publication_blockers(matrix, rows, row_by_id)
    model_gaps = _model_spec_gap_by_id(matrix, capstone)
    categories = _blocker_categories(publication_blockers, rows, model_gaps)
    coverage = _blocker_coverage(publication_blockers, categories)
    criteria = _retire_or_promote_criteria()
    source_artifacts = [
        _source_artifact(root_path, "matrix_v22_authority", MATRIX_V22_REL_PATH, True),
        _source_artifact(root_path, "capstone_v288_authority", CAPSTONE_V288_REL_PATH, True),
        _source_artifact(
            root_path,
            "prior_reduction_ledger_context",
            PRIOR_LEDGER_REL_PATH,
            False,
        ),
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
        criteria=criteria,
    )
    ready = not blocked_reasons

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "source_milestone": SOURCE_MILESTONE,
        "blocker_triage_ready": ready,
        "publication_blocker_count_before": before_count,
        "capstone_publication_blocker_count": capstone_count,
        "blocker_categories": categories,
        "blocker_category_counts": {
            category: len(entries) for category, entries in categories.items()
        },
        "reducible_in_v289": _reducible_in_v289(categories),
        "operator_evidence_required": _operator_evidence_required(categories),
        "retire_or_promote_criteria": criteria,
        "blocker_coverage": coverage,
        "matrix_v23_consumption": _matrix_v23_consumption(before_count, categories),
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
    """Build and persist the Exp 3096 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def _model_spec_gap_by_id(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> dict[str, JsonDict]:
    gaps: dict[str, JsonDict] = {}
    for source_name, payload in (("matrix_v22", matrix), ("capstone_v288", capstone)):
        for gap in _as_list(payload.get("headline_model_spec_gaps")):
            if isinstance(gap, Mapping):
                row_id = str(gap.get("row_id") or "")
                if row_id:
                    entry = _as_mapping(gap)
                    entry["gap_source"] = source_name
                    gaps[row_id] = entry
    return gaps


def _blocker_categories(
    publication_blockers: list[Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
    model_gaps: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[JsonDict]]:
    categories: dict[str, list[JsonDict]] = {category: [] for category in CATEGORY_NAMES}
    categorized_ids: set[str] = set()
    model_gap_ids = set(model_gaps)
    for row in publication_blockers:
        category = _category_for_row(row, model_gap_ids)
        categories[category].append(_ledger_entry(row, category, model_gaps.get(str(row.get("row_id")))))
        categorized_ids.add(str(row.get("row_id") or ""))
    for row in rows:
        if normal_status(str(row.get("status") or "")) == "retired":
            categories["retired_status"].append(_ledger_entry(row, "retired_status", None))
    for row_id, gap in model_gaps.items():
        if row_id not in categorized_ids:
            categories["model_spec_gap"].append(_gap_only_entry(row_id, gap))
    return categories


def _ledger_entry(
    row: Mapping[str, Any],
    category: str,
    model_gap: Mapping[str, Any] | None,
) -> JsonDict:
    status = normal_status(str(row.get("status") or "missing"))
    entry: JsonDict = {
        "row_id": str(row.get("row_id") or ""),
        "status": status,
        "category": category,
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "evidence_class": str(row.get("evidence_class") or ""),
        "blocker_class": str(row.get("blocker_class") or blocker_class(status)),
        "claim_scope": str(row.get("claim_scope") or ""),
        "requires_external_operator_evidence": category == "hardware_evidence",
        "reducible_in_v289": category in REDUCIBLE_CATEGORIES and status != "retired",
        "v289_action": _v289_action(category),
    }
    if model_gap:
        entry["model_spec_gap"] = _as_mapping(model_gap)
    return entry


def _gap_only_entry(row_id: str, gap: Mapping[str, Any]) -> JsonDict:
    return {
        "row_id": row_id,
        "status": "missing",
        "category": "model_spec_gap",
        "source_artifact": str(gap.get("source_artifact") or ""),
        "source_field": "headline_model_spec_gaps",
        "evidence_class": "headline_model_spec_gap",
        "blocker_class": "model_spec_gap",
        "claim_scope": "headline_model_spec_gap",
        "requires_external_operator_evidence": False,
        "reducible_in_v289": True,
        "v289_action": _v289_action("model_spec_gap"),
        "model_spec_gap": _as_mapping(gap),
    }


def _category_for_row(row: Mapping[str, Any], model_gap_ids: set[str]) -> str:
    status = normal_status(str(row.get("status") or "missing"))
    row_id = str(row.get("row_id") or "")
    text = _row_text(row)
    primary_text = _row_primary_text(row)
    if status == "retired":
        return "retired_status"
    if row_id in model_gap_ids:
        return "model_spec_gap"
    if status == "projection_only" or any(
        token in primary_text for token in ("ebt", "arm", "adapter", "sidecar", "future_adapter")
    ):
        return "ebt_arm_projection"
    if "fr11" in primary_text or "controller_only" in primary_text or "self_learning" in primary_text:
        return "fr11_boundary"
    if status == "missing" and ("repair" in text or "xgrammar" in text):
        return "repair_missing_artifact"
    if "formal_feedback" in text or "solver_grounded_repair_feedback" in text:
        return "formal_feedback"
    if "llm_guided_smt" in text or "aquaforte" in text or "dafny" in text or "mcs" in text:
        return "formal_feedback"
    if "paper_readiness" in text:
        return "publication_readiness"
    if status == "bounded":
        return "bounded_status"
    if any(token in primary_text for token in HARDWARE_TOKENS):
        return "hardware_evidence"
    if status == "missing":
        return "missing_artifact"
    return "verifier_repair"


def _row_primary_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("row_id") or ""),
            str(row.get("status") or ""),
            str(row.get("source_artifact") or ""),
            str(row.get("source_field") or ""),
            str(row.get("evidence_class") or ""),
            str(row.get("claim_scope") or ""),
        ]
    ).lower()


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


def _reducible_in_v289(categories: Mapping[str, list[Mapping[str, Any]]]) -> list[JsonDict]:
    return [
        dict(row)
        for category in CATEGORY_NAMES
        if category in REDUCIBLE_CATEGORIES
        for row in categories.get(category, [])
        if row.get("reducible_in_v289") is True
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
    blocker_set = set(blocker_ids)
    covered_ids = [
        str(row.get("row_id") or "")
        for category, rows in categories.items()
        if category != "retired_status"
        for row in rows
        if str(row.get("row_id") or "") in blocker_set
    ]
    covered_counts = Counter(covered_ids)
    return {
        "publication_blocker_ids": blocker_ids,
        "covered_publication_blocker_ids": covered_ids,
        "covered_publication_blocker_count": len(covered_ids),
        "uncategorized_publication_blocker_ids": sorted(blocker_set - set(covered_ids)),
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
    criteria: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if not matrix:
        reasons.append("matrix v22 authority missing or malformed")
    if not capstone:
        reasons.append("capstone .288 authority missing or malformed")
    if matrix and matrix.get("matrix_v22_ready") is not True:
        reasons.append("matrix v22 is not ready")
    if capstone and capstone.get("capstone_ready") is not True:
        reasons.append("capstone .288 is not ready")
    if capstone and capstone_count != before_count:
        reasons.append("matrix v22 and capstone .288 blocker counts disagree")
    if coverage.get("uncategorized_publication_blocker_ids"):
        reasons.append("one or more matrix v22 blockers were not categorized")
    if coverage.get("duplicate_publication_blocker_ids"):
        reasons.append("one or more matrix v22 blockers were categorized more than once")
    if set(CATEGORY_NAMES) - set(criteria):
        reasons.append("one or more categories lack retire-or-promote criteria")
    return reasons


def _source_artifact(root: Path, role: str, rel_path: Path, required: bool) -> JsonDict:
    path = root / rel_path
    payload = read_json_object(path) if rel_path.suffix == ".json" else {}
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "required": required,
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


def _matrix_v23_consumption(
    before_count: int,
    categories: Mapping[str, list[Mapping[str, Any]]],
) -> JsonDict:
    return {
        "publication_blocker_count_authority": "matrix_v22.publication_blocker_count",
        "publication_blocker_count_before": before_count,
        "category_order": list(CATEGORY_NAMES),
        "direct_reduction_categories": sorted(REDUCIBLE_CATEGORIES),
        "external_operator_category": "hardware_evidence",
        "retired_category": "retired_status",
        "category_counts": {category: len(categories.get(category, [])) for category in CATEGORY_NAMES},
    }


def _retire_or_promote_criteria() -> dict[str, str]:
    return {
        "verifier_repair": (
            "Retire stale verifier/repair claims if exact acceptance/rejection evidence "
            "remains flagged; promote only when matrix v23 cites clean verifier or repair "
            "evidence with false-accept and false-reject accounting."
        ),
        "formal_feedback": (
            "Retire formal-feedback claims when guided runs do not beat solver-only "
            "baselines; promote only with localized counterexample/MCS evidence and "
            "positive guided-minus-solver-only lift."
        ),
        "repair_missing_artifact": (
            "Keep missing until the repair micro-panel artifact exists and validates "
            "exact semantics; retire if the upstream gate remains skipped."
        ),
        "fr11_boundary": (
            "Keep FR-11 controller-only unless matrix v23 proves zero soundness mistakes, "
            "zero completeness mistakes, and non-vacuous controls for any broader claim."
        ),
        "ebt_arm_projection": (
            "Keep projection-only until a checked-in adapter implementation, tests, and "
            "live model-integration evidence exist; otherwise retire as future context."
        ),
        "hardware_evidence": (
            "Keep blocked until external operator evidence provides host-visible GateMate "
            "or SSQA smoke/readback artifacts; conductor-side evidence alone cannot promote."
        ),
        "publication_readiness": (
            "Promote only when matrix v23 publication_blocker_count reaches zero and all "
            "headline model-spec and missing-input checks are clear."
        ),
        "model_spec_gap": (
            "Promote only when the source artifact names mandatory_headline_model_ids and "
            "model_specs for every live headline model; otherwise keep as a model_specs gap."
        ),
        "missing_artifact": (
            "Reduce by writing the missing terminal artifact or by retiring the row with "
            "an explicit matrix-v23 citation to the absent source."
        ),
        "bounded_status": (
            "Bounded rows exit only by a mechanical clean or retired matrix-v23 status; "
            "capstone prose cannot promote them."
        ),
        "retired_status": (
            "Keep retired rows out of publication_blocker_count while preserving source "
            "artifact, field, and retired claim wording."
        ),
    }


def _v289_action(category: str) -> str:
    return {
        "verifier_repair": "run exact verifier/repair reducer tasks before headline experiments",
        "formal_feedback": "prove formal-feedback lift over solver-only baselines or retire",
        "repair_missing_artifact": "write the missing repair micro-panel artifact or retire the row",
        "fr11_boundary": "preserve controller-only boundary unless broader evidence is added",
        "ebt_arm_projection": "add adapter implementation and integration evidence or keep projection-only",
        "hardware_evidence": "wait for external operator host-visible hardware evidence",
        "publication_readiness": "recompute only after upstream blockers and model gaps clear",
        "model_spec_gap": "add mandatory_headline_model_ids and complete model_specs",
        "missing_artifact": "write or explicitly retire the missing terminal artifact",
        "bounded_status": "convert to clean or retired with matrix-v23 mechanical evidence",
        "retired_status": "preserve retired provenance outside blocker counts",
    }[category]


def _honest_verdict(artifact: Mapping[str, Any], *, matrix_present: bool) -> str:
    if not matrix_present:
        return "blocked_required_matrix_v22_missing"
    if artifact.get("blocker_triage_ready") is not True:
        return (
            "blocked_triage_preconditions: "
            f"blocked_reasons={_as_list(artifact.get('blocked_reasons'))}"
        )
    return (
        "complete: "
        "blocker_triage_ready=true; "
        f"publication_blocker_count_before={artifact['publication_blocker_count_before']}; "
        f"reducible_in_v289={len(_as_list(artifact.get('reducible_in_v289')))}; "
        "operator_evidence_required="
        f"{len(_as_list(artifact.get('operator_evidence_required')))}"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_artifacts",
        "source": "matrix_v22_and_capstone_v288",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def normal_status(status: str) -> str:
    """Normalize legacy matrix labels into the status vocabulary used here."""

    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str) -> str:
    """Map a normalized status to the blocker class used by matrix artifacts."""

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
    "CAPSTONE_V288_REL_PATH",
    "CATEGORY_NAMES",
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
