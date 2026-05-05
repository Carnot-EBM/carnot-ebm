"""Exp 1391 diagnosis for Exp 1382 semantic-validation failures.

Spec: REQ-VERIFY-1391, SCENARIO-VERIFY-1391.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260505"
EXPERIMENT = "1391_fullscale_pipeline_failure_diagnosis"
SCHEMA = "fullscale_pipeline_failure_diagnosis_v1"
DEFAULT_EXP1382_PATH = (
    REPO_ROOT / "results" / "experiment_1382_fullscale_certificate_semantic_repair_100cases.json"
)
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1391_fullscale_pipeline_failure_diagnosis.json"
)

CATEGORIES = (
    "Z3_CONSTRAINT_MISMATCH",
    "MISSING_CERTIFICATE_FIELD",
    "SEMANTIC_CONTRADICTION",
    "CORPUS_SPECIFIC",
    "VALIDATOR_BUG",
    "OTHER",
)
FIXABLE_CATEGORIES = {
    "MISSING_CERTIFICATE_FIELD",
    "VALIDATOR_BUG",
    "CORPUS_SPECIFIC",
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "total_cases_analyzed",
    "parse_rate_confirmed",
    "semantic_validation_failures_classified",
    "failure_categories",
    "failure_category_counts",
    "top_failure_category",
    "fixable_failure_pct",
    "estimated_semantic_validation_pass_rate_after_fixes",
    "failure_analysis_complete",
    "recommended_fixes",
    "honest_verdict",
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def write_in_progress_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-VERIFY-1391: persist an auditable bootstrap artifact before loading data."""

    artifact = _base_artifact(project_root=Path(project_root), run_date=run_date)
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def build_failure_diagnosis_artifact(
    *,
    exp1382_artifact: Mapping[str, Any],
    fover_case_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Classify Exp 1382 semantic failures from already-recorded artifact rows.

    The Exp 1382 artifact is rich enough to diagnose the main failure source
    without fresh model calls. The key predicate is ``constraint_passed`` in
    ``semantic_validation_rows``; failed rows are joined with certificate rows
    and optional FoVer source metadata to separate parse, certificate-state,
    DVI-boundary, and corpus-specific arithmetic failures.
    """

    semantic_rows = _mapping_rows(exp1382_artifact.get("semantic_validation_rows"))
    certificate_rows = _mapping_rows(exp1382_artifact.get("certificate_rows"))
    certificate_by_case = _rows_by_case_id(certificate_rows)
    source_lookup = fover_case_lookup or {}

    failures = [row for row in semantic_rows if _is_semantic_failure(row)]
    classifications: list[dict[str, Any]] = []
    category_counts = {category: 0 for category in CATEGORIES}

    for row in failures:
        case_id = str(row.get("case_id") or "")
        certificate_row = certificate_by_case.get(case_id, {})
        source_row = source_lookup.get(case_id, {})
        category = _classify_failure(row, certificate_row, source_row)
        category_counts[category] += 1
        classifications.append(
            _case_classification(
                row=row,
                certificate_row=certificate_row,
                source_row=source_row,
                category=category,
            )
        )

    total_cases = int(
        exp1382_artifact.get("total_fover_cases")
        or exp1382_artifact.get("total_cases")
        or len(semantic_rows)
    )
    semantic_pass_count = sum(1 for row in semantic_rows if row.get("constraint_passed") is True)
    fixable_count = sum(category_counts[category] for category in FIXABLE_CATEGORIES)
    top_category = _top_category(category_counts)
    observed_parse_rate = _number(exp1382_artifact.get("certificate_parse_rate"), 0.0)
    estimated_pass_rate = _rate(semantic_pass_count + fixable_count, total_cases)

    artifact = _base_artifact(
        project_root=Path(project_root),
        run_date=run_date,
        status="complete",
    )
    artifact.update(
        {
            "total_cases_analyzed": total_cases,
            "parse_rate_confirmed": abs(observed_parse_rate - 1.0) <= 1e-9,
            "semantic_validation_failures_classified": len(classifications),
            "failure_categories": _failure_category_details(
                classifications,
                category_counts,
            ),
            "failure_category_counts": category_counts,
            "top_failure_category": top_category,
            "fixable_failure_pct": _rate(fixable_count, len(failures)),
            "estimated_semantic_validation_pass_rate_after_fixes": estimated_pass_rate,
            "failure_analysis_complete": len(classifications) == len(failures),
            "recommended_fixes": _recommended_fixes(category_counts),
            "honest_verdict": _honest_verdict(category_counts),
            "artifact_metadata": {
                "project_root": str(project_root),
                "run_date": run_date,
                "source_experiment": "1382",
                "spec": ["REQ-VERIFY-1391", "SCENARIO-VERIFY-1391"],
            },
            "source_data_locations": {
                "primary": str(DEFAULT_EXP1382_PATH),
                "semantic_failures": "semantic_validation_rows[constraint_passed=false]",
                "certificate_rows": "certificate_rows keyed by case_id",
                "fover_metadata": str(DEFAULT_FOVER_PATH),
            },
            "observed_certificate_parse_rate": observed_parse_rate,
            "observed_semantic_validation_pass_rate": _number(
                exp1382_artifact.get("semantic_validation_pass_rate"),
                _rate(semantic_pass_count, total_cases),
            ),
            "semantic_validation_failure_count": len(failures),
            "semantic_validation_pass_count": semantic_pass_count,
            "fixable_failure_count": fixable_count,
            "failure_ids": [entry["case_id"] for entry in classifications],
            "per_case_failure_classifications": classifications,
            "measurement_note": (
                "Diagnostic-only replay over saved Exp 1382 rows. No fresh model "
                "generation, DVI inference, Z3 solving, or repair attempt was run."
            ),
        }
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    exp1382_path: str | Path = DEFAULT_EXP1382_PATH,
    fover_path: str | Path = DEFAULT_FOVER_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write the Exp 1391 artifact from saved Exp 1382 and FoVer metadata."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )
    exp1382_artifact = _load_json(_resolve(root, exp1382_path))
    fover_lookup = load_fover_case_lookup(_resolve(root, fover_path))
    artifact = build_failure_diagnosis_artifact(
        exp1382_artifact=exp1382_artifact,
        fover_case_lookup=fover_lookup,
        project_root=root,
        run_date=run_date,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def load_fover_case_lookup(path: Path | str = DEFAULT_FOVER_PATH) -> dict[str, dict[str, Any]]:
    """Load FoVer rows keyed by the same stable case IDs used by Exp 1382.

    This deliberately duplicates only the lightweight ID/label normalization
    logic from Exp 1382. Importing that runner would also import optional model
    and accelerator dependencies, which is unnecessary for a saved-artifact
    diagnosis.
    """

    rows = _read_rows(Path(path))
    lookup: dict[str, dict[str, Any]] = {}
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        label = _label_from_row(row)
        response = _row_text(row)
        if label is None or not response:
            continue
        raw_id = str(
            row.get("question_id")
            or row.get("case_id")
            or row.get("id")
            or row.get("question_index")
            or f"fover_{index}"
        )
        ordinal = seen.get(raw_id, 0)
        seen[raw_id] = ordinal + 1
        case_id = raw_id if ordinal == 0 else f"{raw_id}_{ordinal}"
        lookup[case_id] = {
            "case_id": case_id,
            "source": str(row.get("source") or "fover_corpus"),
            "label": label,
            "question": str(row.get("question") or row.get("prompt") or ""),
            "response": response,
        }
    return lookup


def _base_artifact(
    *,
    project_root: Path,
    run_date: str,
    status: str = "in_progress",
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": status,
        "total_cases_analyzed": 0,
        "parse_rate_confirmed": None,
        "semantic_validation_failures_classified": 0,
        "failure_categories": [],
        "failure_category_counts": {category: 0 for category in CATEGORIES},
        "top_failure_category": None,
        "fixable_failure_pct": 0.0,
        "estimated_semantic_validation_pass_rate_after_fixes": 0.0,
        "failure_analysis_complete": False,
        "recommended_fixes": [],
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiment": "1382",
            "spec": ["REQ-VERIFY-1391", "SCENARIO-VERIFY-1391"],
        },
    }


def _classify_failure(
    semantic_row: Mapping[str, Any],
    certificate_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> str:
    failure_reason = str(semantic_row.get("failure_reason") or "").lower()
    if _looks_like_z3_mismatch(failure_reason):
        return "Z3_CONSTRAINT_MISMATCH"
    if (
        certificate_row.get("parseable") is not True
        or semantic_row.get("constraint_evaluated") is False
    ):
        return "MISSING_CERTIFICATE_FIELD"
    if failure_reason == "certificate_state_mismatch":
        return "SEMANTIC_CONTRADICTION"
    if _certificate_state(semantic_row) != _expected_state(semantic_row):
        return "SEMANTIC_CONTRADICTION"
    if failure_reason == "dvi_disagrees_with_fover_label":
        if str(semantic_row.get("fover_label")) == "correct":
            return "VALIDATOR_BUG"
        if _source_family(semantic_row, source_row) in {"fover_v4", "math_z3", "math_z3_v3"}:
            return "CORPUS_SPECIFIC"
        return "VALIDATOR_BUG"
    return "OTHER"


def _case_classification(
    *,
    row: Mapping[str, Any],
    certificate_row: Mapping[str, Any],
    source_row: Mapping[str, Any],
    category: str,
) -> dict[str, Any]:
    case_id = str(row.get("case_id") or "")
    source = _source_family(row, source_row)
    return {
        "case_id": case_id,
        "category": category,
        "subcategory": _subcategory(row, category, source),
        "source": source,
        "fover_label": row.get("fover_label"),
        "expected_state": row.get("expected_state"),
        "certificate_state": row.get("certificate_state"),
        "semantic_result": row.get("semantic_result"),
        "failure_reason": row.get("failure_reason"),
        "dvi_incorrect_probability": row.get("dvi_incorrect_probability"),
        "dvi_incorrect_threshold": row.get("dvi_incorrect_threshold"),
        "semantic_margin": row.get("semantic_margin"),
        "certificate_parseable": certificate_row.get("parseable"),
        "certificate_errors": certificate_row.get("errors", []),
        "fix_complexity": _fix_complexity(category),
    }


def _failure_category_details(
    classifications: list[dict[str, Any]],
    category_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    for entry in classifications:
        grouped[str(entry["category"])].append(entry)

    details: list[dict[str, Any]] = []
    for category in sorted(CATEGORIES, key=lambda item: (-category_counts[item], item)):
        count = int(category_counts[category])
        if count == 0:
            continue
        entries = grouped[category]
        details.append(
            {
                "category": category,
                "count": count,
                "pct_of_failures": _rate(count, len(classifications)),
                "fix_complexity": _fix_complexity(category),
                "description": _category_description(category),
                "source_counts": dict(Counter(str(entry["source"]) for entry in entries)),
                "direction_counts": dict(
                    Counter(
                        f"{entry.get('fover_label')}->{entry.get('semantic_result')}"
                        for entry in entries
                    )
                ),
                "example_case_ids": [str(entry["case_id"]) for entry in entries[:8]],
            }
        )
    return details


def _recommended_fixes(category_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    fixes: list[dict[str, Any]] = []
    if category_counts["CORPUS_SPECIFIC"]:
        fixes.append(
            {
                "rank": 1,
                "category": "CORPUS_SPECIFIC",
                "fix_complexity": "medium",
                "expected_failure_reduction_count": category_counts["CORPUS_SPECIFIC"],
                "recommendation": (
                    "For Exp 1396, add an arithmetic-aware fallback before accepting "
                    "DVI SAT on FoVer incorrect rows: extract arithmetic claims and "
                    "run the existing Z3/math verifier or escalate to the full NSVIF "
                    "validator when DVI probability is below threshold on a known "
                    "math/FoVer arithmetic source."
                ),
            }
        )
    if category_counts["VALIDATOR_BUG"]:
        fixes.append(
            {
                "rank": len(fixes) + 1,
                "category": "VALIDATOR_BUG",
                "fix_complexity": "low_medium",
                "expected_failure_reduction_count": category_counts["VALIDATOR_BUG"],
                "recommendation": (
                    "Add a DVI abstention/calibration band around the 0.72 threshold "
                    "for SAT certificates. Correct FoVer rows just above the threshold "
                    "should escalate instead of being counted as semantic failures."
                ),
            }
        )
    if category_counts["MISSING_CERTIFICATE_FIELD"]:
        fixes.append(
            {
                "rank": len(fixes) + 1,
                "category": "MISSING_CERTIFICATE_FIELD",
                "fix_complexity": "low",
                "expected_failure_reduction_count": category_counts["MISSING_CERTIFICATE_FIELD"],
                "recommendation": (
                    "Repair the tag-first certificate parser/regenerator for rows "
                    "with absent or malformed required fields."
                ),
            }
        )
    if category_counts["Z3_CONSTRAINT_MISMATCH"] or category_counts["SEMANTIC_CONTRADICTION"]:
        fixes.append(
            {
                "rank": len(fixes) + 1,
                "category": "HARD_FAILURES",
                "fix_complexity": "high",
                "expected_failure_reduction_count": (
                    category_counts["Z3_CONSTRAINT_MISMATCH"]
                    + category_counts["SEMANTIC_CONTRADICTION"]
                ),
                "recommendation": (
                    "Treat solver UNSAT and intra-certificate contradictions as hard "
                    "semantic failures; inspect them manually before attempting an "
                    "automatic repair rule."
                ),
            }
        )
    fixes.append(
        {
            "rank": len(fixes) + 1,
            "category": "DIAGNOSTIC_INSTRUMENTATION",
            "fix_complexity": "low",
            "expected_failure_reduction_count": 0,
            "recommendation": (
                "Persist source family, extracted arithmetic claims, DVI score, "
                "threshold margin, and fallback solver verdict in Exp 1396 rows so "
                "future semantic failures do not require reconstructing context."
            ),
        }
    )
    return fixes


def _honest_verdict(category_counts: Mapping[str, int]) -> str:
    if category_counts["OTHER"]:
        return "diagnosis_complete_other_bucket_requires_manual_review"
    if category_counts["Z3_CONSTRAINT_MISMATCH"] or category_counts["SEMANTIC_CONTRADICTION"]:
        return "diagnosis_complete_contains_hard_semantic_failures"
    if category_counts["CORPUS_SPECIFIC"] or category_counts["VALIDATOR_BUG"]:
        return "diagnosis_complete_dvi_boundary_failures_dominate_exp1396_should_fix_validator"
    return "diagnosis_complete_no_semantic_failures_found"


def _is_semantic_failure(row: Mapping[str, Any]) -> bool:
    if row.get("constraint_passed") is False:
        return True
    return "constraint_passed" not in row and row.get("failure_reason") is not None


def _looks_like_z3_mismatch(failure_reason: str) -> bool:
    return "z3" in failure_reason or "unsat" in failure_reason or "solver" in failure_reason


def _certificate_state(row: Mapping[str, Any]) -> str:
    return str(row.get("certificate_state") or "").upper()


def _expected_state(row: Mapping[str, Any]) -> str:
    return str(row.get("expected_state") or "").upper()


def _source_family(row: Mapping[str, Any], source_row: Mapping[str, Any]) -> str:
    source = str(source_row.get("source") or "")
    if source:
        return source
    case_id = str(row.get("case_id") or "")
    if case_id.startswith("math_v3_"):
        return "math_z3_v3"
    if case_id.startswith("math_"):
        return "math_z3"
    return "unknown"


def _subcategory(row: Mapping[str, Any], category: str, source: str) -> str:
    if category == "CORPUS_SPECIFIC":
        return f"dvi_false_sat_on_incorrect_{source}"
    if category == "VALIDATOR_BUG":
        return "dvi_false_repair_on_correct_sat_certificate"
    if category == "MISSING_CERTIFICATE_FIELD":
        return "certificate_unparseable_or_constraint_not_evaluated"
    if category == "SEMANTIC_CONTRADICTION":
        return "certificate_state_conflicts_with_expected_state"
    if category == "Z3_CONSTRAINT_MISMATCH":
        return "solver_unsat_or_constraint_mismatch"
    return str(row.get("failure_reason") or "unclassified_semantic_failure")


def _fix_complexity(category: str) -> str:
    return {
        "CORPUS_SPECIFIC": "medium",
        "VALIDATOR_BUG": "low_medium",
        "MISSING_CERTIFICATE_FIELD": "low",
        "Z3_CONSTRAINT_MISMATCH": "high",
        "SEMANTIC_CONTRADICTION": "high",
        "OTHER": "unknown",
    }.get(category, "unknown")


def _category_description(category: str) -> str:
    return {
        "CORPUS_SPECIFIC": (
            "DVI predicts SAT for FoVer rows labeled incorrect, concentrated in "
            "arithmetic-heavy FoVer/math_z3 sources whose certificates parsed and "
            "correctly requested REPAIR_HINT."
        ),
        "VALIDATOR_BUG": (
            "DVI predicts REPAIR_HINT for FoVer rows labeled correct even though "
            "the parsed certificate is a matching SAT certificate."
        ),
        "MISSING_CERTIFICATE_FIELD": (
            "The certificate parser could not evaluate the semantic constraint "
            "because a required certificate field was missing or malformed."
        ),
        "SEMANTIC_CONTRADICTION": (
            "The parsed certificate state contradicts the expected semantic state."
        ),
        "Z3_CONSTRAINT_MISMATCH": (
            "The solver reported UNSAT or an equivalent constraint mismatch for "
            "a certificate expected to be satisfiable."
        ),
        "OTHER": "The row did not match a known failure signature.",
    }.get(category, "Unknown category.")


def _top_category(category_counts: Mapping[str, int]) -> str | None:
    if not category_counts:
        return None
    category, count = max(category_counts.items(), key=lambda item: (item[1], item[0]))
    return category if count > 0 else None


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _rows_by_case_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("case_id")): row for row in rows if row.get("case_id") is not None}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping):
                rows.append(dict(row))
        return rows
    payload = _load_json(path)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "pairs", "items", "examples", "data", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _label_from_row(row: Mapping[str, Any]) -> int | None:
    if "is_correct" in row:
        return 0 if bool(row["is_correct"]) else 1
    if "step_correct" in row:
        return 0 if bool(row["step_correct"]) else 1
    raw = row.get("label")
    if raw is None:
        raw = row.get("verdict") or row.get("z3_label") or row.get("sc_energy_label")
    if isinstance(raw, bool):
        return 0 if raw else 1
    if isinstance(raw, int | float):
        return 0 if int(raw) == 1 else 1
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"correct", "true", "supported", "entailed", "1"}:
            return 0
        if normalized in {"incorrect", "wrong", "false", "violated", "violation", "0"}:
            return 1
    return None


def _row_text(row: Mapping[str, Any]) -> str:
    return str(row.get("step_text") or row.get("response") or row.get("step") or "").strip()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(
    path: Path,
    artifact: dict[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    if write_observer is not None:
        write_observer(path, artifact)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rate(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, float(numerator) / denominator)), 6)
