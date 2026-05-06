"""Exp 1413 repair-hint diagnosis for the missing LLM repair executor.

Spec: REQ-VERIFY-1413, SCENARIO-VERIFY-1413.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260506"
EXPERIMENT = "1413_certificate_repair_execution_diagnosis"
SCHEMA = "certificate_repair_execution_diagnosis_v1"
DEFAULT_EXP1397_PATH = (
    REPO_ROOT / "results" / "experiment_1397_fullscale_pipeline_v2_200cases.json"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "experiment_1413_certificate_repair_execution_diagnosis.json"
)

HINT_CATEGORIES = (
    "FIELD_REWRITE",
    "STEP_REWRITE",
    "CONSTRAINT_REWRITE",
    "CERTIFICATE_REGENERATE",
    "UNKNOWN",
)
EXECUTABLE_CATEGORIES = {
    "FIELD_REWRITE",
    "STEP_REWRITE",
    "CONSTRAINT_REWRITE",
    "CERTIFICATE_REGENERATE",
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "total_cases_analyzed",
    "repair_hint_cases_total",
    "no_repair_cases_total",
    "repair_execution_diagnosis_complete",
    "hint_category_counts",
    "executable_hint_pct",
    "recommended_executor_contract",
    "expected_full_pipeline_pass_rate_if_50pct_repaired",
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
    """REQ-VERIFY-1413: persist a bootstrap artifact before reading Exp 1397.

    This diagnosis is inexpensive, but the in-progress write is still useful:
    it gives the conductor a durable marker that the repair-executor handoff
    started and prevents a missing terminal JSON from looking like the task was
    never attempted.
    """

    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="in_progress")
    artifact["honest_verdict"] = "in_progress"
    _write_json(Path(path), artifact, write_observer=write_observer)
    return artifact


def build_certificate_repair_execution_diagnosis(
    *,
    exp1397_artifact: Mapping[str, Any],
    run_date: str = RUN_DATE,
    project_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Classify Exp1397 repair hints and define the Exp1414 executor contract.

    Exp1397 already proved that certificate parsing and semantic validation can
    both reach 1.0.  Its remaining failure mode is that repair hints are only
    localized; no executor turns those hints into rewritten reasoning plus a
    repaired certificate.  This function keeps the diagnosis deterministic so
    Exp1414 can implement the actual LLM call behind a clear contract.
    """

    repair_rows = _repair_hint_rows(exp1397_artifact)
    total_cases = _total_cases(exp1397_artifact)
    full_pipeline_pass_rate = _float(exp1397_artifact.get("full_pipeline_pass_rate"))
    category_counts = _category_counts(repair_rows)
    repair_hint_cases_total = len(repair_rows)
    executable_count = sum(
        count for category, count in category_counts.items() if category in EXECUTABLE_CATEGORIES
    )
    expected_rate, expected_basis = _expected_rate_if_50pct_repaired(
        current_rate=full_pipeline_pass_rate,
        total_cases=total_cases,
        repair_hint_cases=repair_hint_cases_total,
    )

    artifact = _base_artifact(project_root=project_root, run_date=run_date, status="complete")
    artifact.update(
        {
            "source_experiment": "experiment_1397_fullscale_pipeline_v2_200cases",
            "source_metrics": {
                "certificate_parse_rate": _float(exp1397_artifact.get("certificate_parse_rate")),
                "semantic_validation_pass_rate": _float(
                    exp1397_artifact.get("semantic_validation_pass_rate")
                ),
                "full_pipeline_pass_rate": full_pipeline_pass_rate,
                "scheduler_accept_rate": _float(exp1397_artifact.get("scheduler_accept_rate")),
            },
            "total_cases_analyzed": total_cases,
            "repair_hint_cases_total": repair_hint_cases_total,
            "no_repair_cases_total": max(0, total_cases - repair_hint_cases_total),
            "repair_execution_diagnosis_complete": True,
            "hint_category_counts": category_counts,
            "executable_hint_pct": _rate(executable_count, repair_hint_cases_total),
            "recommended_executor_contract": recommended_executor_contract(),
            "expected_full_pipeline_pass_rate_if_50pct_repaired": expected_rate,
            "expected_rate_basis": expected_basis,
            "diagnostic_evidence": {
                "repair_hint_categories_observed": [
                    category for category, count in category_counts.items() if count
                ],
                "repair_rows_with_accepted_false": sum(
                    1 for row in repair_rows if row.get("accepted") is False
                ),
                "repair_rows_with_semantic_equivalence_false": sum(
                    1 for row in repair_rows if row.get("semantic_equivalence_passed") is False
                ),
                "scheduler_nonrepair_failures": _scheduler_nonrepair_failures(exp1397_artifact),
            },
            "honest_verdict": _honest_verdict(
                repair_hint_cases_total=repair_hint_cases_total,
                executable_hint_pct=_rate(executable_count, repair_hint_cases_total),
                category_counts=category_counts,
                full_pipeline_pass_rate=full_pipeline_pass_rate,
            ),
        }
    )
    return artifact


def classify_hint_category(row: Mapping[str, Any]) -> str:
    """Return the bounded executor category for one VERGE/MCS repair row."""

    text = " ".join(
        str(row.get(key) or "")
        for key in ("localized_constraint", "minimal_local_change", "repair_hint")
    ).lower()
    if any(token in text for token in ("regenerate", "parse failure", "tag-first")):
        return "CERTIFICATE_REGENERATE"
    if any(
        token in text
        for token in (
            "certificate_state",
            "dispatched_state",
            "tag_state",
            "field",
            "set_certificate_state",
        )
    ):
        return "FIELD_REWRITE"
    if any(
        token in text
        for token in ("reasoning step", "arithmetic step", "fover_incorrect_reasoning_step")
    ):
        return "STEP_REWRITE"
    if any(
        token in text
        for token in (
            "constraint",
            "bound",
            "premise",
            "clause",
            "cnf",
            "formula",
            "capacity",
            "z3",
        )
    ):
        return "CONSTRAINT_REWRITE"
    return "UNKNOWN"


def recommended_executor_contract() -> dict[str, Any]:
    """Return the Exp1414 implementation contract for bounded local repair.

    The contract deliberately names a local LLM executor, not a vendor SDK.  The
    core pipeline should depend only on a callable/protocol that can be backed
    by an open local model; closed-weight adapters can be optional wrappers.
    """

    return {
        "inputs": [
            "case_id",
            "original_question",
            "original_reasoning_step",
            "original_certificate_text",
            "semantic_validation_row",
            "repair_localization_row",
            "repair_hint",
            "hint_category",
            "bounded_prompt_budget_tokens",
        ],
        "outputs": [
            "case_id",
            "repair_attempted",
            "hint_category",
            "repaired_reasoning_step",
            "repaired_certificate_text",
            "executor_model_id",
            "executor_runtime_s",
            "validation_result",
            "fallback_reason",
        ],
        "validation_call": (
            "Parse the repaired tag-first certificate, call "
            "calibrated_fover_semantic_validation_row(...), then replay the "
            "repair-localization and scheduler checks; accept only when "
            "constraint_passed is true, semantic_result is SAT, repair_required "
            "is false, full_pipeline_pass is true, and false_acceptance is false."
        ),
        "timeout": {
            "per_case_seconds": 30,
            "max_attempts_per_case": 1,
            "batch_budget_source": "caller_provided",
        },
        "fallback_behavior": {
            "on_timeout": "preserve_original_repair_hint_and_escalate_full_verifier",
            "on_invalid_json_or_certificate": "preserve_original_repair_hint_and_escalate_full_verifier",
            "on_validation_failure": "preserve_original_repair_hint_and_escalate_full_verifier",
            "preserve_original_repair_hint": True,
            "accepted_without_validation": False,
        },
        "data_handling_class": "minimize",
        "executor_model_policy": "local_open_model_first_optional_closed_adapter_only",
    }


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    exp1397_path: str | Path = DEFAULT_EXP1397_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write progress, load Exp1397, then persist the complete diagnosis."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    write_in_progress_artifact(
        output,
        project_root=root,
        run_date=run_date,
        write_observer=write_observer,
    )
    artifact = build_certificate_repair_execution_diagnosis(
        exp1397_artifact=_read_json(_resolve(root, exp1397_path)),
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _base_artifact(*, project_root: str | Path, run_date: str, status: str) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": status,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "spec": ["REQ-VERIFY-1413", "SCENARIO-VERIFY-1413"],
            "source_experiments": ["exp1397"],
        },
        "total_cases_analyzed": 0,
        "repair_hint_cases_total": 0,
        "no_repair_cases_total": 0,
        "repair_execution_diagnosis_complete": False,
        "hint_category_counts": {category: 0 for category in HINT_CATEGORIES},
        "executable_hint_pct": 0.0,
        "recommended_executor_contract": {},
        "expected_full_pipeline_pass_rate_if_50pct_repaired": 0.0,
        "honest_verdict": "not_run",
    }


def _repair_hint_rows(exp1397_artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = exp1397_artifact.get("repair_localization_rows") or []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping) and row.get("repair_hint")]


def _category_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(classify_hint_category(row) for row in rows)
    return {category: int(counter.get(category, 0)) for category in HINT_CATEGORIES}


def _total_cases(exp1397_artifact: Mapping[str, Any]) -> int:
    for key in ("cases_evaluated", "total_fover_cases", "target_cases"):
        value = exp1397_artifact.get(key)
        if value is not None:
            return max(0, int(value))
    for key in ("scheduler_rows", "certificate_rows"):
        rows = exp1397_artifact.get(key)
        if isinstance(rows, list):
            return len(rows)
    return 0


def _expected_rate_if_50pct_repaired(
    *,
    current_rate: float,
    total_cases: int,
    repair_hint_cases: int,
) -> tuple[float, dict[str, Any]]:
    if total_cases > 0 and repair_hint_cases > 0:
        lift = 0.5 * repair_hint_cases / total_cases
        return (
            round(min(1.0, current_rate + lift), 6),
            {
                "used_repair_specific_denominator": True,
                "current_full_pipeline_pass_rate": current_rate,
                "repair_hint_cases_total": repair_hint_cases,
                "total_cases_analyzed": total_cases,
                "formula": "current_full_pipeline_pass_rate + 0.5 * repair_hint_cases_total / total_cases_analyzed",
            },
        )
    return (
        round(min(1.0, current_rate + 0.5 * (1.0 - current_rate)), 6),
        {
            "used_repair_specific_denominator": False,
            "current_full_pipeline_pass_rate": current_rate,
            "formula": "current_full_pipeline_pass_rate + 0.5 * (1 - current_full_pipeline_pass_rate)",
        },
    )


def _scheduler_nonrepair_failures(exp1397_artifact: Mapping[str, Any]) -> int:
    rows = exp1397_artifact.get("scheduler_rows") or []
    if not isinstance(rows, list):
        return 0
    return sum(
        1
        for row in rows
        if isinstance(row, Mapping)
        and row.get("full_pipeline_pass") is not True
        and row.get("repair_required") is not True
    )


def _honest_verdict(
    *,
    repair_hint_cases_total: int,
    executable_hint_pct: float,
    category_counts: Mapping[str, int],
    full_pipeline_pass_rate: float,
) -> str:
    if repair_hint_cases_total == 0:
        return "repair_execution_diagnosis_complete_no_repair_hints_found"
    if executable_hint_pct == 0.0:
        return "repair_execution_diagnosis_complete_missing_repair_execution_but_hints_not_executable"
    dominant = max(HINT_CATEGORIES, key=lambda category: category_counts.get(category, 0))
    return (
        "repair_execution_diagnosis_complete_missing_repair_execution_"
        f"{dominant.lower()}_dominant_full_pipeline_{_rate_label(full_pipeline_pass_rate)}"
    )


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if write_observer is not None:
        write_observer(path, dict(payload))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _float(value: Any) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _rate_label(value: float) -> str:
    return str(round(float(value), 6)).replace(".", "_")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for conductor/manual Exp1413 runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--exp1397-path", type=Path, default=DEFAULT_EXP1397_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)

    artifact = run_experiment(
        project_root=args.project_root,
        run_date=args.run_date,
        exp1397_path=args.exp1397_path,
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {
                "status": artifact.get("status"),
                "repair_hint_cases_total": artifact.get("repair_hint_cases_total"),
                "executable_hint_pct": artifact.get("executable_hint_pct"),
                "honest_verdict": artifact.get("honest_verdict"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
