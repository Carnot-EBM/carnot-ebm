"""Exp 1354 LogicSkills-style certificate skill split replay.

Spec: REQ-VERIFY-1354,
      SCENARIO-VERIFY-1354
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from z3 import Bool, Not, Solver, sat, unsat


DEFAULT_RUN_DATE = "20260505"
DEFAULT_EXP1353_PATH = Path("results/experiment_1353_triggered_certificate_v7_truncproof_sota.json")
DEFAULT_OUTPUT_PATH = Path("results/experiment_1354_logicskills_certificate_skill_split.json")
ARTIFACT_NAME = "experiment_1354_logicskills_certificate_skill_split"
SCHEMA_VERSION = 1
SKILL_NAMES = ("symbolization", "countermodel", "validity_assessment", "unknown_preservation")
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "certificate_cases_used",
    "symbolization_pass_rate",
    "countermodel_pass_rate",
    "validity_pass_rate",
    "z3_verified_case_count",
    "dominant_skill_gap",
    "skill_split_claim_allowed",
    "honest_verdict",
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def build_logicskills_certificate_skill_split_artifact(
    *,
    exp1353_artifact: Mapping[str, Any],
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Build the replay artifact without calling any model.

    LogicSkills is useful here because a missing certificate tag is not the
    same kind of failure as a wrong SAT/UNSAT verdict.  This builder keeps the
    first parser/symbolization gate separate from downstream validity and
    UNKNOWN-preservation evidence, so later repair work can target the actual
    blocked skill instead of a generic "semantic failure" bucket.
    """
    rows = _certificate_rows(exp1353_artifact)
    classifications = [_classification_record(row) for row in rows]
    z3_checks = _z3_checks(rows)
    skill_gap_counts = _skill_gap_counts(classifications)
    dominant_gap = _dominant_skill_gap(skill_gap_counts)
    skill_split_claim_allowed = bool(classifications)

    artifact = _base_artifact(project_root=Path(project_root), run_date=run_date)
    artifact.update(
        {
            "status": "complete",
            "certificate_cases_used": len(rows),
            "symbolization_pass_rate": _rate(
                sum(1 for row in rows if bool(row.get("parseable"))), len(rows)
            ),
            "countermodel_pass_rate": _countermodel_pass_rate(rows),
            "validity_pass_rate": _rate(
                sum(1 for row in rows if bool(row.get("truthful"))), len(rows)
            ),
            "z3_verified_case_count": sum(1 for check in z3_checks if check["verified"]),
            "dominant_skill_gap": dominant_gap,
            "skill_split_claim_allowed": skill_split_claim_allowed,
            "honest_verdict": _honest_verdict(
                dominant_gap=dominant_gap,
                skill_split_claim_allowed=skill_split_claim_allowed,
            ),
            "classification_rows": classifications,
            "classification_counts": _classification_counts(classifications),
            "skill_gap_counts": skill_gap_counts,
            "z3_checks": z3_checks,
            "source_honest_verdicts": {"exp1353": exp1353_artifact.get("honest_verdict")},
            "measurement_note": (
                "Replay-only artifact. Exp 1353 generation rows are not regenerated; "
                "local Z3 checks are limited to cases whose fixture constraints are "
                "formalizable without a certificate body."
            ),
        }
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    exp1353_path: str | Path = DEFAULT_EXP1353_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write an in-progress artifact first, then persist the completed replay."""
    root = Path(project_root)
    output = _resolve(root, output_path)
    _write_json(
        output,
        _base_artifact(project_root=root, run_date=run_date, status="in_progress"),
        write_observer=write_observer,
    )
    artifact = build_logicskills_certificate_skill_split_artifact(
        exp1353_artifact=_read_json(_resolve(root, exp1353_path)),
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _classification_record(row: Mapping[str, Any]) -> dict[str, Any]:
    classification = _classify_row(row)
    skill_gap = _skill_gap(row, classification)
    return {
        "case_id": str(row.get("case_id") or ""),
        "expected_state": _expected_state(row),
        "classification": classification,
        "skill_gap": skill_gap,
        "evidence": _evidence(row, skill_gap),
        "parseable": bool(row.get("parseable")),
        "truthful": bool(row.get("truthful")),
        "unknown_preserved": bool(row.get("unknown_preserved")),
    }


def _classify_row(row: Mapping[str, Any]) -> str:
    if _expected_state(row) == "UNKNOWN" and bool(row.get("unknown_preserved")):
        return "UNKNOWN-preserving"
    if bool(row.get("parseable")) and bool(row.get("truthful")):
        return "semantically truth-preserving"
    if bool(row.get("parseable")):
        return "parsed"
    return "rejected"


def _skill_gap(row: Mapping[str, Any], classification: str) -> str | None:
    if classification in {"semantically truth-preserving", "UNKNOWN-preserving"}:
        return None
    errors = [str(error).lower() for error in row.get("errors") or []]
    if _expected_state(row) == "UNKNOWN" and bool(row.get("parseable")):
        return "unknown_preservation"
    if any("countermodel" in error or "witness" in error for error in errors):
        return "countermodel"
    if not bool(row.get("parseable")):
        return "symbolization"
    return "validity_assessment"


def _evidence(row: Mapping[str, Any], skill_gap: str | None) -> str:
    if skill_gap is None:
        return "local parser row is parseable and truth-preserving"
    errors = [str(error) for error in row.get("errors") or []]
    if skill_gap == "symbolization":
        return "certificate rejected before semantic checking: " + ", ".join(
            errors or ["unparseable"]
        )
    if skill_gap == "unknown_preservation":
        return "UNKNOWN case parsed without preserving UNKNOWN"
    if skill_gap == "countermodel":
        return "countermodel or witness evidence failed: " + ", ".join(errors)
    return "parseable certificate did not preserve the expected verifier state"


def _z3_checks(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        solver_status = _z3_expected_state(case_id)
        if solver_status is None:
            continue
        expected = _expected_state(row)
        checks.append(
            {
                "case_id": case_id,
                "expected_state": expected,
                "solver_state": solver_status,
                "verified": solver_status == expected,
            }
        )
    return checks


def _z3_expected_state(case_id: str) -> str | None:
    x1 = Bool("x1")
    solver = Solver()
    if case_id == "sat_unit_clause":
        solver.add(x1)
        return "SAT" if solver.check() == sat else "UNSAT"
    if case_id == "unsat_unit_conflict":
        solver.add(x1, Not(x1))
        return "UNSAT" if solver.check() == unsat else "SAT"
    return None


def _countermodel_pass_rate(rows: list[Mapping[str, Any]]) -> float:
    applicable = [row for row in rows if _countermodel_applicable(row)]
    passes = sum(
        1 for row in applicable if bool(row.get("truthful")) and bool(row.get("parseable"))
    )
    return _rate(passes, len(applicable))


def _countermodel_applicable(row: Mapping[str, Any]) -> bool:
    errors = [str(error).lower() for error in row.get("errors") or []]
    return _expected_state(row) == "SAT" or any(
        "countermodel" in error or "witness" in error for error in errors
    )


def _certificate_rows(exp1353_artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = exp1353_artifact.get("certificate_rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _classification_counts(classifications: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in classifications:
        label = str(row["classification"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def _skill_gap_counts(classifications: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {skill: 0 for skill in SKILL_NAMES}
    for row in classifications:
        skill = row.get("skill_gap")
        if skill in counts:
            counts[str(skill)] += 1
    return counts


def _dominant_skill_gap(skill_gap_counts: Mapping[str, int]) -> str:
    if not any(skill_gap_counts.values()):
        return "none"
    return max(
        SKILL_NAMES, key=lambda skill: (skill_gap_counts.get(skill, 0), -SKILL_NAMES.index(skill))
    )


def _honest_verdict(*, dominant_gap: str, skill_split_claim_allowed: bool) -> str:
    if not skill_split_claim_allowed:
        return "no_exp1353_certificate_cases_available"
    if dominant_gap == "none":
        return "logic_skill_split_supported_no_measured_gap"
    return f"logic_skill_split_supported_{dominant_gap}_dominates_exp1353"


def _base_artifact(
    *,
    project_root: Path,
    run_date: str,
    status: str = "complete",
) -> dict[str, Any]:
    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "status": status,
        "certificate_cases_used": 0,
        "symbolization_pass_rate": 0.0,
        "countermodel_pass_rate": 0.0,
        "validity_pass_rate": 0.0,
        "z3_verified_case_count": 0,
        "dominant_skill_gap": "none",
        "skill_split_claim_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiments": ["exp1353"],
            "spec": "REQ-VERIFY-1354",
        },
    }


def _expected_state(row: Mapping[str, Any]) -> str:
    return str(row.get("expected_state") or "").upper()


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_observer is not None:
        write_observer(path, payload)


def main() -> None:  # pragma: no cover - thin CLI wrapper covered through run_experiment.
    run_experiment(project_root=Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
