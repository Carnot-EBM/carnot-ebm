"""Exp 1368 FoVer-aligned LogicSkills certificate skill audit.

Spec: REQ-VERIFY-1368,
      SCENARIO-VERIFY-1368
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from z3 import Bool, Not, Solver, sat, unsat

from carnot.reporting.triggered_certificate_v7_truncproof_sota import (
    bounded_certificate_suite,
)


DEFAULT_RUN_DATE = "20260505"
DEFAULT_EXP1366_PATH = Path(
    "results/experiment_1366_certificate_v8_tag_first_prefix_injection_crane.json"
)
DEFAULT_OUTPUT_PATH = Path("results/experiment_1368_fover_aligned_logicskills_skill_audit.json")
ARTIFACT_NAME = "experiment_1368_fover_aligned_logicskills_skill_audit"
SCHEMA_VERSION = 1
EXP1366_PARSE_GATE = 0.75
SKILL_GAPS = ("symbolization", "countermodel", "validity", "unknown")
FOVER_ANALOGS = {
    "symbolization": "formalization_failure",
    "validity": "entailment_failure",
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "certificate_cases_used",
    "symbolization_pass_rate",
    "countermodel_pass_rate",
    "validity_pass_rate",
    "z3_verified_case_count",
    "fover_symbolization_alignment",
    "fover_validity_alignment",
    "dominant_skill_gap",
    "fover_training_data_applicable",
    "honest_verdict",
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def build_fover_aligned_logicskills_skill_audit_artifact(
    *,
    exp1366_artifact: Mapping[str, Any],
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    parse_gate: float = EXP1366_PARSE_GATE,
) -> dict[str, Any]:
    """Build the replay-only Exp 1368 audit without calling a model.

    The purpose of this audit is narrower than a new certificate run: Exp 1366
    already paid for SOTA generation and cleared the tag-first parse gate.  This
    builder only reuses those terminal rows, checks the small CNF fixtures with
    local Z3 when the formula is extractable, and translates any remaining
    LogicSkills failure into the FoVer-style tool-label buckets that could
    provide future training data.
    """

    root = Path(project_root)
    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")
    parse_rate = _float(exp1366_artifact.get("certificate_parse_rate"))
    artifact["source_context"] = _source_context(exp1366_artifact, parse_rate, parse_gate)

    if parse_rate < parse_gate:
        blocker = f"exp1366_parse_gate_failed:{parse_rate:g}_lt_{parse_gate:g}"
        artifact.update(
            {
                "terminal_blocker": blocker,
                "honest_verdict": "exp1366_parse_gate_failed_no_skill_audit",
                "measurement_note": (
                    "Exp 1366 certificate_parse_rate did not satisfy the >= 0.75 gate, "
                    "so its rows are not used as certificate evidence for this audit."
                ),
            }
        )
        return artifact

    rows = _certificate_rows(exp1366_artifact)
    classifications = [_classification_record(row) for row in rows]
    skill_failure_counts = _skill_failure_counts(classifications)
    total_skill_failures = sum(skill_failure_counts.values())
    z3_checks = _z3_checks(rows)
    dominant_gap = _dominant_skill_gap(skill_failure_counts)
    training_data_applicable = _fover_training_data_applicable(
        dominant_gap=dominant_gap,
        skill_failure_counts=skill_failure_counts,
    )

    artifact.update(
        {
            "certificate_cases_used": len(rows),
            "symbolization_pass_rate": _symbolization_pass_rate(rows),
            "countermodel_pass_rate": _countermodel_pass_rate(rows),
            "validity_pass_rate": _validity_pass_rate(rows),
            "z3_verified_case_count": sum(1 for check in z3_checks if check["verified"]),
            "fover_symbolization_alignment": _rate(
                skill_failure_counts["symbolization"], total_skill_failures
            ),
            "fover_validity_alignment": _rate(
                skill_failure_counts["validity"], total_skill_failures
            ),
            "dominant_skill_gap": dominant_gap,
            "fover_training_data_applicable": training_data_applicable,
            "honest_verdict": _honest_verdict(
                dominant_gap=dominant_gap,
                training_data_applicable=training_data_applicable,
                rows_used=len(rows),
            ),
            "classification_rows": classifications,
            "classification_counts": _classification_counts(classifications),
            "skill_failure_counts": skill_failure_counts,
            "z3_checks": z3_checks,
            "fover_alignment_denominator": total_skill_failures,
            "fover_taxonomy_mapping": {
                "symbolization": FOVER_ANALOGS["symbolization"],
                "validity": FOVER_ANALOGS["validity"],
                "countermodel": "no_direct_published_fover_skill_bucket",
                "unknown": "no_direct_published_fover_skill_bucket",
            },
            "fover_published_distribution_reference": {
                "source": "arXiv:2505.15960 Table 2(b)",
                "formal_logic_raw_step_error_rates": [0.437, 0.413],
                "formal_proof_raw_step_error_rates": [0.132, 0.141],
            },
            "measurement_note": (
                "Replay-only audit over Exp 1366 certificate rows. FoVer alignment "
                "fractions use observed Carnot skill failures as the denominator; "
                "no fresh LLM inference is run."
            ),
        }
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    exp1366_path: str | Path = DEFAULT_EXP1366_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Write an in-progress artifact first, then persist the completed audit."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    _write_json(
        output,
        _base_artifact(project_root=root, run_date=run_date, status="in_progress"),
        write_observer=write_observer,
    )
    artifact = build_fover_aligned_logicskills_skill_audit_artifact(
        exp1366_artifact=_read_json(_resolve(root, exp1366_path)),
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _classification_record(row: Mapping[str, Any]) -> dict[str, Any]:
    category = _logicskills_category(row)
    skill_gap = _skill_gap_from_category(category)
    return {
        "case_id": str(row.get("case_id") or ""),
        "expected_state": _expected_state(row),
        "logicskills_category": category,
        "skill_gap": skill_gap,
        "fover_error_taxonomy": FOVER_ANALOGS.get(skill_gap or ""),
        "evidence": _evidence(row, category),
        "parseable": bool(row.get("parseable")),
        "truthful": bool(row.get("truthful")),
        "unknown_preserved": bool(row.get("unknown_preserved")),
    }


def _logicskills_category(row: Mapping[str, Any]) -> str:
    if not bool(row.get("parseable")):
        return "symbolization_failure"
    if _expected_state(row) == "UNKNOWN" and not bool(row.get("unknown_preserved")):
        return "unknown"
    if _has_countermodel_evidence(row):
        return "countermodel_failure"
    if not bool(row.get("truthful")):
        return "validity_failure"
    return "pass"


def _skill_gap_from_category(category: str) -> str | None:
    if category == "pass":
        return None
    if category == "symbolization_failure":
        return "symbolization"
    if category == "countermodel_failure":
        return "countermodel"
    if category == "validity_failure":
        return "validity"
    return "unknown"


def _evidence(row: Mapping[str, Any], category: str) -> str:
    errors = [str(error) for error in row.get("errors") or []]
    if category == "pass":
        return "certificate row is parseable and truth-preserving"
    if category == "symbolization_failure":
        return "certificate rejected before semantic checking: " + ", ".join(
            errors or ["unparseable"]
        )
    if category == "countermodel_failure":
        return "countermodel or witness evidence failed: " + ", ".join(errors)
    if category == "validity_failure":
        return "parseable certificate did not preserve the expected verifier state"
    return "UNKNOWN case parsed without preserving UNKNOWN"


def _z3_checks(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    case_prompts = {case.case_id: case.prompt for case in bounded_certificate_suite()}
    checks: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        prompt = str(row.get("prompt") or case_prompts.get(case_id) or "")
        solver_state = _z3_expected_state_from_prompt(prompt)
        if solver_state is None:
            continue
        expected = _expected_state(row)
        checks.append(
            {
                "case_id": case_id,
                "expected_state": expected,
                "formula_source": "certificate_fixture_prompt",
                "solver_state": solver_state,
                "verified": solver_state == expected,
            }
        )
    return checks


def _z3_expected_state_from_prompt(prompt: str) -> str | None:
    x1 = Bool("x1")
    solver = Solver()
    normalised = " ".join(prompt.split()).lower()
    if "cnf: (x1) and (not x1)" in normalised:
        solver.add(x1, Not(x1))
        return "UNSAT" if solver.check() == unsat else "SAT"
    if "cnf: (x1)" in normalised:
        solver.add(x1)
        return "SAT" if solver.check() == sat else "UNSAT"
    return None


def _symbolization_pass_rate(rows: list[Mapping[str, Any]]) -> float:
    return _rate(sum(1 for row in rows if bool(row.get("parseable"))), len(rows))


def _countermodel_pass_rate(rows: list[Mapping[str, Any]]) -> float:
    applicable = [row for row in rows if _countermodel_applicable(row)]
    passes = sum(1 for row in applicable if bool(row.get("truthful")))
    return _rate(passes, len(applicable))


def _countermodel_applicable(row: Mapping[str, Any]) -> bool:
    if not bool(row.get("parseable")):
        return False
    return _expected_state(row) == "SAT" or _has_countermodel_evidence(row)


def _validity_pass_rate(rows: list[Mapping[str, Any]]) -> float:
    applicable = [row for row in rows if bool(row.get("parseable"))]
    passes = sum(1 for row in applicable if bool(row.get("truthful")))
    return _rate(passes, len(applicable))


def _has_countermodel_evidence(row: Mapping[str, Any]) -> bool:
    errors = [str(error).lower() for error in row.get("errors") or []]
    return any("countermodel" in error or "witness" in error for error in errors)


def _certificate_rows(exp1366_artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = exp1366_artifact.get("certificate_rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _classification_counts(classifications: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in classifications:
        label = str(row["logicskills_category"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def _skill_failure_counts(classifications: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {skill: 0 for skill in SKILL_GAPS}
    for row in classifications:
        skill = row.get("skill_gap")
        if skill in counts:
            counts[str(skill)] += 1
    return counts


def _dominant_skill_gap(skill_failure_counts: Mapping[str, int]) -> str:
    if not any(skill_failure_counts.values()):
        return "none"
    return max(
        SKILL_GAPS,
        key=lambda skill: (skill_failure_counts.get(skill, 0), -SKILL_GAPS.index(skill)),
    )


def _fover_training_data_applicable(
    *,
    dominant_gap: str,
    skill_failure_counts: Mapping[str, int],
) -> bool:
    return dominant_gap in FOVER_ANALOGS and skill_failure_counts.get(dominant_gap, 0) > 0


def _honest_verdict(
    *,
    dominant_gap: str,
    training_data_applicable: bool,
    rows_used: int,
) -> str:
    if rows_used == 0:
        return "no_exp1366_certificate_cases_available"
    if dominant_gap == "none":
        return "fover_aligned_logicskills_audit_no_skill_gap"
    if training_data_applicable:
        return f"fover_aligned_logicskills_audit_{dominant_gap}_gap_has_training_analog"
    return f"fover_aligned_logicskills_audit_{dominant_gap}_gap_no_direct_training_analog"


def _source_context(
    exp1366_artifact: Mapping[str, Any],
    parse_rate: float,
    parse_gate: float,
) -> dict[str, Any]:
    return {
        "source_experiment": "exp1366",
        "exp1366_status": exp1366_artifact.get("status"),
        "exp1366_certificate_parse_rate": parse_rate,
        "required_parse_gate": parse_gate,
        "parse_gate_passed": parse_rate >= parse_gate,
        "exp1366_honest_verdict": exp1366_artifact.get("honest_verdict"),
    }


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
        "fover_symbolization_alignment": 0.0,
        "fover_validity_alignment": 0.0,
        "dominant_skill_gap": "none",
        "fover_training_data_applicable": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "classification_rows": [],
        "classification_counts": {},
        "skill_failure_counts": {skill: 0 for skill in SKILL_GAPS},
        "z3_checks": [],
        "terminal_blocker": None,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiments": ["exp1366"],
            "spec": "REQ-VERIFY-1368",
        },
    }


def _expected_state(row: Mapping[str, Any]) -> str:
    return str(row.get("expected_state") or "").upper()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
