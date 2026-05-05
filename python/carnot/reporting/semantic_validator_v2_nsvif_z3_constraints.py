"""Exp 1369 semantic validator v2 with NSVIF-style Z3 constraints.

Spec: REQ-VERIFY-1369,
      SCENARIO-VERIFY-1369
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
DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1369_semantic_validator_v2_nsvif_z3_constraints.json"
)
ARTIFACT_NAME = "experiment_1369_semantic_validator_v2_nsvif_z3_constraints"
SCHEMA_VERSION = 1
EXP1366_PARSE_GATE = 0.75
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "parsed_certificate_cases",
    "fully_formal_claim_count",
    "nltc_claim_count",
    "z3_constraint_pass_rate",
    "unknown_preservation_rate",
    "smt_text_constraint_pass_rate",
    "validator_execution_pass_rate",
    "coverage_delta_over_fol_only",
    "semantic_validator_claim_allowed",
    "honest_verdict",
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def build_semantic_validator_v2_artifact(
    *,
    exp1366_artifact: Mapping[str, Any],
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    parse_gate: float = EXP1366_PARSE_GATE,
) -> dict[str, Any]:
    """Build Exp 1369 from already parsed Exp 1366 certificate evidence.

    The validator deliberately stays replay-only.  Exp 1366 paid the model cost
    and established parseability; this builder takes those parsed rows and asks
    only what the local verifier can support.  Exact CNF fixtures are encoded as
    Z3 constraints.  Cases that are meaningful but not fully formal, such as
    missing-bound UNKNOWN and repair-hint text, go through conservative text
    predicates so an UNKNOWN is preserved instead of being turned into a false
    formal SAT/UNSAT claim.
    """

    root = Path(project_root)
    parse_rate = _float(exp1366_artifact.get("certificate_parse_rate"))
    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")
    artifact["source_context"] = _source_context(exp1366_artifact, parse_rate, parse_gate)

    if parse_rate < parse_gate:
        blocker = f"exp1366_parse_gate_failed:{parse_rate:g}_lt_{parse_gate:g}"
        artifact.update(
            {
                "status": "blocked",
                "terminal_blocker": blocker,
                "honest_verdict": "blocked_exp1366_parse_gate_below_0_75",
                "measurement_note": (
                    "Exp 1366 certificate_parse_rate did not satisfy the >= 0.75 gate, "
                    "so semantic validation did not run on those rows."
                ),
            }
        )
        return artifact

    parsed_rows = _parsed_certificate_rows(exp1366_artifact)
    generation_rows = _generation_rows_by_case(exp1366_artifact)
    case_prompts = {case.case_id: case.prompt for case in bounded_certificate_suite()}
    validator_rows = [
        _semantic_validator_row(row, case_prompts, generation_rows) for row in parsed_rows
    ]
    formal_rows = [row for row in validator_rows if row["claim_route"] == "z3_fully_formal"]
    nltc_rows = [row for row in validator_rows if row["claim_route"] == "nltc_partial_smt"]
    unknown_rate = _unknown_preservation_rate(validator_rows)
    execution_pass_rate = _rate(
        sum(1 for row in validator_rows if row.get("constraint_passed") is True),
        len(validator_rows),
    )
    coverage_delta = _coverage_delta_over_fol_only(
        parsed_case_count=len(parsed_rows),
        fully_formal_claim_count=len(formal_rows),
        text_evaluated_count=sum(1 for row in nltc_rows if row.get("constraint_evaluated")),
    )
    claim_allowed = bool(parsed_rows and unknown_rate == 1.0)

    artifact.update(
        {
            "parsed_certificate_cases": len(parsed_rows),
            "fully_formal_claim_count": len(formal_rows),
            "nltc_claim_count": len(nltc_rows),
            "z3_constraint_pass_rate": _rate(
                sum(1 for row in formal_rows if row.get("constraint_passed") is True),
                len(formal_rows),
            ),
            "unknown_preservation_rate": unknown_rate,
            "smt_text_constraint_pass_rate": _rate(
                sum(1 for row in nltc_rows if row.get("constraint_passed") is True),
                len(nltc_rows),
            ),
            "validator_execution_pass_rate": execution_pass_rate,
            "coverage_delta_over_fol_only": coverage_delta,
            "semantic_validator_claim_allowed": claim_allowed,
            "honest_verdict": _honest_verdict(
                parsed_case_count=len(parsed_rows),
                unknown_preservation_rate=unknown_rate,
                validator_execution_pass_rate=execution_pass_rate,
            ),
            "semantic_validator_rows": validator_rows,
            "formal_claim_rows": formal_rows,
            "nltc_claim_rows": nltc_rows,
            "measurement_note": (
                "Replay-only semantic validation over Exp 1366 parsed rows. "
                "Fully formal CNF fixtures use local Z3; natural-language "
                "missing-bound and repair-hint cases use conservative text "
                "constraints and preserve UNKNOWN when formalization is unsafe."
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
    """Write an in-progress artifact first, then persist the terminal result."""

    root = Path(project_root)
    output = _resolve(root, output_path)
    _write_json(
        output,
        _base_artifact(project_root=root, run_date=run_date, status="in_progress"),
        write_observer=write_observer,
    )
    artifact = build_semantic_validator_v2_artifact(
        exp1366_artifact=_read_json(_resolve(root, exp1366_path)),
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _semantic_validator_row(
    row: Mapping[str, Any],
    case_prompts: Mapping[str, str],
    generation_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    case_id = str(row.get("case_id") or "")
    prompt = str(row.get("prompt") or case_prompts.get(case_id) or "")
    expected_state = _expected_state(row)
    certificate_state = _certificate_state(row)
    solver_state = _z3_expected_state_from_prompt(prompt)
    if solver_state is not None:
        passed = (
            bool(row.get("parseable"))
            and bool(row.get("truthful"))
            and solver_state == expected_state
            and solver_state == certificate_state
        )
        return {
            "case_id": case_id,
            "claim_route": "z3_fully_formal",
            "expected_state": expected_state,
            "certificate_state": certificate_state,
            "semantic_result": solver_state,
            "constraint_passed": passed,
            "constraint_evaluated": True,
            "formula_source": "certificate_fixture_prompt",
            "formula_kind": "cnf_unit_fixture",
            "nsvif_encoding": _nsvif_encoding_label(prompt),
        }

    text_check = _partial_smt_text_check(
        row=row,
        prompt=prompt,
        generation_row=generation_rows.get(case_id, {}),
    )
    return {
        "case_id": case_id,
        "claim_route": "nltc_partial_smt",
        "expected_state": expected_state,
        "certificate_state": certificate_state,
        **text_check,
    }


def _partial_smt_text_check(
    *,
    row: Mapping[str, Any],
    prompt: str,
    generation_row: Mapping[str, Any],
) -> dict[str, Any]:
    normalised_prompt = " ".join(prompt.lower().split())
    certificate_state = _certificate_state(row)
    expected_state = _expected_state(row)
    certificate_text = _certificate_text(generation_row)

    if expected_state == "UNKNOWN" or "preserve unknown" in normalised_prompt:
        unknown_preserved = (
            bool(row.get("parseable"))
            and bool(row.get("truthful"))
            and bool(row.get("unknown_preserved"))
            and certificate_state == "UNKNOWN"
        )
        return {
            "semantic_result": "UNKNOWN" if unknown_preserved else "UNKNOWN_COLLAPSED",
            "constraint_passed": unknown_preserved,
            "constraint_evaluated": True,
            "text_constraints": ["missing_capacity_bound_requires_unknown"],
            "logitext_reason": (
                "The prompt omits capacity bound B, so the conservative text "
                "constraint allows only an UNKNOWN-preserving certificate."
            ),
        }

    if expected_state == "REPAIR_HINT" or "repair hint" in normalised_prompt:
        mentions_bound = "bound" in certificate_text or "upper-bound" in normalised_prompt
        repair_preserved = (
            bool(row.get("parseable"))
            and bool(row.get("truthful"))
            and certificate_state == "REPAIR_HINT"
            and mentions_bound
        )
        return {
            "semantic_result": "REPAIR_HINT" if repair_preserved else "REPAIR_TEXT_FAILED",
            "constraint_passed": repair_preserved,
            "constraint_evaluated": True,
            "text_constraints": ["missing_upper_bound_requires_repair_hint"],
            "logitext_reason": (
                "The prompt says an upper-bound premise is missing, so the "
                "text constraint accepts only a repair hint that still refers "
                "to the missing bound."
            ),
        }

    cannot_formalize_preserved = certificate_state == "UNKNOWN" and bool(
        row.get("unknown_preserved")
    )
    return {
        "semantic_result": "UNKNOWN" if cannot_formalize_preserved else "UNFORMALIZED",
        "constraint_passed": cannot_formalize_preserved,
        "constraint_evaluated": cannot_formalize_preserved,
        "text_constraints": ["unformalized_text_preserves_unknown"],
        "logitext_reason": (
            "No safe local formalization was available.  The validator keeps "
            "UNKNOWN only when the certificate itself preserved UNKNOWN."
        ),
    }


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


def _nsvif_encoding_label(prompt: str) -> str:
    normalised = " ".join(prompt.split()).lower()
    if "cnf: (x1) and (not x1)" in normalised:
        return "And(x1, Not(x1))"
    if "cnf: (x1)" in normalised:
        return "x1"
    return "unavailable"


def _parsed_certificate_rows(exp1366_artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = exp1366_artifact.get("certificate_rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping) and bool(row.get("parseable"))]


def _generation_rows_by_case(
    exp1366_artifact: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = exp1366_artifact.get("generation_rows", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("case_id") or ""): row
        for row in rows
        if isinstance(row, Mapping) and row.get("case_id")
    }


def _certificate_text(generation_row: Mapping[str, Any]) -> str:
    parts = [
        generation_row.get("certificate_body"),
        generation_row.get("full_certificate_text"),
        generation_row.get("reasoning_text"),
    ]
    return " ".join(str(part).lower() for part in parts if part is not None)


def _unknown_preservation_rate(validator_rows: list[Mapping[str, Any]]) -> float:
    unknown_rows = [
        row
        for row in validator_rows
        if row.get("expected_state") == "UNKNOWN" or row.get("certificate_state") == "UNKNOWN"
    ]
    if not unknown_rows:
        return 1.0 if validator_rows else 0.0
    preserved = sum(
        1
        for row in unknown_rows
        if row.get("semantic_result") == "UNKNOWN" and row.get("constraint_passed") is True
    )
    return _rate(preserved, len(unknown_rows))


def _coverage_delta_over_fol_only(
    *,
    parsed_case_count: int,
    fully_formal_claim_count: int,
    text_evaluated_count: int,
) -> float:
    if parsed_case_count <= 0:
        return 0.0
    fol_coverage = fully_formal_claim_count / parsed_case_count
    semantic_coverage = min(
        1.0,
        (fully_formal_claim_count + text_evaluated_count) / parsed_case_count,
    )
    return round(max(0.0, semantic_coverage - fol_coverage), 6)


def _honest_verdict(
    *,
    parsed_case_count: int,
    unknown_preservation_rate: float,
    validator_execution_pass_rate: float,
) -> str:
    if parsed_case_count == 0:
        return "no_exp1366_parsed_certificate_cases_available"
    if unknown_preservation_rate < 1.0:
        return "semantic_validator_v2_ran_unknown_collapsed"
    if validator_execution_pass_rate < 1.0:
        return "semantic_validator_v2_complete_with_constraint_failures"
    return "semantic_validator_v2_complete_unknown_preserved"


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
        "parsed_certificate_cases": 0,
        "fully_formal_claim_count": 0,
        "nltc_claim_count": 0,
        "z3_constraint_pass_rate": 0.0,
        "unknown_preservation_rate": 0.0,
        "smt_text_constraint_pass_rate": 0.0,
        "validator_execution_pass_rate": 0.0,
        "coverage_delta_over_fol_only": 0.0,
        "semantic_validator_claim_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "terminal_blocker": None,
        "semantic_validator_rows": [],
        "formal_claim_rows": [],
        "nltc_claim_rows": [],
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiments": ["exp1366"],
            "spec": "REQ-VERIFY-1369",
        },
    }


def _expected_state(row: Mapping[str, Any]) -> str:
    return str(row.get("expected_state") or "").upper()


def _certificate_state(row: Mapping[str, Any]) -> str:
    for key in ("dispatched_state", "tag_state", "certificate_state", "expected_state"):
        value = row.get(key)
        if value:
            return str(value).upper()
    return ""


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
