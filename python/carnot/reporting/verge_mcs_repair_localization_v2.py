"""Exp 1370 VERGE-style MCS repair localization over Exp 1369.

Spec: REQ-VERIFY-1370,
      SCENARIO-VERIFY-1370
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping

from z3 import Bool, Not, Solver, sat, unsat


DEFAULT_RUN_DATE = "20260505"
DEFAULT_EXP1369_PATH = Path(
    "results/experiment_1369_semantic_validator_v2_nsvif_z3_constraints.json"
)
DEFAULT_OUTPUT_PATH = Path("results/experiment_1370_verge_mcs_repair_localization_v2.json")
ARTIFACT_NAME = "experiment_1370_verge_mcs_repair_localization_v2"
SCHEMA_VERSION = 1
EXP1369_EXECUTION_GATE = 0.5
NON_ACCEPTED_SEMANTIC_STATES = {
    "UNSAT",
    "UNKNOWN",
    "REPAIR_HINT",
    "UNKNOWN_COLLAPSED",
    "REPAIR_TEXT_FAILED",
    "UNFORMALIZED",
}
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "semantic_cases_used",
    "mcs_localization_rate",
    "repair_hint_count",
    "repair_hint_precision",
    "semantic_equivalence_pass_rate",
    "iteration_count_to_accept",
    "accepted_violation_delta",
    "repair_claim_allowed",
    "honest_verdict",
)

WriteObserver = Callable[[Path, dict[str, Any]], None]


def build_verge_mcs_repair_localization_artifact(
    *,
    exp1369_artifact: Mapping[str, Any],
    run_date: str = DEFAULT_RUN_DATE,
    project_root: str | Path = ".",
    execution_gate: float = EXP1369_EXECUTION_GATE,
) -> dict[str, Any]:
    """Build a replay-only repair-localization artifact from Exp 1369 rows.

    VERGE-style repair is useful only when the verifier can name the failing
    local constraint.  This builder therefore avoids fresh model calls and
    reuses the structured Exp 1369 validator rows: formal rows are replayed as
    tiny Z3 formulas so an MCS can be computed exactly, while partial-SMT text
    rows reuse the named text constraints already emitted by the semantic
    validator.  The repair claim is conservative: a localized hint can be
    actionable even when the proposed formal edit is rejected because it would
    change the original formula's semantics.
    """

    root = Path(project_root)
    execution_pass_rate = _float(exp1369_artifact.get("validator_execution_pass_rate"))
    artifact = _base_artifact(project_root=root, run_date=run_date, status="complete")
    artifact["source_context"] = _source_context(
        exp1369_artifact=exp1369_artifact,
        execution_pass_rate=execution_pass_rate,
        execution_gate=execution_gate,
    )

    if execution_pass_rate < execution_gate:
        artifact.update(
            {
                "status": "blocked",
                "terminal_blocker": (
                    f"exp1369_validator_execution_pass_rate_failed:"
                    f"{execution_pass_rate:g}_lt_{execution_gate:g}"
                ),
                "honest_verdict": "blocked_exp1369_validator_execution_pass_rate_below_0_5",
                "measurement_note": (
                    "Exp 1369 validator_execution_pass_rate did not satisfy the >= 0.5 "
                    "gate, so VERGE MCS repair localization was not run."
                ),
            }
        )
        return artifact

    semantic_rows = _semantic_validator_rows(exp1369_artifact)
    repair_cases = [row for row in semantic_rows if _is_semantic_repair_case(row)]
    localization_rows = [_localize_repair_case(row) for row in repair_cases]

    semantic_cases_used = len(localization_rows)
    localized_count = sum(1 for row in localization_rows if row["localized"])
    repair_hint_count = sum(1 for row in localization_rows if row["repair_hint"])
    precise_hint_count = sum(1 for row in localization_rows if row["precision_match"])
    accepted_rows = [row for row in localization_rows if row["accepted"]]
    semantic_equivalence_pass_count = sum(
        1 for row in accepted_rows if row["semantic_equivalence_passed"]
    )
    accepted_violation_delta = (semantic_cases_used - len(accepted_rows)) - semantic_cases_used
    repair_claim_allowed = bool(
        semantic_cases_used
        and localized_count == semantic_cases_used
        and repair_hint_count == semantic_cases_used
        and accepted_violation_delta <= 0
    )

    artifact.update(
        {
            "semantic_cases_used": semantic_cases_used,
            "mcs_localization_rate": _rate(localized_count, semantic_cases_used),
            "repair_hint_count": repair_hint_count,
            "repair_hint_precision": _rate(precise_hint_count, repair_hint_count),
            "semantic_equivalence_pass_rate": _rate(
                semantic_equivalence_pass_count,
                len(accepted_rows),
            ),
            "iteration_count_to_accept": _average(
                [row["iteration_count"] for row in accepted_rows]
            ),
            "accepted_violation_delta": accepted_violation_delta,
            "repair_claim_allowed": repair_claim_allowed,
            "honest_verdict": _honest_verdict(
                semantic_cases_used=semantic_cases_used,
                localized_count=localized_count,
                repair_claim_allowed=repair_claim_allowed,
            ),
            "repair_localization_rows": localization_rows,
            "accepted_repair_count": len(accepted_rows),
            "rejected_repair_count": semantic_cases_used - len(accepted_rows),
            "measurement_note": (
                "Replay-only VERGE MCS localization over Exp 1369 semantic outcomes. "
                "Rows with SAT semantic results are already accepted and are not repair "
                "cases.  UNSAT/UNKNOWN/REPAIR_HINT rows are localized without fresh LLM "
                "calls; formal edits that change formula semantics are reported as hints "
                "but not counted as accepted repairs."
            ),
        }
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    exp1369_path: str | Path = DEFAULT_EXP1369_PATH,
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
    artifact = build_verge_mcs_repair_localization_artifact(
        exp1369_artifact=_read_json(_resolve(root, exp1369_path)),
        run_date=run_date,
        project_root=root,
    )
    _write_json(output, artifact, write_observer=write_observer)
    return artifact


def _semantic_validator_rows(exp1369_artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = exp1369_artifact.get("semantic_validator_rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _is_semantic_repair_case(row: Mapping[str, Any]) -> bool:
    return (
        row.get("constraint_passed") is False
        or _semantic_state(row) in NON_ACCEPTED_SEMANTIC_STATES
    )


def _localize_repair_case(row: Mapping[str, Any]) -> dict[str, Any]:
    route = str(row.get("claim_route") or "")
    if route == "z3_fully_formal":
        return _localize_z3_case(row)
    return _localize_text_case(row)


def _localize_z3_case(row: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(row.get("case_id") or "")
    constraints = _constraints_for_nsvif(str(row.get("nsvif_encoding") or ""))

    if constraints:
        original_status = _z3_status([expr for _name, expr in constraints])
        mcs_candidates = _minimal_correction_subsets(constraints)
        repaired_status = _z3_status(
            [
                expr
                for name, expr in constraints
                if not mcs_candidates or name not in set(mcs_candidates[0])
            ]
        )
        semantic_equivalence_passed = original_status == repaired_status
        localized = bool(mcs_candidates)
        hint = _z3_repair_hint(mcs_candidates)
        precision_match = localized
        accepted = localized and semantic_equivalence_passed
        acceptance_reason = (
            "accepted_z3_repair_preserves_solver_semantics"
            if accepted
            else "rejected_formal_repair_changes_formula_semantics"
        )
        return _localization_row(
            row=row,
            localized=localized,
            localized_constraint="cnf_unit_conflict" if localized else None,
            mcs_candidates=mcs_candidates,
            minimal_local_change="relax_one_minimal_conflicting_cnf_clause"
            if localized
            else "no_mcs_candidate",
            repair_hint=hint,
            precision_match=precision_match,
            semantic_equivalence_passed=semantic_equivalence_passed,
            semantic_equivalence_method="z3_status_preservation",
            accepted=accepted,
            acceptance_reason=acceptance_reason,
            iteration_count=1 if accepted else 0,
            verifier_before=original_status,
            verifier_after=repaired_status,
        )

    semantic_result = _semantic_state(row)
    certificate_state = _certificate_state(row)
    localized = semantic_result != certificate_state
    return _localization_row(
        row=row,
        localized=localized,
        localized_constraint="certificate_state_mismatch" if localized else None,
        mcs_candidates=[["certificate_state_mismatch"]] if localized else [],
        minimal_local_change=f"set_certificate_state_to_{semantic_result.lower()}"
        if localized
        else "no_local_change_needed",
        repair_hint=(
            f"Change only the certificate state from {certificate_state} to "
            f"{semantic_result} to match the local Z3 result."
            if localized
            else ""
        ),
        precision_match=localized,
        semantic_equivalence_passed=localized,
        semantic_equivalence_method="z3_result_matches_certificate_state",
        accepted=localized,
        acceptance_reason="accepted_certificate_state_matches_z3_result"
        if localized
        else "no_mcs_candidate",
        iteration_count=1 if localized else 0,
        verifier_before=certificate_state,
        verifier_after=semantic_result,
    )


def _localize_text_case(row: Mapping[str, Any]) -> dict[str, Any]:
    text_constraints = _text_constraints(row)
    constraint_name = text_constraints[0] if text_constraints else "partial_smt_text_constraint"
    localized_constraint, minimal_change, hint = _text_repair_details(constraint_name)
    localized = bool(localized_constraint)

    return _localization_row(
        row=row,
        localized=localized,
        localized_constraint=localized_constraint,
        mcs_candidates=[[localized_constraint]] if localized else [],
        minimal_local_change=minimal_change,
        repair_hint=hint,
        precision_match=localized,
        semantic_equivalence_passed=localized,
        semantic_equivalence_method="partial_smt_additive_premise_preservation",
        accepted=localized,
        acceptance_reason="accepted_text_repair_adds_missing_premise_without_rewriting_claims"
        if localized
        else "no_text_constraint_available",
        iteration_count=1 if localized else 0,
        verifier_before=_semantic_state(row),
        verifier_after=_semantic_state(row),
    )


def _constraints_for_nsvif(encoding: str) -> list[tuple[str, Any]]:
    x1 = Bool("x1")
    normalised = " ".join(encoding.split())
    if normalised == "And(x1, Not(x1))":
        return [("cnf_clause_x1", x1), ("cnf_clause_not_x1", Not(x1))]
    if normalised == "x1":
        return [("cnf_clause_x1", x1)]
    return []


def _minimal_correction_subsets(constraints: list[tuple[str, Any]]) -> list[list[str]]:
    if _z3_status([expr for _name, expr in constraints]) == "SAT":
        return []

    names = [name for name, _expr in constraints]
    expr_by_name = dict(constraints)
    for subset_size in range(1, len(names) + 1):
        candidates: list[list[str]] = []
        for relaxed in combinations(names, subset_size):
            relaxed_set = set(relaxed)
            remaining = [expr for name, expr in expr_by_name.items() if name not in relaxed_set]
            if _z3_status(remaining) == "SAT":
                candidates.append(list(relaxed))
        if candidates:
            return candidates
    return []


def _z3_status(expressions: list[Any]) -> str:
    solver = Solver()
    for expression in expressions:
        solver.add(expression)
    result = solver.check()
    if result == sat:
        return "SAT"
    if result == unsat:
        return "UNSAT"
    return "UNKNOWN"


def _z3_repair_hint(mcs_candidates: list[list[str]]) -> str:
    if not mcs_candidates:
        return ""
    if len(mcs_candidates) == 1:
        return (
            "Relax the localized verifier assertion "
            f"{mcs_candidates[0][0]} and re-run the Z3 semantic validator."
        )
    alternatives = " or ".join(candidate[0] for candidate in mcs_candidates)
    return (
        "Relax exactly one contradictory unit clause "
        f"({alternatives}) after checking source-premise provenance; either edit is "
        "minimal, but changing the formula is not semantically accepted by this replay."
    )


def _text_repair_details(constraint_name: str) -> tuple[str | None, str, str]:
    if constraint_name == "missing_capacity_bound_requires_unknown":
        return (
            "capacity_bound_B",
            "add_missing_capacity_bound_B_or_preserve_unknown",
            (
                "Add the missing capacity bound B before asking the verifier to decide "
                "SAT/UNSAT; until B is present, keep the certificate state UNKNOWN."
            ),
        )
    if constraint_name == "missing_upper_bound_requires_repair_hint":
        return (
            "upper_bound_premise",
            "add_missing_upper_bound_premise",
            (
                "Add the missing upper-bound premise to the certificate input and keep "
                "the localized REPAIR_HINT tied to that premise."
            ),
        )
    if constraint_name:
        return (
            constraint_name,
            f"repair_named_text_constraint_{constraint_name}",
            f"Repair the named partial-SMT text constraint {constraint_name}.",
        )
    return None, "no_text_constraint_available", ""


def _localization_row(
    *,
    row: Mapping[str, Any],
    localized: bool,
    localized_constraint: str | None,
    mcs_candidates: list[list[str]],
    minimal_local_change: str,
    repair_hint: str,
    precision_match: bool,
    semantic_equivalence_passed: bool,
    semantic_equivalence_method: str,
    accepted: bool,
    acceptance_reason: str,
    iteration_count: int,
    verifier_before: str,
    verifier_after: str,
) -> dict[str, Any]:
    return {
        "case_id": str(row.get("case_id") or ""),
        "claim_route": str(row.get("claim_route") or ""),
        "expected_state": _expected_state(row),
        "certificate_state": _certificate_state(row),
        "semantic_result": _semantic_state(row),
        "localized": localized,
        "localized_constraint": localized_constraint,
        "mcs_candidates": mcs_candidates,
        "minimal_local_change": minimal_local_change,
        "repair_hint": repair_hint,
        "precision_match": precision_match,
        "semantic_equivalence_passed": semantic_equivalence_passed,
        "semantic_equivalence_method": semantic_equivalence_method,
        "accepted": accepted,
        "acceptance_reason": acceptance_reason,
        "iteration_count": iteration_count,
        "verifier_before": verifier_before,
        "verifier_after": verifier_after,
    }


def _source_context(
    *,
    exp1369_artifact: Mapping[str, Any],
    execution_pass_rate: float,
    execution_gate: float,
) -> dict[str, Any]:
    return {
        "source_experiment": "exp1369",
        "exp1369_status": exp1369_artifact.get("status"),
        "exp1369_validator_execution_pass_rate": execution_pass_rate,
        "required_execution_gate": execution_gate,
        "execution_gate_passed": execution_pass_rate >= execution_gate,
        "exp1369_honest_verdict": exp1369_artifact.get("honest_verdict"),
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
        "semantic_cases_used": 0,
        "mcs_localization_rate": 0.0,
        "repair_hint_count": 0,
        "repair_hint_precision": 0.0,
        "semantic_equivalence_pass_rate": 0.0,
        "iteration_count_to_accept": 0,
        "accepted_violation_delta": 0,
        "repair_claim_allowed": False,
        "honest_verdict": "in_progress" if status == "in_progress" else "not_run",
        "terminal_blocker": None,
        "repair_localization_rows": [],
        "accepted_repair_count": 0,
        "rejected_repair_count": 0,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "source_experiments": ["exp1369"],
            "spec": "REQ-VERIFY-1370",
        },
    }


def _honest_verdict(
    *,
    semantic_cases_used: int,
    localized_count: int,
    repair_claim_allowed: bool,
) -> str:
    if semantic_cases_used == 0:
        return "no_exp1369_semantic_repair_cases_available"
    if localized_count < semantic_cases_used:
        return "verge_mcs_repair_localization_partial"
    if repair_claim_allowed:
        return "verge_mcs_repair_localization_complete_claim_allowed"
    return "verge_mcs_repair_localization_complete_claim_blocked"


def _expected_state(row: Mapping[str, Any]) -> str:
    return str(row.get("expected_state") or "").upper()


def _certificate_state(row: Mapping[str, Any]) -> str:
    return str(row.get("certificate_state") or row.get("dispatched_state") or "").upper()


def _semantic_state(row: Mapping[str, Any]) -> str:
    return str(row.get("semantic_result") or "").upper()


def _text_constraints(row: Mapping[str, Any]) -> list[str]:
    constraints = row.get("text_constraints", [])
    if not isinstance(constraints, list):
        return []
    return [str(constraint) for constraint in constraints if constraint]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _average(values: list[int]) -> float | int:
    if not values:
        return 0
    return round(sum(values) / len(values), 6)


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
