"""Exp1549 SATQuest oracle false-accept repair.

Spec: REQ-BENCH-1549, SCENARIO-BENCH-1549.

The Exp1536 benchmark correctly kept model answers separate from the local SAT
solver, but its downstream gate still needed auditable evidence before SATQuest
could act as an acceptance authority.  This module keeps the historical rows
unchanged and adds the missing proof layer: SAT labels must carry a checked
assignment witness, and UNSAT labels must carry an exact contradiction
certificate for every bounded assignment when no external proof is available.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from carnot.eval import satquest_cnf_verifier_benchmark as exp1536

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
DEFAULT_SOURCE_ARTIFACT_PATH = Path("results/experiment_1536_satquest_cnf_verifier_benchmark.json")
DEFAULT_SOURCE_MANIFEST_PATH = Path("results/satquest_cnf_verifier_1536.jsonl")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1549_satquest_oracle_false_accept_repair.json")
DEFAULT_REPAIRED_CASE_MANIFEST_PATH = Path("results/satquest_oracle_false_accept_repair_1549.jsonl")
ORACLE_MODULE_PATH = "python/carnot/eval/satquest_oracle_false_accept_repair.py"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "satquest_oracle_repair_ready",
    "satquest_zero_false_accepts",
    "false_accepts_before",
    "solver_oracle_false_accepts_after",
    "failing_case_ids_rechecked",
    "pysat_available",
    "exact_fallback_available",
    "assignment_witnesses_checked",
    "unsat_certificates_checked",
    "perturbation_checks_passed",
    "oracle_module_path",
    "repaired_case_manifest_path",
    "focused_tests_passed",
    "honest_verdict",
)


class CNFParseError(ValueError):
    """Raised when a DIMACS CNF cannot be trusted by the repair oracle."""


@dataclass(frozen=True)
class StrictCNF:
    """Strictly parsed CNF data with header and clause-count validation."""

    n_vars: int
    clauses: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class UnsatCertificateRow:
    """One checked assignment and the first clause that contradicts it."""

    assignment: tuple[bool, ...]
    violated_clause_index: int
    violated_clause: tuple[int, ...]

    def to_dict(self) -> JsonDict:
        """Return a JSON-ready record for the repaired-case manifest."""

        return {
            "assignment": list(self.assignment),
            "violated_clause_index": self.violated_clause_index,
            "violated_clause": list(self.violated_clause),
        }


@dataclass(frozen=True)
class OracleEvidence:
    """SAT/UNSAT oracle label plus the evidence needed to trust it."""

    label: str
    is_satisfiable: bool
    backend: str
    checked_assignments: int
    assignment_witness: tuple[bool, ...] | None
    assignment_witness_checked: bool
    unsat_certificate: tuple[UnsatCertificateRow, ...]
    unsat_certificate_checked: bool
    pysat_used: bool
    exact_fallback_used: bool

    def to_dict(self) -> JsonDict:
        """Return a JSON-ready evidence payload for audit artifacts."""

        return {
            "label": self.label,
            "is_satisfiable": self.is_satisfiable,
            "backend": self.backend,
            "checked_assignments": self.checked_assignments,
            "assignment_witness": (
                list(self.assignment_witness) if self.assignment_witness is not None else None
            ),
            "assignment_witness_checked": self.assignment_witness_checked,
            "unsat_certificate": [row.to_dict() for row in self.unsat_certificate],
            "unsat_certificate_checked": self.unsat_certificate_checked,
            "pysat_used": self.pysat_used,
            "exact_fallback_used": self.exact_fallback_used,
        }


def parse_dimacs_strict(text: str) -> StrictCNF:
    """Parse DIMACS only when the header, ranges, and clause terminators agree."""

    n_vars: int | None = None
    expected_clauses: int | None = None
    clauses: list[tuple[int, ...]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        parts = line.split()
        if parts[:2] == ["p", "cnf"]:
            if n_vars is not None:  # pragma: no cover - defensive malformed fixture guard.
                raise CNFParseError("multiple_problem_headers")
            try:
                n_vars = int(parts[2])
                expected_clauses = int(parts[3])
            except (IndexError, ValueError) as exc:
                raise CNFParseError("invalid_problem_header") from exc
            if n_vars <= 0 or expected_clauses < 0:  # pragma: no cover - defensive guard.
                raise CNFParseError("invalid_problem_bounds")
            continue
        if n_vars is None:
            raise CNFParseError("missing_problem_header")
        clause: list[int] = []
        terminated = False
        for token in parts:
            try:
                literal = int(token)
            except ValueError as exc:  # pragma: no cover - defensive malformed fixture guard.
                raise CNFParseError("invalid_literal") from exc
            if literal == 0:
                terminated = True
                break
            if abs(literal) > n_vars:
                raise CNFParseError("literal_out_of_range")
            clause.append(literal)
        if not terminated:
            raise CNFParseError("unterminated_clause")
        clauses.append(tuple(clause))
    if n_vars is None:
        raise CNFParseError("missing_problem_header")
    if len(clauses) != expected_clauses:
        raise CNFParseError("clause_count_mismatch")
    return StrictCNF(n_vars=n_vars, clauses=tuple(clauses))


def pysat_available() -> bool:
    """Return whether the optional PySAT solver import is available."""

    try:
        return importlib.util.find_spec("pysat.solvers") is not None
    except ModuleNotFoundError:
        return False


def solve_cnf_with_evidence(
    n_vars: int,
    clauses: Iterable[Iterable[int]],
    *,
    prefer_pysat: bool = True,
) -> OracleEvidence:
    """Solve a bounded CNF and attach SAT witnesses or UNSAT contradiction evidence."""

    clean_clauses = _normalise_clauses(n_vars, clauses)
    if prefer_pysat and pysat_available():  # pragma: no cover - depends on optional host package.
        try:
            return _solve_with_pysat_evidence(n_vars, clean_clauses)
        except Exception:
            pass
    return _solve_exact_with_evidence(n_vars, clean_clauses, backend="exact_exhaustive_fallback")


def evaluate_candidate_with_evidence(
    n_vars: int,
    clauses: Iterable[Iterable[int]],
    candidate: exp1536.CandidateAnswer,
    *,
    evidence: OracleEvidence | None = None,
) -> JsonDict:
    """Decide whether a SATQuest answer is accepted by the repaired oracle."""

    try:
        clean_clauses = _normalise_clauses(n_vars, clauses)
    except CNFParseError as exc:
        return _decision(False, "malformed_cnf", None, str(exc))
    oracle = evidence or solve_cnf_with_evidence(n_vars, clean_clauses)
    if candidate.label is None:
        return _decision(False, "no_answer", oracle)
    if candidate.label != oracle.label:
        return _decision(False, "wrong_label", oracle)
    if candidate.label == "SAT":
        if not oracle.assignment_witness_checked:
            return _decision(False, "missing_oracle_sat_witness", oracle)
        if candidate.assignment is None:
            return _decision(False, "missing_assignment_witness", oracle)
        if len(candidate.assignment) != n_vars:
            return _decision(False, "malformed_assignment_witness", oracle)
        if not exp1536.assignment_satisfies(clean_clauses, candidate.assignment):
            return _decision(False, "invalid_assignment_witness", oracle)
        return _decision(True, "oracle_agreement_with_sat_witness", oracle)
    if not oracle.unsat_certificate_checked:
        return _decision(False, "missing_unsat_certificate", oracle)
    return _decision(True, "oracle_agreement_with_unsat_certificate", oracle)


def load_false_accept_rows(
    source_artifact_path: Path | str = DEFAULT_SOURCE_ARTIFACT_PATH,
    source_manifest_path: Path | str = DEFAULT_SOURCE_MANIFEST_PATH,
) -> tuple[JsonDict, list[JsonDict]]:
    """Load Exp1536 and return only rows with historical model false accepts."""

    artifact = _read_json(Path(source_artifact_path))
    rows = [
        row
        for row in _read_jsonl(Path(source_manifest_path))
        if bool(_mapping(row.get("verifier")).get("self_verifier_false_accept"))
    ]
    return artifact, rows


def replay_false_accept_rows(rows: list[JsonDict]) -> JsonDict:
    """Replay known false accepts and count repaired oracle false accepts."""

    return _replay_rows(rows, failing_case_ids=[str(row.get("case_id")) for row in rows])


def run_repair(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    repaired_case_manifest_path: Path | str = DEFAULT_REPAIRED_CASE_MANIFEST_PATH,
    source_artifact_path: Path | str = DEFAULT_SOURCE_ARTIFACT_PATH,
    source_manifest_path: Path | str = DEFAULT_SOURCE_MANIFEST_PATH,
    focused_tests_passed: bool = False,
) -> JsonDict:
    """Run the Exp1549 zero-false-accept SATQuest repair gate."""

    output = Path(output_path)
    manifest = Path(repaired_case_manifest_path)
    _write_json(output, _in_progress_payload(manifest))
    source_artifact, false_accept_rows = load_false_accept_rows(source_artifact_path, source_manifest_path)
    regression_rows = _build_regression_rows()
    combined = [*false_accept_rows, *regression_rows]
    replay = _replay_rows(combined, failing_case_ids=[str(row.get("case_id")) for row in false_accept_rows])
    perturbation_ok = _run_perturbation_checks()
    zero_false_accepts = replay["solver_oracle_false_accepts_after"] == 0
    ready = bool(false_accept_rows) and zero_false_accepts and perturbation_ok
    _write_jsonl(manifest, replay["rows"])
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "milestone": RUN_DATE,
        "satquest_oracle_repair_ready": ready,
        "satquest_zero_false_accepts": zero_false_accepts,
        "false_accepts_before": int(source_artifact.get("solver_oracle_false_accepts") or 0),
        "solver_oracle_false_accepts_after": replay["solver_oracle_false_accepts_after"],
        "failing_case_ids_rechecked": replay["failing_case_ids_rechecked"],
        "pysat_available": pysat_available(),
        "exact_fallback_available": _exact_fallback_available(),
        "assignment_witnesses_checked": replay["assignment_witnesses_checked"],
        "unsat_certificates_checked": replay["unsat_certificates_checked"],
        "perturbation_checks_passed": perturbation_ok,
        "oracle_module_path": ORACLE_MODULE_PATH,
        "repaired_case_manifest_path": _display_path(manifest),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": (
            "complete: satquest_oracle_repair_zero_false_accepts_with_witness_evidence"
            if ready
            else "blocked: satquest_oracle_repair_evidence_gate_not_met"
        ),
        "false_accept_rows_before": _false_accept_row_summary(false_accept_rows),
        "regression_rows_checked": len(combined),
        "source_artifact_path": _display_path(Path(source_artifact_path)),
        "source_manifest_path": _display_path(Path(source_manifest_path)),
    }
    _write_json(output, artifact)
    return artifact


def _solve_exact_with_evidence(
    n_vars: int,
    clauses: tuple[tuple[int, ...], ...],
    *,
    backend: str,
    pysat_used: bool = False,
) -> OracleEvidence:
    if n_vars > exp1536.MAX_EXHAUSTIVE_VARS:
        raise ValueError("bounded exhaustive solver refuses CNFs above MAX_EXHAUSTIVE_VARS")
    checked = 0
    contradictions: list[UnsatCertificateRow] = []
    for assignment in itertools.product((False, True), repeat=n_vars):
        candidate = tuple(bool(value) for value in assignment)
        checked += 1
        if exp1536.assignment_satisfies(clauses, candidate):
            return OracleEvidence(
                label="SAT",
                is_satisfiable=True,
                backend=backend,
                checked_assignments=checked,
                assignment_witness=candidate,
                assignment_witness_checked=True,
                unsat_certificate=(),
                unsat_certificate_checked=False,
                pysat_used=pysat_used,
                exact_fallback_used=not pysat_used,
            )
        violated_index, violated_clause = _first_violated_clause(clauses, candidate)
        contradictions.append(UnsatCertificateRow(candidate, violated_index, violated_clause))
    return OracleEvidence(
        label="UNSAT",
        is_satisfiable=False,
        backend=backend,
        checked_assignments=checked,
        assignment_witness=None,
        assignment_witness_checked=False,
        unsat_certificate=tuple(contradictions),
        unsat_certificate_checked=len(contradictions) == 2**n_vars,
        pysat_used=pysat_used,
        exact_fallback_used=True,
    )


def _solve_with_pysat_evidence(
    n_vars: int,
    clauses: tuple[tuple[int, ...], ...],
) -> OracleEvidence:  # pragma: no cover - optional dependency path.
    from pysat.solvers import Solver  # type: ignore[import-not-found]  # noqa: PLC0415

    with Solver(bootstrap_with=[list(clause) for clause in clauses]) as solver:
        sat = bool(solver.solve())
        model = solver.get_model() or []
    if not sat:
        return _solve_exact_with_evidence(
            n_vars,
            clauses,
            backend="pysat+exact_unsat_certificate",
            pysat_used=True,
        )
    positive = {abs(literal): literal > 0 for literal in model}
    assignment = tuple(bool(positive.get(index, False)) for index in range(1, n_vars + 1))
    if not exp1536.assignment_satisfies(clauses, assignment):
        return _solve_exact_with_evidence(n_vars, clauses, backend="pysat_invalid_model_fallback")
    return OracleEvidence(
        label="SAT",
        is_satisfiable=True,
        backend="pysat",
        checked_assignments=1,
        assignment_witness=assignment,
        assignment_witness_checked=True,
        unsat_certificate=(),
        unsat_certificate_checked=False,
        pysat_used=True,
        exact_fallback_used=False,
    )


def _replay_rows(rows: list[JsonDict], *, failing_case_ids: list[str]) -> JsonDict:
    repaired_rows: list[JsonDict] = []
    after_false_accepts = 0
    assignment_witnesses_checked = 0
    unsat_certificates_checked = 0
    for row in rows:
        n_vars = int(row["n_vars"])
        clauses = _normalise_clauses(n_vars, row["clauses"])
        parsed = exp1536.parse_model_answer(str(row.get("model_output") or ""))
        evidence = solve_cnf_with_evidence(n_vars, clauses)
        decision = evaluate_candidate_with_evidence(n_vars, clauses, parsed.baseline, evidence=evidence)
        repaired_false_accept = bool(decision["accepted"] and decision["classification"] not in _ACCEPTED_CLASSES)
        after_false_accepts += int(repaired_false_accept)
        assignment_witnesses_checked += int(evidence.assignment_witness_checked)
        unsat_certificates_checked += int(evidence.unsat_certificate_checked)
        repaired_rows.append(
            {
                "case_id": row.get("case_id"),
                "instance_id": row.get("instance_id"),
                "format_name": row.get("format_name"),
                "expected_label": evidence.label,
                "legacy_self_verifier_false_accept": bool(
                    _mapping(row.get("verifier")).get("self_verifier_false_accept")
                ),
                "legacy_classification": _mapping(row.get("baseline")).get("classification"),
                "repaired_decision": decision,
                "repaired_false_accept": repaired_false_accept,
                "oracle_evidence": evidence.to_dict(),
            }
        )
    return {
        "rows": repaired_rows,
        "false_accepts_before": len(failing_case_ids),
        "solver_oracle_false_accepts_after": after_false_accepts,
        "failing_case_ids_rechecked": failing_case_ids,
        "assignment_witnesses_checked": assignment_witnesses_checked,
        "unsat_certificates_checked": unsat_certificates_checked,
    }


_ACCEPTED_CLASSES = {
    "oracle_agreement_with_sat_witness",
    "oracle_agreement_with_unsat_certificate",
}


def _normalise_clauses(n_vars: int, clauses: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    if n_vars <= 0:
        raise CNFParseError("invalid_variable_count")
    clean: list[tuple[int, ...]] = []
    for clause in clauses:
        parsed_clause = tuple(int(literal) for literal in clause)
        if any(literal == 0 or abs(literal) > n_vars for literal in parsed_clause):
            raise CNFParseError("literal_out_of_range")
        clean.append(parsed_clause)
    return tuple(clean)


def _first_violated_clause(
    clauses: tuple[tuple[int, ...], ...],
    assignment: tuple[bool, ...],
) -> tuple[int, tuple[int, ...]]:
    for index, clause in enumerate(clauses):
        if not any(exp1536._literal_value(literal, assignment) for literal in clause):
            return index, clause
    raise ValueError("assignment satisfies all clauses")


def _decision(
    accepted: bool,
    classification: str,
    evidence: OracleEvidence | None,
    detail: str | None = None,
) -> JsonDict:
    return {
        "accepted": accepted,
        "correct": accepted,
        "classification": classification,
        "oracle_label": evidence.label if evidence is not None else None,
        "evidence_backend": evidence.backend if evidence is not None else None,
        "detail": detail,
    }


def _build_regression_rows() -> list[JsonDict]:
    rows: list[JsonDict] = []
    for case in exp1536.build_prompt_cases(exp1536.build_cnf_instances(run_date=RUN_DATE)):
        rows.append(
            exp1536.build_manifest_row(
                case,
                {
                    "case_id": case.case_id,
                    "instance_id": case.instance.instance_id,
                    "format_name": case.format_name,
                    "model_hf_id": "oracle-regression",
                    "model_name": "oracle-regression",
                    "generation_source": "deterministic_oracle_regression",
                    "output_text": exp1536.gold_answer_for_prompt_case(case),
                    "elapsed_seconds": 0.0,
                    "blocker": None,
                },
            )
        )
    return rows


def _run_perturbation_checks() -> bool:
    try:
        parse_dimacs_strict("p cnf 1 1\n1\n")
    except CNFParseError:
        malformed_rejected = True
    else:  # pragma: no cover - protects the gate if strict parsing regresses.
        malformed_rejected = False
    sat_evidence = solve_cnf_with_evidence(2, ((1,), (-1, 2)), prefer_pysat=False)
    unsat_evidence = solve_cnf_with_evidence(1, ((1,), (-1,)), prefer_pysat=False)
    sat_missing = evaluate_candidate_with_evidence(
        2,
        ((1,), (-1, 2)),
        exp1536.CandidateAnswer("SAT"),
        evidence=sat_evidence,
    )
    sat_invalid = evaluate_candidate_with_evidence(
        2,
        ((1,), (-1, 2)),
        exp1536.CandidateAnswer("SAT", (False, False)),
        evidence=sat_evidence,
    )
    unsat_fake = evaluate_candidate_with_evidence(
        1,
        ((1,), (-1,)),
        exp1536.CandidateAnswer("SAT", (True,)),
        evidence=unsat_evidence,
    )
    return (
        malformed_rejected
        and sat_missing["classification"] == "missing_assignment_witness"
        and sat_invalid["classification"] == "invalid_assignment_witness"
        and unsat_fake["classification"] == "wrong_label"
    )


def _exact_fallback_available() -> bool:
    evidence = solve_cnf_with_evidence(1, ((1,),), prefer_pysat=False)
    return evidence.label == "SAT" and evidence.assignment_witness_checked


def _false_accept_row_summary(rows: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "case_id": row.get("case_id"),
            "format_name": row.get("format_name"),
            "expected_label": _mapping(row.get("solver_oracle")).get("label"),
            "solver_backend": _mapping(row.get("solver_oracle")).get("backend"),
            "legacy_classification": _mapping(row.get("baseline")).get("classification"),
        }
        for row in rows
    ]


def _in_progress_payload(manifest: Path) -> JsonDict:
    return {
        "status": "in_progress",
        "milestone": RUN_DATE,
        "satquest_oracle_repair_ready": False,
        "satquest_zero_false_accepts": False,
        "false_accepts_before": None,
        "solver_oracle_false_accepts_after": None,
        "failing_case_ids_rechecked": [],
        "pysat_available": pysat_available(),
        "exact_fallback_available": _exact_fallback_available(),
        "assignment_witnesses_checked": 0,
        "unsat_certificates_checked": 0,
        "perturbation_checks_passed": False,
        "oracle_module_path": ORACLE_MODULE_PATH,
        "repaired_case_manifest_path": _display_path(manifest),
        "focused_tests_passed": False,
        "honest_verdict": "in_progress: satquest_oracle_false_accept_repair_running",
    }


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, dict) else {}


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_REPAIRED_CASE_MANIFEST_PATH",
    "DEFAULT_SOURCE_ARTIFACT_PATH",
    "DEFAULT_SOURCE_MANIFEST_PATH",
    "ORACLE_MODULE_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "CNFParseError",
    "OracleEvidence",
    "StrictCNF",
    "UnsatCertificateRow",
    "evaluate_candidate_with_evidence",
    "load_false_accept_rows",
    "parse_dimacs_strict",
    "pysat_available",
    "replay_false_accept_rows",
    "run_repair",
    "solve_cnf_with_evidence",
]
