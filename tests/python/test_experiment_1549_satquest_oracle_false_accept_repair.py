"""Tests for Exp1549 SATQuest oracle false-accept repair.

Spec: REQ-BENCH-1549, SCENARIO-BENCH-1549.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from carnot.eval import satquest_cnf_verifier_benchmark as exp1536
from carnot.eval import satquest_oracle_false_accept_repair as exp1549


def test_req_bench_1549_strict_dimacs_rejects_malformed_cnf() -> None:
    """REQ-BENCH-1549: malformed CNF inputs fail before oracle acceptance."""

    parsed = exp1549.parse_dimacs_strict("c ok\np cnf 2 2\n1 -2 0\n2 0\n")

    assert parsed.n_vars == 2
    assert parsed.clauses == ((1, -2), (2,))

    malformed_inputs = [
        "1 0\n",
        "p cnf 2 1\n1 2\n",
        "p cnf 2 2\n1 0\n",
        "p cnf 1 1\n2 0\n",
        "p cnf two 1\n1 0\n",
    ]
    for text in malformed_inputs:
        with pytest.raises(exp1549.CNFParseError):
            exp1549.parse_dimacs_strict(text)


def test_req_bench_1549_sat_success_requires_checked_assignment_witness() -> None:
    """REQ-BENCH-1549: SAT acceptance requires a full checked assignment witness."""

    clauses = ((1,), (-1, 2))
    evidence = exp1549.solve_cnf_with_evidence(2, clauses, prefer_pysat=False)

    assert evidence.label == "SAT"
    assert evidence.assignment_witness == (True, True)
    assert evidence.assignment_witness_checked is True
    assert evidence.unsat_certificate_checked is False

    accepted = exp1549.evaluate_candidate_with_evidence(
        2,
        clauses,
        exp1536.CandidateAnswer("SAT", (True, True)),
        evidence=evidence,
    )
    missing = exp1549.evaluate_candidate_with_evidence(
        2,
        clauses,
        exp1536.CandidateAnswer("SAT"),
        evidence=evidence,
    )
    invalid = exp1549.evaluate_candidate_with_evidence(
        2,
        clauses,
        exp1536.CandidateAnswer("SAT", (False, False)),
        evidence=evidence,
    )
    witnessless_oracle = exp1549.evaluate_candidate_with_evidence(
        2,
        clauses,
        exp1536.CandidateAnswer("SAT", (True, True)),
        evidence=replace(evidence, assignment_witness=None, assignment_witness_checked=False),
    )

    assert accepted["accepted"] is True
    assert accepted["classification"] == "oracle_agreement_with_sat_witness"
    assert missing["classification"] == "missing_assignment_witness"
    assert invalid["classification"] == "invalid_assignment_witness"
    assert witnessless_oracle["classification"] == "missing_oracle_sat_witness"


def test_req_bench_1549_unsat_success_requires_exact_certificate() -> None:
    """REQ-BENCH-1549: UNSAT acceptance requires exhaustive contradiction evidence."""

    clauses = ((1,), (-1,))
    evidence = exp1549.solve_cnf_with_evidence(1, clauses, prefer_pysat=False)

    assert evidence.label == "UNSAT"
    assert evidence.assignment_witness is None
    assert evidence.unsat_certificate_checked is True
    assert len(evidence.unsat_certificate) == 2

    accepted = exp1549.evaluate_candidate_with_evidence(
        1,
        clauses,
        exp1536.CandidateAnswer("UNSAT"),
        evidence=evidence,
    )
    fake_sat = exp1549.evaluate_candidate_with_evidence(
        1,
        clauses,
        exp1536.CandidateAnswer("SAT", (True,)),
        evidence=evidence,
    )
    proofless_unsat = exp1549.evaluate_candidate_with_evidence(
        1,
        clauses,
        exp1536.CandidateAnswer("UNSAT"),
        evidence=replace(evidence, unsat_certificate=(), unsat_certificate_checked=False),
    )

    assert accepted["accepted"] is True
    assert accepted["classification"] == "oracle_agreement_with_unsat_certificate"
    assert fake_sat["classification"] == "wrong_label"
    assert proofless_unsat["classification"] == "missing_unsat_certificate"


def test_scenario_bench_1549_replays_three_known_false_accept_shapes() -> None:
    """SCENARIO-BENCH-1549: Exp1536 false accepts replay to zero repaired accepts."""

    artifact, rows = exp1549.load_false_accept_rows(
        Path("results/experiment_1536_satquest_cnf_verifier_benchmark.json"),
        Path("results/satquest_cnf_verifier_1536.jsonl"),
    )
    replay = exp1549.replay_false_accept_rows(rows)

    assert artifact["solver_oracle_false_accepts"] == 3
    assert [row["case_id"] for row in rows] == [
        "cnf-1536-unsat-unit-clash-narrative",
        "cnf-1536-unsat-two-var-cover-narrative",
        "cnf-1536-sat-negated-path-narrative",
    ]
    assert replay["false_accepts_before"] == 3
    assert replay["solver_oracle_false_accepts_after"] == 0
    assert replay["assignment_witnesses_checked"] >= 1
    assert replay["unsat_certificates_checked"] >= 2
    assert replay["failing_case_ids_rechecked"] == [row["case_id"] for row in rows]
    assert {row["repaired_decision"]["classification"] for row in replay["rows"]} == {
        "wrong_label",
        "invalid_assignment_witness",
    }


def test_scenario_bench_1549_runner_writes_required_artifact_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-BENCH-1549: runner writes the zero-false-accept repair gate."""

    output_path = tmp_path / "experiment_1549.json"
    manifest_path = tmp_path / "satquest_repair.jsonl"

    artifact = exp1549.run_repair(
        output_path=output_path,
        repaired_case_manifest_path=manifest_path,
        source_artifact_path=Path("results/experiment_1536_satquest_cnf_verifier_benchmark.json"),
        source_manifest_path=Path("results/satquest_cnf_verifier_1536.jsonl"),
        focused_tests_passed=True,
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact == persisted
    assert set(exp1549.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "20260508"
    assert artifact["satquest_oracle_repair_ready"] is True
    assert artifact["satquest_zero_false_accepts"] is True
    assert artifact["false_accepts_before"] == 3
    assert artifact["solver_oracle_false_accepts_after"] == 0
    assert artifact["failing_case_ids_rechecked"] == [
        "cnf-1536-unsat-unit-clash-narrative",
        "cnf-1536-unsat-two-var-cover-narrative",
        "cnf-1536-sat-negated-path-narrative",
    ]
    assert artifact["exact_fallback_available"] is True
    assert artifact["assignment_witnesses_checked"] > 0
    assert artifact["unsat_certificates_checked"] > 0
    assert artifact["perturbation_checks_passed"] is True
    assert artifact["oracle_module_path"] == "python/carnot/eval/satquest_oracle_false_accept_repair.py"
    assert artifact["repaired_case_manifest_path"] == str(manifest_path)
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) >= 18
    assert all(row["repaired_false_accept"] is False for row in rows)


def test_req_bench_1549_defensive_oracle_paths_are_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-BENCH-1549: defensive parser, fallback, and PySAT paths fail closed."""

    for text in [
        "p cnf 1 0\np cnf 1 0\n",
        "p cnf 0 0\n",
        "p cnf 1 1\nx 0\n",
        "",
    ]:
        with pytest.raises(exp1549.CNFParseError):
            exp1549.parse_dimacs_strict(text)

    sat_evidence = exp1549.solve_cnf_with_evidence(2, ((1,), (-1, 2)), prefer_pysat=False)
    malformed_cnf = exp1549.evaluate_candidate_with_evidence(
        1,
        ((0,),),
        exp1536.CandidateAnswer("SAT", (True,)),
    )
    no_answer = exp1549.evaluate_candidate_with_evidence(
        2,
        ((1,), (-1, 2)),
        exp1536.CandidateAnswer(None),
        evidence=sat_evidence,
    )
    bad_length = exp1549.evaluate_candidate_with_evidence(
        2,
        ((1,), (-1, 2)),
        exp1536.CandidateAnswer("SAT", (True,)),
        evidence=sat_evidence,
    )

    assert malformed_cnf["classification"] == "malformed_cnf"
    assert no_answer["classification"] == "no_answer"
    assert bad_length["classification"] == "malformed_assignment_witness"
    with pytest.raises(ValueError, match="bounded exhaustive"):
        exp1549.solve_cnf_with_evidence(exp1536.MAX_EXHAUSTIVE_VARS + 1, ((1,),), prefer_pysat=False)
    with pytest.raises(exp1549.CNFParseError, match="invalid_variable_count"):
        exp1549.solve_cnf_with_evidence(0, (), prefer_pysat=False)
    with pytest.raises(ValueError, match="satisfies all clauses"):
        exp1549._first_violated_clause(((1,),), (True,))

    monkeypatch.setattr(
        exp1549,
        "parse_dimacs_strict",
        lambda _text: exp1549.StrictCNF(n_vars=1, clauses=((1,),)),
    )
    assert exp1549._run_perturbation_checks() is False

    class FakeSolver:
        def __init__(self, bootstrap_with: list[list[int]]) -> None:
            self.bootstrap_with = bootstrap_with

        def __enter__(self) -> "FakeSolver":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def solve(self) -> bool:
            return self.bootstrap_with != [[1], [-1]]

        def get_model(self) -> list[int]:
            return [1, 2]

    pysat_module = types.ModuleType("pysat")
    solvers_module = types.ModuleType("pysat.solvers")
    solvers_module.Solver = FakeSolver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pysat", pysat_module)
    monkeypatch.setitem(sys.modules, "pysat.solvers", solvers_module)
    monkeypatch.setattr(exp1549, "pysat_available", lambda: True)

    assert exp1549.solve_cnf_with_evidence(2, ((1,),), prefer_pysat=True).backend == "pysat"
    assert (
        exp1549.solve_cnf_with_evidence(1, ((1,), (-1,)), prefer_pysat=True).backend
        == "pysat+exact_unsat_certificate"
    )

    class InvalidModelSolver(FakeSolver):
        def get_model(self) -> list[int]:
            return [-1]

    solvers_module.Solver = InvalidModelSolver  # type: ignore[attr-defined]
    assert (
        exp1549.solve_cnf_with_evidence(1, ((1,),), prefer_pysat=True).backend
        == "pysat_invalid_model_fallback"
    )

    monkeypatch.setattr(
        exp1549,
        "_solve_with_pysat_evidence",
        lambda _n_vars, _clauses: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert (
        exp1549.solve_cnf_with_evidence(1, ((1,),), prefer_pysat=True).backend
        == "exact_exhaustive_fallback"
    )

    source_artifact = tmp_path / "source.json"
    source_manifest = tmp_path / "source.jsonl"
    source_artifact.write_text('{"solver_oracle_false_accepts": 0}\n', encoding="utf-8")
    source_manifest.write_text("", encoding="utf-8")
    blocked = exp1549.run_repair(
        output_path=tmp_path / "blocked.json",
        repaired_case_manifest_path=tmp_path / "blocked.jsonl",
        source_artifact_path=source_artifact,
        source_manifest_path=source_manifest,
    )

    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
