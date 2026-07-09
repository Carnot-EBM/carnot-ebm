"""Exp5462: minimal-core p-bit/p-dit assumption bridge.

Spec refs: REQ-VERIFY-5462, SCENARIO-VERIFY-5462.

This diagnostic keeps the V495 boundary intact: active constraints, binary
p-bit samples, and multi-state p-dit samples are allowed to suggest temporary
assumptions, but the exact solver remains the only authority for final labels
and assignments.  The V496 addition is minimal-core feedback for bad advisory
assumptions, plus a small assignment fixture whose controls are naturally
categorical p-dit states rather than one-hot binary spins.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5407_pbit_qubo_active_constraint_stress_v492 as exp5407
from carnot import experiment_5433_active_constraint_diversity_lns_v494 as exp5433
from carnot.analysis import pdit_certificate_state_mapping as pdit_mapping


JsonDict = dict[str, Any]
RowOverride = Callable[[list[JsonDict]], list[JsonDict]]
Candidate = tuple[bool | str, ...]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json"
)
EXPERIMENT = 5462
EXPERIMENT_ID = "exp5462-active-constraint-minimal-core-pdit-bridge-v496"
MILESTONE = "2026.07.496"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5462
SCHEMA = "carnot.experiment_5462.active_constraint_minimal_core_pdit_bridge.v496"
SPEC_REFS = ("REQ-VERIFY-5462", "SCENARIO-VERIFY-5462")
INFERENCE_SUBSTRATE = "deterministic_solver_pbit_pdit_fixture"
TERMINAL_PREFIXES = ("complete:", "blocked:")
ASSUMPTION_SOURCES = ("active_constraint", "pbit_binary", "pdit_multistate")
EXPECTED_FIXTURE_COUNT = 5
EXPECTED_SOURCE_COUNT = len(ASSUMPTION_SOURCES)
EXPECTED_CONSTRAINT_FAMILIES = {"assignment", "csp", "lns", "sat"}

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_count": "bounded coverage",
    "constraint_family_counts": "SAT/CSP/LNS/assignment diversity",
    "assumption_source_counts": "active vs p-bit vs p-dit provenance",
    "pdit_variable_count": "multi-state assignment coverage",
    "minimal_core_count": "bad-assumption diagnosis",
    "density_before_after": "restored-sparsity accounting",
    "solver_authoritative": "exact solver final authority",
    "fallback_completeness_rate": "correctness rescue",
    "rejected_assumption_count": "advisory boundary",
    "solver_work_delta": "utility measurement",
    "unsafe_false_accepts": "correctness boundary",
    "minimal_core_pbit_bridge_ready": "downstream hardware gate",
    "hardware_speedup_claim": "no unsupported hardware claim",
    "inference_substrate": "no hidden hardware inference",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class BridgeFixture:
    """One bounded exact fixture plus advisory assumption samples."""

    fixture_id: str
    constraint_family: str
    source_module: str
    source_fixture_id: str
    variables: tuple[str, ...]
    clauses: tuple[tuple[int, ...], ...]
    precedence: tuple[tuple[str, str], ...]
    assignment_domain: tuple[str, ...]
    assignment_costs: tuple[tuple[str, str, int], ...]
    pairwise_costs: tuple[tuple[str, str, str, str, int], ...]
    expected_status: str
    expected_solution: Candidate
    active_assumptions: tuple[str, ...]
    pbit_control_names: tuple[str, ...]
    pbit_true_assumptions: tuple[str, ...]
    pbit_false_assumptions: tuple[str, ...]
    pbit_samples: tuple[tuple[bool, ...], ...]
    pdit_control_names: tuple[str, ...]
    pdit_samples: tuple[tuple[str, ...], ...]
    pdit_domains: tuple[tuple[str, tuple[str, ...]], ...]
    pdit_state_codes: tuple[tuple[str, int], ...]
    density_note: str


@dataclass(frozen=True)
class SolveMetrics:
    """Exact-solver outcome and local work counters."""

    status: str
    solution: Candidate | None
    objective_value: int | None
    conflicts: int
    propagations: int
    iterations: int

    @property
    def solver_work(self) -> int:
        """Local deterministic effort scalar for within-fixture comparison."""

        return self.conflicts + self.propagations + self.iterations

    def as_serializable(self, fixture: BridgeFixture) -> JsonDict:
        """Return JSON-safe counters and the candidate solution."""

        return {
            "status": self.status,
            "solution": _serialize_solution(fixture, self.solution),
            "objective_value": self.objective_value,
            "conflicts": self.conflicts,
            "propagations": self.propagations,
            "iterations": self.iterations,
            "solver_work": self.solver_work,
        }


def build_bridge_fixtures() -> tuple[BridgeFixture, ...]:
    """Build bounded SAT, CSP/LNS, and assignment fixtures with exact outcomes."""

    stress = exp5407.build_stress_fixtures()[1]
    lns_source = exp5433.build_diversity_fixtures()[1]
    csp_expected = tuple(stress.expected_sequence)
    lns_expected = ("invoice", "verify", "receive", "ship")
    assignment_domain = ("pack", "test", "ship")
    assignment_codes = tuple(
        (state, int(row["pdit_code"]))
        for state, row in pdit_mapping.build_pdit_mapping(assignment_domain).items()
    )
    return (
        BridgeFixture(
            fixture_id="sat_consensus_accept",
            constraint_family="sat",
            source_module="local_sat_fixture",
            source_fixture_id="sat:consensus_accept",
            variables=("x1", "x2", "x3", "x4"),
            clauses=((1,), (2, 3), (-2, 4), (1, 4)),
            precedence=(),
            assignment_domain=(),
            assignment_costs=(),
            pairwise_costs=(),
            expected_status="sat",
            expected_solution=(True, True, True, True),
            active_assumptions=("x1",),
            pbit_control_names=("x1", "x2", "x3", "x4"),
            pbit_true_assumptions=("x1", "x2", "x3", "x4"),
            pbit_false_assumptions=("!x1", "!x2", "!x3", "!x4"),
            pbit_samples=(
                (True, True, True, True),
                (True, True, True, True),
                (True, True, False, True),
                (True, False, True, True),
            ),
            pdit_control_names=("x1", "x2", "x3", "x4"),
            pdit_samples=(
                ("true", "true", "true", "true"),
                ("true", "true", "true", "true"),
                ("true", "true", "unknown", "true"),
                ("true", "unknown", "true", "true"),
            ),
            pdit_domains=tuple(
                (name, ("false", "unknown", "true")) for name in ("x1", "x2", "x3", "x4")
            ),
            pdit_state_codes=(("false", 0), ("unknown", 1), ("true", 2)),
            density_note="SAT clauses are lifted to dense pairwise terms before sparse restoration.",
        ),
        BridgeFixture(
            fixture_id="sat_false_basin_rescue",
            constraint_family="sat",
            source_module="local_sat_fixture",
            source_fixture_id="sat:false_basin_rescue",
            variables=("x1", "x2", "x3"),
            clauses=((1,), (2, -3), (1, 2)),
            precedence=(),
            assignment_domain=(),
            assignment_costs=(),
            pairwise_costs=(),
            expected_status="sat",
            expected_solution=(True, True, False),
            active_assumptions=("x1",),
            pbit_control_names=("x1", "x2", "x3"),
            pbit_true_assumptions=("x1", "x2", "x3"),
            pbit_false_assumptions=("!x1", "!x2", "!x3"),
            pbit_samples=(
                (False, True, False),
                (False, True, False),
                (False, True, True),
                (False, False, False),
            ),
            pdit_control_names=("x1", "x2", "x3"),
            pdit_samples=(
                ("true", "true", "false"),
                ("true", "true", "false"),
                ("true", "true", "unknown"),
                ("true", "unknown", "false"),
            ),
            pdit_domains=tuple((name, ("false", "unknown", "true")) for name in ("x1", "x2", "x3")),
            pdit_state_codes=(("false", 0), ("unknown", 1), ("true", 2)),
            density_note="Wrong binary p-bit consensus fixes a unit-clause variable and must be rescued.",
        ),
        BridgeFixture(
            fixture_id="csp_active_join_order",
            constraint_family="csp",
            source_module="experiment_5407_pbit_qubo_active_constraint_stress_v492",
            source_fixture_id=stress.fixture_id,
            variables=tuple(stress.actions),
            clauses=(),
            precedence=tuple(stress.precedence),
            assignment_domain=(),
            assignment_costs=(),
            pairwise_costs=(),
            expected_status="sat",
            expected_solution=csp_expected,
            active_assumptions=(f"{csp_expected[0]}@0", f"{csp_expected[1]}@1"),
            pbit_control_names=("pos0", "pos1"),
            pbit_true_assumptions=(f"{csp_expected[0]}@0", f"{csp_expected[1]}@1"),
            pbit_false_assumptions=(f"{csp_expected[1]}@0", f"{csp_expected[0]}@1"),
            pbit_samples=((True, True), (True, True), (True, True), (True, False)),
            pdit_control_names=("pos0", "pos1"),
            pdit_samples=(
                (str(csp_expected[0]), str(csp_expected[1])),
                (str(csp_expected[0]), str(csp_expected[1])),
                (str(csp_expected[0]), str(csp_expected[1])),
                (str(csp_expected[0]), str(csp_expected[2])),
            ),
            pdit_domains=(
                ("pos0", tuple(str(action) for action in stress.actions)),
                ("pos1", tuple(str(action) for action in stress.actions)),
            ),
            pdit_state_codes=tuple(
                (str(action), index) for index, action in enumerate(stress.actions)
            ),
            density_note="Exp5407 QUBO ordering terms are restored to direct precedence edges.",
        ),
        BridgeFixture(
            fixture_id="lns_valid_but_suboptimal",
            constraint_family="lns",
            source_module="experiment_5433_active_constraint_diversity_lns_v494",
            source_fixture_id=lns_source.fixture_id,
            variables=lns_expected,
            clauses=(),
            precedence=(("invoice", "ship"), ("verify", "ship"), ("receive", "ship")),
            assignment_domain=(),
            assignment_costs=(),
            pairwise_costs=(),
            expected_status="sat",
            expected_solution=lns_expected,
            active_assumptions=("invoice@0", "verify@1"),
            pbit_control_names=("pos0",),
            pbit_true_assumptions=("invoice@0",),
            pbit_false_assumptions=("verify@0",),
            pbit_samples=((False,), (False,), (False,), (True,)),
            pdit_control_names=("pos0", "pos1"),
            pdit_samples=(
                ("invoice", "verify"),
                ("invoice", "verify"),
                ("invoice", "verify"),
                ("verify", "invoice"),
            ),
            pdit_domains=(
                ("pos0", lns_expected),
                ("pos1", lns_expected),
            ),
            pdit_state_codes=tuple((action, index) for index, action in enumerate(lns_expected)),
            density_note="The LNS projection restores sparse direct constraints from a dense projection.",
        ),
        BridgeFixture(
            fixture_id="assignment_pdit_tradeoff",
            constraint_family="assignment",
            source_module="pdit_assignment_fixture",
            source_fixture_id="assignment:qap_style_tradeoff",
            variables=("ana", "ben", "cy"),
            clauses=(),
            precedence=(),
            assignment_domain=assignment_domain,
            assignment_costs=(
                ("ana", "pack", 0),
                ("ana", "test", 4),
                ("ana", "ship", 6),
                ("ben", "pack", 5),
                ("ben", "test", 0),
                ("ben", "ship", 4),
                ("cy", "pack", 6),
                ("cy", "test", 5),
                ("cy", "ship", 0),
            ),
            pairwise_costs=(
                ("ana", "test", "ben", "pack", 2),
                ("ana", "pack", "ben", "test", 0),
                ("ben", "test", "cy", "ship", 0),
            ),
            expected_status="sat",
            expected_solution=("pack", "test", "ship"),
            active_assumptions=("ana=pack",),
            pbit_control_names=("ana", "ben", "cy"),
            pbit_true_assumptions=("ana=pack", "ben=test", "cy=ship"),
            pbit_false_assumptions=("ana=test", "ben=pack", "cy=test"),
            pbit_samples=(
                (True, True, True),
                (True, True, True),
                (True, True, True),
                (True, False, True),
            ),
            pdit_control_names=("ana", "ben", "cy"),
            pdit_samples=(
                ("test", "pack", "ship"),
                ("test", "pack", "ship"),
                ("test", "pack", "ship"),
                ("pack", "test", "ship"),
            ),
            pdit_domains=tuple((worker, assignment_domain) for worker in ("ana", "ben", "cy")),
            pdit_state_codes=assignment_codes,
            density_note="Assignment/QAP-style p-dit categorical controls restore unary and pairwise terms.",
        ),
    )


def constraint_family_counts(fixtures: Sequence[BridgeFixture]) -> dict[str, int]:
    """Count fixtures by constraint family in stable key order."""

    counts = Counter(fixture.constraint_family for fixture in fixtures)
    return {key: counts[key] for key in sorted(counts)}


def pdit_variable_count(fixtures: Sequence[BridgeFixture]) -> int:
    """Count categorical p-dit controls exposed by the bounded fixtures."""

    return sum(len(fixture.pdit_control_names) for fixture in fixtures)


def density_before_after(fixtures: Sequence[BridgeFixture]) -> JsonDict:
    """Measure dense lifted terms before and sparse terms after restoration."""

    by_fixture: dict[str, JsonDict] = {}
    before_values: list[float] = []
    after_values: list[float] = []
    for fixture in fixtures:
        possible_edges = _possible_edge_count(fixture)
        before_edges = possible_edges
        after_edges = len(_restored_edges(fixture))
        before = _rate(before_edges, possible_edges)
        after = _rate(after_edges, possible_edges)
        before_values.append(before)
        after_values.append(after)
        by_fixture[fixture.fixture_id] = {
            "constraint_family": fixture.constraint_family,
            "before": before,
            "after": after,
            "before_edges": before_edges,
            "after_edges": after_edges,
            "possible_edges": possible_edges,
            "density_note": fixture.density_note,
        }
    mean_before = _rate(sum(before_values), len(before_values))
    mean_after = _rate(sum(after_values), len(after_values))
    return {
        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_restored_sparsity_delta": round(mean_before - mean_after, 6),
        "by_fixture": by_fixture,
    }


def run_diagnostic(row_overrides: RowOverride | None = None) -> JsonDict:
    """Evaluate all advisory sources while keeping exact solves authoritative."""

    fixtures = build_bridge_fixtures()
    rows = [
        evaluate_fixture_source(fixture, source)
        for fixture in fixtures
        for source in ASSUMPTION_SOURCES
    ]
    if row_overrides is not None:
        rows = row_overrides(rows)
    summary = _summarize_rows(fixtures, rows)
    blockers = readiness_blockers(summary)
    summary["minimal_core_pbit_bridge_ready"] = not blockers
    summary["readiness_blockers"] = blockers
    summary["row_records"] = rows
    return summary


def evaluate_fixture_source(fixture: BridgeFixture, assumption_source: str) -> JsonDict:
    """Run one fixture/source row and rescue bad assumptions with exact fallback."""

    if assumption_source not in ASSUMPTION_SOURCES:
        raise ValueError(f"assumption_source: {assumption_source}")

    baseline = solve_exact(fixture, ())
    assumptions = _assumptions_for_source(fixture, assumption_source)
    attempt = solve_exact(fixture, assumptions)
    fallback_used = False
    rejected = 0
    overwritten = 0
    decision = "accepted"
    final = attempt
    core_assumptions: tuple[str, ...] = ()
    core_ids: tuple[str, ...] = ()
    core_evidence: list[JsonDict] = []

    if attempt.status != baseline.status:
        decision = "rejected"
        fallback_used = True
    elif attempt.status == "sat" and attempt.solution != baseline.solution:
        decision = "overwritten"
        fallback_used = True

    if fallback_used:
        core_assumptions, core_ids, core_evidence = minimal_core_for_assumptions(
            fixture,
            assumptions,
            baseline,
        )
        rejected = len(core_assumptions) if decision == "rejected" else 0
        overwritten = len(core_assumptions) if decision == "overwritten" else 0
        final = baseline

    final_matches_exact = _same_exact_outcome(final, baseline)
    solution_valid = _solution_valid_for_final(fixture, final)
    objective_preserved = final.objective_value == baseline.objective_value
    unsafe_false_accept = bool(decision == "accepted" and not final_matches_exact)
    guided_work = attempt.solver_work + (baseline.solver_work if fallback_used else 0)
    density = density_before_after((fixture,))["by_fixture"][fixture.fixture_id]
    return {
        "fixture_id": fixture.fixture_id,
        "constraint_family": fixture.constraint_family,
        "source_module": fixture.source_module,
        "source_fixture_id": fixture.source_fixture_id,
        "assumption_source": assumption_source,
        "assumptions": list(assumptions),
        "pbit_binary_sample_count": len(fixture.pbit_samples)
        if assumption_source == "pbit_binary"
        else 0,
        "pdit_sample_count": len(fixture.pdit_samples)
        if assumption_source == "pdit_multistate"
        else 0,
        "pdit_control_names": list(fixture.pdit_control_names)
        if assumption_source == "pdit_multistate"
        else [],
        "pdit_state_codes": dict(fixture.pdit_state_codes)
        if assumption_source == "pdit_multistate"
        else {},
        "assumption_decision": decision,
        "solver_authoritative": True,
        "accepted_without_verification": False,
        "fallback_used": fallback_used,
        "minimal_core_assumptions": list(core_assumptions),
        "minimal_core_ids": list(core_ids),
        "minimal_core_evidence": core_evidence,
        "rejected_assumptions": rejected,
        "overwritten_assumptions": overwritten,
        "baseline_status": baseline.status,
        "assumption_attempt_status": attempt.status,
        "final_status": final.status,
        "baseline_solution": _serialize_solution(fixture, baseline.solution),
        "assumption_solution": _serialize_solution(fixture, attempt.solution),
        "final_solution": _serialize_solution(fixture, final.solution),
        "baseline_metrics": baseline.as_serializable(fixture),
        "assumption_metrics": attempt.as_serializable(fixture),
        "fallback_metrics": baseline.as_serializable(fixture) if fallback_used else None,
        "conflicts": attempt.conflicts,
        "propagations": attempt.propagations,
        "iterations": attempt.iterations,
        "baseline_solver_work": baseline.solver_work,
        "guided_solver_work": guided_work,
        "work_delta": baseline.solver_work - guided_work,
        "density_before": density["before"],
        "density_after": density["after"],
        "density_before_edges": density["before_edges"],
        "density_after_edges": density["after_edges"],
        "solution_valid": solution_valid,
        "objective_preserved": objective_preserved,
        "final_matches_exact": final_matches_exact,
        "unsafe_false_accept": unsafe_false_accept,
        "hardware_speedup_claim": False,
    }


def solve_exact(fixture: BridgeFixture, assumptions: Sequence[str]) -> SolveMetrics:
    """Exhaustively solve one bounded fixture under temporary assumptions."""

    candidates = [
        candidate
        for candidate in _candidate_space(fixture)
        if _satisfies_assumptions(fixture, candidate, assumptions)
    ]
    conflicts = 0
    propagations = max(1, len(assumptions)) * max(1, _constraint_count(fixture))
    iterations = 0
    best: Candidate | None = None
    best_score: int | None = None

    for candidate in candidates:
        iterations += 1
        violations = _constraint_violation_count(fixture, candidate)
        propagations += max(1, _constraint_count(fixture))
        if violations:
            conflicts += violations
            continue
        score = _objective_value(fixture, candidate)
        if best is None or score < _require_int(best_score, "best_score"):
            best = candidate
            best_score = score

    if best is None:
        return SolveMetrics(
            status="unsat",
            solution=None,
            objective_value=None,
            conflicts=conflicts + max(1, len(candidates)),
            propagations=propagations,
            iterations=iterations,
        )
    return SolveMetrics(
        status="sat",
        solution=best,
        objective_value=best_score,
        conflicts=conflicts,
        propagations=propagations,
        iterations=iterations,
    )


def minimal_core_for_assumptions(
    fixture: BridgeFixture,
    assumptions: Sequence[str],
    baseline: SolveMetrics | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], list[JsonDict]]:
    """Return a smallest assumption subset that reproduces the bad guided outcome."""

    exact = baseline or solve_exact(fixture, ())
    normalized = tuple(str(assumption) for assumption in assumptions)
    if not _attempt_disagrees(solve_exact(fixture, normalized), exact):
        return (), (), []
    for size in range(1, len(normalized) + 1):
        for subset in itertools.combinations(normalized, size):
            if not _attempt_disagrees(solve_exact(fixture, subset), exact):
                continue
            evidence = []
            for assumption in subset:
                without = tuple(item for item in subset if item != assumption)
                without_metrics = solve_exact(fixture, without)
                evidence.append(
                    {
                        "assumption": assumption,
                        "core_id": _assumption_core_id(fixture, assumption),
                        "without_core_assumption_matches_exact": _same_exact_outcome(
                            without_metrics,
                            exact,
                        ),
                    }
                )
            return (
                tuple(subset),
                tuple(_assumption_core_id(fixture, assumption) for assumption in subset),
                evidence,
            )
    return (
        normalized,
        tuple(_assumption_core_id(fixture, assumption) for assumption in normalized),
        [],
    )


def build_artifact(
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    row_overrides: RowOverride | None = None,
) -> JsonDict:
    """Build the terminal Exp5462 artifact from deterministic rows."""

    diagnostic = run_diagnostic(row_overrides=row_overrides)
    tests = [_normalize_test_run(item) for item in tests_run]
    blockers = list(diagnostic["readiness_blockers"])
    if diagnostic["minimal_core_pbit_bridge_ready"] and not tests:
        blockers.append("tests_not_recorded")
    ready = bool(diagnostic["minimal_core_pbit_bridge_ready"] and not blockers)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": "complete" if ready else "blocked",
        "fixture_count": diagnostic["fixture_count"],
        "constraint_family_counts": diagnostic["constraint_family_counts"],
        "assumption_source_counts": diagnostic["assumption_source_counts"],
        "pdit_variable_count": diagnostic["pdit_variable_count"],
        "minimal_core_count": diagnostic["minimal_core_count"],
        "density_before_after": diagnostic["density_before_after"],
        "solver_authoritative": diagnostic["solver_authoritative"],
        "fallback_completeness_rate": diagnostic["fallback_completeness_rate"],
        "rejected_assumption_count": diagnostic["rejected_assumption_count"],
        "overwritten_assumption_count": diagnostic["overwritten_assumption_count"],
        "solver_work_delta": diagnostic["solver_work_delta"],
        "unsafe_false_accepts": diagnostic["unsafe_false_accepts"],
        "minimal_core_pbit_bridge_ready": ready,
        "hardware_speedup_claim": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, blockers, diagnostic),
        "row_count": diagnostic["row_count"],
        "assumption_sources": list(ASSUMPTION_SOURCES),
        "row_records": diagnostic["row_records"],
        "mode_summaries": diagnostic["mode_summaries"],
        "readiness_blockers": blockers,
        "tests_run": tests,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [
            str(exp5407.RESULT_RELATIVE_PATH),
            str(exp5433.RESULT_RELATIVE_PATH),
            str(pdit_mapping.DELIVERABLE_PATH.relative_to(REPO_ROOT)),
        ],
        "claim_limits": [
            "deterministic CPU-local solver fixture",
            "active constraints, p-bit samples, and p-dit samples are advisory only",
            "minimal cores diagnose bad assumptions but do not certify final answers",
            "exact unrestricted solving rejects or overwrites unsafe assumptions",
            "no hardware timing-ratio receipt or hardware speedup claim",
        ],
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5462 artifact and return it."""

    artifact = build_artifact(tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when schema, authority, or claim-boundary fields drift."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require("REQ-VERIFY-5462" in artifact.get("spec_refs", []), "spec_refs")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor.py")
    _require(len(str(artifact.get("reproducibility_checksum", ""))) == 64, "checksum")

    rows = artifact.get("row_records")
    _require(isinstance(rows, Sequence), "row_records")
    _validate_rows(rows)
    fixtures = build_bridge_fixtures()
    summary = _summarize_rows(fixtures, rows)
    for field in (
        "fixture_count",
        "constraint_family_counts",
        "assumption_source_counts",
        "pdit_variable_count",
        "minimal_core_count",
        "density_before_after",
        "solver_authoritative",
        "fallback_completeness_rate",
        "rejected_assumption_count",
        "overwritten_assumption_count",
        "solver_work_delta",
        "unsafe_false_accepts",
    ):
        _require(artifact.get(field) == summary[field], field)
    _validate_density(artifact["density_before_after"])

    blockers = readiness_blockers(summary)
    if summary["minimal_core_pbit_bridge_ready"] and not artifact.get("tests_run"):
        blockers.append("tests_not_recorded")
    expected_ready = not blockers
    _require(artifact.get("minimal_core_pbit_bridge_ready") is expected_ready, "readiness")
    _require(artifact.get("readiness_blockers") == blockers, "readiness_blockers")
    if expected_ready:
        _require(artifact.get("status") == "complete", "status")
        _require(str(artifact.get("honest_verdict")).startswith("complete:"), "honest_verdict")
    else:
        _require(artifact.get("status") == "blocked", "status")
        _require(str(artifact.get("honest_verdict")).startswith("blocked:"), "honest_verdict")


def readiness_blockers(summary: Mapping[str, Any]) -> list[str]:
    """Return precise blockers for the bridge-ready gate."""

    blockers: list[str] = []
    if summary["fixture_count"] != EXPECTED_FIXTURE_COUNT:
        blockers.append("fixture_count_mismatch")
    if set(summary["constraint_family_counts"]) != EXPECTED_CONSTRAINT_FAMILIES:
        blockers.append("constraint_family_missing")
    expected_sources = {source: EXPECTED_FIXTURE_COUNT for source in ASSUMPTION_SOURCES}
    if summary["assumption_source_counts"] != expected_sources:
        blockers.append("assumption_source_coverage_mismatch")
    if summary["pdit_variable_count"] <= 0:
        blockers.append("pdit_variables_missing")
    if summary["minimal_core_count"] <= 0:
        blockers.append("minimal_cores_missing")
    density = summary["density_before_after"]
    if density["mean_before"] <= density["mean_after"]:
        blockers.append("density_restoration_not_measured")
    if summary["solver_authoritative"] is not True:
        blockers.append("solver_not_authoritative")
    if summary["fallback_completeness_rate"] != 1.0:
        blockers.append("fallback_completeness_incomplete")
    if summary["rejected_assumption_count"] <= 0:
        blockers.append("no_rejected_assumptions")
    if summary["solver_work_delta"] <= 0:
        blockers.append("solver_work_not_reduced")
    if summary["unsafe_false_accepts"] != 0:
        blockers.append("unsafe_false_accepts_present")
    return blockers


def _summarize_rows(
    fixtures: Sequence[BridgeFixture],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    source_counts = Counter(str(row["assumption_source"]) for row in rows)
    mode_summaries = {
        source: _source_summary([row for row in rows if row["assumption_source"] == source])
        for source in ASSUMPTION_SOURCES
    }
    completeness = _rate(sum(bool(row["final_matches_exact"]) for row in rows), len(rows))
    solver_authoritative = all(
        bool(row["solver_authoritative"])
        and not bool(row["accepted_without_verification"])
        and bool(row["solution_valid"])
        for row in rows
    )
    summary = {
        "fixture_count": len({row["fixture_id"] for row in rows}),
        "constraint_family_counts": constraint_family_counts(fixtures),
        "assumption_source_counts": {
            source: source_counts[source] for source in ASSUMPTION_SOURCES
        },
        "pdit_variable_count": pdit_variable_count(fixtures),
        "minimal_core_count": sum(int(bool(row["minimal_core_ids"])) for row in rows),
        "density_before_after": density_before_after(fixtures),
        "solver_authoritative": solver_authoritative,
        "fallback_completeness_rate": completeness,
        "rejected_assumption_count": sum(int(row["rejected_assumptions"]) for row in rows),
        "overwritten_assumption_count": sum(int(row["overwritten_assumptions"]) for row in rows),
        "solver_work_delta": sum(int(row["work_delta"]) for row in rows),
        "unsafe_false_accepts": sum(int(row["unsafe_false_accept"]) for row in rows),
        "row_count": len(rows),
        "mode_summaries": mode_summaries,
    }
    summary["minimal_core_pbit_bridge_ready"] = not readiness_blockers(summary)
    return summary


def _source_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "row_count": len(rows),
        "accepted_count": sum(int(row["assumption_decision"] == "accepted") for row in rows),
        "rejected_count": sum(int(row["assumption_decision"] == "rejected") for row in rows),
        "overwritten_count": sum(int(row["assumption_decision"] == "overwritten") for row in rows),
        "fallback_count": sum(int(row["fallback_used"]) for row in rows),
        "minimal_core_count": sum(int(bool(row["minimal_core_ids"])) for row in rows),
        "solver_work_delta": sum(int(row["work_delta"]) for row in rows),
        "unsafe_false_accepts": sum(int(row["unsafe_false_accept"]) for row in rows),
    }


def _assumptions_for_source(fixture: BridgeFixture, source: str) -> tuple[str, ...]:
    if source == "active_constraint":
        return fixture.active_assumptions
    if source == "pbit_binary":
        return _pbit_binary_assumptions(fixture)
    return _pdit_multistate_assumptions(fixture)


def _pbit_binary_assumptions(fixture: BridgeFixture) -> tuple[str, ...]:
    threshold = 0.75
    assumptions: list[str] = []
    for index in range(len(fixture.pbit_control_names)):
        true_count = sum(int(sample[index]) for sample in fixture.pbit_samples)
        true_rate = true_count / len(fixture.pbit_samples)
        if true_rate >= threshold:
            assumptions.append(fixture.pbit_true_assumptions[index])
        elif true_rate <= 1.0 - threshold:
            assumptions.append(fixture.pbit_false_assumptions[index])
    return tuple(assumptions)


def _pdit_multistate_assumptions(fixture: BridgeFixture) -> tuple[str, ...]:
    threshold = 0.75
    assumptions: list[str] = []
    for index, control in enumerate(fixture.pdit_control_names):
        counts = Counter(str(sample[index]) for sample in fixture.pdit_samples)
        value, count = counts.most_common(1)[0]
        if count / len(fixture.pdit_samples) >= threshold:
            assumption = _pdit_value_to_assumption(fixture, control, value)
            if assumption:
                assumptions.append(assumption)
    return tuple(assumptions)


def _pdit_value_to_assumption(
    fixture: BridgeFixture,
    control: str,
    value: str,
) -> str:
    if value in {"unknown", "abstain"}:
        return ""
    if fixture.constraint_family == "sat":
        return control if value == "true" else f"!{control}"
    if fixture.constraint_family == "assignment":
        return f"{control}={value}"
    position = int(control.removeprefix("pos"))
    return f"{value}@{position}"


def _candidate_space(fixture: BridgeFixture) -> tuple[Candidate, ...]:
    if fixture.constraint_family == "sat":
        return tuple(itertools.product((False, True), repeat=len(fixture.variables)))
    if fixture.constraint_family == "assignment":
        return tuple(itertools.permutations(fixture.assignment_domain))
    return tuple(itertools.permutations(fixture.variables))


def _satisfies_assumptions(
    fixture: BridgeFixture,
    candidate: Candidate,
    assumptions: Sequence[str],
) -> bool:
    for assumption in assumptions:
        if fixture.constraint_family == "sat":
            if assumption.startswith("!"):
                index = fixture.variables.index(assumption[1:])
                if bool(candidate[index]) is not False:
                    return False
            else:
                index = fixture.variables.index(assumption)
                if bool(candidate[index]) is not True:
                    return False
        elif fixture.constraint_family == "assignment":
            worker, job = assumption.split("=", 1)
            if str(candidate[fixture.variables.index(worker)]) != job:
                return False
        else:
            action, position_text = assumption.rsplit("@", 1)
            if str(candidate[int(position_text)]) != action:
                return False
    return True


def _constraint_violation_count(fixture: BridgeFixture, candidate: Candidate) -> int:
    if fixture.constraint_family == "sat":
        return sum(int(not _clause_satisfied(candidate, clause)) for clause in fixture.clauses)
    if fixture.constraint_family == "assignment":
        return int(set(str(job) for job in candidate) != set(fixture.assignment_domain))
    positions = {str(action): index for index, action in enumerate(candidate)}
    return sum(int(positions[before] > positions[after]) for before, after in fixture.precedence)


def _clause_satisfied(candidate: Candidate, clause: Sequence[int]) -> bool:
    for literal in clause:
        value = bool(candidate[abs(literal) - 1])
        if value == (literal > 0):
            return True
    return False


def _objective_value(fixture: BridgeFixture, candidate: Candidate) -> int:
    if fixture.constraint_family == "sat":
        return sum(
            (index + 1) * int(candidate[index] != fixture.expected_solution[index])
            for index in range(len(candidate))
        )
    if fixture.constraint_family == "assignment":
        return _assignment_objective(fixture, candidate)
    preferred = {value: index for index, value in enumerate(fixture.expected_solution)}
    return sum((index + 1) * abs(index - preferred[value]) for index, value in enumerate(candidate))


def _assignment_objective(fixture: BridgeFixture, candidate: Candidate) -> int:
    assignment = {worker: str(candidate[index]) for index, worker in enumerate(fixture.variables)}
    cost_lookup = {(worker, job): cost for worker, job, cost in fixture.assignment_costs}
    total = sum(cost_lookup[(worker, assignment[worker])] for worker in fixture.variables)
    for left_worker, left_job, right_worker, right_job, cost in fixture.pairwise_costs:
        if assignment[left_worker] == left_job and assignment[right_worker] == right_job:
            total += cost
    return total


def _constraint_count(fixture: BridgeFixture) -> int:
    if fixture.constraint_family == "sat":
        return len(fixture.clauses)
    if fixture.constraint_family == "assignment":
        return len(fixture.assignment_costs) + len(fixture.pairwise_costs)
    return len(fixture.precedence)


def _possible_edge_count(fixture: BridgeFixture) -> int:
    if fixture.constraint_family == "assignment":
        workers = len(fixture.variables)
        jobs = len(fixture.assignment_domain)
        pairwise = workers * (workers - 1) // 2 * jobs * jobs
        return max(1, workers * jobs + pairwise)
    return max(1, len(fixture.variables) * (len(fixture.variables) - 1) // 2)


def _restored_edges(fixture: BridgeFixture) -> set[tuple[str, str]]:
    if fixture.constraint_family == "sat":
        edges: set[tuple[str, str]] = set()
        for clause in fixture.clauses:
            variables = sorted({fixture.variables[abs(literal) - 1] for literal in clause})
            for left, right in itertools.combinations(variables, 2):
                edges.add((left, right))
        return edges
    if fixture.constraint_family == "assignment":
        edges = {(worker, job) for worker, job, _cost in fixture.assignment_costs}
        for left_worker, left_job, right_worker, right_job, _cost in fixture.pairwise_costs:
            edges.add((f"{left_worker}={left_job}", f"{right_worker}={right_job}"))
        return edges
    return {tuple(sorted(edge)) for edge in fixture.precedence}


def _solution_valid_for_final(fixture: BridgeFixture, metrics: SolveMetrics) -> bool:
    if metrics.status == "unsat":
        return fixture.expected_status == "unsat"
    return (
        metrics.solution is not None and _constraint_violation_count(fixture, metrics.solution) == 0
    )


def _attempt_disagrees(candidate: SolveMetrics, baseline: SolveMetrics) -> bool:
    return not _same_exact_outcome(candidate, baseline)


def _same_exact_outcome(candidate: SolveMetrics, baseline: SolveMetrics) -> bool:
    return candidate.status == baseline.status and candidate.solution == baseline.solution


def _serialize_solution(
    fixture: BridgeFixture,
    solution: Candidate | None,
) -> list[bool] | list[str] | None:
    if solution is None:
        return None
    if fixture.constraint_family == "sat":
        return [bool(value) for value in solution]
    return [str(value) for value in solution]


def _validate_density(density: Mapping[str, Any]) -> None:
    _require(density["mean_before"] > density["mean_after"], "density_before_after")
    _require(density["mean_restored_sparsity_delta"] > 0, "density_before_after")
    for row in density["by_fixture"].values():
        _require(0.0 <= row["after"] <= row["before"] <= 1.0, "density_before_after")
        _require(row["before_edges"] >= row["after_edges"], "density_before_after")
        _require(row["possible_edges"] > 0, "density_before_after")


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == EXPECTED_FIXTURE_COUNT * EXPECTED_SOURCE_COUNT, "row_records")
    for row in rows:
        _require(row["assumption_source"] in ASSUMPTION_SOURCES, "assumption_source")
        _require(row["solver_authoritative"] is True, "solver_authoritative")
        _require(row["accepted_without_verification"] is False, "accepted_without_verification")
        _require(row["solution_valid"] is True, "solution_valid")
        _require(row["final_matches_exact"] is True, "fallback_completeness_rate")
        _require(row["unsafe_false_accept"] is False, "unsafe_false_accepts")
        _require(row["hardware_speedup_claim"] is False, "hardware_speedup_claim")
        _require(row["density_before"] >= row["density_after"], "density_before_after")
        if row["assumption_decision"] == "accepted":
            _require(not row["minimal_core_ids"], "minimal_core")
        else:
            _require(row["fallback_used"] is True, "fallback_used")
            _require(row["minimal_core_ids"], "minimal_core")
            _require(row["minimal_core_assumptions"], "minimal_core")
            _require(row["minimal_core_evidence"], "minimal_core")
            _require(
                all(
                    item["without_core_assumption_matches_exact"]
                    for item in row["minimal_core_evidence"]
                ),
                "minimal_core",
            )


def _normalize_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "passed"}
    return dict(item)


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _honest_verdict(
    ready: bool,
    blockers: Sequence[str],
    diagnostic: Mapping[str, Any],
) -> str:
    if ready:
        return (
            "complete: active, p-bit, and p-dit assumptions stayed advisory across "
            f"{diagnostic['fixture_count']} fixtures, minimal cores diagnosed bad "
            "assumptions, fallback completeness was 1.0, and no hardware speedup "
            "was claimed"
        )
    return "blocked: minimal-core p-bit/p-dit bridge blocked by " + ", ".join(blockers)


def _rate(numerator: float, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(float(numerator) / denominator, 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_int(value: int | None, label: str) -> int:
    if value is None:
        raise ValueError(label)
    return int(value)


def _assumption_core_id(fixture: BridgeFixture, assumption: str) -> str:
    safe = (
        assumption.replace("!", "not_").replace("@", "_at_").replace("=", "_eq_").replace(" ", "_")
    )
    return f"core:{fixture.fixture_id}:{safe}"


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)
    run(result_path=args.output, tests_run=args.test_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main())
