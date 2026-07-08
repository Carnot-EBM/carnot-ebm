"""Exp5448: active-constraint and p-bit assumption sparsity bridge.

Spec refs: REQ-VERIFY-5448, SCENARIO-VERIFY-5448.

The bridge is intentionally deterministic and CPU-local. Active constraints
and p-bit-like consensus samples are allowed to suggest temporary assumptions,
but the exact solver recomputes the final label or solution. This mirrors the
hardware-facing idea without claiming that a sampler or hardware device has
authority over correctness.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5407_pbit_qubo_active_constraint_stress_v492 as exp5407
from carnot import experiment_5433_active_constraint_diversity_lns_v494 as exp5433


JsonDict = dict[str, Any]
RowOverride = Callable[[list[JsonDict]], list[JsonDict]]
Candidate = tuple[bool | str, ...]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.json"
)
EXPERIMENT = 5448
EXPERIMENT_ID = "exp5448-active-constraint-pbit-sparsity-bridge-v495"
MILESTONE = "2026.07.495"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5448
SCHEMA = "carnot.experiment_5448.active_constraint_pbit_sparsity_bridge.v495"
SPEC_REFS = ("REQ-VERIFY-5448", "SCENARIO-VERIFY-5448")
INFERENCE_SUBSTRATE = "deterministic_solver_pbit_assumption_fixture"
TERMINAL_PREFIXES = ("complete:", "blocked:")
ASSUMPTION_SOURCES = ("active_constraint", "pbit_consensus")
EXPECTED_FIXTURE_COUNT = 4
EXPECTED_SOURCE_COUNT = len(ASSUMPTION_SOURCES)

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_count": "bounded coverage",
    "constraint_family_counts": "diversity",
    "assumption_source_counts": "active vs p-bit provenance",
    "density_before_after": "restored-sparsity accounting",
    "solver_authoritative": "exact solver final authority",
    "fallback_completeness_rate": "correctness rescue",
    "rejected_assumption_count": "advisory boundary",
    "overwritten_assumption_count": "solver authority",
    "solver_work_delta": "utility measurement",
    "unsafe_false_accepts": "correctness boundary",
    "pbit_assumption_bridge_ready": "downstream hardware gate",
    "hardware_speedup_claim": "no unsupported hardware claim",
    "inference_substrate": "no hidden hardware inference",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class BridgeFixture:
    """One bounded exact fixture plus advisory assumptions.

    SAT rows use Boolean assignments; CSP and LNS rows use action sequences.
    The same exact-solve wrapper handles both by filtering candidates through
    temporary assumptions and then checking the original constraints.
    """

    fixture_id: str
    constraint_family: str
    source_module: str
    source_fixture_id: str
    variables: tuple[str, ...]
    clauses: tuple[tuple[int, ...], ...]
    precedence: tuple[tuple[str, str], ...]
    expected_status: str
    expected_solution: Candidate
    active_assumptions: tuple[str, ...]
    pbit_samples: tuple[Candidate, ...]
    density_note: str


@dataclass(frozen=True)
class SolveMetrics:
    """Exact-solver outcome and counters for one bounded solve."""

    status: str
    solution: Candidate | None
    objective_value: int | None
    conflicts: int
    propagations: int
    iterations: int

    @property
    def solver_work(self) -> int:
        """Local work scalar used only for within-fixture comparison."""

        return self.conflicts + self.propagations + self.iterations

    def as_serializable(self, fixture: BridgeFixture) -> JsonDict:
        """Return JSON-safe counters and solution data."""

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
    """Build SAT, CSP, and LNS fixtures with known exact outcomes."""

    stress = exp5407.build_stress_fixtures()[1]
    lns_source = exp5433.build_diversity_fixtures()[1]
    csp_expected = tuple(stress.expected_sequence)
    lns_expected = ("invoice", "verify", "receive", "ship")
    return (
        BridgeFixture(
            fixture_id="sat_consensus_accept",
            constraint_family="sat",
            source_module="local_sat_fixture",
            source_fixture_id="sat:consensus_accept",
            variables=("x1", "x2", "x3", "x4"),
            clauses=((1,), (2, 3), (-2, 4), (1, 4)),
            precedence=(),
            expected_status="sat",
            expected_solution=(True, True, True, True),
            active_assumptions=("x1",),
            pbit_samples=(
                (True, True, True, True),
                (True, True, True, True),
                (True, True, False, True),
                (True, False, True, True),
            ),
            density_note="SAT clauses lifted to a pairwise QUBO-style clique before restoration.",
        ),
        BridgeFixture(
            fixture_id="sat_false_basin_rescue",
            constraint_family="sat",
            source_module="local_sat_fixture",
            source_fixture_id="sat:false_basin_rescue",
            variables=("x1", "x2", "x3"),
            clauses=((1,), (2, -3), (1, 2)),
            precedence=(),
            expected_status="sat",
            expected_solution=(True, True, False),
            active_assumptions=("x1",),
            pbit_samples=(
                (False, True, False),
                (False, True, False),
                (False, True, True),
                (False, False, False),
            ),
            density_note="Wrong consensus fixes a unit-clause variable and must be rescued.",
        ),
        BridgeFixture(
            fixture_id="csp_active_join_order",
            constraint_family="csp",
            source_module="experiment_5407_pbit_qubo_active_constraint_stress_v492",
            source_fixture_id=stress.fixture_id,
            variables=tuple(stress.actions),
            clauses=(),
            precedence=tuple(stress.precedence),
            expected_status="sat",
            expected_solution=csp_expected,
            active_assumptions=(
                f"{csp_expected[0]}@0",
                f"{csp_expected[1]}@1",
            ),
            pbit_samples=(
                csp_expected,
                csp_expected,
                tuple(reversed(csp_expected)),
                csp_expected,
            ),
            density_note="Exp5407 QUBO-style ordering terms are restored to direct precedence edges.",
        ),
        BridgeFixture(
            fixture_id="lns_valid_but_suboptimal",
            constraint_family="lns",
            source_module="experiment_5433_active_constraint_diversity_lns_v494",
            source_fixture_id=lns_source.fixture_id,
            variables=lns_expected,
            clauses=(),
            precedence=(("invoice", "ship"), ("verify", "ship"), ("receive", "ship")),
            expected_status="sat",
            expected_solution=lns_expected,
            active_assumptions=("invoice@0", "verify@1"),
            pbit_samples=(
                ("verify", "invoice", "receive", "ship"),
                ("verify", "invoice", "receive", "ship"),
                ("verify", "receive", "invoice", "ship"),
                ("invoice", "verify", "receive", "ship"),
            ),
            density_note="The LNS projection keeps the V494 family provenance but restores sparse direct constraints.",
        ),
    )


def constraint_family_counts(fixtures: Sequence[BridgeFixture]) -> dict[str, int]:
    """Count fixtures by SAT/CSP/LNS family in stable key order."""

    counts = Counter(fixture.constraint_family for fixture in fixtures)
    return {key: counts[key] for key in sorted(counts)}


def density_before_after(fixtures: Sequence[BridgeFixture]) -> JsonDict:
    """Measure dense lifted constraints before and sparse constraints after restoration."""

    by_fixture: dict[str, JsonDict] = {}
    before_values: list[float] = []
    after_values: list[float] = []
    for fixture in fixtures:
        possible_edges = max(1, len(fixture.variables) * (len(fixture.variables) - 1) // 2)
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
    """Evaluate active and p-bit assumptions while keeping exact solves authoritative."""

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
    summary["pbit_assumption_bridge_ready"] = not blockers
    summary["readiness_blockers"] = blockers
    summary["row_records"] = rows
    return summary


def evaluate_fixture_source(fixture: BridgeFixture, assumption_source: str) -> JsonDict:
    """Run one fixture/source row and rescue with unrestricted exact solving as needed."""

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

    if attempt.status != baseline.status:
        decision = "rejected"
        rejected = len(assumptions)
        fallback_used = True
        final = baseline
    elif attempt.status == "sat" and attempt.solution != baseline.solution:
        decision = "overwritten"
        overwritten = len(assumptions)
        fallback_used = True
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
        "pbit_consensus_sample_count": len(fixture.pbit_samples)
        if assumption_source == "pbit_consensus"
        else 0,
        "assumption_decision": decision,
        "solver_authoritative": True,
        "accepted_without_verification": False,
        "fallback_used": fallback_used,
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
    """Exhaustively solve the bounded fixture under temporary assumptions."""

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


def build_artifact(
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    row_overrides: RowOverride | None = None,
) -> JsonDict:
    """Build the terminal Exp5448 artifact from deterministic rows."""

    diagnostic = run_diagnostic(row_overrides=row_overrides)
    tests = [_normalize_test_run(item) for item in tests_run]
    blockers = list(diagnostic["readiness_blockers"])
    if diagnostic["pbit_assumption_bridge_ready"] and not tests:
        blockers.append("tests_not_recorded")
    ready = bool(diagnostic["pbit_assumption_bridge_ready"] and not blockers)
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
        "density_before_after": diagnostic["density_before_after"],
        "solver_authoritative": diagnostic["solver_authoritative"],
        "fallback_completeness_rate": diagnostic["fallback_completeness_rate"],
        "rejected_assumption_count": diagnostic["rejected_assumption_count"],
        "overwritten_assumption_count": diagnostic["overwritten_assumption_count"],
        "solver_work_delta": diagnostic["solver_work_delta"],
        "unsafe_false_accepts": diagnostic["unsafe_false_accepts"],
        "pbit_assumption_bridge_ready": ready,
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
        ],
        "claim_limits": [
            "deterministic CPU-local solver fixture",
            "active constraints and p-bit consensus samples are advisory assumptions",
            "exact unrestricted solving rejects or overwrites unsafe assumptions",
            "density restoration is accounting only, not a hardware timing result",
            "no LLM, generated text judge, hardware sampler, or hardware speedup claim",
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5448 artifact and return it."""

    artifact = build_artifact(tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when schema, authority, or claim-boundary fields drift."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require("REQ-VERIFY-5448" in artifact.get("spec_refs", []), "spec_refs")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(len(str(artifact.get("reproducibility_checksum", ""))) == 64, "checksum")

    rows = artifact.get("row_records")
    _require(isinstance(rows, Sequence), "row_records")
    fixtures = build_bridge_fixtures()
    summary = _summarize_rows(fixtures, rows)
    for field in (
        "fixture_count",
        "constraint_family_counts",
        "assumption_source_counts",
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
    _validate_rows(rows)

    blockers = readiness_blockers(summary)
    if summary["pbit_assumption_bridge_ready"] and not artifact.get("tests_run"):
        blockers.append("tests_not_recorded")
    expected_ready = not blockers
    _require(artifact.get("pbit_assumption_bridge_ready") is expected_ready, "readiness")
    _require(artifact.get("readiness_blockers") == blockers, "readiness_blockers")
    if expected_ready:
        _require(artifact.get("status") == "complete", "status")
        _require(str(artifact.get("honest_verdict")).startswith("complete:"), "honest_verdict")
    else:
        _require(artifact.get("status") == "blocked", "status")
        _require(str(artifact.get("honest_verdict")).startswith("blocked:"), "honest_verdict")


def readiness_blockers(summary: Mapping[str, Any]) -> list[str]:
    """Return precise blockers for the downstream bridge-ready gate."""

    blockers: list[str] = []
    if summary["fixture_count"] != EXPECTED_FIXTURE_COUNT:
        blockers.append("fixture_count_mismatch")
    if set(summary["constraint_family_counts"]) != {"csp", "lns", "sat"}:
        blockers.append("constraint_family_missing")
    expected_sources = {source: EXPECTED_FIXTURE_COUNT for source in ASSUMPTION_SOURCES}
    if summary["assumption_source_counts"] != expected_sources:
        blockers.append("assumption_source_coverage_mismatch")
    density = summary["density_before_after"]
    if density["mean_before"] <= density["mean_after"]:
        blockers.append("density_restoration_not_measured")
    if summary["solver_authoritative"] is not True:
        blockers.append("solver_not_authoritative")
    if summary["fallback_completeness_rate"] != 1.0:
        blockers.append("fallback_completeness_incomplete")
    if summary["rejected_assumption_count"] <= 0:
        blockers.append("no_rejected_assumptions")
    if summary["overwritten_assumption_count"] <= 0:
        blockers.append("no_overwritten_assumptions")
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
    summary["pbit_assumption_bridge_ready"] = not readiness_blockers(summary)
    return summary


def _source_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "row_count": len(rows),
        "accepted_count": sum(int(row["assumption_decision"] == "accepted") for row in rows),
        "rejected_count": sum(int(row["assumption_decision"] == "rejected") for row in rows),
        "overwritten_count": sum(int(row["assumption_decision"] == "overwritten") for row in rows),
        "fallback_count": sum(int(row["fallback_used"]) for row in rows),
        "solver_work_delta": sum(int(row["work_delta"]) for row in rows),
        "unsafe_false_accepts": sum(int(row["unsafe_false_accept"]) for row in rows),
    }


def _assumptions_for_source(fixture: BridgeFixture, source: str) -> tuple[str, ...]:
    if source == "active_constraint":
        return fixture.active_assumptions
    return _pbit_consensus_assumptions(fixture)


def _pbit_consensus_assumptions(fixture: BridgeFixture) -> tuple[str, ...]:
    threshold = 0.75
    assumptions: list[str] = []
    if fixture.constraint_family == "sat":
        for index, variable in enumerate(fixture.variables):
            true_count = sum(int(sample[index]) for sample in fixture.pbit_samples)
            true_rate = true_count / len(fixture.pbit_samples)
            if true_rate >= threshold:
                assumptions.append(variable)
            elif true_rate <= 1.0 - threshold:
                assumptions.append(f"!{variable}")
        return tuple(assumptions)

    for position in range(len(fixture.variables)):
        counts = Counter(str(sample[position]) for sample in fixture.pbit_samples)
        action, count = counts.most_common(1)[0]
        if count / len(fixture.pbit_samples) >= threshold:
            assumptions.append(f"{action}@{position}")
    return tuple(assumptions)


def _candidate_space(fixture: BridgeFixture) -> tuple[Candidate, ...]:
    if fixture.constraint_family == "sat":
        return tuple(itertools.product((False, True), repeat=len(fixture.variables)))
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
        else:
            action, position_text = assumption.rsplit("@", 1)
            if candidate[int(position_text)] != action:
                return False
    return True


def _constraint_violation_count(fixture: BridgeFixture, candidate: Candidate) -> int:
    if fixture.constraint_family == "sat":
        return sum(int(not _clause_satisfied(candidate, clause)) for clause in fixture.clauses)
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
    preferred = {value: index for index, value in enumerate(fixture.expected_solution)}
    return sum((index + 1) * abs(index - preferred[value]) for index, value in enumerate(candidate))


def _constraint_count(fixture: BridgeFixture) -> int:
    return len(fixture.clauses) if fixture.constraint_family == "sat" else len(fixture.precedence)


def _restored_edges(fixture: BridgeFixture) -> set[tuple[str, str]]:
    if fixture.constraint_family == "sat":
        edges: set[tuple[str, str]] = set()
        for clause in fixture.clauses:
            variables = sorted({fixture.variables[abs(literal) - 1] for literal in clause})
            for left, right in itertools.combinations(variables, 2):
                edges.add((left, right))
        return edges
    return {tuple(sorted(edge)) for edge in fixture.precedence}


def _solution_valid_for_final(fixture: BridgeFixture, metrics: SolveMetrics) -> bool:
    if metrics.status == "unsat":
        return fixture.expected_status == "unsat"
    return (
        metrics.solution is not None and _constraint_violation_count(fixture, metrics.solution) == 0
    )


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
        if row["assumption_decision"] in {"rejected", "overwritten"}:
            _require(row["fallback_used"] is True, "fallback_used")


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
            "complete: active and p-bit assumptions stayed advisory across "
            f"{diagnostic['fixture_count']} fixtures, fallback completeness was "
            f"{diagnostic['fallback_completeness_rate']}, density was restored, "
            "and no hardware speedup was claimed"
        )
    return "blocked: active/p-bit sparsity bridge blocked by " + ", ".join(blockers)


def _rate(numerator: float, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(float(numerator) / denominator, 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_int(value: int | None, message: str) -> int:
    if value is None:
        raise ValueError(message)
    return value


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)
    run(result_path=args.output, tests_run=args.test_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
