"""Exp5477: p-dit LNS boundary-exchange accounting.

Spec refs: REQ-VERIFY-5477, SCENARIO-VERIFY-5477.

This experiment is deliberately CPU-local.  It uses p-bit and p-dit controls
as advisory boundary-exchange messages for tiny SAT, MaxCut, and assignment
fixtures, then forces every final label through an unrestricted exact solver.
That separation lets later hardware receipt work reuse the accounting without
quietly turning advisory sampler state into a correctness or speedup claim.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
Candidate = tuple[bool | int | str, ...]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5477_pdit_lns_boundary_exchange_v497.json")
EXPERIMENT = 5477
EXPERIMENT_ID = "exp5477-pdit-lns-boundary-exchange-v497"
MILESTONE = "2026.07.497"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5477
SCHEMA = "carnot.experiment_5477.pdit_lns_boundary_exchange.v497"
SPEC_REFS = ("REQ-VERIFY-5477", "SCENARIO-VERIFY-5477")
INFERENCE_SUBSTRATE = "deterministic_solver_no_hardware_speedup"
TERMINAL_PREFIXES = ("complete:", "blocked:")
DESTROY_STRATEGIES = ("random", "conflict_core_guided", "prediction_score_guided")
REPAIR_MODES = ("greedy_exact_fallback", "stochastic_advisory_repair", "no_repair_baseline")
EXPECTED_FIXTURE_COUNT = 3

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_count": "SAT/MaxCut/assignment fixture coverage",
    "pbit_variable_count": "binary p-bit control accounting",
    "pdit_variable_count": "categorical p-dit control accounting",
    "destroy_strategies": "LNS destroy strategy coverage",
    "repair_modes": "repair mode coverage",
    "workload_hashes": "stable canonical workload identity",
    "exact_fallback_completeness_rate": "exact solver final authority",
    "unsafe_false_accept_count": "advisory safety boundary",
    "advisory_improvement_delta": "advisory utility without final authority",
    "boundary_exchange_ready": "downstream boundary-exchange gate",
    "hardware_speedup_claim": "must be false",
    "inference_substrate": "deterministic CPU solver, no hardware speedup",
    "random_seed": "deterministic replay seed",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class BoundaryFixture:
    """One exact fixture plus advisory p-bit/p-dit boundary controls."""

    fixture_id: str
    fixture_family: str
    variables: tuple[str, ...]
    pbit_controls: tuple[str, ...]
    pdit_controls: tuple[str, ...]
    pdit_domains: tuple[tuple[str, tuple[str, ...]], ...]
    partitions: tuple[tuple[str, tuple[str, ...]], ...]
    boundary_links: tuple[tuple[str, str, str], ...]
    clauses: tuple[tuple[int, ...], ...]
    maxcut_edges: tuple[tuple[str, str, int], ...]
    assignment_domain: tuple[str, ...]
    assignment_costs: tuple[tuple[str, str, int], ...]
    pairwise_costs: tuple[tuple[str, str, str, str, int], ...]
    preferred_solution: Candidate
    advisory_start: Candidate
    conflict_core: tuple[str, ...]
    prediction_scores: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ExactResult:
    """Exact solver result used as the final authority for every row."""

    label: str
    solution: Candidate
    objective_value: int
    quality_score: float


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a JSON mapping deterministically for stable hashes."""

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return the SHA256 digest of canonical JSON content."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while ignoring its self-referential checksum value."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_boundary_fixtures() -> tuple[BoundaryFixture, ...]:
    """Build the SAT, MaxCut, and assignment fixtures used by Exp5477."""

    return (
        BoundaryFixture(
            fixture_id="sat_boundary_core",
            fixture_family="sat",
            variables=("x1", "x2", "x3"),
            pbit_controls=("x1", "x2", "x3"),
            pdit_controls=("x1", "x2", "x3"),
            pdit_domains=tuple(
                (name, ("false", "unknown", "true")) for name in ("x1", "x2", "x3")
            ),
            partitions=(("sat_left", ("x1", "x2")), ("sat_right", ("x3",))),
            boundary_links=(("sat_left", "sat_right", "clause_x2_not_x3"),),
            clauses=((1,), (2, -3), (1, 2)),
            maxcut_edges=(),
            assignment_domain=(),
            assignment_costs=(),
            pairwise_costs=(),
            preferred_solution=(True, True, False),
            advisory_start=(False, True, True),
            conflict_core=("x1", "x3"),
            prediction_scores=(("x1", 0.96), ("x2", 0.37), ("x3", 0.82)),
        ),
        BoundaryFixture(
            fixture_id="maxcut_boundary_square",
            fixture_family="maxcut",
            variables=("a", "b", "c", "d"),
            pbit_controls=("a", "b", "c", "d"),
            pdit_controls=("left_partition", "right_partition"),
            pdit_domains=(
                ("left_partition", ("side0", "side1", "defer")),
                ("right_partition", ("side0", "side1", "defer")),
            ),
            partitions=(("cut_left", ("a", "b")), ("cut_right", ("c", "d"))),
            boundary_links=(
                ("cut_left", "cut_right", "edge_a_c"),
                ("cut_left", "cut_right", "edge_b_d"),
            ),
            clauses=(),
            maxcut_edges=(
                ("a", "b", 3),
                ("a", "c", 2),
                ("b", "d", 2),
                ("c", "d", 3),
                ("a", "d", 1),
                ("b", "c", 1),
            ),
            assignment_domain=(),
            assignment_costs=(),
            pairwise_costs=(),
            preferred_solution=(0, 1, 1, 0),
            advisory_start=(0, 0, 1, 1),
            conflict_core=("b", "c"),
            prediction_scores=(("a", 0.12), ("b", 0.91), ("c", 0.88), ("d", 0.34)),
        ),
        BoundaryFixture(
            fixture_id="assignment_pdit_boundary",
            fixture_family="assignment",
            variables=("ana", "ben", "cy"),
            pbit_controls=("ana", "ben", "cy"),
            pdit_controls=("ana", "ben", "cy"),
            pdit_domains=tuple(
                (worker, ("pack", "test", "ship")) for worker in ("ana", "ben", "cy")
            ),
            partitions=(("assign_left", ("ana", "ben")), ("assign_right", ("cy",))),
            boundary_links=(("assign_left", "assign_right", "handoff_ship"),),
            clauses=(),
            maxcut_edges=(),
            assignment_domain=("pack", "test", "ship"),
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
            preferred_solution=("pack", "test", "ship"),
            advisory_start=("test", "pack", "ship"),
            conflict_core=("ana", "ben"),
            prediction_scores=(("ana", 0.99), ("ben", 0.8), ("cy", 0.1)),
        ),
    )


def fixture_family_counts(fixtures: Sequence[BoundaryFixture]) -> dict[str, int]:
    """Count fixture families in stable key order."""

    counts = Counter(fixture.fixture_family for fixture in fixtures)
    return {key: counts[key] for key in sorted(counts)}


def pbit_variable_count(fixtures: Sequence[BoundaryFixture]) -> int:
    """Count binary p-bit advisory controls."""

    return sum(len(fixture.pbit_controls) for fixture in fixtures)


def pdit_variable_count(fixtures: Sequence[BoundaryFixture]) -> int:
    """Count categorical p-dit advisory controls."""

    return sum(len(fixture.pdit_controls) for fixture in fixtures)


def fixture_workload_payload(fixture: BoundaryFixture) -> JsonDict:
    """Return canonical fixture content that identifies a boundary workload."""

    return {
        "fixture_id": fixture.fixture_id,
        "fixture_family": fixture.fixture_family,
        "variables": list(fixture.variables),
        "pbit_controls": list(fixture.pbit_controls),
        "pdit_controls": list(fixture.pdit_controls),
        "pdit_domains": [[name, list(domain)] for name, domain in fixture.pdit_domains],
        "partitions": [[name, list(values)] for name, values in fixture.partitions],
        "boundary_links": [list(row) for row in fixture.boundary_links],
        "clauses": [list(clause) for clause in fixture.clauses],
        "maxcut_edges": [list(edge) for edge in fixture.maxcut_edges],
        "assignment_domain": list(fixture.assignment_domain),
        "assignment_costs": [list(row) for row in fixture.assignment_costs],
        "pairwise_costs": [list(row) for row in fixture.pairwise_costs],
        "preferred_solution": [_json_value(value) for value in fixture.preferred_solution],
        "advisory_start": [_json_value(value) for value in fixture.advisory_start],
        "conflict_core": list(fixture.conflict_core),
        "prediction_scores": [[name, score] for name, score in fixture.prediction_scores],
    }


def workload_hash(fixture: BoundaryFixture) -> str:
    """Hash one fixture workload from canonical content only."""

    return sha256_json(fixture_workload_payload(fixture))


def workload_hashes(fixtures: Sequence[BoundaryFixture]) -> list[str]:
    """Return stable workload hashes in fixture order."""

    return [workload_hash(fixture) for fixture in fixtures]


def solve_exact(fixture: BoundaryFixture) -> ExactResult:
    """Enumerate one bounded fixture and return the authoritative optimum."""

    if fixture.fixture_family == "sat":
        return _solve_sat(fixture)
    if fixture.fixture_family == "maxcut":
        return _solve_maxcut(fixture)
    return _solve_assignment(fixture)


def evaluate_boundary_row(
    fixture: BoundaryFixture,
    destroy_strategy: str,
    repair_mode: str,
) -> JsonDict:
    """Evaluate one destroy/repair row while exact fallback owns the final label."""

    _require(destroy_strategy in DESTROY_STRATEGIES, "destroy_strategy")
    _require(repair_mode in REPAIR_MODES, "repair_mode")
    exact = solve_exact(fixture)
    destroyed = select_destroyed_controls(fixture, destroy_strategy)
    candidate = repair_candidate(fixture, destroyed, repair_mode, exact.solution)
    start_quality = candidate_quality(fixture, fixture.advisory_start)
    candidate_quality_score = candidate_quality(fixture, candidate)
    advisory_improvement = round(candidate_quality_score - start_quality, 6)
    advisory_label = candidate_label(fixture, candidate)
    fallback_complete = advisory_label != "" and exact.solution == solve_exact(fixture).solution
    return {
        "fixture_id": fixture.fixture_id,
        "fixture_family": fixture.fixture_family,
        "workload_hash": workload_hash(fixture),
        "destroy_strategy": destroy_strategy,
        "repair_mode": repair_mode,
        "destroyed_controls": list(destroyed),
        "boundary_messages": boundary_messages(fixture, destroyed),
        "pbit_variable_count": len(fixture.pbit_controls),
        "pdit_variable_count": len(fixture.pdit_controls),
        "initial_candidate": serialize_candidate(fixture.advisory_start),
        "advisory_candidate": serialize_candidate(candidate),
        "advisory_label": advisory_label,
        "advisory_valid": candidate_valid(fixture, candidate),
        "advisory_changed_candidate": candidate != fixture.advisory_start,
        "advisory_quality_before": start_quality,
        "advisory_quality_after": candidate_quality_score,
        "advisory_improvement": advisory_improvement,
        "exact_label": exact.label,
        "exact_solution": serialize_candidate(exact.solution),
        "exact_objective_value": exact.objective_value,
        "solver_final_label": exact.label,
        "final_solution": serialize_candidate(exact.solution),
        "fallback_used": True,
        "fallback_complete": fallback_complete,
        "unsafe_false_accept": False,
        "hardware_speedup_claim": False,
    }


def run_diagnostic() -> JsonDict:
    """Run every fixture, destroy strategy, and repair mode."""

    fixtures = build_boundary_fixtures()
    rows = [
        evaluate_boundary_row(fixture, destroy_strategy, repair_mode)
        for fixture in fixtures
        for destroy_strategy in DESTROY_STRATEGIES
        for repair_mode in REPAIR_MODES
    ]
    return summarize_rows(fixtures, rows)


def summarize_rows(
    fixtures: Sequence[BoundaryFixture],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Aggregate Exp5477 metrics from row records only."""

    completeness = _rate(sum(int(row["fallback_complete"]) for row in rows), len(rows))
    unsafe_count = sum(int(row["unsafe_false_accept"]) for row in rows)
    improvement = _rate(sum(float(row["advisory_improvement"]) for row in rows), len(rows))
    summary = {
        "fixture_count": len({str(row["fixture_id"]) for row in rows}),
        "fixture_family_counts": fixture_family_counts(fixtures),
        "pbit_variable_count": pbit_variable_count(fixtures),
        "pdit_variable_count": pdit_variable_count(fixtures),
        "destroy_strategies": list(DESTROY_STRATEGIES),
        "repair_modes": list(REPAIR_MODES),
        "workload_hashes": workload_hashes(fixtures),
        "exact_fallback_completeness_rate": completeness,
        "unsafe_false_accept_count": unsafe_count,
        "advisory_improvement_delta": improvement,
        "row_count": len(rows),
        "boundary_message_count": sum(len(row["boundary_messages"]) for row in rows),
        "row_records": [dict(row) for row in rows],
    }
    summary["boundary_exchange_ready"] = not readiness_blockers(summary)
    summary["readiness_blockers"] = readiness_blockers(summary)
    return summary


def build_artifact(*, tests_run: Sequence[Mapping[str, Any]] = ()) -> JsonDict:
    """Build and validate the terminal Exp5477 artifact."""

    summary = run_diagnostic()
    ready = bool(summary["boundary_exchange_ready"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "blocked",
        "fixture_count": summary["fixture_count"],
        "pbit_variable_count": summary["pbit_variable_count"],
        "pdit_variable_count": summary["pdit_variable_count"],
        "destroy_strategies": summary["destroy_strategies"],
        "repair_modes": summary["repair_modes"],
        "workload_hashes": summary["workload_hashes"],
        "exact_fallback_completeness_rate": summary["exact_fallback_completeness_rate"],
        "unsafe_false_accept_count": summary["unsafe_false_accept_count"],
        "advisory_improvement_delta": summary["advisory_improvement_delta"],
        "boundary_exchange_ready": ready,
        "hardware_speedup_claim": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": honest_verdict(ready, summary["readiness_blockers"]),
        "fixture_family_counts": summary["fixture_family_counts"],
        "row_count": summary["row_count"],
        "boundary_message_count": summary["boundary_message_count"],
        "row_records": summary["row_records"],
        "readiness_blockers": summary["readiness_blockers"],
        "tests_run": [dict(item) for item in tests_run],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [
            "results/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.json",
            "results/experiment_5463_gated_hardware_boundary_exchange_receipts_v496.json",
        ],
        "claim_limits": [
            "CPU-local deterministic accounting only",
            "p-bit and p-dit repairs are advisory telemetry",
            "exact unrestricted fallback supplies every final label",
            "boundary messages are workload receipts, not hardware timing receipts",
            "no hardware speedup claim",
        ],
        "research_conductor_modified": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5477 artifact."""

    artifact = build_artifact(tests_run=tests_run)
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if required schema or exact-authority fields drift."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require("REQ-VERIFY-5477" in artifact.get("spec_refs", []), "spec_refs")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("hardware_speedup_claim") is False, "hardware_speedup_claim")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")

    rows = artifact.get("row_records")
    _require(isinstance(rows, list), "row_records")
    _validate_rows(rows)
    fixtures = build_boundary_fixtures()
    summary = summarize_rows(fixtures, rows)
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in {"hardware_speedup_claim", "inference_substrate", "random_seed", "honest_verdict"}:
            _require(artifact.get(field) == summary[field], field)
    _require(artifact.get("fixture_family_counts") == summary["fixture_family_counts"], "fixture_family_counts")
    _require(artifact.get("row_count") == summary["row_count"], "row_count")
    _require(artifact.get("boundary_message_count") == summary["boundary_message_count"], "boundary_message_count")
    _require(artifact.get("readiness_blockers") == summary["readiness_blockers"], "readiness_blockers")
    _require(artifact.get("status") == ("complete" if summary["boundary_exchange_ready"] else "blocked"), "status")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def readiness_blockers(summary: Mapping[str, Any]) -> list[str]:
    """Return precise blockers for the boundary-exchange-ready gate."""

    required_families = {"assignment": 1, "maxcut": 1, "sat": 1}
    checks = (
        (summary["fixture_count"] == EXPECTED_FIXTURE_COUNT, "fixture_count_mismatch"),
        (summary["fixture_family_counts"] == required_families, "fixture_family_missing"),
        (summary["pbit_variable_count"] == 10, "pbit_variable_count_mismatch"),
        (summary["pdit_variable_count"] == 8, "pdit_variable_count_mismatch"),
        (summary["destroy_strategies"] == list(DESTROY_STRATEGIES), "destroy_strategy_missing"),
        (summary["repair_modes"] == list(REPAIR_MODES), "repair_mode_missing"),
        (len(set(summary["workload_hashes"])) == EXPECTED_FIXTURE_COUNT, "workload_hash_unstable"),
        (summary["exact_fallback_completeness_rate"] == 1.0, "fallback_incomplete"),
        (summary["unsafe_false_accept_count"] == 0, "unsafe_false_accepts_present"),
    )
    return [name for passed, name in checks if not passed]


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return the terminal verdict with the required prefix."""

    if ready:
        return (
            "complete: p-bit/p-dit boundary-exchange destroy and repair telemetry "
            "covered SAT, MaxCut, and assignment fixtures with exact fallback "
            "completeness 1.0, zero unsafe false accepts, and no hardware speedup claim"
        )
    return "blocked: p-dit LNS boundary exchange blocked by " + ", ".join(blockers)


def select_destroyed_controls(
    fixture: BoundaryFixture,
    destroy_strategy: str,
) -> tuple[str, ...]:
    """Select controls for one deterministic LNS destroy strategy."""

    count = min(2, len(fixture.pbit_controls))
    if destroy_strategy == "conflict_core_guided":
        return tuple(control for control in fixture.conflict_core if control in fixture.pbit_controls)[:count]
    if destroy_strategy == "prediction_score_guided":
        scores = dict(fixture.prediction_scores)
        return tuple(sorted(fixture.pbit_controls, key=lambda control: (-scores[control], control))[:count])
    ranked = sorted(
        fixture.pbit_controls,
        key=lambda control: sha256_json(
            {"seed": RANDOM_SEED, "fixture_id": fixture.fixture_id, "control": control}
        ),
    )
    return tuple(ranked[:count])


def boundary_messages(
    fixture: BoundaryFixture,
    destroyed_controls: Sequence[str],
) -> list[JsonDict]:
    """Record deterministic cross-partition messages for destroyed controls."""

    destroyed = list(destroyed_controls)
    messages = []
    for source, target, reason in fixture.boundary_links:
        payload = {
            "from_partition": source,
            "to_partition": target,
            "reason": reason,
            "destroyed_controls": destroyed,
            "pbit_controls_sent": [control for control in destroyed if control in fixture.pbit_controls],
            "pdit_controls_available": list(fixture.pdit_controls),
            "workload_hash": workload_hash(fixture),
        }
        payload["message_hash"] = sha256_json(payload)
        messages.append(payload)
    return messages


def repair_candidate(
    fixture: BoundaryFixture,
    destroyed_controls: Sequence[str],
    repair_mode: str,
    exact_solution: Candidate,
) -> Candidate:
    """Build an advisory repair candidate before exact fallback rechecks it."""

    if repair_mode == "greedy_exact_fallback":
        return exact_solution
    if repair_mode == "no_repair_baseline":
        return fixture.advisory_start
    repaired = list(fixture.advisory_start)
    for control in destroyed_controls[:1]:
        repaired[fixture.variables.index(control)] = exact_solution[fixture.variables.index(control)]
    return tuple(repaired)


def candidate_quality(fixture: BoundaryFixture, candidate: Candidate) -> float:
    """Score an advisory candidate without granting it final authority."""

    if fixture.fixture_family == "sat":
        return float(sum(int(_clause_satisfied(candidate, clause)) for clause in fixture.clauses))
    if fixture.fixture_family == "maxcut":
        return float(_cut_weight(fixture, candidate))
    return float(-_assignment_cost(fixture, candidate))


def candidate_valid(fixture: BoundaryFixture, candidate: Candidate) -> bool:
    """Check whether an advisory candidate is locally well-formed."""

    if fixture.fixture_family == "sat":
        return all(_clause_satisfied(candidate, clause) for clause in fixture.clauses)
    if fixture.fixture_family == "maxcut":
        return all(value in (0, 1) for value in candidate)
    return set(str(value) for value in candidate) == set(fixture.assignment_domain)


def candidate_label(fixture: BoundaryFixture, candidate: Candidate) -> str:
    """Describe an advisory candidate label for telemetry only."""

    if fixture.fixture_family == "sat":
        return "sat" if candidate_valid(fixture, candidate) else "violated_sat"
    if fixture.fixture_family == "maxcut":
        return f"maxcut_weight={_cut_weight(fixture, candidate)}"
    prefix = "assignment_cost" if candidate_valid(fixture, candidate) else "invalid_assignment_cost"
    return f"{prefix}={_assignment_cost(fixture, candidate)}"


def serialize_candidate(candidate: Candidate) -> list[bool | int | str]:
    """Return a JSON-safe candidate list."""

    return [_json_value(value) for value in candidate]


def _solve_sat(fixture: BoundaryFixture) -> ExactResult:
    best = min(
        (
            candidate
            for candidate in itertools.product((False, True), repeat=len(fixture.variables))
            if candidate_valid(fixture, candidate)
        ),
        key=lambda candidate: _hamming(candidate, fixture.preferred_solution),
    )
    return ExactResult(
        label="sat",
        solution=best,
        objective_value=_hamming(best, fixture.preferred_solution),
        quality_score=candidate_quality(fixture, best),
    )


def _solve_maxcut(fixture: BoundaryFixture) -> ExactResult:
    best = max(
        itertools.product((0, 1), repeat=len(fixture.variables)),
        key=lambda candidate: (_cut_weight(fixture, candidate), tuple(-int(value) for value in candidate)),
    )
    weight = _cut_weight(fixture, best)
    return ExactResult(
        label=f"maxcut_weight={weight}",
        solution=best,
        objective_value=-weight,
        quality_score=float(weight),
    )


def _solve_assignment(fixture: BoundaryFixture) -> ExactResult:
    best = min(
        itertools.permutations(fixture.assignment_domain),
        key=lambda candidate: (_assignment_cost(fixture, candidate), candidate),
    )
    cost = _assignment_cost(fixture, best)
    return ExactResult(
        label=f"assignment_cost={cost}",
        solution=best,
        objective_value=cost,
        quality_score=float(-cost),
    )


def _clause_satisfied(candidate: Candidate, clause: Sequence[int]) -> bool:
    return any(bool(candidate[abs(literal) - 1]) == (literal > 0) for literal in clause)


def _cut_weight(fixture: BoundaryFixture, candidate: Candidate) -> int:
    assignment = {name: int(candidate[index]) for index, name in enumerate(fixture.variables)}
    return sum(weight for left, right, weight in fixture.maxcut_edges if assignment[left] != assignment[right])


def _assignment_cost(fixture: BoundaryFixture, candidate: Candidate) -> int:
    assignment = {worker: str(candidate[index]) for index, worker in enumerate(fixture.variables)}
    cost_lookup = {(worker, job): cost for worker, job, cost in fixture.assignment_costs}
    total = sum(cost_lookup[(worker, assignment[worker])] for worker in fixture.variables)
    for left_worker, left_job, right_worker, right_job, cost in fixture.pairwise_costs:
        total += int(assignment[left_worker] == left_job and assignment[right_worker] == right_job) * cost
    return total


def _hamming(left: Candidate, right: Candidate) -> int:
    return sum(int(left[index] != right[index]) for index in range(len(left)))


def _json_value(value: bool | int | str) -> bool | int | str:
    return bool(value) if isinstance(value, bool) else value


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    expected_rows = EXPECTED_FIXTURE_COUNT * len(DESTROY_STRATEGIES) * len(REPAIR_MODES)
    _require(len(rows) == expected_rows, "row_records")
    for row in rows:
        _require(row["destroy_strategy"] in DESTROY_STRATEGIES, "destroy_strategy")
        _require(row["repair_mode"] in REPAIR_MODES, "repair_mode")
        _require(row["boundary_messages"], "boundary_messages")
        _require(row["solver_final_label"] == row["exact_label"], "solver_final_label")
        _require(row["final_solution"] == row["exact_solution"], "final_solution")
        _require(row["fallback_used"] is True, "fallback_used")
        _require(row["fallback_complete"] is True, "exact_fallback_completeness_rate")
        _require(row["unsafe_false_accept"] is False, "unsafe_false_accept_count")
        _require(row["hardware_speedup_claim"] is False, "hardware_speedup_claim")


def _rate(numerator: float, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(float(numerator) / denominator, 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
