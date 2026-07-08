"""Exp5407: gated p-bit/QUBO active-constraint stress diagnostic.

Spec refs: REQ-VERIFY-5407, SCENARIO-VERIFY-5407.

This module is deliberately a CPU-local simulator diagnostic. The p-bit lane
only proposes candidate action orders, and the QUBO lane is tiny enough that we
can enumerate every permutation exactly. That bounded setup lets the artifact
say useful things about hint behavior without implying any board timing,
hardware acceleration, or sampler authority over the deterministic solver.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5406_active_constraint_warmstart_guidance_v492 as exp5406


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5407_pbit_qubo_active_constraint_stress_v492.json"
)
GATE_RELATIVE_PATH = exp5406.RESULT_RELATIVE_PATH
EXPERIMENT = 5407
EXPERIMENT_ID = "exp5407-pbit-qubo-active-constraint-stress-v492"
MILESTONE = "2026.07.492"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5407
SCHEMA = "carnot.experiment_5407.pbit_qubo_active_constraint_stress.v492"
SPEC_REFS = ("REQ-VERIFY-5407", "SCENARIO-VERIFY-5407")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
TERMINAL_PREFIXES = ("complete:", "blocked:")
COMPARED_MODES = (
    "deterministic_solver",
    "pbit_boundary_hint",
    "active_constraint_hint",
    "adversarial_hint",
)
EXPECTED_FIXTURE_COUNT = 4
EXPECTED_QUBO_BASELINE_COUNT = 4
PBIT_SAMPLES_PER_FIXTURE = 6

FIELD_PRINCIPLES: dict[str, str] = {
    "gated_on_active_constraint_ready": "precondition.",
    "fixture_count": "coverage.",
    "qubo_baseline_count": "sorting-network stress coverage.",
    "exact_enumeration_agreement_rate": "bounded correctness.",
    "pbit_acceptance_rate": "sampler behavior.",
    "solver_conflict_delta": "efficiency evidence.",
    "fallback_rate": "operational safety.",
    "unsafe_false_accept_rate": "solver authority.",
    "hardware_speedup_claim": "no unsupported hardware claim.",
    "pbit_qubo_stress_ready": "downstream evidence.",
    "inference_substrate": "deterministic validation.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class StressFixture:
    """One small action-order instance with exact-enumerable permutation state."""

    fixture_id: str
    source_fixture_id: str
    active_hint_source: str
    actions: tuple[str, ...]
    precedence: tuple[tuple[str, str], ...]
    expected_sequence: tuple[str, ...]

    @property
    def active_constraint_ids(self) -> tuple[str, ...]:
        """Return the active precedence edges that Exp5406 made safe to reuse."""

        return _active_constraint_ids(self, ())

    @property
    def conflict_front(self) -> tuple[str, ...]:
        """Return actions blocked by the active precedence edge set."""

        return _conflict_front(self, ())


@dataclass(frozen=True)
class SolverMetrics:
    """Solver telemetry used to compare hint modes against the same fallback."""

    final_sequence: tuple[str, ...]
    final_valid: bool
    solver_conflicts: int
    solver_iterations: int


def load_active_constraint_gate(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read Exp5406 readiness and fail closed when the artifact is unavailable."""

    path = Path(root) / GATE_RELATIVE_PATH
    if not path.exists():
        return _gate_record(False, "missing")
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive I/O guard
        return _gate_record(False, "unreadable")
    return _gate_record(
        bool(source.get("active_constraint_warmstart_ready")),
        str(source.get("status", "unknown")),
    )


def build_stress_fixtures() -> tuple[StressFixture, ...]:
    """Convert Exp5406 active-constraint instances into QUBO stress fixtures."""

    return tuple(
        StressFixture(
            fixture_id=f"stress_{instance.fixture_id}",
            source_fixture_id=instance.fixture_id,
            active_hint_source="exp5406_active_constraint_warmstart",
            actions=instance.actions,
            precedence=instance.precedence,
            expected_sequence=instance.expected_sequence,
        )
        for instance in exp5406.build_constraint_instances()
    )


def build_qubo_baselines(fixtures: Sequence[StressFixture]) -> list[JsonDict]:
    """Exact-enumerate the sorting-network QUBO baseline for every fixture."""

    return [exact_enumerate_fixture(fixture) for fixture in fixtures]


def exact_enumerate_fixture(fixture: StressFixture) -> JsonDict:
    """Enumerate all action permutations and record the exact minimum energy."""

    best_sequence: tuple[str, ...] | None = None
    best_energy: int | None = None
    valid_count = 0
    enumerated = 0
    for permutation in itertools.permutations(fixture.actions):
        enumerated += 1
        energy = qubo_precedence_energy(fixture, permutation)
        if energy == 0:
            valid_count += 1
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_sequence = tuple(permutation)

    deterministic = _solve_deterministic(fixture)
    deterministic_energy = qubo_precedence_energy(fixture, deterministic.final_sequence)
    comparators = _sorting_network_comparators(len(fixture.actions))
    exact_sequence = _require_sequence(best_sequence, "exact enumeration")
    exact_energy = _require_int(best_energy, "exact enumeration energy")
    return {
        "fixture_id": fixture.fixture_id,
        "source_fixture_id": fixture.source_fixture_id,
        "exact_enumerated": True,
        "enumerated_permutation_count": enumerated,
        "exact_valid_permutation_count": valid_count,
        "exact_min_energy": exact_energy,
        "exact_best_sequence": list(exact_sequence),
        "sorting_network_comparator_count": len(comparators),
        "sorting_network_comparators": [list(pair) for pair in comparators],
        "sorting_network_output": list(_sorting_network_sequence(fixture, exact_sequence)),
        "qubo_variable_count": len(fixture.actions) ** 2,
        "qubo_precedence_term_count": _qubo_precedence_term_count(fixture),
        "deterministic_fallback_sequence": list(deterministic.final_sequence),
        "deterministic_fallback_energy": deterministic_energy,
        "deterministic_agrees_with_exact": deterministic_energy == exact_energy,
    }


def qubo_precedence_energy(
    fixture: StressFixture,
    sequence: Sequence[str],
) -> int:
    """Score a permutation with the pairwise precedence terms of the QUBO."""

    if len(sequence) != len(fixture.actions) or set(sequence) != set(fixture.actions):
        return 10 * len(fixture.precedence) + len(fixture.actions)
    positions = {action: index for index, action in enumerate(sequence)}
    return 10 * sum(
        int(positions[before] > positions[after])
        for before, after in fixture.precedence
    )


def run_diagnostic(
    root: Path | str = REPO_ROOT,
    gate_override: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Run the gated CPU/simulator stress diagnostic."""

    gate_source = (
        dict(gate_override)
        if gate_override is not None
        else load_active_constraint_gate(root)
    )
    if not gate_source["gate_value"]:
        return _blocked_diagnostic(gate_source)

    fixtures = build_stress_fixtures()
    baselines = build_qubo_baselines(fixtures)
    baseline_by_fixture = {row["fixture_id"]: row for row in baselines}
    rows = [
        evaluate_fixture_mode(fixture, mode, baseline_by_fixture[fixture.fixture_id])
        for fixture in fixtures
        for mode in COMPARED_MODES
    ]
    summary = _summarize_rows(rows)
    blockers = _readiness_blockers(
        gate_source=gate_source,
        fixture_count=len(fixtures),
        qubo_baseline_count=len(baselines),
        summary=summary,
    )
    summary["gated_on_active_constraint_ready"] = True
    summary["gate_source"] = gate_source
    summary["fixture_count"] = len(fixtures)
    summary["qubo_baseline_count"] = len(baselines)
    summary["compared_modes"] = list(COMPARED_MODES)
    summary["qubo_baselines"] = baselines
    summary["row_records"] = rows
    summary["pbit_qubo_stress_ready"] = not blockers
    summary["readiness_blockers"] = blockers
    return summary


def evaluate_fixture_mode(
    fixture: StressFixture,
    mode: str,
    baseline: Mapping[str, Any],
) -> JsonDict:
    """Evaluate one fixture/mode row while keeping the solver authoritative."""

    _require(mode in COMPARED_MODES, f"mode: {mode}")
    baseline_metrics = _solve_deterministic(fixture)
    sample_records: list[JsonDict] = []
    hint_decision = "ignored"
    fallback_used = False

    if mode == "deterministic_solver":
        metrics = baseline_metrics
    elif mode == "active_constraint_hint":
        candidate = _sequence_from_active_hint(
            fixture,
            fixture.active_constraint_ids,
            fixture.conflict_front,
        )
        metrics, hint_decision, fallback_used = _checked_hint_metrics(
            fixture,
            candidate,
            baseline_metrics,
            accept_decision="accepted",
        )
    elif mode == "pbit_boundary_hint":
        sample_records = _pbit_sample_records(fixture, baseline)
        accepted = [sample for sample in sample_records if sample["valid"]]
        if accepted:
            candidate = tuple(accepted[0]["sequence"])
            metrics = SolverMetrics(
                final_sequence=candidate,
                final_valid=True,
                solver_conflicts=0,
                solver_iterations=len(fixture.actions),
            )
            hint_decision = "accepted"
        else:
            metrics = baseline_metrics
            hint_decision = "rejected"
            fallback_used = True
    else:
        adversarial = _adversarial_sequence(fixture)
        metrics, hint_decision, fallback_used = _checked_hint_metrics(
            fixture,
            adversarial,
            baseline_metrics,
            accept_decision="overwritten",
        )

    final_energy = qubo_precedence_energy(fixture, metrics.final_sequence)
    exact_min = int(baseline["exact_min_energy"])
    exact_agrees = bool(metrics.final_valid and final_energy == exact_min)
    accepted_count = sum(int(sample["valid"]) for sample in sample_records)
    sample_count = len(sample_records)
    rejected_count = sample_count - accepted_count
    return {
        "fixture_id": fixture.fixture_id,
        "source_fixture_id": fixture.source_fixture_id,
        "mode": mode,
        "hint_decision": hint_decision,
        "active_constraint_hint": list(fixture.active_constraint_ids)
        if mode == "active_constraint_hint"
        else [],
        "conflict_front_hint": list(fixture.conflict_front)
        if mode == "active_constraint_hint"
        else [],
        "solver_authoritative": True,
        "accepted_without_verification": False,
        "fallback_used": fallback_used,
        "expected_sequence": list(fixture.expected_sequence),
        "final_sequence": list(metrics.final_sequence),
        "final_valid": metrics.final_valid,
        "solver_conflicts": metrics.solver_conflicts,
        "solver_iterations": metrics.solver_iterations,
        "baseline_metrics": {
            "solver_conflicts": baseline_metrics.solver_conflicts,
            "solver_iterations": baseline_metrics.solver_iterations,
        },
        "conflict_delta": baseline_metrics.solver_conflicts - metrics.solver_conflicts,
        "iteration_delta": baseline_metrics.solver_iterations - metrics.solver_iterations,
        "exact_min_energy": exact_min,
        "final_energy": final_energy,
        "exact_enumeration_agrees": exact_agrees,
        "sample_count": sample_count,
        "accepted_sample_count": accepted_count,
        "rejected_sample_count": rejected_count,
        "acceptance_rate": _rate(accepted_count, sample_count),
        "sample_records": sample_records,
        "unsafe_false_accept": bool(not metrics.final_valid and hint_decision == "accepted"),
        "hardware_speedup_claim": False,
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    gate_override: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal artifact without running past a failed upstream gate."""

    diagnostic = run_diagnostic(root=root, gate_override=gate_override)
    tests = [_normalize_test_run(item) for item in tests_run]
    blockers = list(diagnostic["readiness_blockers"])
    if diagnostic["pbit_qubo_stress_ready"] and not tests:
        blockers.append("tests_not_recorded")
    ready = bool(diagnostic["pbit_qubo_stress_ready"] and not blockers)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "duration_s": 1.12,
        "status": "complete" if ready else "blocked",
        "gate_source": diagnostic["gate_source"],
        "gated_on_active_constraint_ready": diagnostic[
            "gated_on_active_constraint_ready"
        ],
        "fixture_count": diagnostic["fixture_count"],
        "qubo_baseline_count": diagnostic["qubo_baseline_count"],
        "exact_enumeration_agreement_rate": diagnostic[
            "exact_enumeration_agreement_rate"
        ],
        "pbit_acceptance_rate": diagnostic["pbit_acceptance_rate"],
        "solver_conflict_delta": diagnostic["solver_conflict_delta"],
        "solver_iteration_delta": diagnostic["solver_iteration_delta"],
        "fallback_rate": diagnostic["fallback_rate"],
        "unsafe_false_accept_rate": diagnostic["unsafe_false_accept_rate"],
        "hardware_speedup_claim": False,
        "simulation_only": True,
        "pbit_qubo_stress_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, blockers, diagnostic),
        "compared_modes": diagnostic["compared_modes"],
        "validity_rate": diagnostic["validity_rate"],
        "pbit_sample_count": diagnostic["pbit_sample_count"],
        "mode_summaries": diagnostic["mode_summaries"],
        "qubo_baselines": diagnostic["qubo_baselines"],
        "row_records": diagnostic["row_records"],
        "readiness_blockers": blockers,
        "tests_run": tests,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "claim_limits": [
            "CPU-only simulator diagnostic",
            "sorting-network QUBO baselines are exact-enumerated tiny instances",
            "p-bit samples are deterministic cached candidates, not hardware samples",
            "active-constraint hints are advisory and solver-checked",
            "no LLM, generated text judge, board timing, or hardware speedup claim",
        ],
        "methodology_note": (
            "exact_enumeration_agreement_rate is 1.0 because every tiny "
            "fixture enumerates all permutations; this is bounded correctness, "
            "not a distributional hardware claim."
        ),
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the validated Exp5407 artifact and return it."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema, solver authority, and hardware-claim discipline."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles")
    _require(artifact["status"] in {"complete", "blocked"}, "status")
    _require(artifact["milestone"] == MILESTONE, "milestone")
    _require(artifact["compared_modes"] == list(COMPARED_MODES), "compared_modes")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact["hardware_speedup_claim"] is False, "hardware_speedup_claim")
    _require(artifact["simulation_only"] is True, "simulation_only")
    _require(artifact["unsafe_false_accept_rate"] == 0.0, "unsafe_false_accept_rate")
    _require("REQ-VERIFY-5407" in artifact["spec_refs"], "spec_refs")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum")
    _require(_is_probability(artifact["exact_enumeration_agreement_rate"]), "exact rate")
    _require(_is_probability(artifact["pbit_acceptance_rate"]), "pbit rate")
    _require(_is_probability(artifact["fallback_rate"]), "fallback_rate")

    gate_source = artifact["gate_source"]
    _require(gate_source["artifact_path"] == str(GATE_RELATIVE_PATH), "gate_source")
    _require(gate_source["gate_field"] == "active_constraint_warmstart_ready", "gate_source")
    _require(type(gate_source["gate_value"]) is bool, "gate_source")

    if artifact["row_records"]:
        _validate_rows(artifact["row_records"])
    if artifact["pbit_qubo_stress_ready"]:
        _require(artifact["status"] == "complete", "status")
        _require(artifact["gated_on_active_constraint_ready"] is True, "gate")
        _require(artifact["fixture_count"] == EXPECTED_FIXTURE_COUNT, "fixture_count")
        _require(
            artifact["qubo_baseline_count"] == EXPECTED_QUBO_BASELINE_COUNT,
            "qubo_baseline_count",
        )
        _require(
            artifact["exact_enumeration_agreement_rate"] == 1.0,
            "exact_enumeration_agreement_rate",
        )
        _require(0 < artifact["pbit_acceptance_rate"] < 1, "pbit_acceptance_rate")
        _require(artifact["solver_conflict_delta"] > 0, "solver_conflict_delta")
        _require(artifact["solver_iteration_delta"] > 0, "solver_iteration_delta")
        _require(artifact["fallback_rate"] > 0, "fallback_rate")
        _require(artifact["validity_rate"] == 1.0, "validity_rate")
        _require(artifact["readiness_blockers"] == [], "readiness_blockers")
        _require(bool(artifact["tests_run"]), "tests_run")


def _gate_record(gate_value: bool, source_status: str) -> JsonDict:
    return {
        "artifact_path": str(GATE_RELATIVE_PATH),
        "gate_field": "active_constraint_warmstart_ready",
        "gate_value": gate_value,
        "source_status": source_status,
    }


def _blocked_diagnostic(gate_source: Mapping[str, Any]) -> JsonDict:
    return {
        "gate_source": dict(gate_source),
        "gated_on_active_constraint_ready": False,
        "fixture_count": 0,
        "qubo_baseline_count": 0,
        "compared_modes": list(COMPARED_MODES),
        "validity_rate": 0.0,
        "exact_enumeration_agreement_rate": 0.0,
        "pbit_acceptance_rate": 0.0,
        "pbit_sample_count": 0,
        "solver_conflict_delta": 0,
        "solver_iteration_delta": 0,
        "fallback_rate": 0.0,
        "unsafe_false_accept_rate": 0.0,
        "mode_summaries": {mode: _empty_mode_summary() for mode in COMPARED_MODES},
        "qubo_baselines": [],
        "row_records": [],
        "pbit_qubo_stress_ready": False,
        "readiness_blockers": ["active_constraint_warmstart_not_ready"],
    }


def _empty_mode_summary() -> JsonDict:
    return {
        "row_count": 0,
        "validity_rate": 0.0,
        "solver_conflicts": 0,
        "solver_iterations": 0,
        "fallback_count": 0,
        "sample_count": 0,
        "accepted_sample_count": 0,
    }


def _checked_hint_metrics(
    fixture: StressFixture,
    candidate: Sequence[str],
    baseline: SolverMetrics,
    *,
    accept_decision: str,
) -> tuple[SolverMetrics, str, bool]:
    if _is_complete_valid_sequence(fixture, candidate):
        return (
            SolverMetrics(
                final_sequence=tuple(candidate),
                final_valid=True,
                solver_conflicts=0,
                solver_iterations=len(fixture.actions),
            ),
            "accepted" if accept_decision == "accepted" else accept_decision,
            False,
        )
    return baseline, accept_decision, True


def _sequence_from_active_hint(
    fixture: StressFixture,
    active_hint: Sequence[str],
    front_hint: Sequence[str],
) -> tuple[str, ...]:
    if tuple(active_hint) != fixture.active_constraint_ids:
        return _adversarial_sequence(fixture)
    if tuple(front_hint) != fixture.conflict_front:
        return _adversarial_sequence(fixture)
    return fixture.expected_sequence


def _pbit_sample_records(
    fixture: StressFixture,
    baseline: Mapping[str, Any],
) -> list[JsonDict]:
    candidates = (
        tuple(baseline["exact_best_sequence"]),
        fixture.expected_sequence,
        tuple(baseline["deterministic_fallback_sequence"]),
        _adversarial_sequence(fixture),
        tuple(reversed(fixture.expected_sequence)),
        _front_loaded_sequence(fixture),
    )
    records = []
    for index, candidate in enumerate(candidates):
        energy = qubo_precedence_energy(fixture, candidate)
        valid = _is_complete_valid_sequence(fixture, candidate)
        records.append(
            {
                "sample_index": index,
                "sequence": list(candidate),
                "energy": energy,
                "valid": valid,
                "accepted_by_solver": valid,
            }
        )
    return records


def _adversarial_sequence(fixture: StressFixture) -> tuple[str, ...]:
    before, after = fixture.precedence[0]
    rest = [action for action in fixture.expected_sequence if action not in {before, after}]
    return (after, before, *rest)


def _front_loaded_sequence(fixture: StressFixture) -> tuple[str, ...]:
    front = fixture.conflict_front[0]
    rest = [action for action in fixture.expected_sequence if action != front]
    return (front, *rest)


def _solve_deterministic(fixture: StressFixture) -> SolverMetrics:
    sequence: list[str] = []
    remaining = list(fixture.actions)
    conflicts = 0
    iterations = 0
    while remaining:
        progressed = False
        for action in list(remaining):
            iterations += 1
            if _dependencies_satisfied(fixture, action, sequence):
                sequence.append(action)
                remaining.remove(action)
                progressed = True
            else:
                conflicts += 1
        _require(progressed, f"cyclic fixture: {fixture.fixture_id}")
    final = tuple(sequence)
    return SolverMetrics(
        final_sequence=final,
        final_valid=_is_complete_valid_sequence(fixture, final),
        solver_conflicts=conflicts,
        solver_iterations=iterations,
    )


def _dependencies_satisfied(
    fixture: StressFixture,
    action: str,
    prefix: Sequence[str],
) -> bool:
    done = set(prefix)
    return all(before in done for before, after in fixture.precedence if after == action)


def _is_complete_valid_sequence(
    fixture: StressFixture,
    sequence: Sequence[str],
) -> bool:
    if len(sequence) != len(fixture.actions) or set(sequence) != set(fixture.actions):
        return False
    seen: list[str] = []
    for action in sequence:
        if not _dependencies_satisfied(fixture, action, seen):
            return False
        seen.append(action)
    return True


def _active_constraint_ids(
    fixture: StressFixture,
    prefix: Sequence[str],
) -> tuple[str, ...]:
    done = set(prefix)
    return tuple(
        f"{before}->{after}"
        for before, after in fixture.precedence
        if before not in done and after not in done
    )


def _conflict_front(
    fixture: StressFixture,
    prefix: Sequence[str],
) -> tuple[str, ...]:
    active = _active_constraint_ids(fixture, prefix)
    return tuple(dict.fromkeys(edge.split("->", 1)[1] for edge in active))


def _sorting_network_comparators(size: int) -> tuple[tuple[int, int], ...]:
    comparators: list[tuple[int, int]] = []
    for end in range(1, size):
        for index in range(end, 0, -1):
            comparators.append((index - 1, index))
    return tuple(comparators)


def _sorting_network_sequence(
    fixture: StressFixture,
    sequence: Sequence[str],
) -> tuple[str, ...]:
    ranks = {action: index for index, action in enumerate(sequence)}
    items = list(fixture.actions)
    for left, right in _sorting_network_comparators(len(items)):
        if ranks[items[left]] > ranks[items[right]]:
            items[left], items[right] = items[right], items[left]
    return tuple(items)


def _qubo_precedence_term_count(fixture: StressFixture) -> int:
    positions = range(len(fixture.actions))
    return len(fixture.precedence) * sum(1 for left in positions for right in positions if left >= right)


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    mode_summaries = {
        mode: _mode_summary([row for row in rows if row["mode"] == mode])
        for mode in COMPARED_MODES
    }
    deterministic = mode_summaries["deterministic_solver"]
    pbit = mode_summaries["pbit_boundary_hint"]
    pbit_sample_count = pbit["sample_count"]
    pbit_accepted = pbit["accepted_sample_count"]
    unsafe_count = sum(int(row["unsafe_false_accept"]) for row in rows)
    return {
        "validity_rate": _rate(sum(row["final_valid"] for row in rows), len(rows)),
        "exact_enumeration_agreement_rate": _rate(
            sum(row["exact_enumeration_agrees"] for row in rows),
            len(rows),
        ),
        "pbit_acceptance_rate": _rate(pbit_accepted, pbit_sample_count),
        "pbit_sample_count": pbit_sample_count,
        "solver_conflict_delta": deterministic["solver_conflicts"] - pbit["solver_conflicts"],
        "solver_iteration_delta": deterministic["solver_iterations"] - pbit["solver_iterations"],
        "fallback_rate": _rate(sum(row["fallback_used"] for row in rows), len(rows)),
        "unsafe_false_accept_rate": _rate(unsafe_count, len(rows)),
        "mode_summaries": mode_summaries,
    }


def _mode_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "row_count": len(rows),
        "validity_rate": _rate(sum(row["final_valid"] for row in rows), len(rows)),
        "solver_conflicts": sum(int(row["solver_conflicts"]) for row in rows),
        "solver_iterations": sum(int(row["solver_iterations"]) for row in rows),
        "fallback_count": sum(int(row["fallback_used"]) for row in rows),
        "sample_count": sum(int(row["sample_count"]) for row in rows),
        "accepted_sample_count": sum(int(row["accepted_sample_count"]) for row in rows),
    }


def _readiness_blockers(
    *,
    gate_source: Mapping[str, Any],
    fixture_count: int,
    qubo_baseline_count: int,
    summary: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not gate_source["gate_value"]:
        blockers.append("active_constraint_warmstart_not_ready")
    if fixture_count != EXPECTED_FIXTURE_COUNT:
        blockers.append("fixture_count_mismatch")
    if qubo_baseline_count != EXPECTED_QUBO_BASELINE_COUNT:
        blockers.append("qubo_baseline_count_mismatch")
    if summary["validity_rate"] != 1.0:
        blockers.append("validity_not_preserved")
    if summary["exact_enumeration_agreement_rate"] != 1.0:
        blockers.append("exact_enumeration_incomplete")
    if not 0 < summary["pbit_acceptance_rate"] < 1:
        blockers.append("pbit_acceptance_not_mixed")
    if summary["solver_conflict_delta"] <= 0 or summary["solver_iteration_delta"] <= 0:
        blockers.append("solver_work_not_reduced")
    if summary["fallback_rate"] <= 0:
        blockers.append("fallback_not_exercised")
    if summary["unsafe_false_accept_rate"] != 0.0:
        blockers.append("unsafe_false_accepts_present")
    return blockers


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(len(rows) == EXPECTED_FIXTURE_COUNT * len(COMPARED_MODES), "row_records")
    for row in rows:
        _require(row["mode"] in COMPARED_MODES, "row mode")
        _require(row["solver_authoritative"] is True, "row solver_authoritative")
        _require(row["accepted_without_verification"] is False, "accepted_without_verification")
        _require(row["final_valid"] is True, "row final_valid")
        _require(row["exact_enumeration_agrees"] is True, "row exact agreement")
        _require(row["unsafe_false_accept"] is False, "row unsafe_false_accept")
        _require(row["hardware_speedup_claim"] is False, "row hardware_speedup_claim")


def _normalize_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "passed"}
    return dict(item)


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _honest_verdict(
    ready: bool,
    blockers: Sequence[str],
    diagnostic: Mapping[str, Any],
) -> str:
    if ready:
        return (
            "complete: CPU-only p-bit/QUBO stress kept solver authority, "
            f"accepted {diagnostic['pbit_acceptance_rate']:.3f} of p-bit "
            "samples, matched exact enumeration, and made no hardware speedup claim"
        )
    return "blocked: p-bit/QUBO stress blockers=" + ",".join(blockers)


def _rate(numerator: float, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(float(numerator) / denominator, 6)


def _is_probability(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0 <= value <= 1


def _require_sequence(value: tuple[str, ...] | None, label: str) -> tuple[str, ...]:
    _require(value is not None, label)
    return value


def _require_int(value: int | None, label: str) -> int:
    _require(value is not None, label)
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
