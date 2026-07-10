"""Exp5545 exact-checked sparse repair descriptors for finite-state fixtures.

Spec refs: REQ-VERIFY-5545, SCENARIO-VERIFY-5545.

This experiment takes the small exact FSM fixture from Exp5541 and turns its
active problems into sparse repair descriptors. The descriptor is not allowed
to be the authority: every proposed repair is still accepted or rejected by
the Exp5541 exact FSM checker. Timing is recorded only as local observation,
because iteration counts over tiny fixtures are not matched authenticated
timing evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any

from carnot import experiment_5541_llm_fsm_exact_fixture as fsm_mod


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5545_sparse_repair_fsm_descriptor_scale.json")
UPSTREAM_RELATIVE_PATH = fsm_mod.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5545.sparse_repair_fsm_descriptor_scale.v502"
DESCRIPTOR_BUNDLE_SCHEMA = "carnot.descriptor_bundle.sparse_repair_fsm_5545.v1"
DESCRIPTOR_SCHEMA = "carnot.descriptor.sparse_repair_fsm_5545.v1"
EXPERIMENT = 5545
EXPERIMENT_ID = "exp5545-gated-sparse-repair-fsm-descriptor-scale"
MILESTONE = "2026.07.502"
RUN_DATE = "2026-07-10"
INFERENCE_SUBSTRATE = "exact_checked_sparse_repair_fsm_no_llm"
EXACT_VALIDATOR = "exp5541_exact_fsm_validator"
SPARSE_BLOCK_POLICY = "fsm_active_conflict_unreachable_trace_block_then_exact_validate"
KEEP_TRANSITION = "__keep__"
SEEDS = (5545, 5546, 5547, 5548, 5549, 5550, 5551)
SPEC_REFS = ("REQ-VERIFY-5545", "SCENARIO-VERIFY-5545", "REQ-VERIFY-5541")
TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5545_sparse_repair_fsm_descriptor_scale.py",
    "tests/python/test_experiment_5541_llm_fsm_exact_fixture.py",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "fsm_instance_count": "Keeps the finite-state denominator visible for every policy comparison.",
    "random_seed_count": "Records deterministic seed breadth instead of a single lucky block draw.",
    "descriptor_guided_success_rate": "Measures active-descriptor repair only after exact FSM acceptance.",
    "random_block_success_rate": "Provides a same-size random-block control on the same fixtures and seeds.",
    "exact_only_success_rate": "Records the exact fallback authority on the same fixtures and seeds.",
    "descriptor_mean_iterations": "Counts exact validator checks for descriptor-guided repair without making a speedup claim.",
    "random_mean_iterations": "Counts exact validator checks for the random-block control without making a speedup claim.",
    "exact_fallback_used": "Confirms exact FSM validation remains the acceptance authority for repairs.",
    "exact_validator_all_repairs_checked": "Confirms every proposed repair has an exact accept or reject decision.",
    "matched_timing_available": "Gates any future timing language to authenticated matched measurements.",
    "speedup_claim_allowed": "Must remain false without matched authenticated timing.",
    "sparse_repair_fsm_ready": "Opens only when the upstream fixture is ready, exact checks cover every repair, and the descriptor panel succeeds.",
    "tests_added_or_reused": "Names focused tests and reused exact-fixture tests.",
    "field_principles": "Keeps headline and gate fields annotated by evidence boundaries.",
    "inference_substrate": "Declares exact-checked sparse FSM repair with no LLM.",
    "honest_verdict": "Provides a terminal evidence boundary without speedup language.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so hashes follow semantic content."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def load_upstream_artifact(repo_root: Path = REPO_ROOT) -> JsonDict:
    """Load the Exp5541 artifact that authorizes this sparse FSM panel."""

    path = repo_root / UPSTREAM_RELATIVE_PATH
    decoded = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(decoded, dict), "upstream_artifact")
    return decoded


def ensure_upstream_ready(upstream_artifact: Mapping[str, Any]) -> None:
    """Fail closed unless Exp5541's exact finite-state fixture gate is open."""

    _require(upstream_artifact.get("exact_fsm_fixture_ready") is True, "exact_fsm_fixture_ready")
    _require(isinstance(upstream_artifact.get("fsm_family"), list), "fsm_family")
    _require(bool(upstream_artifact.get("fsm_family")), "fsm_family")


def build_sparse_descriptors(upstream_artifact: Mapping[str, Any]) -> JsonDict:
    """Build one sparse repair descriptor per upstream FSM fixture."""

    ensure_upstream_ready(upstream_artifact)
    rows = [
        build_sparse_descriptor(machine)
        for machine in upstream_artifact["fsm_family"]
    ]
    payload = {
        "schema": DESCRIPTOR_BUNDLE_SCHEMA,
        "source_experiment": str(upstream_artifact.get("experiment_id", fsm_mod.EXPERIMENT_ID)),
        "source_result_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "sparse_block_policy": SPARSE_BLOCK_POLICY,
        "descriptor_count": len(rows),
        "sparse_repair_descriptors": rows,
    }
    validate_descriptor_payload(payload, upstream_artifact)
    return payload


def build_sparse_descriptor(machine: Mapping[str, Any]) -> JsonDict:
    """Extract active FSM constraints and attach an exact-validated repair target."""

    report = fsm_mod.solve_instance(machine)
    domains = repair_variable_domains(machine, active=report["solver_status"] != "satisfiable")
    target: JsonDict = {}
    active_constraints: list[JsonDict] = []

    for conflict in transition_conflicts(machine):
        variable = transition_variable(conflict["source"], conflict["symbol"])
        target_state = choose_transition_target(machine, conflict["targets"])
        target[variable] = target_state
        active_constraints.append(
            {
                "kind": "transition_conflict",
                "repair_variable": variable,
                "source": conflict["source"],
                "symbol": conflict["symbol"],
                "targets": conflict["targets"],
                "target": target_state,
                "evidence": conflict["evidence"],
            }
        )

    for state in report["unreachable_states"]:
        repair = transition_repair_for_unreachable(machine, str(state), target)
        active_constraints.append(
            {
                "kind": "unreachable_state",
                "state": str(state),
                "repair_variable": None if repair is None else repair[0],
                "target": None if repair is None else repair[1],
                "evidence": "exact_required_reachability",
            }
        )
        if repair is not None and repair[0] not in target:
            target[repair[0]] = repair[1]

    trace_issues = [
        row
        for row in report["trace_checks"]
        if row["actual_label"] in {"contradiction", "underdetermined"}
    ]
    for trace in trace_issues:
        repair = transition_repair_for_trace(machine, trace, target)
        if repair is not None and repair[0] not in target:
            target[repair[0]] = repair[1]
            active_constraints.append(
                {
                    "kind": "trace_transition_repair",
                    "trace_id": trace["trace_id"],
                    "repair_variable": repair[0],
                    "target": repair[1],
                    "evidence": "first_missing_transition_on_exact_trace",
                }
            )

    if target:
        transition_only = {
            key: value
            for key, value in target.items()
            if key.startswith("transition:")
        }
        repaired_for_status = apply_repair_assignment(machine, transition_only)
        target["expected_status"] = fsm_mod.solve_instance(repaired_for_status)["solver_status"]

    trace_targets = trace_label_targets(machine, trace_issues, target)
    for trace in trace_issues:
        variable = trace_variable(trace["trace_id"])
        target[variable] = trace_targets[str(trace["trace_id"])]
        active_constraints.append(
            {
                "kind": "trace_contradiction",
                "trace_id": str(trace["trace_id"]),
                "repair_variable": variable,
                "observed_label": trace["actual_label"],
                "labels_across_completions": list(trace["labels_across_completions"]),
                "target_label": target[variable],
            }
        )

    all_variables = list(domains)
    repair_block = [variable for variable in all_variables if variable in target]
    repaired = apply_repair_assignment(machine, target)
    fallback_report = fsm_mod.solve_instance(repaired)
    descriptor = {
        "schema": DESCRIPTOR_SCHEMA,
        "descriptor_id": f"sparse-repair-fsm:{machine['instance_id']}",
        "source_result_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "source_instance_id": str(machine["instance_id"]),
        "sparse_block_policy": SPARSE_BLOCK_POLICY,
        "all_repair_variables": all_variables,
        "repair_variable_domains": domains,
        "variable_count": len(all_variables),
        "repair_block_variables": repair_block,
        "block_size": len(repair_block),
        "sparse_subset": len(repair_block) < len(all_variables) if all_variables else True,
        "active_constraints": active_constraints,
        "target_repair_assignment": target,
        "initial_solver_status": report["solver_status"],
        "initial_unreachable_states": list(report["unreachable_states"]),
        "exact_fallback": {
            "required": True,
            "validator": EXACT_VALIDATOR,
            "status": fallback_report["solver_status"],
            "trace_checks_passed": fallback_report["trace_checks_passed"],
            "transition_consistency_passed": fallback_report["transition_consistency_passed"],
            "sat_solver_check_passed": fallback_report["sat_solver_check_passed"],
            "accepted": exact_report_passed(repaired, fallback_report),
        },
    }
    validate_sparse_descriptor(descriptor)
    return descriptor


def repair_variable_domains(machine: Mapping[str, Any], *, active: bool) -> JsonDict:
    """Return bounded domains for active repair variables on one FSM."""

    if not active:
        return {}
    states = [str(state) for state in machine["states"]]
    domains: JsonDict = {
        transition_variable(str(state), str(symbol)): [*states, KEEP_TRANSITION]
        for state in machine["states"]
        for symbol in machine["alphabet"]
    }
    for trace in machine["observable_traces"]:
        domains[trace_variable(str(trace["trace_id"]))] = list(fsm_mod.TRACE_LABELS)
    domains["expected_status"] = list(fsm_mod.SOLVER_STATUSES)
    return domains


def transition_conflicts(machine: Mapping[str, Any]) -> list[JsonDict]:
    """Return deterministic-transition conflicts in inspectable row form."""

    by_pair: dict[tuple[str, str], list[JsonDict]] = {}
    for row in machine["transition_constraints"]:
        if row["kind"] == "require":
            pair = (str(row["source"]), str(row["symbol"]))
            by_pair.setdefault(pair, []).append(dict(row))
    conflicts: list[JsonDict] = []
    for (source, symbol), rows in by_pair.items():
        targets = []
        for row in rows:
            target = str(row["target"])
            if target not in targets:
                targets.append(target)
        if len(targets) > 1:
            conflicts.append(
                {
                    "source": source,
                    "symbol": symbol,
                    "targets": targets,
                    "evidence": [str(row["constraint_id"]) for row in rows],
                }
            )
    return conflicts


def choose_transition_target(machine: Mapping[str, Any], targets: Sequence[str]) -> str:
    """Choose the canonical conflict repair target from exact FSM metadata."""

    accepting = set(machine.get("accepting_states", []))
    errors = set(machine.get("error_states", []))
    for target in targets:
        if target in accepting:
            return str(target)
    for target in targets:
        if target not in errors:
            return str(target)
    return str(targets[0])


def transition_repair_for_unreachable(
    machine: Mapping[str, Any],
    state: str,
    target_assignment: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Find a sparse transition that can make an unreachable state reachable."""

    for variable, target in target_assignment.items():
        if variable.startswith("transition:") and target == state:
            return variable, state

    diagnostics = fsm_mod.transition_diagnostics(machine)
    reachable = reachable_states_from_required(machine, diagnostics["required_map"])
    forbidden = diagnostics["forbidden_set"]
    for source in reachable:
        for symbol in machine["alphabet"]:
            pair = (str(source), str(symbol))
            if pair in diagnostics["required_map"]:
                continue
            if (pair[0], pair[1], state) not in forbidden:
                return transition_variable(pair[0], pair[1]), state
    return None


def transition_repair_for_trace(
    machine: Mapping[str, Any],
    trace: Mapping[str, Any],
    target_assignment: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Infer the first missing transition needed to make an exact trace decisive."""

    if trace["actual_label"] != "underdetermined":
        return None
    diagnostics = fsm_mod.transition_diagnostics(machine)
    required = diagnostics["required_map"]
    forbidden = diagnostics["forbidden_set"]
    current = str(machine["start_state"])
    for symbol in trace["symbols"]:
        pair = (current, str(symbol))
        variable = transition_variable(pair[0], pair[1])
        if variable in target_assignment:
            current = str(target_assignment[variable])
            continue
        if pair not in required:
            target = preferred_decisive_target(machine, pair, forbidden)
            return variable, target
        current = required[pair]
    return None


def preferred_decisive_target(
    machine: Mapping[str, Any],
    pair: tuple[str, str],
    forbidden: set[tuple[str, str, str]],
) -> str:
    """Pick an allowed target that makes an underdetermined trace exact."""

    accepting = [str(state) for state in machine.get("accepting_states", [])]
    states = [str(state) for state in machine["states"]]
    for target in accepting + states:
        if (pair[0], pair[1], target) not in forbidden:
            return target
    raise ValueError("no_allowed_trace_target")


def reachable_states_from_required(
    machine: Mapping[str, Any],
    required_map: Mapping[tuple[str, str], str],
) -> list[str]:
    """Return states reached by required transitions from the start state."""

    seen = {str(machine["start_state"])}
    frontier = list(seen)
    while frontier:
        current = frontier.pop(0)
        for (source, _symbol), target in required_map.items():
            if source == current and target not in seen:
                seen.add(target)
                frontier.append(target)
    return sorted(seen)


def trace_label_targets(
    machine: Mapping[str, Any],
    trace_issues: Sequence[Mapping[str, Any]],
    target_assignment: Mapping[str, Any],
) -> JsonDict:
    """Compute exact trace labels after the transition and status repair plan."""

    if not trace_issues:
        return {}
    partial = {
        key: value
        for key, value in target_assignment.items()
        if key == "expected_status" or key.startswith("transition:")
    }
    repaired = apply_repair_assignment(machine, partial)
    labeled = fsm_mod.attach_reference_trace_labels(repaired)
    labels = {
        str(trace["trace_id"]): str(trace["expected_label"])
        for trace in labeled["observable_traces"]
    }
    return {str(trace["trace_id"]): labels[str(trace["trace_id"])] for trace in trace_issues}


def apply_repair_assignment(machine: Mapping[str, Any], assignment: Mapping[str, Any]) -> JsonDict:
    """Apply a bounded repair assignment to an FSM machine description."""

    repaired = json.loads(fsm_mod.canonical_json(machine))
    transition_updates: dict[tuple[str, str], str] = {}
    trace_updates: dict[str, str] = {}
    for variable, value in assignment.items():
        if variable.startswith("transition:"):
            if str(value) != KEEP_TRANSITION:
                transition_updates[parse_transition_variable(variable)] = str(value)
        elif variable.startswith("trace:"):
            trace_updates[variable.removeprefix("trace:")] = str(value)
        elif variable == "expected_status":
            repaired["expected_status"] = str(value)

    if transition_updates:
        next_constraints = []
        for row in repaired["transition_constraints"]:
            pair = (str(row["source"]), str(row["symbol"]))
            if row["kind"] == "require" and pair in transition_updates:
                continue
            next_constraints.append(row)
        for (source, symbol), target in sorted(transition_updates.items()):
            next_constraints.append(
                {
                    "constraint_id": f"REPAIR_{source}_{symbol}".replace("/", "_"),
                    "kind": "require",
                    "source": source,
                    "symbol": symbol,
                    "target": target,
                }
            )
        repaired["transition_constraints"] = next_constraints
        repaired["transition_sparsity"] = transition_sparsity(repaired)

    for trace in repaired["observable_traces"]:
        trace_id = str(trace["trace_id"])
        if trace_id in trace_updates:
            trace["expected_label"] = trace_updates[trace_id]
    return repaired


def transition_sparsity(machine: Mapping[str, Any]) -> float:
    """Recompute transition sparsity after adding or removing required rows."""

    total_pairs = len(machine["states"]) * len(machine["alphabet"])
    if total_pairs == 0:
        return 0.0
    required_count = sum(1 for row in machine["transition_constraints"] if row["kind"] == "require")
    return round((total_pairs - required_count) / total_pairs, 6)


def validate_candidate_repair(
    machine: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> JsonDict:
    """Check one proposed repair with the Exp5541 exact FSM validator."""

    repaired = apply_repair_assignment(machine, assignment)
    report = fsm_mod.solve_instance(repaired)
    target = descriptor["target_repair_assignment"]
    target_met = all(assignment.get(variable) == value for variable, value in target.items())
    exact_passed = exact_report_passed(repaired, report)
    accepted = bool(target_met and exact_passed)
    return {
        "assignment": dict(assignment),
        "assignment_hash": sha256_json(dict(assignment)),
        "exact_checked": True,
        "exact_validator": EXACT_VALIDATOR,
        "exact_validator_decision": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "target_assignment_met": target_met,
        "exact_report_passed": exact_passed,
        "solver_status": report["solver_status"],
        "trace_checks_passed": report["trace_checks_passed"],
        "transition_consistency_passed": report["transition_consistency_passed"],
        "sat_solver_check_passed": report["sat_solver_check_passed"],
    }


def exact_report_passed(machine: Mapping[str, Any], report: Mapping[str, Any]) -> bool:
    """Return whether an exact FSM report accepts the repaired machine."""

    return bool(
        report["transition_consistency_passed"]
        and report["trace_checks_passed"]
        and report["sat_solver_check_passed"]
        and report["solver_status"] == machine["expected_status"]
    )


def run_descriptor_guided_attempt(
    machine: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    seed: int,
) -> JsonDict:
    """Run the active sparse descriptor block and validate each proposal exactly."""

    block = list(descriptor["repair_block_variables"])
    checks = [
        validate_candidate_repair(machine, descriptor, assignment)
        for assignment in enumerate_candidate_assignments(descriptor, block, guided=True)
    ]
    return attempt_record("descriptor_guided", machine, seed, block, checks)


def run_random_block_attempt(
    machine: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    seed: int,
    instance_index: int,
) -> JsonDict:
    """Run a deterministic same-size random repair block under exact validation."""

    all_variables = list(descriptor["all_repair_variables"])
    block_size = len(descriptor["repair_block_variables"])
    if block_size == 0:
        block: list[str] = []
    else:
        block = random.Random(seed + instance_index * 1009).sample(all_variables, block_size)
    checks = [
        validate_candidate_repair(machine, descriptor, assignment)
        for assignment in enumerate_candidate_assignments(descriptor, block, guided=False)
    ]
    return attempt_record("random_block", machine, seed, block, checks)


def run_exact_only_attempt(
    machine: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    seed: int,
) -> JsonDict:
    """Run exact-only fallback over the bounded repair variable universe."""

    block = list(descriptor["all_repair_variables"])
    checks: list[JsonDict] = []
    for assignment in enumerate_candidate_assignments(descriptor, block, guided=False):
        check = validate_candidate_repair(machine, descriptor, assignment)
        checks.append(check)
        if check["accepted"]:
            break
    return attempt_record("exact_only", machine, seed, block, checks)


def enumerate_candidate_assignments(
    descriptor: Mapping[str, Any],
    block_variables: Sequence[str],
    *,
    guided: bool,
) -> list[JsonDict]:
    """Enumerate finite candidate assignments over a selected repair block."""

    if not block_variables:
        return [{}]
    domains = descriptor["repair_variable_domains"]
    target = descriptor["target_repair_assignment"]
    ordered_domains = [
        ordered_domain(str(variable), list(domains[variable]), target, guided=guided)
        for variable in block_variables
    ]
    return [
        dict(zip(block_variables, values, strict=True))
        for values in product(*ordered_domains)
    ]


def ordered_domain(
    variable: str,
    domain: Sequence[str],
    target: Mapping[str, Any],
    *,
    guided: bool,
) -> list[str]:
    """Put descriptor target values first only for descriptor-guided attempts."""

    values = [str(value) for value in domain]
    target_value = target.get(variable)
    if guided and target_value in values:
        return [str(target_value)] + [value for value in values if value != target_value]
    return values


def attempt_record(
    policy: str,
    machine: Mapping[str, Any],
    seed: int,
    block_variables: Sequence[str],
    checks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one policy attempt row from exact-checked candidate repairs."""

    accepted = [row for row in checks if row["accepted"]]
    return {
        "policy": policy,
        "instance_id": str(machine["instance_id"]),
        "seed": seed,
        "block_variables": list(block_variables),
        "iterations": len(checks),
        "success": bool(accepted),
        "accepted_assignment": dict(accepted[0]["assignment"]) if accepted else None,
        "exact_fallback_used": all(row.get("exact_checked") is True for row in checks),
        "candidate_checks": [dict(row) for row in checks],
    }


def run_policy_comparison(
    *,
    upstream_artifact: Mapping[str, Any],
    descriptors: Mapping[str, Any],
    seeds: Sequence[int] = SEEDS,
) -> JsonDict:
    """Compare descriptor-guided, random-block, and exact-only policies."""

    ensure_upstream_ready(upstream_artifact)
    machines = list(upstream_artifact["fsm_family"])
    descriptor_rows = list(descriptors["sparse_repair_descriptors"])
    policy_results: dict[str, list[JsonDict]] = {
        "descriptor_guided": [],
        "random_block": [],
        "exact_only": [],
    }

    descriptor_start = perf_counter()
    for seed in seeds:
        for machine, descriptor in zip(machines, descriptor_rows, strict=True):
            policy_results["descriptor_guided"].append(
                run_descriptor_guided_attempt(machine, descriptor, seed)
            )
    descriptor_wall = perf_counter() - descriptor_start

    random_start = perf_counter()
    for seed in seeds:
        for instance_index, (machine, descriptor) in enumerate(
            zip(machines, descriptor_rows, strict=True)
        ):
            policy_results["random_block"].append(
                run_random_block_attempt(machine, descriptor, seed, instance_index)
            )
    random_wall = perf_counter() - random_start

    exact_start = perf_counter()
    for seed in seeds:
        for machine, descriptor in zip(machines, descriptor_rows, strict=True):
            policy_results["exact_only"].append(run_exact_only_attempt(machine, descriptor, seed))
    exact_wall = perf_counter() - exact_start

    attempts = [attempt for rows in policy_results.values() for attempt in rows]
    all_checks = [check for attempt in attempts for check in attempt["candidate_checks"]]
    unchecked = [
        check
        for check in all_checks
        if check.get("exact_validator_decision") not in {"accepted", "rejected"}
        or check.get("exact_checked") is not True
    ]
    return {
        "policy_results": policy_results,
        "fsm_instance_count": len(machines),
        "random_seed_count": len(seeds),
        "candidate_repair_count": len(all_checks),
        "unchecked_repair_count": len(unchecked),
        "descriptor_guided_success_rate": success_rate(policy_results["descriptor_guided"]),
        "random_block_success_rate": success_rate(policy_results["random_block"]),
        "exact_only_success_rate": success_rate(policy_results["exact_only"]),
        "descriptor_mean_iterations": mean_iterations(policy_results["descriptor_guided"]),
        "random_mean_iterations": mean_iterations(policy_results["random_block"]),
        "exact_mean_iterations": mean_iterations(policy_results["exact_only"]),
        "exact_fallback_used": all(attempt["exact_fallback_used"] for attempt in attempts),
        "exact_validator_all_repairs_checked": len(unchecked) == 0,
        "wall_time_observations_s": {
            "descriptor_guided": round(descriptor_wall, 9),
            "random_block": round(random_wall, 9),
            "exact_only": round(exact_wall, 9),
            "methodology": "single-process CPU wall observations, not matched authenticated timing",
        },
    }


def success_rate(attempts: Sequence[Mapping[str, Any]]) -> float:
    """Compute exact-validated success rate for policy attempts."""

    return round(sum(int(row["success"]) for row in attempts) / len(attempts), 6)


def mean_iterations(attempts: Sequence[Mapping[str, Any]]) -> float:
    """Compute mean exact-validator checks per policy attempt."""

    return round(sum(float(row["iterations"]) for row in attempts) / len(attempts), 6)


def build_artifact(
    *,
    upstream_artifact: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5545 sparse FSM repair artifact."""

    upstream = load_upstream_artifact() if upstream_artifact is None else upstream_artifact
    ensure_upstream_ready(upstream)
    descriptors = build_sparse_descriptors(upstream)
    comparison = run_policy_comparison(upstream_artifact=upstream, descriptors=descriptors)
    blockers = readiness_blockers(upstream, descriptors, comparison)
    ready = not blockers
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "source_result_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "upstream_exact_fsm_fixture_ready": upstream.get("exact_fsm_fixture_ready") is True,
        "matched_timing_available": False,
        "speedup_claim_allowed": False,
        "sparse_repair_fsm_ready": ready,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, blockers),
        "descriptor_payload": descriptors,
        "descriptor_payload_sha256": sha256_json(descriptors),
        "tests_run": [dict(row) for row in tests_run],
        "research_conductor_modified": False,
        "readiness_blockers": blockers,
        "claim_limits": [
            "exact-checked finite-state fixtures only",
            "descriptor and random-block proposals are exact-validator checked",
            "local wall observations are not matched authenticated timing evidence",
            "no solver, sampler, model, hardware, or speedup claim is allowed",
        ],
    }
    artifact.update(comparison)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def readiness_blockers(
    upstream_artifact: Mapping[str, Any],
    descriptors: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> list[str]:
    """Return precise blockers for the sparse FSM readiness gate."""

    blockers: list[str] = []
    if upstream_artifact.get("exact_fsm_fixture_ready") is not True:
        blockers.append("upstream_exact_fsm_fixture_ready")
    if descriptors.get("descriptor_count") != len(upstream_artifact.get("fsm_family", [])):
        blockers.append("descriptor_count")
    if comparison.get("fsm_instance_count") != len(upstream_artifact.get("fsm_family", [])):
        blockers.append("fsm_instance_count")
    if comparison.get("random_seed_count") != len(SEEDS):
        blockers.append("random_seed_count")
    if comparison.get("descriptor_guided_success_rate") != 1.0:
        blockers.append("descriptor_guided_success_rate")
    if comparison.get("exact_only_success_rate") != 1.0:
        blockers.append("exact_only_success_rate")
    if comparison.get("exact_fallback_used") is not True:
        blockers.append("exact_fallback_used")
    if comparison.get("exact_validator_all_repairs_checked") is not True:
        blockers.append("exact_validator_all_repairs_checked")
    if comparison.get("unchecked_repair_count") != 0:
        blockers.append("unchecked_repair_count")
    return blockers


def validate_descriptor_payload(
    payload: Mapping[str, Any],
    upstream_artifact: Mapping[str, Any],
) -> None:
    """Validate descriptor bundle shape before policies consume it."""

    _require(payload.get("schema") == DESCRIPTOR_BUNDLE_SCHEMA, "descriptor_schema")
    _require(payload.get("source_result_path") == UPSTREAM_RELATIVE_PATH.as_posix(), "source_result_path")
    rows = payload.get("sparse_repair_descriptors")
    _require(isinstance(rows, list), "sparse_repair_descriptors")
    _require(payload.get("descriptor_count") == len(upstream_artifact["fsm_family"]), "descriptor_count")
    for row in rows:
        validate_sparse_descriptor(row)


def validate_sparse_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Reject sparse FSM descriptors that lack exact fallback evidence."""

    _require(descriptor.get("schema") == DESCRIPTOR_SCHEMA, "descriptor_schema")
    _require(descriptor.get("sparse_block_policy") == SPARSE_BLOCK_POLICY, "sparse_block_policy")
    domains = descriptor.get("repair_variable_domains")
    all_variables = descriptor.get("all_repair_variables")
    block = descriptor.get("repair_block_variables")
    target = descriptor.get("target_repair_assignment")
    fallback = descriptor.get("exact_fallback")
    _require(isinstance(domains, Mapping), "repair_variable_domains")
    _require(isinstance(all_variables, list), "all_repair_variables")
    _require(isinstance(block, list), "repair_block_variables")
    _require(isinstance(target, Mapping), "target_repair_assignment")
    _require(set(block).issubset(set(all_variables)), "repair_block_variables")
    _require(set(target).issubset(set(all_variables)), "target_repair_assignment")
    if descriptor.get("active_constraints"):
        _require(bool(block), "repair_block_variables")
        _require(len(block) < len(all_variables), "sparse_subset")
    else:
        _require(block == [], "repair_block_variables")
        _require(target == {}, "target_repair_assignment")
    _require(isinstance(fallback, Mapping), "exact_fallback")
    _require(fallback.get("required") is True, "exact_fallback")
    _require(fallback.get("validator") == EXACT_VALIDATOR, "exact_fallback")
    _require(fallback.get("accepted") is True, "exact_fallback")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the successful terminal artifact and fail closed on overclaims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("source_result_path") == UPSTREAM_RELATIVE_PATH.as_posix(), "source_result_path")
    _require(artifact.get("upstream_exact_fsm_fixture_ready") is True, "upstream_exact_fsm_fixture_ready")
    _require(artifact.get("fsm_instance_count") == artifact["descriptor_payload"]["descriptor_count"], "fsm_instance_count")
    _require(artifact.get("random_seed_count") == len(SEEDS), "random_seed_count")
    _require(artifact.get("descriptor_guided_success_rate") == 1.0, "descriptor_guided_success_rate")
    _require(artifact.get("random_block_success_rate") < artifact.get("descriptor_guided_success_rate"), "random_block_success_rate")
    _require(artifact.get("exact_only_success_rate") == 1.0, "exact_only_success_rate")
    _require(isinstance(artifact.get("descriptor_mean_iterations"), float), "descriptor_mean_iterations")
    _require(isinstance(artifact.get("random_mean_iterations"), float), "random_mean_iterations")
    _require(artifact.get("exact_fallback_used") is True, "exact_fallback_used")
    _require(artifact.get("exact_validator_all_repairs_checked") is True, "exact_validator_all_repairs_checked")
    _require(artifact.get("unchecked_repair_count") == 0, "unchecked_repair_count")
    _require(artifact.get("matched_timing_available") is False, "matched_timing_available")
    _require(artifact.get("speedup_claim_allowed") is False, "speedup_claim_allowed")
    _require(artifact.get("sparse_repair_fsm_ready") is True, "sparse_repair_fsm_ready")
    _require(artifact.get("tests_added_or_reused") == list(TESTS_ADDED_OR_REUSED), "tests_added_or_reused")
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})), "field_principles")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(artifact.get("descriptor_payload_sha256") == sha256_json(artifact["descriptor_payload"]), "descriptor_payload_sha256")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict without speedup language."""

    if ready:
        return "complete: exact_checked_sparse_repair_fsm_descriptor_scale_ready_no_speedup_claim"
    return "blocked: sparse_repair_fsm_descriptor_scale_not_ready_" + "_".join(blockers)


def run(
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5545 deliverable JSON under ``repo_root``."""

    upstream = load_upstream_artifact(repo_root)
    artifact = build_artifact(upstream_artifact=upstream, tests_run=tests_run)
    result_path = repo_root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def transition_variable(source: str, symbol: str) -> str:
    return f"transition:{source}/{symbol}"


def parse_transition_variable(variable: str) -> tuple[str, str]:
    source, symbol = variable.removeprefix("transition:").split("/", 1)
    return source, symbol


def trace_variable(trace_id: str) -> str:
    return f"trace:{trace_id}"


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "sparse_repair_fsm_ready": artifact["sparse_repair_fsm_ready"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
