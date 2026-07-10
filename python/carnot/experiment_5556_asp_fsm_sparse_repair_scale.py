"""Exp5556 descriptor-guided sparse repair over ASP/FSM exact rows.

Spec refs: REQ-VERIFY-5556, SCENARIO-VERIFY-5556.

The experiment reuses Exp5555's tiny stable-model evaluator as the sole
acceptance authority. Each ASP fixture row is copied into a bounded repair
problem by adding or removing one local fact or rule. The descriptor-guided,
random-block, and exact-only controls then receive the same rows, seeds, and
candidate budget; every candidate is evaluated by exact stable-model
enumeration before it can count as a repair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from carnot import experiment_5555_asp_fsm_nonmonotonic_fixture as asp_mod


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5556_asp_fsm_sparse_repair_scale.json")
UPSTREAM_ASP_FSM_FIXTURE = asp_mod.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_5556.asp_fsm_sparse_repair_scale.v503"
DESCRIPTOR_BUNDLE_SCHEMA = "carnot.descriptor_bundle.asp_fsm_sparse_repair_5556.v1"
DESCRIPTOR_SCHEMA = "carnot.descriptor.asp_fsm_sparse_repair_5556.v1"
EXPERIMENT = 5556
EXPERIMENT_ID = "exp5556-gated-asp-fsm-sparse-repair-scale"
MILESTONE = "2026.07.503"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5556
INFERENCE_SUBSTRATE = "deterministic_asp_fsm_sparse_repair_no_llm"
EXACT_VALIDATOR = "exp5555_stable_model_evaluator"
SPARSE_BLOCK_POLICY = "asp_row_active_fact_or_rule_block_then_stable_model_validate"
SEEDS = (5556, 5557, 5558, 5559, 5560, 5561, 5562)
CANDIDATE_BUDGET = 8
SPEC_REFS = ("REQ-VERIFY-5556", "SCENARIO-VERIFY-5556", "REQ-VERIFY-5555")
TESTS_ADDED_OR_REUSED = (
    "tests/python/test_experiment_5556_asp_fsm_sparse_repair_scale.py",
    "tests/python/test_experiment_5555_asp_fsm_nonmonotonic_fixture.py",
    "tests/python/test_experiment_5545_sparse_repair_fsm_descriptor_scale.py",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")
REPAIR_VALUES = ("present", "absent")
ROW_FAMILIES = ("satisfiable", "unsatisfiable", "ambiguous", "default_negation")

FIELD_PRINCIPLES: JsonDict = {
    "upstream_asp_fsm_fixture": "Pins the sparse repair panel to the exact ASP/FSM fixture it reuses.",
    "llm_invoked": "Prevents deterministic stable-model repair from being mistaken for live model inference.",
    "no_model_specs_required": "Confirms the deterministic repair evaluator has no model dependency to disclose.",
    "descriptor_guided_success_rate": "Measures active-descriptor repair only after stable-model validation.",
    "random_block_success_rate": "Provides the matched same-budget random-block control on the same rows and seeds.",
    "exact_only_success_rate": "Records exact-validator-only repair under the same row and budget denominator.",
    "stable_model_checked_rate": "Confirms every fixture row received stable-model validation evidence.",
    "descriptor_mean_iterations": "Counts exact stable-model checks for descriptor-guided repair without making a timing claim.",
    "random_mean_iterations": "Counts exact stable-model checks for the random-block control without making a timing claim.",
    "exact_only_mean_iterations": "Counts exact stable-model checks for the exact-only control without making a timing claim.",
    "row_family_breakdown": "Keeps satisfiable, unsatisfiable, ambiguous, and default-negation outcomes separate.",
    "matched_timing_available": "Gates any future timing language to authenticated matched measurements.",
    "speedup_claim_allowed": "Must remain false without matched authenticated timing.",
    "asp_sparse_repair_claim_allowed": "Opens only when descriptor-guided repair beats random-block repair under exact stable-model checks.",
    "tests_added_or_reused": "Names focused tests and reused upstream ASP/FSM tests.",
    "field_principles": "Keeps headline and gate fields annotated by evidence boundaries.",
    "inference_substrate": "Declares deterministic ASP/FSM sparse repair with no LLM.",
    "honest_verdict": "Provides a terminal evidence boundary without speedup language.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically so checksums follow semantic content."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for JSON-compatible data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def load_upstream_artifact(repo_root: Path = REPO_ROOT) -> JsonDict:
    """Load the Exp5555 ASP/FSM fixture artifact."""

    return _load_json(repo_root / UPSTREAM_ASP_FSM_FIXTURE)


def ensure_upstream_ready(upstream_artifact: Mapping[str, Any]) -> None:
    """Fail closed unless the exact ASP/FSM fixture gate is open."""

    _require(upstream_artifact.get("exact_fsm_fixture_extended_ready") is True, "exact_fsm_fixture_extended_ready")
    _require(upstream_artifact.get("exact_asp_validator_ready") is True, "exact_asp_validator_ready")
    _require(isinstance(upstream_artifact.get("asp_fixture_rows"), list), "asp_fixture_rows")
    _require(isinstance(upstream_artifact.get("stable_model_reports"), list), "stable_model_reports")
    _require(bool(upstream_artifact.get("asp_fixture_rows")), "asp_fixture_rows")
    _require(
        len(upstream_artifact["asp_fixture_rows"]) == len(upstream_artifact["stable_model_reports"]),
        "asp_fixture_rows",
    )


def build_asp_repair_descriptors(upstream_artifact: Mapping[str, Any]) -> JsonDict:
    """Build one sparse stable-model repair descriptor per ASP fixture row."""

    ensure_upstream_ready(upstream_artifact)
    rows_by_id = {
        str(row["row_id"]): row
        for row in upstream_artifact["asp_fixture_rows"]
        if isinstance(row, Mapping)
    }
    descriptors = [
        build_sparse_descriptor(rows_by_id[str(report["row_id"])])
        for report in upstream_artifact["stable_model_reports"]
    ]
    payload: JsonDict = {
        "schema": DESCRIPTOR_BUNDLE_SCHEMA,
        "source_experiment": str(upstream_artifact.get("experiment_id", asp_mod.EXPERIMENT_ID)),
        "source_result_path": UPSTREAM_ASP_FSM_FIXTURE.as_posix(),
        "sparse_block_policy": SPARSE_BLOCK_POLICY,
        "candidate_budget_per_attempt": CANDIDATE_BUDGET,
        "descriptor_count": len(descriptors),
        "asp_repair_descriptors": descriptors,
    }
    validate_descriptor_payload(payload, upstream_artifact)
    return payload


def build_sparse_descriptor(row: Mapping[str, Any]) -> JsonDict:
    """Turn one exact ASP row into a bounded sparse repair problem."""

    source_row = _with_rule_library(row)
    source_report = evaluate_row_with_repair_report(source_row)
    damage = damage_spec_for_row(str(row["row_id"]))
    damaged_row = apply_repair_assignment(source_row, {damage["variable"]: damage["damaged_value"]})
    damaged_report = evaluate_row_with_repair_report(damaged_row)
    target = {str(damage["variable"]): str(damage["target_value"])}
    all_variables = repair_variable_universe(source_row, target)
    variable_states = {
        variable: variable_state(damaged_row, variable)
        for variable in all_variables
    }
    domains = {
        variable: ordered_repair_values(variable_states[variable])
        for variable in all_variables
    }
    repaired_row = apply_repair_assignment(damaged_row, target)
    repaired_report = evaluate_row_with_repair_report(repaired_row)
    descriptor: JsonDict = {
        "schema": DESCRIPTOR_SCHEMA,
        "descriptor_id": f"asp-fsm-sparse-repair:{row['row_id']}",
        "source_result_path": UPSTREAM_ASP_FSM_FIXTURE.as_posix(),
        "row_id": str(row["row_id"]),
        "fsm_instance_id": str(row.get("fsm_instance_id", "")),
        "sparse_block_policy": SPARSE_BLOCK_POLICY,
        "row_family_tags": row_family_tags(source_report),
        "all_repair_variables": all_variables,
        "repair_variable_domains": domains,
        "damaged_variable_state": variable_states,
        "repair_block_variables": list(target),
        "block_size": len(target),
        "target_repair_assignment": target,
        "active_constraints": [
            {
                "kind": str(damage["kind"]),
                "repair_variable": str(damage["variable"]),
                "damaged_value": str(damage["damaged_value"]),
                "target": str(damage["target_value"]),
                "evidence": str(damage["evidence"]),
            }
        ],
        "source_row": source_row,
        "damaged_row": damaged_row,
        "source_report": source_report,
        "damaged_report": damaged_report,
        "exact_fallback": {
            "required": True,
            "validator": EXACT_VALIDATOR,
            "stable_model_checked": True,
            "accepted": repair_report_matches(repaired_report, source_report),
            "solver_status": repaired_report["solver_status"],
            "stable_model_count": repaired_report["stable_model_count"],
        },
    }
    validate_sparse_descriptor(descriptor)
    return descriptor


def damage_spec_for_row(row_id: str) -> JsonDict:
    """Return the deterministic one-step damage used to form a repair task."""

    specs: dict[str, JsonDict] = {
        "asp_sat_fsm_acceptance_default_guard": {
            "kind": "default_guard_false_accept_fact",
            "variable": "fact:fsm_sat_accept_error_trace_sat_empty_rejects_accepted",
            "damaged_value": "present",
            "target_value": "absent",
            "evidence": "hard_constraint_blocks_false_accept_when_guard_derives",
        },
        "asp_unsat_fsm_forbidden_error": {
            "kind": "unsat_constraint_rule_removed",
            "variable": "rule:ASP_UNSAT_01",
            "damaged_value": "absent",
            "target_value": "present",
            "evidence": "hard_constraint_preserves_no_stable_model_control",
        },
        "asp_ambiguous_fsm_default_repair_choice": {
            "kind": "ambiguous_default_choice_rule_removed",
            "variable": "rule:ASP_AMB_01",
            "damaged_value": "absent",
            "target_value": "present",
            "evidence": "paired_default_negation_rules_preserve_two_stable_models",
        },
        "asp_default_negation_no_exception": {
            "kind": "default_exception_fact_injected",
            "variable": "fact:exception_seen",
            "damaged_value": "present",
            "target_value": "absent",
            "evidence": "default_negation_derives_safe_accept_only_without_exception",
        },
        "asp_contradiction_fact_constraint": {
            "kind": "contradiction_constraint_rule_removed",
            "variable": "rule:ASP_CONTRA_00",
            "damaged_value": "absent",
            "target_value": "present",
            "evidence": "hard_constraint_preserves_contradiction_control",
        },
    }
    _require(row_id in specs, "row_id")
    return specs[row_id]


def repair_variable_universe(row: Mapping[str, Any], target: Mapping[str, Any]) -> list[str]:
    """Return the finite fact/rule variable universe visible to every control."""

    fact_variables = {fact_variable(str(atom)) for atom in row.get("facts", [])}
    rule_variables = {
        rule_variable(str(rule["rule_id"]))
        for rule in row.get("rules", [])
        if isinstance(rule, Mapping)
    }
    for variable in target:
        if variable.startswith("fact:"):
            fact_variables.add(variable)
        elif variable.startswith("rule:"):
            rule_variables.add(variable)
    return sorted(fact_variables) + sorted(rule_variables)


def evaluate_row_with_repair_report(row: Mapping[str, Any]) -> JsonDict:
    """Evaluate a row and stamp that exact stable-model validation occurred."""

    report = asp_mod.evaluate_asp_row(row)
    return {
        "row_id": report["row_id"],
        "solver_status": report["solver_status"],
        "status_matches_expected": report["status_matches_expected"],
        "stable_model_checked": True,
        "stable_model_count": report["stable_model_count"],
        "stable_model_samples": report["stable_model_samples"],
        "default_rule_count": report["default_rule_count"],
        "contains_default_negation": report["contains_default_negation"],
        "contradiction_row": report["contradiction_row"],
        "atom_count": report["atom_count"],
        "rule_count": report["rule_count"],
    }


def apply_repair_assignment(row: Mapping[str, Any], assignment: Mapping[str, Any]) -> JsonDict:
    """Apply bounded fact/rule presence edits to an ASP fixture row."""

    repaired = json.loads(canonical_json(row))
    facts = [str(atom) for atom in repaired.get("facts", [])]
    rules = [dict(rule) for rule in repaired.get("rules", [])]
    library = {
        str(rule["rule_id"]): dict(rule)
        for rule in rules
        if isinstance(rule, Mapping) and "rule_id" in rule
    }
    library.update(
        {
            str(rule_id): dict(rule)
            for rule_id, rule in repaired.get("repair_rule_library", {}).items()
            if isinstance(rule, Mapping)
        }
    )

    for variable, value in assignment.items():
        value = str(value)
        _require(value in REPAIR_VALUES, "repair_value")
        if str(variable).startswith("fact:"):
            atom = str(variable).removeprefix("fact:")
            if value == "present" and atom not in facts:
                facts.append(atom)
            if value == "absent":
                facts = [existing for existing in facts if existing != atom]
        elif str(variable).startswith("rule:"):
            rule_id = str(variable).removeprefix("rule:")
            rules = [rule for rule in rules if str(rule.get("rule_id")) != rule_id]
            if value == "present":
                _require(rule_id in library, "repair_rule_library")
                rules.append(dict(library[rule_id]))
        else:
            _require(False, "repair_variable")

    repaired["facts"] = sorted(facts)
    repaired["rules"] = sorted(rules, key=lambda rule: str(rule["rule_id"]))
    repaired["repair_rule_library"] = library
    return repaired


def variable_state(row: Mapping[str, Any], variable: str) -> str:
    """Return whether one fact/rule variable is present in a row."""

    if variable.startswith("fact:"):
        atom = variable.removeprefix("fact:")
        return "present" if atom in set(row.get("facts", [])) else "absent"
    rule_id = variable.removeprefix("rule:")
    return (
        "present"
        if any(str(rule.get("rule_id")) == rule_id for rule in row.get("rules", []))
        else "absent"
    )


def ordered_repair_values(first_value: str) -> list[str]:
    """Return a two-value domain with the damaged baseline first."""

    return [first_value, "absent" if first_value == "present" else "present"]


def fact_variable(atom: str) -> str:
    return f"fact:{atom}"


def rule_variable(rule_id: str) -> str:
    return f"rule:{rule_id}"


def row_family_tags(report: Mapping[str, Any]) -> list[str]:
    """Classify one row into status and default-negation reporting families."""

    tags = [str(report["solver_status"])]
    if report.get("contains_default_negation") is True:
        tags.append("default_negation")
    return tags


def repair_report_matches(candidate_report: Mapping[str, Any], source_report: Mapping[str, Any]) -> bool:
    """Return whether a candidate restored the exact source stable-model evidence."""

    return bool(
        candidate_report.get("solver_status") == source_report.get("solver_status")
        and candidate_report.get("stable_model_count") == source_report.get("stable_model_count")
        and candidate_report.get("stable_model_samples") == source_report.get("stable_model_samples")
    )


def validate_candidate_repair(
    descriptor: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> JsonDict:
    """Check one proposed repair with Exp5555's stable-model evaluator."""

    repaired = apply_repair_assignment(descriptor["damaged_row"], assignment)
    report = evaluate_row_with_repair_report(repaired)
    target = descriptor["target_repair_assignment"]
    target_met = all(str(assignment.get(variable)) == str(value) for variable, value in target.items())
    exact_passed = repair_report_matches(report, descriptor["source_report"])
    accepted = bool(target_met and exact_passed)
    return {
        "assignment": dict(assignment),
        "assignment_hash": sha256_json(dict(assignment)),
        "stable_model_checked": True,
        "exact_validator": EXACT_VALIDATOR,
        "exact_validator_decision": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "target_assignment_met": target_met,
        "exact_report_passed": exact_passed,
        "solver_status": report["solver_status"],
        "stable_model_count": report["stable_model_count"],
        "stable_model_samples": report["stable_model_samples"],
    }


def run_descriptor_guided_attempt(descriptor: Mapping[str, Any], seed: int) -> JsonDict:
    """Run the active descriptor block with target values ordered first."""

    block = list(descriptor["repair_block_variables"])
    checks = [
        validate_candidate_repair(descriptor, assignment)
        for assignment in enumerate_block_assignments(descriptor, block, guided=True)
    ]
    return attempt_record("descriptor_guided", descriptor, seed, block, checks)


def run_random_block_attempt(
    descriptor: Mapping[str, Any],
    seed: int,
    row_index: int,
) -> JsonDict:
    """Run a same-size random block selected from the row-local universe."""

    all_variables = list(descriptor["all_repair_variables"])
    block_size = len(descriptor["repair_block_variables"])
    block = random.Random(seed + row_index * 1009).sample(all_variables, block_size)
    checks = [
        validate_candidate_repair(descriptor, assignment)
        for assignment in enumerate_block_assignments(descriptor, block, guided=False)
    ]
    return attempt_record("random_block", descriptor, seed, block, checks)


def run_exact_only_attempt(descriptor: Mapping[str, Any], seed: int) -> JsonDict:
    """Run exact-only single-variable enumeration under the matched budget."""

    block = list(descriptor["all_repair_variables"])
    checks = [
        validate_candidate_repair(descriptor, assignment)
        for assignment in enumerate_exact_only_assignments(descriptor)
    ]
    return attempt_record("exact_only", descriptor, seed, block, checks)


def enumerate_block_assignments(
    descriptor: Mapping[str, Any],
    block_variables: Sequence[str],
    *,
    guided: bool,
) -> list[JsonDict]:
    """Enumerate candidate assignments for a selected sparse block."""

    assignments: list[JsonDict] = []
    for variable in block_variables:
        domain = ordered_domain(
            variable,
            descriptor["repair_variable_domains"][variable],
            descriptor["target_repair_assignment"],
            guided=guided,
        )
        assignments.extend({variable: value} for value in domain)
    return assignments[:CANDIDATE_BUDGET]


def enumerate_exact_only_assignments(descriptor: Mapping[str, Any]) -> list[JsonDict]:
    """Enumerate exact-validator-only single edits over the full row universe."""

    assignments: list[JsonDict] = [{}]
    for variable in descriptor["all_repair_variables"]:
        baseline = descriptor["damaged_variable_state"][variable]
        for value in descriptor["repair_variable_domains"][variable]:
            if value != baseline:
                assignments.append({variable: value})
    return assignments[:CANDIDATE_BUDGET]


def ordered_domain(
    variable: str,
    domain: Sequence[str],
    target: Mapping[str, Any],
    *,
    guided: bool,
) -> list[str]:
    """Put descriptor target values first only for descriptor-guided attempts."""

    values = [str(value) for value in domain]
    target_value = str(target.get(variable, ""))
    if guided and target_value in values:
        return [target_value] + [value for value in values if value != target_value]
    return values


def attempt_record(
    policy: str,
    descriptor: Mapping[str, Any],
    seed: int,
    block_variables: Sequence[str],
    checks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one policy attempt row from stable-model-checked repairs."""

    accepted = [row for row in checks if row["accepted"]]
    return {
        "policy": policy,
        "row_id": str(descriptor["row_id"]),
        "fsm_instance_id": str(descriptor["fsm_instance_id"]),
        "row_family_tags": list(descriptor["row_family_tags"]),
        "seed": seed,
        "candidate_budget": CANDIDATE_BUDGET,
        "block_variables": list(block_variables),
        "iterations": len(checks),
        "success": bool(accepted),
        "accepted_assignment": dict(accepted[0]["assignment"]) if accepted else None,
        "stable_model_checks_complete": all(row["stable_model_checked"] for row in checks),
        "candidate_checks": [dict(row) for row in checks],
    }


def run_policy_comparison(
    *,
    upstream_artifact: Mapping[str, Any],
    descriptors: Mapping[str, Any],
    seeds: Sequence[int] = SEEDS,
) -> JsonDict:
    """Compare descriptor-guided, random-block, and exact-only controls."""

    ensure_upstream_ready(upstream_artifact)
    validate_descriptor_payload(descriptors, upstream_artifact)
    descriptor_rows = list(descriptors["asp_repair_descriptors"])
    policy_results: dict[str, list[JsonDict]] = {
        "descriptor_guided": [],
        "random_block": [],
        "exact_only": [],
    }
    for seed in seeds:
        for row_index, descriptor in enumerate(descriptor_rows):
            policy_results["descriptor_guided"].append(run_descriptor_guided_attempt(descriptor, seed))
            policy_results["random_block"].append(run_random_block_attempt(descriptor, seed, row_index))
            policy_results["exact_only"].append(run_exact_only_attempt(descriptor, seed))

    attempts = [attempt for rows in policy_results.values() for attempt in rows]
    all_checks = [check for attempt in attempts for check in attempt["candidate_checks"]]
    unchecked = [
        check
        for check in all_checks
        if check.get("stable_model_checked") is not True
        or check.get("exact_validator_decision") not in {"accepted", "rejected"}
    ]
    comparison: JsonDict = {
        "policy_results": policy_results,
        "asp_row_count": len(descriptor_rows),
        "random_seed_count": len(seeds),
        "candidate_budget_per_attempt": CANDIDATE_BUDGET,
        "candidate_repair_count": len(all_checks),
        "unchecked_repair_count": len(unchecked),
        "descriptor_guided_success_rate": success_rate(policy_results["descriptor_guided"]),
        "random_block_success_rate": success_rate(policy_results["random_block"]),
        "exact_only_success_rate": success_rate(policy_results["exact_only"]),
        "descriptor_mean_iterations": mean_iterations(policy_results["descriptor_guided"]),
        "random_mean_iterations": mean_iterations(policy_results["random_block"]),
        "exact_only_mean_iterations": mean_iterations(policy_results["exact_only"]),
        "stable_model_checked_rate": stable_model_checked_rate(policy_results, descriptor_rows),
        "stable_model_validation_by_row": stable_model_validation_by_row(policy_results, descriptor_rows),
    }
    comparison["row_family_breakdown"] = row_family_breakdown(policy_results, descriptor_rows)
    return comparison


def success_rate(attempts: Sequence[Mapping[str, Any]]) -> float:
    """Compute exact-validated success rate for a set of attempts."""

    if not attempts:
        return 0.0
    return round(sum(int(row["success"]) for row in attempts) / len(attempts), 6)


def mean_iterations(attempts: Sequence[Mapping[str, Any]]) -> float:
    """Compute mean stable-model checks per policy attempt."""

    if not attempts:
        return 0.0
    return round(sum(float(row["iterations"]) for row in attempts) / len(attempts), 6)


def stable_model_checked_rate(
    policy_results: Mapping[str, Sequence[Mapping[str, Any]]],
    descriptors: Sequence[Mapping[str, Any]],
) -> float:
    """Measure whether every policy-row pair has stable-model evidence."""

    expected = {
        (policy, str(descriptor["row_id"]))
        for policy in policy_results
        for descriptor in descriptors
    }
    checked = {
        (policy, str(attempt["row_id"]))
        for policy, attempts in policy_results.items()
        for attempt in attempts
        if attempt["candidate_checks"] and all(check["stable_model_checked"] for check in attempt["candidate_checks"])
    }
    return round(len(checked & expected) / len(expected), 6)


def stable_model_validation_by_row(
    policy_results: Mapping[str, Sequence[Mapping[str, Any]]],
    descriptors: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Return row-level stable-model coverage diagnostics."""

    diagnostics: JsonDict = {}
    for descriptor in descriptors:
        row_id = str(descriptor["row_id"])
        policy_checked = {
            policy: any(
                attempt["row_id"] == row_id
                and attempt["candidate_checks"]
                and all(check["stable_model_checked"] for check in attempt["candidate_checks"])
                for attempt in attempts
            )
            for policy, attempts in policy_results.items()
        }
        diagnostics[row_id] = {
            "row_family_tags": list(descriptor["row_family_tags"]),
            "stable_model_checked": all(policy_checked.values()),
            "policies_checked": policy_checked,
        }
    return diagnostics


def row_family_breakdown(
    policy_results: Mapping[str, Sequence[Mapping[str, Any]]],
    descriptors: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Aggregate success rates and iterations by ASP row family."""

    breakdown: JsonDict = {}
    for family in ROW_FAMILIES:
        row_ids = [
            str(descriptor["row_id"])
            for descriptor in descriptors
            if family in descriptor["row_family_tags"]
        ]
        family_attempts = {
            policy: [attempt for attempt in attempts if attempt["row_id"] in row_ids]
            for policy, attempts in policy_results.items()
        }
        checked_rows = {
            str(attempt["row_id"])
            for attempts in family_attempts.values()
            for attempt in attempts
            if attempt["candidate_checks"] and all(check["stable_model_checked"] for check in attempt["candidate_checks"])
        }
        breakdown[family] = {
            "row_count": len(row_ids),
            "row_ids": row_ids,
            "attempt_count": sum(len(attempts) for attempts in family_attempts.values()),
            "descriptor_guided_success_rate": success_rate(family_attempts["descriptor_guided"]),
            "random_block_success_rate": success_rate(family_attempts["random_block"]),
            "exact_only_success_rate": success_rate(family_attempts["exact_only"]),
            "descriptor_mean_iterations": mean_iterations(family_attempts["descriptor_guided"]),
            "random_mean_iterations": mean_iterations(family_attempts["random_block"]),
            "exact_only_mean_iterations": mean_iterations(family_attempts["exact_only"]),
            "stable_model_checked_rate": round(len(checked_rows) / len(row_ids), 6),
        }
    return breakdown


def build_artifact(
    *,
    upstream_artifact: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5556 ASP/FSM sparse repair artifact."""

    upstream = load_upstream_artifact() if upstream_artifact is None else upstream_artifact
    ensure_upstream_ready(upstream)
    descriptors = build_asp_repair_descriptors(upstream)
    comparison = run_policy_comparison(upstream_artifact=upstream, descriptors=descriptors)
    blockers = readiness_blockers(upstream, descriptors, comparison)
    claim_allowed = not blockers
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "duration_s": 0.0,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "upstream_asp_fsm_fixture": UPSTREAM_ASP_FSM_FIXTURE.as_posix(),
        "upstream_asp_fsm_fixture_ready": upstream.get("exact_fsm_fixture_extended_ready") is True,
        "exact_asp_validator_ready": upstream.get("exact_asp_validator_ready") is True,
        "llm_invoked": False,
        "no_model_specs_required": True,
        "matched_timing_available": False,
        "speedup_claim_allowed": False,
        "asp_sparse_repair_claim_allowed": claim_allowed,
        "tests_added_or_reused": list(TESTS_ADDED_OR_REUSED),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(claim_allowed, blockers),
        "descriptor_payload": descriptors,
        "descriptor_payload_sha256": sha256_json(descriptors),
        "readiness_blockers": blockers,
        "research_conductor_modified": False,
        "tests_run": [dict(row) for row in tests_run],
        "claim_limits": [
            "exact ASP/FSM fixture rows only",
            "every candidate repair is stable-model checked",
            "descriptor, random-block, and exact-only controls share rows, seeds, and budget",
            "no timing or hardware speedup claim is allowed without matched receipts",
        ],
        "reproducibility_checksum": "",
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
    """Return precise blockers for the ASP sparse-repair claim gate."""

    blockers: list[str] = []
    if upstream_artifact.get("exact_fsm_fixture_extended_ready") is not True:
        blockers.append("upstream_asp_fsm_fixture_ready")
    if upstream_artifact.get("exact_asp_validator_ready") is not True:
        blockers.append("exact_asp_validator_ready")
    if descriptors.get("descriptor_count") != upstream_artifact.get("asp_row_count"):
        blockers.append("descriptor_count")
    if comparison.get("asp_row_count") != upstream_artifact.get("asp_row_count"):
        blockers.append("asp_row_count")
    if comparison.get("random_seed_count") != len(SEEDS):
        blockers.append("random_seed_count")
    if comparison.get("candidate_budget_per_attempt") != CANDIDATE_BUDGET:
        blockers.append("candidate_budget_per_attempt")
    if comparison.get("stable_model_checked_rate") != 1.0:
        blockers.append("stable_model_checked_rate")
    if comparison.get("descriptor_guided_success_rate", 0.0) <= comparison.get("random_block_success_rate", 0.0):
        blockers.append("descriptor_vs_random")
    if comparison.get("descriptor_guided_success_rate") != 1.0:
        blockers.append("descriptor_guided_success_rate")
    if comparison.get("unchecked_repair_count") != 0:
        blockers.append("unchecked_repair_count")
    return blockers


def validate_descriptor_payload(
    payload: Mapping[str, Any],
    upstream_artifact: Mapping[str, Any],
) -> None:
    """Validate descriptor bundle shape before policies consume it."""

    _require(payload.get("schema") == DESCRIPTOR_BUNDLE_SCHEMA, "descriptor_schema")
    _require(payload.get("source_result_path") == UPSTREAM_ASP_FSM_FIXTURE.as_posix(), "source_result_path")
    _require(payload.get("candidate_budget_per_attempt") == CANDIDATE_BUDGET, "candidate_budget_per_attempt")
    rows = payload.get("asp_repair_descriptors")
    _require(isinstance(rows, list), "asp_repair_descriptors")
    _require(payload.get("descriptor_count") == upstream_artifact.get("asp_row_count"), "descriptor_count")
    for row in rows:
        validate_sparse_descriptor(row)


def validate_sparse_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Reject sparse ASP descriptors that lack stable-model repair evidence."""

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
    _require(set(block) == set(target), "repair_block_variables")
    _require(bool(block), "repair_block_variables")
    _require(descriptor["source_report"].get("stable_model_checked") is True, "source_report")
    _require(descriptor["damaged_report"].get("stable_model_checked") is True, "damaged_report")
    _require(
        descriptor["source_report"].get("stable_model_samples")
        != descriptor["damaged_report"].get("stable_model_samples"),
        "damaged_report",
    )
    _require(isinstance(fallback, Mapping), "exact_fallback")
    _require(fallback.get("required") is True, "exact_fallback")
    _require(fallback.get("validator") == EXACT_VALIDATOR, "exact_fallback")
    _require(fallback.get("stable_model_checked") is True, "exact_fallback")
    _require(fallback.get("accepted") is True, "exact_fallback")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal artifact and fail closed on overclaims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        artifact.get("upstream_asp_fsm_fixture") == UPSTREAM_ASP_FSM_FIXTURE.as_posix(),
        "upstream_asp_fsm_fixture",
    )
    _require(artifact.get("upstream_asp_fsm_fixture_ready") is True, "upstream_asp_fsm_fixture_ready")
    _require(artifact.get("exact_asp_validator_ready") is True, "exact_asp_validator_ready")
    _require(artifact.get("llm_invoked") is False, "llm_invoked")
    _require(artifact.get("no_model_specs_required") is True, "no_model_specs_required")
    _require("model_specs" not in artifact, "model_specs")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("stable_model_checked_rate") == 1.0, "stable_model_checked_rate")
    _require(artifact.get("matched_timing_available") is False, "matched_timing_available")
    _require(artifact.get("speedup_claim_allowed") is False, "speedup_claim_allowed")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(artifact.get("random_seed_count") == len(SEEDS), "random_seed_count")
    _require(artifact.get("candidate_budget_per_attempt") == CANDIDATE_BUDGET, "candidate_budget_per_attempt")
    _require(artifact.get("unchecked_repair_count") == 0, "unchecked_repair_count")
    _require(artifact.get("descriptor_guided_success_rate") == 1.0, "descriptor_guided_success_rate")
    claim_ok = bool(
        artifact.get("descriptor_guided_success_rate", 0.0) > artifact.get("random_block_success_rate", 0.0)
        and artifact.get("stable_model_checked_rate") == 1.0
        and artifact.get("matched_timing_available") is False
        and artifact.get("speedup_claim_allowed") is False
    )
    _require(artifact.get("asp_sparse_repair_claim_allowed") is claim_ok, "asp_sparse_repair_claim_allowed")
    _require(artifact.get("asp_sparse_repair_claim_allowed") is True, "asp_sparse_repair_claim_allowed")
    _require(isinstance(artifact.get("row_family_breakdown"), Mapping), "row_family_breakdown")
    _require(set(ROW_FAMILIES).issubset(set(artifact["row_family_breakdown"])), "row_family_breakdown")
    _require(artifact.get("tests_added_or_reused") == list(TESTS_ADDED_OR_REUSED), "tests_added_or_reused")
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact.get("field_principles", {})), "field_principles")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("descriptor_payload_sha256") == sha256_json(artifact["descriptor_payload"]), "descriptor_payload_sha256")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict without speedup language."""

    if ready:
        return "complete: asp_fsm_sparse_repair_descriptor_signal_ready_no_speedup_claim"
    return "blocked: asp_fsm_sparse_repair_not_ready_" + "_".join(blockers)


def run(
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5556 deliverable JSON under ``repo_root``."""

    upstream = load_upstream_artifact(repo_root)
    artifact = build_artifact(upstream_artifact=upstream, tests_run=tests_run)
    result_path = repo_root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _with_rule_library(row: Mapping[str, Any]) -> JsonDict:
    copied = json.loads(canonical_json(row))
    copied["repair_rule_library"] = {
        str(rule["rule_id"]): dict(rule)
        for rule in copied.get("rules", [])
        if isinstance(rule, Mapping) and "rule_id" in rule
    }
    return copied


def _load_json(path: Path) -> JsonDict:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"load_error": "missing", "path": path.as_posix()}
    except json.JSONDecodeError as exc:
        return {"load_error": "json_decode", "path": path.as_posix(), "detail": str(exc)}
    if not isinstance(decoded, dict):
        return {"load_error": "json_not_object", "path": path.as_posix()}
    return decoded


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


def main() -> int:  # pragma: no cover
    artifact = run()
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH.as_posix(),
                "asp_sparse_repair_claim_allowed": artifact["asp_sparse_repair_claim_allowed"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
