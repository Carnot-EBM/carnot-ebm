"""Exp5531 sparse repair scale-up with confidence intervals.

Spec refs: REQ-VERIFY-5531, SCENARIO-VERIFY-5531.

This module scales the Exp5518 exact-checked sparse repair interface without
turning descriptor selection into a speedup headline. Each fixture is still a
small finite-domain hard/soft constraint problem. Sparse repair proposes edits
only to variables named by violated hard constraints, and the exact validator
accepts or rejects every candidate before any aggregate is counted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any

from carnot import experiment_5505_active_constraint_milp_descriptor_v499 as descriptor_mod
from carnot import experiment_5518_block_gibbs_sparse_repair_descriptors as sparse5518


JsonDict = dict[str, Any]
Assignment = dict[str, str]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5531_sparse_repair_scaleup_ci.json")

SCHEMA = "carnot.experiment_5531.sparse_repair_scaleup_ci.v501"
FIXTURE_SCHEMA = "carnot.fixture.sparse_repair_scaleup_5531.v1"
DESCRIPTOR_BUNDLE_SCHEMA = "carnot.descriptor_bundle.sparse_repair_scaleup_5531.v1"
EXPERIMENT = 5531
EXPERIMENT_ID = "exp5531-sparse-repair-scaleup-ci"
MILESTONE = "2026.07.501"
RUN_DATE = "2026-07-10"
INFERENCE_SUBSTRATE = "exact_checked_sparse_repair_scaleup"
SPARSE_BLOCK_POLICY = sparse5518.SPARSE_BLOCK_POLICY
EXACT_VALIDATOR = sparse5518.EXACT_VALIDATOR
SEEDS = (5531, 5532, 5533, 5534, 5535, 5536, 5537)
SPEC_REFS = ("REQ-VERIFY-5531", "SCENARIO-VERIFY-5531", "REQ-VERIFY-5518")
TERMINAL_PREFIXES = ("complete:", "blocked:")
TEST_PATHS = (
    "tests/python/test_experiment_5531_sparse_repair_scaleup_ci.py",
    "tests/python/test_experiment_5518_block_gibbs_sparse_repair_descriptors.py",
    "tests/python/test_experiment_5499_preference_maxsat_minimal_fixture_v499.py",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_families": "names the larger exact-checkable fixture families behind the aggregate.",
    "n_instances": "bounds the scale-up evidence size without implying benchmark breadth.",
    "n_seeds": "records deterministic multi-seed evidence rather than a single lucky row.",
    "exact_only_success_rate": "full exact fallback acceptance reference.",
    "sparse_repair_success_rate": "descriptor-guided repair acceptance under exact checking.",
    "random_block_success_rate": "same-size random block control under exact checking.",
    "mean_iterations_exact_only": "iteration evidence for the exact-only baseline.",
    "mean_iterations_sparse_repair": "iteration evidence for sparse repair; not a speedup claim.",
    "exact_fallback_rate": "fraction of policy attempts with exact accept/reject validation.",
    "confidence_intervals": "uncertainty over success and iteration summaries, not headline proof.",
    "matched_timing_available": "matched timing gate for any future speedup language.",
    "speedup_claim_allowed": "must remain false without matched timing evidence.",
    "active_constraint_sparse_repair_ready": (
        "readiness gate for exact-checked active-constraint repair."
    ),
    "tests_added_or_reused": "names the tests supporting this artifact.",
    "field_principles": "keeps headline and gate fields annotated by evidence boundaries.",
    "inference_substrate": "declares exact-checked sparse repair scale-up, no live inference.",
    "honest_verdict": "terminal status; start with complete: or blocked: and avoid speedup claims.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
CI_FIELDS = (
    "exact_only_success_rate",
    "sparse_repair_success_rate",
    "random_block_success_rate",
    "mean_iterations_exact_only",
    "mean_iterations_sparse_repair",
    "exact_fallback_rate",
)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so checksums review semantic content."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(payload: Any) -> str:
    """Hash a JSON-compatible payload after stable serialization."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_scaleup_fixtures() -> JsonDict:
    """Build exact-checkable active hard/soft fixtures for the scale-up panel."""

    instances = [_with_family(row, "exp5518_typed_claims") for row in sparse5518.build_selected_fixtures()["instances"]]
    instances.extend(
        [
            make_instance(
                instance_id="active_claim_four_variable_support",
                fixture_family="four_variable_active_claims",
                typed_claims=[
                    claim("support", "evidence_support", ["unsupported", "entailed"]),
                    claim("citation", "citation_state", ["missing", "present"]),
                    claim("scope", "claim_scope", ["overbroad", "bounded"]),
                    claim("source_quality", "provenance_quality", ["secondary", "primary"]),
                ],
                hard_constraints=[
                    clause("HC_SUPPORT_ENTAILED", "support", "entailed"),
                    clause("HC_CITATION_PRESENT", "citation", "present"),
                ],
                soft_preferences=[
                    value_reward("SP_PRIMARY_SOURCE", "source_quality", "primary", 6),
                    value_reward("SP_ENTAILED_SUPPORT", "support", "entailed", 3),
                    value_reward("SP_CITATION_PRESENT", "citation", "present", 2),
                    value_reward("SP_BOUNDED_SCOPE", "scope", "bounded", 1),
                ],
                initial_assignment={
                    "support": "unsupported",
                    "citation": "missing",
                    "scope": "bounded",
                    "source_quality": "primary",
                },
            ),
            make_instance(
                instance_id="active_claim_ternary_evidence_risk",
                fixture_family="ternary_active_claims",
                typed_claims=[
                    claim("risk", "risk_label", ["high", "medium", "low"]),
                    claim("evidence", "evidence_state", ["absent", "partial", "complete"]),
                    claim("freshness", "evidence_freshness", ["stale", "current"]),
                    claim("audience", "claim_audience", ["broad", "bounded"]),
                    claim("action", "validator_action", ["reject", "review", "accept"]),
                ],
                hard_constraints=[
                    clause("HC_RISK_LOW", "risk", "low"),
                    clause("HC_EVIDENCE_COMPLETE", "evidence", "complete"),
                    clause("HC_FRESHNESS_CURRENT", "freshness", "current"),
                ],
                soft_preferences=[
                    value_reward("SP_ACCEPT", "action", "accept", 5),
                    value_reward("SP_BOUNDED_AUDIENCE", "audience", "bounded", 2),
                    value_reward("SP_LOW_RISK", "risk", "low", 2),
                    value_reward("SP_COMPLETE_EVIDENCE", "evidence", "complete", 2),
                    value_reward("SP_CURRENT_EVIDENCE", "freshness", "current", 1),
                ],
                initial_assignment={
                    "risk": "high",
                    "evidence": "absent",
                    "freshness": "stale",
                    "audience": "bounded",
                    "action": "accept",
                },
            ),
            make_instance(
                instance_id="active_claim_six_variable_quality_scope",
                fixture_family="four_variable_active_claims",
                typed_claims=[
                    claim("numeric_consistency", "numeric_consistency", ["bad", "ok"]),
                    claim("citation", "citation_state", ["missing", "present"]),
                    claim("scope", "claim_scope", ["overbroad", "bounded"]),
                    claim("recency", "evidence_recency", ["old", "recent"]),
                    claim("source_quality", "provenance_quality", ["secondary", "primary"]),
                    claim("tone", "claim_tone", ["speculative", "cautious"]),
                ],
                hard_constraints=[
                    clause("HC_NUMERIC_OK", "numeric_consistency", "ok"),
                    clause("HC_CITATION_PRESENT", "citation", "present"),
                    clause("HC_SCOPE_BOUNDED", "scope", "bounded"),
                ],
                soft_preferences=[
                    value_reward("SP_RECENT", "recency", "recent", 4),
                    value_reward("SP_PRIMARY", "source_quality", "primary", 4),
                    value_reward("SP_CAUTIOUS", "tone", "cautious", 2),
                    value_reward("SP_NUMERIC_OK", "numeric_consistency", "ok", 2),
                    value_reward("SP_CITATION_PRESENT", "citation", "present", 2),
                    value_reward("SP_SCOPE_BOUNDED", "scope", "bounded", 1),
                ],
                initial_assignment={
                    "numeric_consistency": "bad",
                    "citation": "missing",
                    "scope": "overbroad",
                    "recency": "recent",
                    "source_quality": "primary",
                    "tone": "cautious",
                },
            ),
        ]
    )
    payload = {
        "schema": FIXTURE_SCHEMA,
        "source_experiment": sparse5518.EXPERIMENT_ID,
        "fixture_families": fixture_families({"instances": instances}),
        "exact_validator": EXACT_VALIDATOR,
        "instances": instances,
    }
    validate_scaleup_fixture_payload(payload)
    return payload


def _with_family(instance: Mapping[str, Any], fixture_family: str) -> JsonDict:
    row = dict(instance)
    row["fixture_family"] = fixture_family
    validate_scaleup_instance(row)
    return row


def claim(name: str, claim_type: str, domain: Sequence[str]) -> JsonDict:
    """Build one typed variable declaration for an exact finite-domain fixture."""

    return {"name": name, "claim_type": claim_type, "domain": list(domain)}


def clause(identifier: str, variable: str, value: str) -> JsonDict:
    """Build the clause shape already consumed by Exp5518 sparse block derivation."""

    return {"id": identifier, "type": "clause", "literals": [{"variable": variable, "equals": value}]}


def value_reward(identifier: str, variable: str, value: str, weight: int) -> JsonDict:
    """Build a soft preference row that exact solving scores only after hard feasibility."""

    return {
        "id": identifier,
        "type": "value_reward",
        "variable": variable,
        "value": value,
        "weight": weight,
    }


def make_instance(
    *,
    instance_id: str,
    fixture_family: str,
    typed_claims: Sequence[Mapping[str, Any]],
    hard_constraints: Sequence[Mapping[str, Any]],
    soft_preferences: Sequence[Mapping[str, Any]],
    initial_assignment: Mapping[str, str],
) -> JsonDict:
    """Assemble one fixture and attach its exact optimum before validation."""

    instance: JsonDict = {
        "schema": sparse5518.FIXTURE_SCHEMA,
        "instance_id": instance_id,
        "fixture_family": fixture_family,
        "source_fixture_path": "built_in_exp5531_scaleup_fixture",
        "typed_claims": [dict(row) for row in typed_claims],
        "domains": {str(row["name"]): list(row["domain"]) for row in typed_claims},
        "hard_constraints": [dict(row) for row in hard_constraints],
        "soft_preferences": [dict(row) for row in soft_preferences],
        "initial_assignment": dict(initial_assignment),
    }
    instance["exact_reference"] = solve_reference(instance)
    instance["violated_hard_constraints"] = sparse5518.violated_hard_constraint_ids(instance)
    instance["expected_repair_block_variables"] = sparse5518.derive_repair_block(instance)
    validate_scaleup_instance(instance)
    return instance


def solve_reference(instance: Mapping[str, Any]) -> JsonDict:
    """Exhaustively solve one finite-domain hard/soft fixture."""

    names = [str(row["name"]) for row in instance["typed_claims"]]
    feasible: list[JsonDict] = []
    for assignment in enumerate_full_assignments(instance):
        if descriptor_mod.constraints_satisfied(assignment, instance["hard_constraints"]):
            feasible.append(
                {
                    "assignment": assignment,
                    "objective_score": score_assignment(instance, assignment),
                    "assignment_hash": sparse5518.sha256_json(assignment),
                }
            )
    _require(bool(feasible), "exact_reference")
    feasible.sort(key=lambda row: (-float(row["objective_score"]), sparse5518.canonical_json(row["assignment"])))
    best = feasible[0]
    return {
        "status": "optimal",
        "assignment": best["assignment"],
        "objective_score": best["objective_score"],
        "assignment_hash": best["assignment_hash"],
        "feasible_assignment_count": len(feasible),
        "variable_order": names,
    }


def enumerate_full_assignments(instance: Mapping[str, Any]) -> list[Assignment]:
    """Enumerate the full finite domain in the typed-claim declaration order."""

    domains = instance["domains"]
    names = [str(row["name"]) for row in instance["typed_claims"]]
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(domains[name] for name in names))
    ]


def score_assignment(instance: Mapping[str, Any], assignment: Mapping[str, str]) -> float:
    """Score soft preferences for an assignment after hard constraints pass."""

    return round(
        sum(
            descriptor_mod.preference_score(assignment, preference)
            for preference in instance["soft_preferences"]
        ),
        6,
    )


def fixture_families(fixtures: Mapping[str, Any]) -> list[str]:
    """Return stable fixture family names in first-seen order."""

    families: list[str] = []
    for instance in fixtures["instances"]:
        family = str(instance["fixture_family"])
        if family not in families:
            families.append(family)
    return families


def validate_scaleup_fixture_payload(payload: Mapping[str, Any]) -> None:
    """Validate the scale-up fixture bundle before policy comparisons use it."""

    _require(payload.get("schema") == FIXTURE_SCHEMA, "fixture_schema")
    instances = payload.get("instances")
    _require(isinstance(instances, list) and len(instances) >= 5, "n_instances")
    for instance in instances:
        validate_scaleup_instance(instance)
    _require(payload.get("fixture_families") == fixture_families(payload), "fixture_families")


def validate_scaleup_instance(instance: Mapping[str, Any]) -> None:
    """Reject fixtures that cannot be exactly solved by a sparse hard-violation block."""

    variables = [str(row["name"]) for row in instance["typed_claims"]]
    domains = instance["domains"]
    initial = instance["initial_assignment"]
    reference = instance["exact_reference"]
    _require(bool(instance.get("fixture_family")), "fixture_family")
    _require(set(domains) == set(variables), "domains")
    _require(set(initial) == set(variables), "initial_assignment")
    _require(reference.get("status") == "optimal", "exact_reference")
    _require(bool(instance.get("violated_hard_constraints")), "violated_hard_constraints")
    block = list(instance.get("expected_repair_block_variables", []))
    _require(0 < len(block) < len(variables), "repair_block_variables")
    optimal = reference["assignment"]
    for variable in variables:
        _require(initial[variable] in domains[variable], "initial_domain")
        if variable not in block:
            _require(initial[variable] == optimal[variable], "sparse_reachable")


def build_sparse_descriptors(fixtures: Mapping[str, Any]) -> JsonDict:
    """Build Exp5518-shaped sparse repair descriptors for every scale-up instance."""

    validate_scaleup_fixture_payload(fixtures)
    rows: list[JsonDict] = []
    for instance in fixtures["instances"]:
        variables = [str(row["name"]) for row in instance["typed_claims"]]
        repair_block = sparse5518.derive_repair_block(instance)
        descriptor = {
            "schema": sparse5518.DESCRIPTOR_SCHEMA,
            "descriptor_id": f"sparse-repair-scaleup:{instance['instance_id']}",
            "source_fixture_path": instance["source_fixture_path"],
            "source_instance_id": instance["instance_id"],
            "fixture_family": instance["fixture_family"],
            "sparse_block_policy": SPARSE_BLOCK_POLICY,
            "variables": variables,
            "variable_count": len(variables),
            "repair_block_variables": repair_block,
            "block_size": len(repair_block),
            "sparse_subset": 0 < len(repair_block) < len(variables),
            "violated_hard_constraints": sparse5518.violated_hard_constraint_ids(instance),
            "initial_assignment": dict(instance["initial_assignment"]),
            "exact_fallback": {
                "required": True,
                "validator": EXACT_VALIDATOR,
                "status": instance["exact_reference"]["status"],
                "accepted_assignment": instance["exact_reference"]["assignment"],
                "objective_score": instance["exact_reference"]["objective_score"],
            },
        }
        validate_sparse_descriptor(descriptor)
        rows.append(descriptor)
    return {
        "schema": DESCRIPTOR_BUNDLE_SCHEMA,
        "sparse_block_policy": SPARSE_BLOCK_POLICY,
        "descriptor_count": len(rows),
        "sparse_repair_descriptors": rows,
    }


def validate_sparse_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Use the Exp5518 descriptor validator and preserve scale-up metadata."""

    sparse5518.validate_sparse_descriptor(descriptor)
    _require(str(descriptor.get("descriptor_id", "")).startswith("sparse-repair"), "descriptor_id")
    _require(bool(descriptor.get("fixture_family")), "fixture_family")


def run_exact_only_attempt(instance: Mapping[str, Any], seed: int) -> JsonDict:
    """Run the full exact fallback over all variables and count checked rows."""

    variables = [str(row["name"]) for row in instance["typed_claims"]]
    checks: list[JsonDict] = []
    for assignment in sparse5518.enumerate_assignments(instance, variables):
        check = sparse5518.validate_candidate_assignment(instance, assignment)
        checks.append(check)
        if check["accepted"]:
            break
    return attempt_record("exact_only", instance, seed, variables, checks)


def run_sparse_repair_attempt(
    instance: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    seed: int,
) -> JsonDict:
    """Run descriptor-guided sparse repair from the violated starting state."""

    block = list(descriptor["repair_block_variables"])
    checks = [
        sparse5518.validate_candidate_assignment(instance, assignment)
        for assignment in sparse5518.enumerate_assignments(
            instance, block, instance["initial_assignment"]
        )
    ]
    return attempt_record("sparse_repair", instance, seed, block, checks)


def run_random_block_attempt(
    instance: Mapping[str, Any],
    seed: int,
    instance_index: int,
) -> JsonDict:
    """Run a deterministic same-size random block control under exact validation."""

    variables = [str(row["name"]) for row in instance["typed_claims"]]
    block_size = len(sparse5518.derive_repair_block(instance))
    block = random.Random(seed + instance_index * 1009).sample(variables, block_size)
    checks = [
        sparse5518.validate_candidate_assignment(instance, assignment)
        for assignment in sparse5518.enumerate_assignments(
            instance, block, instance["initial_assignment"]
        )
    ]
    return attempt_record("random_block", instance, seed, block, checks)


def attempt_record(
    policy: str,
    instance: Mapping[str, Any],
    seed: int,
    block_variables: Sequence[str],
    checks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one exact-validated policy attempt row."""

    accepted_checks = [row for row in checks if row["accepted"]]
    exact_decisions = {row.get("exact_validator_decision") for row in checks}
    return {
        "policy": policy,
        "instance_id": instance["instance_id"],
        "fixture_family": instance["fixture_family"],
        "seed": seed,
        "block_variables": list(block_variables),
        "iterations": len(checks),
        "success": bool(accepted_checks),
        "accepted_assignment": dict(accepted_checks[0]["assignment"]) if accepted_checks else None,
        "exact_fallback_used": exact_decisions <= {"accepted", "rejected"},
        "candidate_checks": [dict(row) for row in checks],
    }


def run_policy_comparison(
    *,
    fixtures: Mapping[str, Any] | None = None,
    descriptors: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = SEEDS,
) -> JsonDict:
    """Compare exact-only, sparse repair, and random blocks over all seeds."""

    fixture_payload = build_scaleup_fixtures() if fixtures is None else fixtures
    descriptor_payload = (
        build_sparse_descriptors(fixture_payload) if descriptors is None else descriptors
    )
    validate_scaleup_fixture_payload(fixture_payload)
    instances = list(fixture_payload["instances"])
    descriptor_rows = list(descriptor_payload["sparse_repair_descriptors"])
    policy_results: dict[str, list[JsonDict]] = {
        "exact_only": [],
        "sparse_repair": [],
        "random_block": [],
    }

    exact_start = perf_counter()
    for seed in seeds:
        for instance in instances:
            policy_results["exact_only"].append(run_exact_only_attempt(instance, seed))
    exact_wall = perf_counter() - exact_start

    sparse_start = perf_counter()
    for seed in seeds:
        for instance, descriptor in zip(instances, descriptor_rows, strict=True):
            policy_results["sparse_repair"].append(
                run_sparse_repair_attempt(instance, descriptor, seed)
            )
    sparse_wall = perf_counter() - sparse_start

    random_start = perf_counter()
    for seed in seeds:
        for instance_index, instance in enumerate(instances):
            policy_results["random_block"].append(
                run_random_block_attempt(instance, seed, instance_index)
            )
    random_wall = perf_counter() - random_start

    attempts = [attempt for rows in policy_results.values() for attempt in rows]
    all_checks = [check for attempt in attempts for check in attempt["candidate_checks"]]
    unchecked_count = sum(
        int(check.get("exact_validator_decision") not in {"accepted", "rejected"})
        for check in all_checks
    )
    metrics: JsonDict = {
        "policy_results": policy_results,
        "n_instances": len(instances),
        "n_seeds": len(seeds),
        "candidate_count": len(all_checks),
        "all_candidates_exact_checked": unchecked_count == 0,
        "unchecked_candidate_count": unchecked_count,
        "exact_only_success_rate": success_rate(policy_results["exact_only"]),
        "sparse_repair_success_rate": success_rate(policy_results["sparse_repair"]),
        "random_block_success_rate": success_rate(policy_results["random_block"]),
        "mean_iterations_exact_only": mean_iterations(policy_results["exact_only"]),
        "mean_iterations_sparse_repair": mean_iterations(policy_results["sparse_repair"]),
        "mean_iterations_random_block": mean_iterations(policy_results["random_block"]),
        "exact_fallback_rate": success_rate(
            [{"success": attempt["exact_fallback_used"]} for attempt in attempts]
        ),
        "wall_time_observations_s": {
            "exact_only": round(exact_wall, 9),
            "sparse_repair": round(sparse_wall, 9),
            "random_block": round(random_wall, 9),
            "methodology": "single-process CPU wall observations, not matched timing evidence",
        },
    }
    metrics["confidence_intervals"] = confidence_intervals(metrics, policy_results, attempts)
    return metrics


def success_rate(attempts: Sequence[Mapping[str, Any]]) -> float:
    """Compute the exact-validated success rate for policy attempts."""

    return round(sum(int(row["success"]) for row in attempts) / len(attempts), 6)


def mean_iterations(attempts: Sequence[Mapping[str, Any]]) -> float:
    """Compute mean exact validator checks per policy attempt."""

    return round(sum(float(row["iterations"]) for row in attempts) / len(attempts), 6)


def confidence_intervals(
    metrics: Mapping[str, Any],
    policy_results: Mapping[str, Sequence[Mapping[str, Any]]],
    attempts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Compute cautious 95 percent intervals from deterministic attempt rows."""

    fallback_attempts = [{"success": attempt["exact_fallback_used"]} for attempt in attempts]
    return {
        "exact_only_success_rate": wilson_interval(
            policy_results["exact_only"], metrics["exact_only_success_rate"]
        ),
        "sparse_repair_success_rate": wilson_interval(
            policy_results["sparse_repair"], metrics["sparse_repair_success_rate"]
        ),
        "random_block_success_rate": wilson_interval(
            policy_results["random_block"], metrics["random_block_success_rate"]
        ),
        "mean_iterations_exact_only": mean_interval(
            [float(row["iterations"]) for row in policy_results["exact_only"]],
            metrics["mean_iterations_exact_only"],
        ),
        "mean_iterations_sparse_repair": mean_interval(
            [float(row["iterations"]) for row in policy_results["sparse_repair"]],
            metrics["mean_iterations_sparse_repair"],
        ),
        "exact_fallback_rate": wilson_interval(fallback_attempts, metrics["exact_fallback_rate"]),
    }


def wilson_interval(attempts: Sequence[Mapping[str, Any]], estimate: float) -> JsonDict:
    """Return a Wilson 95 percent interval for a binary attempt metric."""

    n = len(attempts)
    successes = sum(int(row["success"]) for row in attempts)
    z = 1.959963984540054
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)
    return {
        "method": "wilson_95",
        "n": n,
        "low": round(min(low, estimate), 6),
        "high": round(max(high, estimate), 6),
    }


def mean_interval(values: Sequence[float], estimate: float) -> JsonDict:
    """Return a normal-approximation 95 percent interval for iteration means."""

    n = len(values)
    mean = fmean(values)
    standard_error = pstdev(values) / math.sqrt(n)
    margin = 1.959963984540054 * standard_error
    low = max(0.0, mean - margin)
    high = mean + margin
    return {
        "method": "normal_approx_95",
        "n": n,
        "low": round(min(low, estimate), 6),
        "high": round(max(high, estimate), 6),
    }


def build_artifact(tests_run: Sequence[Mapping[str, Any]] = ()) -> JsonDict:
    """Build the terminal Exp5531 artifact without allowing a speedup headline."""

    fixtures = build_scaleup_fixtures()
    descriptors = build_sparse_descriptors(fixtures)
    comparison = run_policy_comparison(fixtures=fixtures, descriptors=descriptors)
    blockers = readiness_blockers(fixtures, descriptors, comparison)
    ready = not blockers
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "fixture_families": fixture_families(fixtures),
        "matched_timing_available": False,
        "speedup_claim_allowed": False,
        "active_constraint_sparse_repair_ready": ready,
        "tests_added_or_reused": list(TEST_PATHS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, blockers),
        "fixture_payload": fixtures,
        "descriptor_payload": descriptors,
        "fixture_payload_sha256": sha256_json(fixtures),
        "descriptor_payload_sha256": sha256_json(descriptors),
        "tests_run": [dict(row) for row in tests_run],
        "research_conductor_modified": False,
        "readiness_blockers": blockers,
        "claim_limits": [
            "exact-checked finite-domain active hard/soft fixtures only",
            "single-process wall observations are not matched timing evidence",
            "no solver, sampler, model, or hardware speedup claim is allowed",
        ],
    }
    artifact.update(comparison)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def readiness_blockers(
    fixtures: Mapping[str, Any],
    descriptors: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> list[str]:
    """Return precise blockers for the scale-up readiness gate."""

    blockers: list[str] = []
    if comparison.get("n_instances", 0) < 5:
        blockers.append("n_instances")
    if comparison.get("n_seeds", 0) < 2:
        blockers.append("n_seeds")
    if descriptors.get("descriptor_count") != len(fixtures.get("instances", [])):
        blockers.append("descriptor_count")
    if comparison.get("all_candidates_exact_checked") is not True:
        blockers.append("all_candidates_exact_checked")
    if comparison.get("unchecked_candidate_count") != 0:
        blockers.append("unchecked_candidate_count")
    if comparison.get("sparse_repair_success_rate") != 1.0:
        blockers.append("sparse_repair_success_rate")
    if comparison.get("exact_fallback_rate") != 1.0:
        blockers.append("exact_fallback_rate")
    ci = comparison.get("confidence_intervals")
    if not isinstance(ci, Mapping) or any(field not in ci for field in CI_FIELDS):
        blockers.append("confidence_intervals")
    return blockers


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal-prefix verdict that refuses speedup wording."""

    if ready:
        return "complete: exact_checked_sparse_repair_scaleup_ci_ready_no_speedup_claim"
    return "blocked: sparse_repair_scaleup_not_ready_" + "_".join(blockers)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the successful terminal artifact and fail closed on overclaims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("fixture_families") == fixture_families(artifact["fixture_payload"]), "fixture_families")
    _require(artifact.get("n_instances") == len(artifact["fixture_payload"]["instances"]), "n_instances")
    _require(artifact.get("n_seeds") == len(SEEDS), "n_seeds")
    _require(artifact.get("exact_only_success_rate") == 1.0, "exact_only_success_rate")
    _require(artifact.get("sparse_repair_success_rate") == 1.0, "sparse_repair_success_rate")
    _require(artifact.get("random_block_success_rate") < artifact.get("sparse_repair_success_rate"), "random_block_success_rate")
    _require(artifact.get("mean_iterations_exact_only") > artifact.get("mean_iterations_sparse_repair"), "mean_iterations")
    _require(artifact.get("exact_fallback_rate") == 1.0, "exact_fallback_rate")
    _require(artifact.get("matched_timing_available") is False, "matched_timing_available")
    _require(artifact.get("speedup_claim_allowed") is False, "speedup_claim_allowed")
    _require(artifact.get("active_constraint_sparse_repair_ready") is True, "active_constraint_sparse_repair_ready")
    _require(artifact.get("tests_added_or_reused") == list(TEST_PATHS), "tests_added_or_reused")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("all_candidates_exact_checked") is True, "all_candidates_exact_checked")
    _require(artifact.get("unchecked_candidate_count") == 0, "unchecked_candidate_count")
    _require_confidence_intervals(artifact)
    _require(artifact.get("fixture_payload_sha256") == sha256_json(artifact["fixture_payload"]), "fixture_payload_sha256")
    _require(artifact.get("descriptor_payload_sha256") == sha256_json(artifact["descriptor_payload"]), "descriptor_payload_sha256")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def _require_confidence_intervals(artifact: Mapping[str, Any]) -> None:
    ci = artifact.get("confidence_intervals")
    _require(isinstance(ci, Mapping), "confidence_intervals")
    for field in CI_FIELDS:
        row = ci.get(field)
        _require(isinstance(row, Mapping), "confidence_intervals")
        _require(bool(row.get("method")), "confidence_intervals")
        _require(row["low"] <= artifact[field] <= row["high"], "confidence_intervals")


def run(
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the terminal Exp5531 result JSON under ``repo_root``."""

    artifact = build_artifact(tests_run=tests_run)
    result_path = repo_root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


if __name__ == "__main__":  # pragma: no cover
    run()
