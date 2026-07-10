"""Exp5518 exact-checked sparse block repair descriptors.

Spec refs: REQ-VERIFY-5518, SCENARIO-VERIFY-5518.

This module is deliberately small and exact-checkable. It does not train a
diffusion model and does not replace the verifier with a learned shortcut.
Instead, it turns two Exp5499 hard/soft claim fixtures into sparse repair
descriptors, proposes repairs over a violated variable block, and lets the
exact finite-domain validator accept or reject every candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Any

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as fixture_mod
from carnot import experiment_5505_active_constraint_milp_descriptor_v499 as descriptor_mod


JsonDict = dict[str, Any]
Assignment = dict[str, str]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5518_block_gibbs_sparse_repair_descriptors.json")
PAYLOAD_DIR = Path("results/block_gibbs_sparse_repair_5518")
DESCRIPTOR_RELATIVE_PATH = PAYLOAD_DIR / "descriptors.json"
FIXTURE_RELATIVE_PATH = PAYLOAD_DIR / "fixtures.json"
EXP5499_FIXTURE_PATH = fixture_mod.FIXTURE_RELATIVE_PATH

SCHEMA = "carnot.experiment_5518.block_gibbs_sparse_repair_descriptors.v500"
FIXTURE_SCHEMA = "carnot.fixture.block_gibbs_sparse_repair_5518.v1"
DESCRIPTOR_SCHEMA = "carnot.descriptor.sparse_block_repair.v1"
EXPERIMENT = 5518
EXPERIMENT_ID = "exp5518-block-gibbs-sparse-repair-descriptors"
MILESTONE = "2026.07.500"
RUN_DATE = "2026-07-10"
INFERENCE_SUBSTRATE = "exact_checked_sparse_repair"
SPARSE_BLOCK_POLICY = "violated_hard_constraint_variables_then_exact_validate"
EXACT_VALIDATOR = "exp5499_finite_domain_hard_soft_exact_validator"
SEEDS = (5518, 5519, 5520, 5521, 5522)
SPEC_REFS = ("REQ-VERIFY-5518", "SCENARIO-VERIFY-5518", "REQ-VERIFY-5499")
TEST_PATHS = ("tests/python/test_experiment_5518_block_gibbs_sparse_repair_descriptors.py",)
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "descriptor_path": "points to the executable sparse block descriptor payload.",
    "fixture_paths": "names the exact-checkable hard/soft fixtures used by every policy.",
    "exact_fallback_used": "keeps exact validators as the acceptance authority.",
    "sparse_block_policy": "names the non-trained block-selection rule after violations.",
    "seeds": "records deterministic candidate-generation seeds.",
    "exact_only_success_rate": "full exact fallback acceptance reference.",
    "sparse_repair_success_rate": "descriptor-guided repair acceptance under exact checking.",
    "random_block_success_rate": "same-size random block control under exact checking.",
    "mean_iterations_exact_only": "iteration evidence for the exact-only baseline.",
    "mean_iterations_sparse_repair": "iteration evidence for sparse repair; not a speedup claim.",
    "speedup_claim_allowed": "must remain false without matched timing evidence.",
    "active_constraint_sparse_repair_ready": "interface readiness gate, not a performance headline.",
    "inference_substrate": "declares exact-checked sparse repair, no model training or live inference.",
    "honest_verdict": "terminal status; start with complete: or blocked: and do not claim speedup.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so descriptor hashes are reviewable."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(payload: Any) -> str:
    """Hash a JSON-compatible payload after stable serialization."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_selected_fixtures() -> JsonDict:
    """Select two small Exp5499 hard/soft fixtures with deliberate violations."""

    source_fixture = fixture_mod.build_fixture()
    fixture_mod.validate_fixture(source_fixture)
    by_id = {str(row["instance_id"]): row for row in source_fixture["instances"]}
    initial_assignments = {
        "claim_support_preference": {
            "support": "unsupported",
            "source_quality": "primary",
            "scope": "overbroad",
        },
        "claim_safety_conflict": {
            "safety": "unsafe",
            "citation": "missing",
            "action": "accept",
        },
    }
    instances: list[JsonDict] = []
    for instance_id, initial in initial_assignments.items():
        source = by_id[instance_id]
        reference = fixture_mod.solve_reference(source)
        instance = {
            "schema": fixture_mod.FIXTURE_SCHEMA,
            "instance_id": instance_id,
            "source_fixture_path": EXP5499_FIXTURE_PATH.as_posix(),
            "typed_claims": [dict(row) for row in source["typed_claims"]],
            "domains": {str(row["name"]): list(row["domain"]) for row in source["typed_claims"]},
            "hard_constraints": [dict(row) for row in source["hard_constraints"]],
            "soft_preferences": [dict(row) for row in source["soft_preferences"]],
            "initial_assignment": dict(initial),
            "exact_reference": dict(reference),
        }
        instance["violated_hard_constraints"] = violated_hard_constraint_ids(instance)
        instance["expected_repair_block_variables"] = derive_repair_block(instance)
        instances.append(instance)
    payload = {
        "schema": FIXTURE_SCHEMA,
        "source_fixture_path": EXP5499_FIXTURE_PATH.as_posix(),
        "fixture_selection": "two optimal Exp5499 rows with two hard-constraint violations each",
        "exact_validator": EXACT_VALIDATOR,
        "instances": instances,
    }
    validate_fixture_payload(payload)
    return payload


def validate_fixture_payload(payload: Mapping[str, Any]) -> None:
    """Validate the local fixture subset before repair policies use it."""

    _require(payload.get("schema") == FIXTURE_SCHEMA, "fixture_schema")
    instances = payload.get("instances")
    _require(isinstance(instances, list) and len(instances) == 2, "instances")
    for instance in instances:
        _require(instance.get("exact_reference", {}).get("status") == "optimal", "exact_reference")
        _require(bool(instance.get("violated_hard_constraints")), "violated_hard_constraints")
        _require(bool(instance.get("expected_repair_block_variables")), "repair_block_variables")


def violated_hard_constraint_ids(instance: Mapping[str, Any]) -> list[str]:
    """List hard rows violated by the fixture's starting assignment."""

    initial = instance["initial_assignment"]
    return [
        str(row["id"])
        for row in instance["hard_constraints"]
        if not descriptor_mod.constraint_satisfied(initial, row)
    ]


def derive_repair_block(instance: Mapping[str, Any]) -> list[str]:
    """Derive the sparse block from violated hard-constraint variables."""

    initial = instance["initial_assignment"]
    variables: list[str] = []
    for constraint in instance["hard_constraints"]:
        if not descriptor_mod.constraint_satisfied(initial, constraint):
            for variable in clause_variables(constraint):
                if variable not in variables:
                    variables.append(variable)
    return variables


def clause_variables(constraint: Mapping[str, Any]) -> list[str]:
    """Extract variables from the Exp5499 clause shape used by this prototype."""

    return [str(literal["variable"]) for literal in constraint["literals"]]


def build_sparse_descriptors(fixtures: Mapping[str, Any]) -> JsonDict:
    """Build one sparse repair descriptor for each selected fixture."""

    validate_fixture_payload(fixtures)
    rows: list[JsonDict] = []
    for instance in fixtures["instances"]:
        variables = [str(row["name"]) for row in instance["typed_claims"]]
        repair_block = derive_repair_block(instance)
        descriptor = {
            "schema": DESCRIPTOR_SCHEMA,
            "descriptor_id": f"sparse-repair:{instance['instance_id']}",
            "source_fixture_path": fixtures["source_fixture_path"],
            "source_instance_id": instance["instance_id"],
            "sparse_block_policy": SPARSE_BLOCK_POLICY,
            "variables": variables,
            "variable_count": len(variables),
            "repair_block_variables": repair_block,
            "block_size": len(repair_block),
            "sparse_subset": 0 < len(repair_block) < len(variables),
            "violated_hard_constraints": violated_hard_constraint_ids(instance),
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
        "schema": "carnot.descriptor_bundle.sparse_block_repair_5518.v1",
        "sparse_block_policy": SPARSE_BLOCK_POLICY,
        "descriptor_count": len(rows),
        "sparse_repair_descriptors": rows,
    }


def validate_sparse_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Reject descriptor rows that are not sparse or exact-fallback checked."""

    variables = descriptor.get("variables")
    block = descriptor.get("repair_block_variables")
    fallback = descriptor.get("exact_fallback")
    _require(descriptor.get("schema") == DESCRIPTOR_SCHEMA, "descriptor_schema")
    _require(descriptor.get("sparse_block_policy") == SPARSE_BLOCK_POLICY, "sparse_block_policy")
    _require(isinstance(variables, list) and bool(variables), "variables")
    _require(isinstance(block, list) and bool(block), "repair_block_variables")
    _require(
        descriptor.get("sparse_subset") is True and len(block) < len(variables), "sparse_subset"
    )
    _require(isinstance(fallback, Mapping) and fallback.get("required") is True, "exact_fallback")
    _require(fallback.get("validator") == EXACT_VALIDATOR, "exact_fallback")
    _require(fallback.get("status") == "optimal", "exact_fallback")


def validate_candidate_assignment(
    instance: Mapping[str, Any],
    assignment: Mapping[str, str],
) -> JsonDict:
    """Accept only the exact optimum; reject hard failures and soft suboptimal rows."""

    hard_pass = descriptor_mod.constraints_satisfied(assignment, instance["hard_constraints"])
    soft_score = sum(
        descriptor_mod.preference_score(assignment, preference)
        for preference in instance["soft_preferences"]
    )
    reference = instance["exact_reference"]
    exact_assignment = dict(reference["assignment"])
    accepted = hard_pass and dict(assignment) == exact_assignment
    reject_reason = "accepted" if accepted else "not_exact_optimum"
    if not hard_pass:
        reject_reason = "hard_constraints_failed"
    return {
        "assignment": dict(assignment),
        "assignment_hash": sha256_json(dict(assignment)),
        "hard_constraints_pass": hard_pass,
        "soft_score": float(soft_score),
        "exact_reference_score": reference["objective_score"],
        "accepted": accepted,
        "exact_validator_decision": "accepted" if accepted else "rejected",
        "reject_reason": reject_reason,
    }


def enumerate_assignments(
    instance: Mapping[str, Any],
    variables: Sequence[str],
    base_assignment: Mapping[str, str] | None = None,
) -> list[Assignment]:
    """Enumerate finite-domain assignments over the requested variable block."""

    base = {} if base_assignment is None else dict(base_assignment)
    domains = instance["domains"]
    assignments: list[Assignment] = []
    for values in itertools.product(*(domains[name] for name in variables)):
        candidate = dict(base)
        candidate.update(dict(zip(variables, values, strict=True)))
        assignments.append(candidate)
    return assignments


def run_exact_only_attempt(instance: Mapping[str, Any], seed: int) -> JsonDict:
    """Run the full finite-domain exact baseline and count checked assignments."""

    variables = [str(row["name"]) for row in instance["typed_claims"]]
    checks: list[JsonDict] = []
    for assignment in enumerate_assignments(instance, variables):
        check = validate_candidate_assignment(instance, assignment)
        checks.append(check)
        if check["accepted"]:
            break
    return attempt_record("exact_only", instance, seed, variables, checks)


def run_sparse_repair_attempt(
    instance: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    seed: int,
) -> JsonDict:
    """Run descriptor-guided block repair from the violated starting assignment."""

    block = list(descriptor["repair_block_variables"])
    checks = [
        validate_candidate_assignment(instance, assignment)
        for assignment in enumerate_assignments(instance, block, instance["initial_assignment"])
    ]
    return attempt_record("sparse_repair", instance, seed, block, checks)


def run_random_block_attempt(
    instance: Mapping[str, Any], seed: int, instance_index: int
) -> JsonDict:
    """Run the same-size deterministic random block repair control."""

    variables = [str(row["name"]) for row in instance["typed_claims"]]
    block_size = len(derive_repair_block(instance))
    block = random.Random(seed + instance_index * 1009).sample(variables, block_size)
    checks = [
        validate_candidate_assignment(instance, assignment)
        for assignment in enumerate_assignments(instance, block, instance["initial_assignment"])
    ]
    return attempt_record("random_block", instance, seed, block, checks)


def attempt_record(
    policy: str,
    instance: Mapping[str, Any],
    seed: int,
    block_variables: Sequence[str],
    checks: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a compact policy attempt row from exact validator decisions."""

    accepted_checks = [row for row in checks if row["accepted"]]
    return {
        "policy": policy,
        "instance_id": instance["instance_id"],
        "seed": seed,
        "block_variables": list(block_variables),
        "iterations": len(checks),
        "success": bool(accepted_checks),
        "accepted_assignment": dict(accepted_checks[0]["assignment"]) if accepted_checks else None,
        "candidate_checks": [dict(row) for row in checks],
    }


def run_policy_comparison(
    *,
    fixtures: Mapping[str, Any] | None = None,
    descriptors: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = SEEDS,
) -> JsonDict:
    """Compare exact-only, sparse repair, and deterministic random-block policies."""

    fixture_payload = build_selected_fixtures() if fixtures is None else fixtures
    descriptor_payload = (
        build_sparse_descriptors(fixture_payload) if descriptors is None else descriptors
    )
    instances = list(fixture_payload["instances"])
    descriptor_rows = list(descriptor_payload["sparse_repair_descriptors"])
    policy_results = {"exact_only": [], "sparse_repair": [], "random_block": []}

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

    all_checks = [
        check
        for attempts in policy_results.values()
        for attempt in attempts
        for check in attempt["candidate_checks"]
    ]
    unchecked_count = sum(
        int(check.get("exact_validator_decision") not in {"accepted", "rejected"})
        for check in all_checks
    )
    return {
        "policy_results": policy_results,
        "candidate_count": len(all_checks),
        "all_candidates_exact_checked": unchecked_count == 0,
        "unchecked_candidate_count": unchecked_count,
        "exact_only_success_rate": success_rate(policy_results["exact_only"]),
        "sparse_repair_success_rate": success_rate(policy_results["sparse_repair"]),
        "random_block_success_rate": success_rate(policy_results["random_block"]),
        "mean_iterations_exact_only": mean_iterations(policy_results["exact_only"]),
        "mean_iterations_sparse_repair": mean_iterations(policy_results["sparse_repair"]),
        "mean_iterations_random_block": mean_iterations(policy_results["random_block"]),
        "wall_time_exact_only_s": round(exact_wall, 9),
        "wall_time_sparse_repair_s": round(sparse_wall, 9),
        "wall_time_random_block_s": round(random_wall, 9),
    }


def success_rate(attempts: Sequence[Mapping[str, Any]]) -> float:
    """Compute policy acceptance rate from exact-validated attempt rows."""

    return round(sum(int(row["success"]) for row in attempts) / len(attempts), 6)


def mean_iterations(attempts: Sequence[Mapping[str, Any]]) -> float:
    """Compute mean candidate checks per attempt."""

    return round(sum(float(row["iterations"]) for row in attempts) / len(attempts), 6)


def build_artifact(tests_run: Sequence[Mapping[str, Any]] = ()) -> JsonDict:
    """Build the terminal Exp5518 result artifact."""

    fixtures = build_selected_fixtures()
    descriptors = build_sparse_descriptors(fixtures)
    comparison = run_policy_comparison(fixtures=fixtures, descriptors=descriptors)
    blockers = readiness_blockers(descriptors, comparison)
    ready = not blockers
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "descriptor_path": DESCRIPTOR_RELATIVE_PATH.as_posix(),
        "fixture_paths": [FIXTURE_RELATIVE_PATH.as_posix(), EXP5499_FIXTURE_PATH.as_posix()],
        "exact_fallback_used": True,
        "sparse_block_policy": SPARSE_BLOCK_POLICY,
        "seeds": list(SEEDS),
        "speedup_claim_allowed": False,
        "active_constraint_sparse_repair_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, blockers),
        "field_principles": dict(FIELD_PRINCIPLES),
        "descriptor_payload": descriptors,
        "fixture_payload": fixtures,
        "descriptor_payload_sha256": sha256_json(descriptors),
        "fixture_payload_sha256": sha256_json(fixtures),
        "claim_limits": [
            "small CPU-local hard/soft fixtures only",
            "exact fallback accepts or rejects every candidate",
            "random block is a deterministic same-size control",
            "wall-time observations are recorded but no speedup claim is allowed",
        ],
        "research_conductor_modified": False,
        "test_paths": list(TEST_PATHS),
        "tests_run": [dict(row) for row in tests_run],
        "readiness_blockers": blockers,
    }
    artifact.update(comparison)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def readiness_blockers(
    descriptors: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> list[str]:
    """Return precise blockers for the sparse repair readiness gate."""

    blockers: list[str] = []
    if descriptors.get("descriptor_count") != 2:
        blockers.append("descriptor_count")
    if comparison.get("unchecked_candidate_count") != 0:
        blockers.append("unchecked_candidate_count")
    if comparison.get("sparse_repair_success_rate") != 1.0:
        blockers.append("sparse_repair_success_rate")
    if comparison.get("all_candidates_exact_checked") is not True:
        blockers.append("all_candidates_exact_checked")
    return blockers


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal-prefix verdict without turning timing into speedup."""

    if ready:
        return "complete: exact_checked_sparse_repair_descriptor_interface_ready_no_speedup_claim"
    return "blocked: sparse_repair_not_ready_" + "_".join(blockers)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the successful terminal artifact and reject overclaims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        artifact.get("descriptor_path") == DESCRIPTOR_RELATIVE_PATH.as_posix(), "descriptor_path"
    )
    _require(
        artifact.get("fixture_paths")
        == [FIXTURE_RELATIVE_PATH.as_posix(), EXP5499_FIXTURE_PATH.as_posix()],
        "fixture_paths",
    )
    _require(artifact.get("exact_fallback_used") is True, "exact_fallback_used")
    _require(artifact.get("sparse_block_policy") == SPARSE_BLOCK_POLICY, "sparse_block_policy")
    _require(artifact.get("seeds") == list(SEEDS), "seeds")
    _require(artifact.get("exact_only_success_rate") == 1.0, "exact_only_success_rate")
    _require(artifact.get("sparse_repair_success_rate") == 1.0, "sparse_repair_success_rate")
    _require(artifact.get("random_block_success_rate") < 1.0, "random_block_success_rate")
    _require(
        artifact.get("mean_iterations_exact_only") > artifact.get("mean_iterations_sparse_repair"),
        "mean_iterations",
    )
    _require(artifact.get("speedup_claim_allowed") is False, "speedup_claim_allowed")
    _require(
        artifact.get("active_constraint_sparse_repair_ready") is True,
        "active_constraint_sparse_repair_ready",
    )
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(
        str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict"
    )
    _require(artifact.get("unchecked_candidate_count") == 0, "unchecked_candidate_count")
    _require(artifact.get("all_candidates_exact_checked") is True, "all_candidates_exact_checked")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def run(
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write descriptor, fixture, and terminal result payloads under ``repo_root``."""

    artifact = build_artifact(tests_run=tests_run)
    descriptor_path = repo_root / DESCRIPTOR_RELATIVE_PATH
    fixture_path = repo_root / FIXTURE_RELATIVE_PATH
    result_path = repo_root / RESULT_RELATIVE_PATH
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_text(
        json.dumps(artifact["descriptor_payload"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fixture_path.write_text(
        json.dumps(artifact["fixture_payload"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _require(condition: bool, field: str) -> None:
    """Raise a compact validation error for the field that drifted."""

    if not condition:
        raise ValueError(field)
