"""Exp5499 Preference-MaxSAT typed claim-state fixture.

Spec refs: REQ-VERIFY-5499, SCENARIO-VERIFY-5499.

This module keeps the verification problem intentionally small: each claim is
represented by a finite typed domain, hard rows decide feasibility, and soft
rows only rank assignments after every hard row passes. The exact enumerator is
the final authority because it ignores cached candidate labels and recomputes
the optimum from the fixture domains.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
Assignment = dict[str, str]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5499_preference_maxsat_minimal_fixture_v499.json")
FIXTURE_RELATIVE_PATH = Path("results/preference_maxsat_minimal_fixture_5499/fixture.json")
REFERENCE_SOLVER_PATH = (
    "python/carnot/experiment_5499_preference_maxsat_minimal_fixture_v499.py::solve_reference"
)
TEST_PATHS = ("tests/python/test_experiment_5499_preference_maxsat_minimal_fixture_v499.py",)

SCHEMA = "carnot.experiment_5499.preference_maxsat_minimal_fixture.v499"
FIXTURE_SCHEMA = "carnot.fixture.preference_maxsat_minimal_typed_claim_state.v1"
EXPERIMENT = 5499
EXPERIMENT_ID = "exp5499-preference-maxsat-minimal-fixture-v499"
MILESTONE = "2026.07.499"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5499
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ("REQ-VERIFY-5499", "SCENARIO-VERIFY-5499")

REQUIRED_ARTIFACT_FIELDS = (
    "fixture_path",
    "reference_solver_path",
    "test_paths",
    "num_instances",
    "hard_constraint_pass_rate",
    "preference_optimality_rate",
    "independent_reference_agreement_rate",
    "false_accept_rate",
    "preference_maxsat_fixture_ready",
    "guided_decoding_used",
    "inference_substrate",
    "honest_verdict",
)


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize mappings deterministically so hashes reflect semantic drift."""

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Hash a mapping after stable JSON serialization."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_fixture() -> JsonDict:
    """Return the deterministic typed claim-state Preference-MaxSAT fixture."""

    return {
        "schema": FIXTURE_SCHEMA,
        "fixture_id": "preference_maxsat_minimal_typed_claim_state_5499",
        "guided_decoding_used": False,
        "token_steering_used": False,
        "instances": [
            {
                "instance_id": "claim_support_preference",
                "expected_status": "optimal",
                "typed_claims": [
                    {
                        "name": "support",
                        "claim_type": "evidence_support",
                        "domain": ["unsupported", "entailed"],
                    },
                    {
                        "name": "source_quality",
                        "claim_type": "provenance_quality",
                        "domain": ["secondary", "primary"],
                    },
                    {
                        "name": "scope",
                        "claim_type": "claim_scope",
                        "domain": ["overbroad", "bounded"],
                    },
                ],
                "hard_constraints": [
                    {
                        "id": "HC_SUPPORT_ENTAILED",
                        "type": "clause",
                        "literals": [{"variable": "support", "equals": "entailed"}],
                    },
                    {
                        "id": "HC_SCOPE_BOUNDED",
                        "type": "clause",
                        "literals": [{"variable": "scope", "equals": "bounded"}],
                    },
                ],
                "soft_preferences": [
                    {
                        "id": "SP_PRIMARY_SOURCE",
                        "type": "value_reward",
                        "variable": "source_quality",
                        "value": "primary",
                        "weight": 7,
                    },
                    {
                        "id": "SP_ENTAILED_SUPPORT",
                        "type": "value_reward",
                        "variable": "support",
                        "value": "entailed",
                        "weight": 3,
                    },
                    {
                        "id": "SP_BOUNDED_SCOPE",
                        "type": "value_reward",
                        "variable": "scope",
                        "value": "bounded",
                        "weight": 1,
                    },
                ],
                "candidates": [
                    {
                        "candidate_id": "support_exact_optimum",
                        "assignment": {
                            "support": "entailed",
                            "source_quality": "primary",
                            "scope": "bounded",
                        },
                        "accept": True,
                    },
                    {
                        "candidate_id": "support_hard_violation",
                        "assignment": {
                            "support": "unsupported",
                            "source_quality": "primary",
                            "scope": "bounded",
                        },
                        "accept": False,
                    },
                    {
                        "candidate_id": "support_soft_suboptimal",
                        "assignment": {
                            "support": "entailed",
                            "source_quality": "secondary",
                            "scope": "bounded",
                        },
                        "accept": False,
                    },
                ],
            },
            {
                "instance_id": "claim_safety_conflict",
                "expected_status": "optimal",
                "typed_claims": [
                    {
                        "name": "safety",
                        "claim_type": "safety_label",
                        "domain": ["unsafe", "safe"],
                    },
                    {
                        "name": "citation",
                        "claim_type": "citation_state",
                        "domain": ["missing", "present"],
                    },
                    {
                        "name": "action",
                        "claim_type": "validator_action",
                        "domain": ["reject", "accept"],
                    },
                ],
                "hard_constraints": [
                    {
                        "id": "HC_SAFETY_SAFE",
                        "type": "clause",
                        "literals": [{"variable": "safety", "equals": "safe"}],
                    },
                    {
                        "id": "HC_CITATION_PRESENT",
                        "type": "clause",
                        "literals": [{"variable": "citation", "equals": "present"}],
                    },
                ],
                "soft_preferences": [
                    {
                        "id": "SP_ACCEPT_WHEN_SAFE",
                        "type": "value_reward",
                        "variable": "action",
                        "value": "accept",
                        "weight": 5,
                    },
                    {
                        "id": "SP_SAFE_LABEL",
                        "type": "value_reward",
                        "variable": "safety",
                        "value": "safe",
                        "weight": 4,
                    },
                    {
                        "id": "SP_CITATION_PRESENT",
                        "type": "value_reward",
                        "variable": "citation",
                        "value": "present",
                        "weight": 3,
                    },
                ],
                "candidates": [
                    {
                        "candidate_id": "safety_exact_optimum",
                        "assignment": {
                            "safety": "safe",
                            "citation": "present",
                            "action": "accept",
                        },
                        "accept": True,
                    },
                    {
                        "candidate_id": "safety_hard_violation_high_soft",
                        "assignment": {
                            "safety": "unsafe",
                            "citation": "present",
                            "action": "accept",
                        },
                        "accept": False,
                    },
                    {
                        "candidate_id": "safety_soft_suboptimal",
                        "assignment": {
                            "safety": "safe",
                            "citation": "present",
                            "action": "reject",
                        },
                        "accept": False,
                    },
                ],
            },
            {
                "instance_id": "claim_infeasible_negative_control",
                "expected_status": "infeasible",
                "typed_claims": [
                    {
                        "name": "verdict",
                        "claim_type": "validator_verdict",
                        "domain": ["reject", "accept"],
                    },
                    {
                        "name": "evidence",
                        "claim_type": "evidence_state",
                        "domain": ["absent", "present"],
                    },
                ],
                "hard_constraints": [
                    {
                        "id": "HC_VERDICT_ACCEPT",
                        "type": "clause",
                        "literals": [{"variable": "verdict", "equals": "accept"}],
                    },
                    {
                        "id": "HC_VERDICT_REJECT",
                        "type": "clause",
                        "literals": [{"variable": "verdict", "equals": "reject"}],
                    },
                ],
                "soft_preferences": [
                    {
                        "id": "SP_ACCEPT_WRONG_IF_HARD_IGNORED",
                        "type": "value_reward",
                        "variable": "verdict",
                        "value": "accept",
                        "weight": 10,
                    },
                    {
                        "id": "SP_EVIDENCE_PRESENT",
                        "type": "value_reward",
                        "variable": "evidence",
                        "value": "present",
                        "weight": 2,
                    },
                ],
                "candidates": [
                    {
                        "candidate_id": "infeasible_false_accept_probe",
                        "assignment": {"verdict": "accept", "evidence": "present"},
                        "accept": False,
                    }
                ],
            },
        ],
    }


def validate_fixture(fixture: Mapping[str, Any]) -> None:
    """Validate the fixture shape before exact solving."""

    _require(fixture.get("schema") == FIXTURE_SCHEMA, "fixture_schema")
    _require(fixture.get("guided_decoding_used") is False, "guided_decoding_used")
    _require(fixture.get("token_steering_used") is False, "token_steering_used")
    instances = fixture.get("instances")
    _require(isinstance(instances, list) and bool(instances), "instances")
    negative_seen = False
    for instance in instances:
        validate_instance(instance)
        negative_seen = negative_seen or instance.get("expected_status") == "infeasible"
    _require(negative_seen, "negative_control")


def validate_instance(instance: Mapping[str, Any]) -> None:
    """Validate one typed claim-state instance."""

    _require(bool(instance.get("instance_id")), "instance_id")
    _require(instance.get("expected_status") in {"optimal", "infeasible"}, "expected_status")
    domains = domains_from_instance(instance)
    _require(bool(domains), "typed_claims")
    _require(bool(instance.get("hard_constraints")), "hard_constraints")
    _require(bool(instance.get("soft_preferences")), "soft_preferences")
    _require(bool(instance.get("candidates")), "candidates")
    for constraint in instance["hard_constraints"]:
        _require(constraint.get("type") == "clause", "constraint_type")
        _require(bool(constraint.get("literals")), "constraint_literals")
        for literal in constraint["literals"]:
            _require(literal.get("variable") in domains, "constraint_variable")
            _require(literal.get("equals") in domains[literal["variable"]], "constraint_value")
    for preference in instance["soft_preferences"]:
        _require(preference.get("type") == "value_reward", "preference_type")
        _require(preference.get("variable") in domains, "preference_variable")
        _require(preference.get("value") in domains[preference["variable"]], "preference_value")
        _require(float(preference.get("weight", 0.0)) > 0.0, "preference_weight")
    for candidate in instance["candidates"]:
        assignment = candidate.get("assignment")
        _require(isinstance(assignment, dict), "candidate_assignment")
        _require(set(assignment) == set(domains), "candidate_assignment_keys")
        _require(all(value in domains[name] for name, value in assignment.items()), "candidate_domain")
        _require(isinstance(candidate.get("accept"), bool), "candidate_accept")


def domains_from_instance(instance: Mapping[str, Any]) -> dict[str, list[str]]:
    """Extract finite typed domains from one fixture instance."""

    domains: dict[str, list[str]] = {}
    for claim in instance.get("typed_claims", []):
        name = str(claim.get("name", ""))
        domain = list(claim.get("domain", []))
        _require(bool(name) and bool(domain), "typed_claim")
        domains[name] = [str(value) for value in domain]
    return domains


def solve_reference(instance: Mapping[str, Any]) -> JsonDict:
    """Enumerate all typed states and return the exact hard/soft optimum."""

    validate_instance(instance)
    domains = domains_from_instance(instance)
    names = list(domains)
    feasible_rows: list[JsonDict] = []
    for values in itertools.product(*(domains[name] for name in names)):
        assignment = dict(zip(names, values, strict=True))
        if hard_constraints_pass(instance, assignment):
            feasible_rows.append(
                {
                    "assignment": assignment,
                    "objective_score": soft_score(instance, assignment),
                    "assignment_hash": assignment_hash(assignment),
                }
            )
    if not feasible_rows:
        return {
            "status": "infeasible",
            "assignment": None,
            "objective_score": None,
            "assignment_hash": None,
            "feasible_assignment_count": 0,
        }
    feasible_rows.sort(key=lambda row: (-float(row["objective_score"]), canonical_json(row["assignment"])))
    best = feasible_rows[0]
    return {
        "status": "optimal",
        "assignment": best["assignment"],
        "objective_score": best["objective_score"],
        "assignment_hash": best["assignment_hash"],
        "feasible_assignment_count": len(feasible_rows),
    }


def hard_constraints_pass(instance: Mapping[str, Any], assignment: Mapping[str, str]) -> bool:
    """Return true only when every hard Preference-MaxSAT clause is satisfied."""

    return all(
        any(assignment[literal["variable"]] == literal["equals"] for literal in constraint["literals"])
        for constraint in instance["hard_constraints"]
    )


def soft_score(instance: Mapping[str, Any], assignment: Mapping[str, str]) -> float:
    """Score soft preferences after the caller has handled hard feasibility."""

    return round(
        sum(
            float(preference["weight"])
            for preference in instance["soft_preferences"]
            if assignment[preference["variable"]] == preference["value"]
        ),
        6,
    )


def assignment_hash(assignment: Mapping[str, str]) -> str:
    """Hash a claim-state assignment independently from candidate metadata."""

    return sha256_json({"assignment": dict(assignment)})


def evaluate_fixture(fixture: Mapping[str, Any]) -> JsonDict:
    """Evaluate cached candidates against exact references and compute rates."""

    validate_fixture(fixture)
    instance_reports: list[JsonDict] = []
    accepted_evaluations: list[JsonDict] = []
    accepted_candidate_ids: list[str] = []
    rejected_candidate_ids: list[str] = []
    for instance in fixture["instances"]:
        reference = solve_reference(instance)
        candidate_reports = []
        accepted_for_instance = []
        for candidate in instance["candidates"]:
            accepted = bool(candidate["accept"])
            assignment = candidate["assignment"]
            hard_ok = hard_constraints_pass(instance, assignment)
            score = soft_score(instance, assignment)
            optimal = (
                reference["status"] == "optimal"
                and hard_ok
                and score == reference["objective_score"]
            )
            agrees = (
                optimal
                and assignment_hash(assignment) == reference["assignment_hash"]
                and assignment == reference["assignment"]
            )
            false_accept = accepted and (reference["status"] != "optimal" or not hard_ok or not agrees)
            row = {
                "candidate_id": candidate["candidate_id"],
                "accepted": accepted,
                "hard_constraints_pass": hard_ok,
                "soft_score": score,
                "soft_optimal": optimal,
                "reference_agreement": agrees,
                "false_accept": false_accept,
            }
            candidate_reports.append(row)
            if accepted:
                accepted_evaluations.append(row)
                accepted_for_instance.append(row)
                accepted_candidate_ids.append(str(candidate["candidate_id"]))
            else:
                rejected_candidate_ids.append(str(candidate["candidate_id"]))
        instance_agrees = (
            reference["status"] == "infeasible"
            and not accepted_for_instance
            or len(accepted_for_instance) == 1
            and accepted_for_instance[0]["reference_agreement"]
        )
        instance_reports.append(
            {
                "instance_id": instance["instance_id"],
                "expected_status": instance["expected_status"],
                "reference": reference,
                "reference_agreement": bool(instance_agrees),
                "candidates": candidate_reports,
            }
        )
    hard_rate = _rate(
        sum(int(row["hard_constraints_pass"]) for row in accepted_evaluations),
        len(accepted_evaluations),
    )
    hard_accepted = [row for row in accepted_evaluations if row["hard_constraints_pass"]]
    optimality_rate = _rate(sum(int(row["soft_optimal"]) for row in hard_accepted), len(hard_accepted))
    agreement_rate = _rate(
        sum(int(row["reference_agreement"]) for row in instance_reports),
        len(instance_reports),
    )
    false_accept_rate = _rate(
        sum(int(row["false_accept"]) for row in accepted_evaluations),
        len(accepted_evaluations),
    )
    report = {
        "num_instances": len(instance_reports),
        "hard_constraint_pass_rate": hard_rate,
        "preference_optimality_rate": optimality_rate,
        "independent_reference_agreement_rate": agreement_rate,
        "false_accept_rate": false_accept_rate,
        "accepted_candidate_ids": accepted_candidate_ids,
        "rejected_candidate_ids": rejected_candidate_ids,
        "instance_reports": instance_reports,
        "readiness_blockers": readiness_blockers(
            hard_rate=hard_rate,
            optimality_rate=optimality_rate,
            agreement_rate=agreement_rate,
            false_accept_rate=false_accept_rate,
            fixture=fixture,
        ),
    }
    report["preference_maxsat_fixture_ready"] = not report["readiness_blockers"]
    return report


def readiness_blockers(
    *,
    hard_rate: float,
    optimality_rate: float,
    agreement_rate: float,
    false_accept_rate: float,
    fixture: Mapping[str, Any],
) -> list[str]:
    """Return precise blockers for the fixture-ready gate."""

    checks = (
        (bool(fixture.get("instances")), "fixture_missing"),
        (REFERENCE_SOLVER_PATH.endswith("::solve_reference"), "reference_solver_missing"),
        (hard_rate == 1.0, "hard_constraint_pass_rate_below_1"),
        (optimality_rate == 1.0, "preference_optimality_rate_below_1"),
        (agreement_rate == 1.0, "independent_reference_agreement_rate_below_1"),
        (false_accept_rate == 0.0, "false_accept_rate_nonzero"),
        (fixture.get("guided_decoding_used") is False, "guided_decoding_used"),
        (fixture.get("token_steering_used") is False, "token_steering_used"),
    )
    return [name for passed, name in checks if not passed]


def build_artifact(
    *,
    fixture: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5499 artifact from the exact fixture report."""

    fixture_payload = build_fixture() if fixture is None else dict(fixture)
    report = evaluate_fixture(fixture_payload)
    ready = bool(report["preference_maxsat_fixture_ready"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "fixture_path": FIXTURE_RELATIVE_PATH.as_posix(),
        "reference_solver_path": REFERENCE_SOLVER_PATH,
        "test_paths": list(TEST_PATHS),
        "num_instances": report["num_instances"],
        "hard_constraint_pass_rate": report["hard_constraint_pass_rate"],
        "preference_optimality_rate": report["preference_optimality_rate"],
        "independent_reference_agreement_rate": report["independent_reference_agreement_rate"],
        "false_accept_rate": report["false_accept_rate"],
        "preference_maxsat_fixture_ready": ready,
        "guided_decoding_used": False,
        "token_steering_used": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, report["readiness_blockers"]),
        "fixture": fixture_payload,
        "reference_rows": report["instance_reports"],
        "accepted_candidate_ids": report["accepted_candidate_ids"],
        "rejected_candidate_ids": report["rejected_candidate_ids"],
        "readiness_blockers": report["readiness_blockers"],
        "tests_run": [dict(item) for item in tests_run],
        "research_conductor_modified": False,
        "fixture_sha256": sha256_json(fixture_payload),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str | None = None,
    fixture_path: Path | str | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the fixture JSON and the terminal Exp5499 result JSON."""

    fixture_payload = build_fixture()
    fixture_output = Path(fixture_path) if fixture_path is not None else repo_root / FIXTURE_RELATIVE_PATH
    fixture_output.parent.mkdir(parents=True, exist_ok=True)
    fixture_output.write_text(
        json.dumps(fixture_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    artifact = build_artifact(fixture=fixture_payload, tests_run=tests_run)
    result_output = Path(result_path) if result_path is not None else repo_root / RESULT_RELATIVE_PATH
    result_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required result fields against the embedded exact fixture."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(artifact.get("fixture_path") == FIXTURE_RELATIVE_PATH.as_posix(), "fixture_path")
    _require(artifact.get("reference_solver_path") == REFERENCE_SOLVER_PATH, "reference_solver_path")
    _require(artifact.get("test_paths") == list(TEST_PATHS), "test_paths")
    _require(artifact.get("guided_decoding_used") is False, "guided_decoding_used")
    _require(artifact.get("token_steering_used") is False, "token_steering_used")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(str(artifact.get("honest_verdict", "")).startswith(("complete:", "blocked:")), "honest_verdict")
    report = evaluate_fixture(artifact["fixture"])
    for field in (
        "num_instances",
        "hard_constraint_pass_rate",
        "preference_optimality_rate",
        "independent_reference_agreement_rate",
        "false_accept_rate",
        "preference_maxsat_fixture_ready",
    ):
        _require(artifact.get(field) == report[field], field)
    _require(artifact.get("readiness_blockers") == report["readiness_blockers"], "readiness_blockers")
    _require(artifact.get("fixture_sha256") == sha256_json(artifact["fixture"]), "fixture_sha256")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict that does not overstate blocked evidence."""

    if ready:
        return "complete: preference_maxsat_minimal_fixture_ready_exact_validators_authoritative"
    return "blocked: preference_maxsat_fixture_not_ready_" + "_".join(blockers)


def _rate(numerator: int | float, denominator: int) -> float:
    """Return a rounded rate with a zero denominator treated as zero evidence."""

    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
