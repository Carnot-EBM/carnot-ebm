"""Exp5501 hierarchical helper-contract claim fixture.

Spec refs: REQ-VERIFY-5501, SCENARIO-VERIFY-5501.

Natural-language helper text is useful only when it points to checks that can
actually run. This module treats each helper statement as a tree: the helper is
the root, typed claim spans are the leaves, and every supported leaf must
compile to an exact predicate over the Exp5499 hard/soft claim-state fixture.
Unsupported prose stays visible as refused evidence instead of becoming an
authority by sounding plausible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as fixture_mod


JsonDict = dict[str, Any]
PredicateFn = Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[str, JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5501_helper_contract_hierarchical_claim_fixture_v499.json"
)
FIXTURE_RELATIVE_PATH = Path("results/helper_contract_hierarchical_claim_fixture_5501/fixture.json")
SOURCE_ARTIFACT_RELATIVE_PATH = fixture_mod.RESULT_RELATIVE_PATH

MODULE_PATH = "python/carnot/experiment_5501_helper_contract_hierarchical_claim_fixture_v499.py"
EXECUTABLE_PREDICATE_PATHS = (
    f"{MODULE_PATH}::predicate_assignment_equals",
    f"{MODULE_PATH}::predicate_soft_preference_declared",
    f"{MODULE_PATH}::predicate_candidate_soft_suboptimal",
    f"{MODULE_PATH}::predicate_repair_to_reference",
    f"{MODULE_PATH}::predicate_candidate_acceptance_authorized",
    f"{MODULE_PATH}::predicate_soft_preference_sufficient",
)

SCHEMA = "carnot.experiment_5501.helper_contract_hierarchical_claim_fixture.v499"
FIXTURE_SCHEMA = "carnot.fixture.helper_contract_hierarchical_claims.v1"
EXPERIMENT = 5501
EXPERIMENT_ID = "exp5501-helper-contract-hierarchical-claim-fixture-v499"
MILESTONE = "2026.07.499"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5501
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ("REQ-VERIFY-5501", "SCENARIO-VERIFY-5501", "REQ-VERIFY-5499")

LOCAL_LABELS = ("entailed", "contradicted", "unsupported", "overbroad")
HELPER_VERDICTS = (
    "accepted",
    "refused_unsupported",
    "rejected_contradicted",
    "rejected_overbroad",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "helper_contract_fixture_path": (
        "points to the checked helper-contract fixture rather than relying on prose in the "
        "result artifact."
    ),
    "executable_predicate_paths": (
        "lists the exact predicate entry points that make helper text auditable."
    ),
    "test_paths": "identifies the tests that assert the REQ/SCENARIO contract.",
    "num_helper_contracts": "bounds the fixture size and prevents silent row loss.",
    "unsupported_contract_count": (
        "counts helper text refused because it lacked executable support."
    ),
    "local_claim_label_accuracy": (
        "measures local span labels against exact predicates before rolling up."
    ),
    "rolled_up_verdict_accuracy": ("measures RT4CHART-style local-to-global verdict agreement."),
    "useful_repair_rate": "counts only repair suggestions that become exact-reference agreements.",
    "false_accept_rate": (
        "guards against unsupported, contradicted, or overbroad helper text being accepted."
    ),
    "helper_contract_fixture_ready": (
        "downstream gate for the hierarchical helper-contract fixture."
    ),
    "inference_substrate": (
        "declares verifier-scoring against cached candidates rather than live model inference."
    ),
    "honest_verdict": (
        "terminal status; start with complete: or blocked: and do not launder unsupported text."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
TEST_PATHS = (
    "tests/python/test_experiment_5501_helper_contract_hierarchical_claim_fixture_v499.py",
)


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically so checksums notice semantic drift."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(payload: Any) -> str:
    """Return a stable content hash for JSON-compatible payloads."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def build_fixture() -> JsonDict:
    """Build the deterministic helper-contract fixture over Exp5499 rows."""

    source_fixture = fixture_mod.build_fixture()
    return {
        "schema": FIXTURE_SCHEMA,
        "fixture_id": "helper_contract_hierarchical_claim_fixture_5501",
        "source_artifact": SOURCE_ARTIFACT_RELATIVE_PATH.as_posix(),
        "source_fixture_sha256": fixture_mod.sha256_json(source_fixture),
        "guided_decoding_used": False,
        "token_steering_used": False,
        "helper_contracts": [
            _contract(
                contract_id="support_acceptance_contract",
                contract_kind="acceptance_rule",
                statement=(
                    "For claim_support_preference, the helper accepts only when "
                    "support='entailed' and scope='bounded'; source_quality='primary' "
                    "is a soft preference after hard support and scope pass."
                ),
                expected_verdict="accepted",
                claim_specs=[
                    _claim_spec(
                        claim_id="support_entailment_required",
                        span_text="support='entailed'",
                        typed_claim="support",
                        expected_label="entailed",
                        predicate={
                            "type": "assignment_equals",
                            "instance_id": "claim_support_preference",
                            "candidate_id": "support_exact_optimum",
                            "variable": "support",
                            "equals": "entailed",
                        },
                    ),
                    _claim_spec(
                        claim_id="scope_bounded_required",
                        span_text="scope='bounded'",
                        typed_claim="scope",
                        expected_label="entailed",
                        predicate={
                            "type": "assignment_equals",
                            "instance_id": "claim_support_preference",
                            "candidate_id": "support_exact_optimum",
                            "variable": "scope",
                            "equals": "bounded",
                        },
                    ),
                    _claim_spec(
                        claim_id="primary_source_soft_only",
                        span_text="source_quality='primary' is a soft preference",
                        typed_claim="source_quality",
                        expected_label="entailed",
                        predicate={
                            "type": "soft_preference_declared",
                            "instance_id": "claim_support_preference",
                            "preference_id": "SP_PRIMARY_SOURCE",
                            "variable": "source_quality",
                            "value": "primary",
                        },
                    ),
                ],
            ),
            _contract(
                contract_id="safety_acceptance_contract",
                contract_kind="acceptance_rule",
                statement=(
                    "For claim_safety_conflict, the helper accepts only when safety='safe' "
                    "and citation='present'; action='accept' is the soft-preferred "
                    "validator action."
                ),
                expected_verdict="accepted",
                claim_specs=[
                    _claim_spec(
                        claim_id="safe_label_required",
                        span_text="safety='safe'",
                        typed_claim="safety",
                        expected_label="entailed",
                        predicate={
                            "type": "assignment_equals",
                            "instance_id": "claim_safety_conflict",
                            "candidate_id": "safety_exact_optimum",
                            "variable": "safety",
                            "equals": "safe",
                        },
                    ),
                    _claim_spec(
                        claim_id="citation_present_required",
                        span_text="citation='present'",
                        typed_claim="citation",
                        expected_label="entailed",
                        predicate={
                            "type": "assignment_equals",
                            "instance_id": "claim_safety_conflict",
                            "candidate_id": "safety_exact_optimum",
                            "variable": "citation",
                            "equals": "present",
                        },
                    ),
                    _claim_spec(
                        claim_id="accept_action_soft_preferred",
                        span_text="action='accept' is the soft-preferred",
                        typed_claim="action",
                        expected_label="entailed",
                        predicate={
                            "type": "soft_preference_declared",
                            "instance_id": "claim_safety_conflict",
                            "preference_id": "SP_ACCEPT_WHEN_SAFE",
                            "variable": "action",
                            "value": "accept",
                        },
                    ),
                ],
            ),
            _contract(
                contract_id="support_soft_repair_contract",
                contract_kind="repair_rule",
                statement=(
                    "For support_soft_suboptimal, changing source_quality from secondary "
                    "to primary while preserving support and scope repairs the helper to "
                    "the exact reference assignment."
                ),
                expected_verdict="accepted",
                repair_attempt=True,
                claim_specs=[
                    _claim_spec(
                        claim_id="support_candidate_is_soft_suboptimal",
                        span_text="support_soft_suboptimal",
                        typed_claim="source_quality",
                        expected_label="entailed",
                        predicate={
                            "type": "candidate_soft_suboptimal",
                            "instance_id": "claim_support_preference",
                            "candidate_id": "support_soft_suboptimal",
                        },
                    ),
                    _claim_spec(
                        claim_id="support_repair_matches_reference",
                        span_text="changing source_quality from secondary to primary",
                        typed_claim="source_quality",
                        expected_label="entailed",
                        predicate={
                            "type": "repair_to_reference",
                            "instance_id": "claim_support_preference",
                            "candidate_id": "support_soft_suboptimal",
                            "repair_assignment": {
                                "support": "entailed",
                                "source_quality": "primary",
                                "scope": "bounded",
                            },
                        },
                    ),
                ],
            ),
            _contract(
                contract_id="safety_soft_repair_contract",
                contract_kind="repair_rule",
                statement=(
                    "For safety_soft_suboptimal, changing action from reject to accept "
                    "while preserving safety and citation repairs the helper to the exact "
                    "reference assignment."
                ),
                expected_verdict="accepted",
                repair_attempt=True,
                claim_specs=[
                    _claim_spec(
                        claim_id="safety_candidate_is_soft_suboptimal",
                        span_text="safety_soft_suboptimal",
                        typed_claim="action",
                        expected_label="entailed",
                        predicate={
                            "type": "candidate_soft_suboptimal",
                            "instance_id": "claim_safety_conflict",
                            "candidate_id": "safety_soft_suboptimal",
                        },
                    ),
                    _claim_spec(
                        claim_id="safety_repair_matches_reference",
                        span_text="changing action from reject to accept",
                        typed_claim="action",
                        expected_label="entailed",
                        predicate={
                            "type": "repair_to_reference",
                            "instance_id": "claim_safety_conflict",
                            "candidate_id": "safety_soft_suboptimal",
                            "repair_assignment": {
                                "safety": "safe",
                                "citation": "present",
                                "action": "accept",
                            },
                        },
                    ),
                ],
            ),
            _contract(
                contract_id="baseless_confidence_contract",
                contract_kind="negative_control_baseless",
                statement=(
                    "Because the helper sounds confident, claim_support_preference should "
                    "be accepted."
                ),
                expected_verdict="refused_unsupported",
                claim_specs=[
                    _claim_spec(
                        claim_id="confidence_has_no_typed_evidence",
                        span_text="sounds confident",
                        typed_claim="confidence",
                        expected_label="unsupported",
                        predicate={
                            "type": "unsupported",
                            "reason": "confidence prose has no Exp5499 typed claim or executable evidence",
                        },
                        evidence_refs=[],
                    ),
                ],
            ),
            _contract(
                contract_id="contradicted_support_contract",
                contract_kind="negative_control_contradicted",
                statement=(
                    "The support_hard_violation candidate should be accepted because "
                    "source_quality is primary even when support is unsupported."
                ),
                expected_verdict="rejected_contradicted",
                claim_specs=[
                    _claim_spec(
                        claim_id="hard_violation_primary_source_present",
                        span_text="source_quality is primary",
                        typed_claim="source_quality",
                        expected_label="entailed",
                        predicate={
                            "type": "assignment_equals",
                            "instance_id": "claim_support_preference",
                            "candidate_id": "support_hard_violation",
                            "variable": "source_quality",
                            "equals": "primary",
                        },
                    ),
                    _claim_spec(
                        claim_id="hard_violation_acceptance_authorized",
                        span_text="candidate should be accepted",
                        typed_claim="support",
                        expected_label="contradicted",
                        predicate={
                            "type": "candidate_acceptance_authorized",
                            "instance_id": "claim_support_preference",
                            "candidate_id": "support_hard_violation",
                        },
                    ),
                ],
            ),
            _contract(
                contract_id="overbroad_primary_contract",
                contract_kind="negative_control_overbroad",
                statement=(
                    "Any candidate with source_quality='primary' can be accepted "
                    "regardless of support, making the primary-source soft preference "
                    "sufficient for helper acceptance."
                ),
                expected_verdict="rejected_overbroad",
                claim_specs=[
                    _claim_spec(
                        claim_id="primary_source_soft_preference_not_sufficient",
                        span_text="primary-source soft preference sufficient",
                        typed_claim="source_quality",
                        expected_label="overbroad",
                        predicate={
                            "type": "soft_preference_sufficient",
                            "instance_id": "claim_support_preference",
                            "preference_id": "SP_PRIMARY_SOURCE",
                            "variable": "source_quality",
                            "value": "primary",
                            "counterexample_candidate_id": "support_hard_violation",
                        },
                    ),
                ],
            ),
        ],
    }


def validate_fixture(fixture: Mapping[str, Any]) -> None:
    """Validate fixture shape before helper text is allowed to compile."""

    _require(fixture.get("schema") == FIXTURE_SCHEMA, "fixture_schema")
    _require(
        fixture.get("source_artifact") == SOURCE_ARTIFACT_RELATIVE_PATH.as_posix(),
        "source_artifact",
    )
    _require(
        fixture.get("source_fixture_sha256")
        == fixture_mod.sha256_json(fixture_mod.build_fixture()),
        "source_fixture_sha256",
    )
    _require(fixture.get("guided_decoding_used") is False, "guided_decoding_used")
    _require(fixture.get("token_steering_used") is False, "token_steering_used")
    contracts = fixture.get("helper_contracts")
    _require(isinstance(contracts, list) and bool(contracts), "helper_contracts")
    ids = [str(contract.get("contract_id")) for contract in contracts]
    _require(len(ids) == len(set(ids)), "contract_id_unique")
    for contract in contracts:
        _validate_contract(contract)


def evaluate_fixture(fixture: Mapping[str, Any]) -> JsonDict:
    """Evaluate local claim labels and helper rollups with exact predicates."""

    validate_fixture(fixture)
    contract_reports: list[JsonDict] = []
    claim_matches = 0
    claim_count = 0
    verdict_matches = 0
    false_accept_count = 0
    unsupported_count = 0
    repair_attempt_count = 0
    useful_repair_count = 0
    for contract in fixture["helper_contracts"]:
        claim_reports = [
            evaluate_claim_span(fixture, contract, claim) for claim in contract["claim_spans"]
        ]
        observed_verdict = roll_up_helper_verdict(claim_reports)
        expected_verdict = str(contract["expected_verdict"])
        repair_attempted = bool(contract.get("repair_attempt"))
        useful_repair = any(
            row["predicate_type"] == "repair_to_reference"
            and row["observed_label"] == "entailed"
            and row["predicate_result"].get("before_reference_agreement") is False
            and row["predicate_result"].get("after_reference_agreement") is True
            for row in claim_reports
        )
        false_accept = observed_verdict == "accepted" and expected_verdict != "accepted"
        claim_matches += sum(int(row["label_matches"]) for row in claim_reports)
        claim_count += len(claim_reports)
        verdict_matches += int(observed_verdict == expected_verdict)
        false_accept_count += int(false_accept)
        unsupported_count += int(observed_verdict == "refused_unsupported")
        repair_attempt_count += int(repair_attempted)
        useful_repair_count += int(repair_attempted and useful_repair)
        contract_reports.append(
            {
                "contract_id": contract["contract_id"],
                "contract_kind": contract["contract_kind"],
                "expected_verdict": expected_verdict,
                "observed_verdict": observed_verdict,
                "verdict_matches": observed_verdict == expected_verdict,
                "repair_attempted": repair_attempted,
                "useful_repair": bool(repair_attempted and useful_repair),
                "false_accept": bool(false_accept),
                "claim_reports": claim_reports,
            }
        )
    local_accuracy = _rate(claim_matches, claim_count)
    verdict_accuracy = _rate(verdict_matches, len(contract_reports))
    useful_repair_rate = _rate(useful_repair_count, repair_attempt_count)
    false_accept_rate = _rate(false_accept_count, len(contract_reports))
    readiness_blockers = readiness_blockers_for_report(
        num_contracts=len(contract_reports),
        unsupported_contract_count=unsupported_count,
        local_claim_label_accuracy=local_accuracy,
        rolled_up_verdict_accuracy=verdict_accuracy,
        useful_repair_rate=useful_repair_rate,
        false_accept_rate=false_accept_rate,
        repair_attempt_count=repair_attempt_count,
    )
    return {
        "num_helper_contracts": len(contract_reports),
        "unsupported_contract_count": unsupported_count,
        "local_claim_label_accuracy": local_accuracy,
        "rolled_up_verdict_accuracy": verdict_accuracy,
        "repair_attempt_count": repair_attempt_count,
        "useful_repair_count": useful_repair_count,
        "useful_repair_rate": useful_repair_rate,
        "false_accept_count": false_accept_count,
        "false_accept_rate": false_accept_rate,
        "contract_reports": contract_reports,
        "readiness_blockers": readiness_blockers,
        "helper_contract_fixture_ready": not readiness_blockers,
    }


def evaluate_claim_span(
    fixture: Mapping[str, Any],
    contract: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> JsonDict:
    """Compile one claim span to an exact predicate or unsupported label."""

    predicate = _mapping(claim.get("predicate"))
    predicate_type = str(predicate.get("type", ""))
    expected_label = str(claim.get("expected_label"))
    if predicate_type == "unsupported":
        observed_label = "unsupported"
        result = {"reason": str(predicate.get("reason", "unsupported"))}
        compiled_to = "unsupported_label"
        predicate_path = ""
    else:
        predicate_fn = PREDICATES.get(predicate_type)
        if predicate_fn is None:
            raise ValueError(predicate_type)
        observed_label, result = predicate_fn(fixture, predicate)
        compiled_to = "executable_predicate"
        predicate_path = PREDICATE_PATH_BY_TYPE[predicate_type]
    return {
        "contract_id": contract["contract_id"],
        "claim_id": claim["claim_id"],
        "span_text": claim["span_text"],
        "typed_claim": claim["typed_claim"],
        "expected_label": expected_label,
        "observed_label": observed_label,
        "label_matches": observed_label == expected_label,
        "compiled_to": compiled_to,
        "predicate_type": predicate_type,
        "predicate_path": predicate_path,
        "predicate_result": result,
        "evidence_refs": list(claim.get("evidence_refs", [])),
    }


def roll_up_helper_verdict(claim_reports: Sequence[Mapping[str, Any]]) -> str:
    """Roll local labels into the helper verdict used by the fixture."""

    labels = [str(row["observed_label"]) for row in claim_reports]
    if "unsupported" in labels:
        return "refused_unsupported"
    if "contradicted" in labels:
        return "rejected_contradicted"
    if "overbroad" in labels:
        return "rejected_overbroad"
    return "accepted"


def predicate_assignment_equals(
    fixture: Mapping[str, Any],
    predicate: Mapping[str, Any],
) -> tuple[str, JsonDict]:
    """Check that a cached Exp5499 assignment has the claimed typed value."""

    candidate = _candidate_for_predicate(predicate)
    assignment = _mapping(candidate["assignment"])
    variable = str(predicate["variable"])
    expected = str(predicate["equals"])
    observed = str(assignment.get(variable, ""))
    label = "entailed" if observed == expected else "contradicted"
    return label, {
        "candidate_id": candidate["candidate_id"],
        "variable": variable,
        "expected": expected,
        "observed": observed,
    }


def predicate_soft_preference_declared(
    fixture: Mapping[str, Any],
    predicate: Mapping[str, Any],
) -> tuple[str, JsonDict]:
    """Check that a soft preference exists without treating it as a hard rule."""

    instance = _instance_for_predicate(predicate)
    preference_id = str(predicate["preference_id"])
    variable = str(predicate["variable"])
    value = str(predicate["value"])
    matches = [
        preference
        for preference in instance["soft_preferences"]
        if preference["id"] == preference_id
        and preference["variable"] == variable
        and preference["value"] == value
    ]
    label = "entailed" if matches else "unsupported"
    return label, {
        "instance_id": instance["instance_id"],
        "preference_id": preference_id,
        "variable": variable,
        "value": value,
        "declared": bool(matches),
    }


def predicate_candidate_soft_suboptimal(
    fixture: Mapping[str, Any],
    predicate: Mapping[str, Any],
) -> tuple[str, JsonDict]:
    """Check that a cached candidate passes hard rows but misses the soft optimum."""

    instance = _instance_for_predicate(predicate)
    candidate = _candidate_for_predicate(predicate)
    evaluation = _evaluate_assignment(instance, _mapping(candidate["assignment"]))
    is_suboptimal = (
        evaluation["hard_constraints_pass"] is True
        and evaluation["reference_agreement"] is False
        and float(evaluation["soft_score"]) < float(evaluation["reference"]["objective_score"])
    )
    label = "entailed" if is_suboptimal else "contradicted"
    return label, {
        "candidate_id": candidate["candidate_id"],
        "hard_constraints_pass": evaluation["hard_constraints_pass"],
        "soft_score": evaluation["soft_score"],
        "reference_objective_score": evaluation["reference"]["objective_score"],
        "reference_agreement": evaluation["reference_agreement"],
    }


def predicate_repair_to_reference(
    fixture: Mapping[str, Any],
    predicate: Mapping[str, Any],
) -> tuple[str, JsonDict]:
    """Check that a proposed repair becomes the exact Exp5499 reference state."""

    instance = _instance_for_predicate(predicate)
    candidate = _candidate_for_predicate(predicate)
    before = _evaluate_assignment(instance, _mapping(candidate["assignment"]))
    repair_assignment = {
        str(key): str(value) for key, value in predicate["repair_assignment"].items()
    }
    after = _evaluate_assignment(instance, repair_assignment)
    useful = before["reference_agreement"] is False and after["reference_agreement"] is True
    label = "entailed" if useful else "contradicted"
    return label, {
        "candidate_id": candidate["candidate_id"],
        "before_assignment": dict(candidate["assignment"]),
        "after_assignment": repair_assignment,
        "before_reference_agreement": before["reference_agreement"],
        "after_reference_agreement": after["reference_agreement"],
        "after_hard_constraints_pass": after["hard_constraints_pass"],
        "after_soft_score": after["soft_score"],
    }


def predicate_candidate_acceptance_authorized(
    fixture: Mapping[str, Any],
    predicate: Mapping[str, Any],
) -> tuple[str, JsonDict]:
    """Check whether a helper may accept the cached candidate under exact rules."""

    instance = _instance_for_predicate(predicate)
    candidate = _candidate_for_predicate(predicate)
    evaluation = _evaluate_assignment(instance, _mapping(candidate["assignment"]))
    authorized = (
        evaluation["hard_constraints_pass"] is True and evaluation["reference_agreement"] is True
    )
    label = "entailed" if authorized else "contradicted"
    return label, {
        "candidate_id": candidate["candidate_id"],
        "hard_constraints_pass": evaluation["hard_constraints_pass"],
        "reference_agreement": evaluation["reference_agreement"],
        "authorized": authorized,
    }


def predicate_soft_preference_sufficient(
    fixture: Mapping[str, Any],
    predicate: Mapping[str, Any],
) -> tuple[str, JsonDict]:
    """Reject a helper that upgrades a soft preference into sufficient evidence."""

    soft_label, soft_result = predicate_soft_preference_declared(fixture, predicate)
    candidate = _candidate_by_id(
        _instance_for_predicate(predicate),
        str(predicate["counterexample_candidate_id"]),
    )
    assignment = _mapping(candidate["assignment"])
    variable = str(predicate["variable"])
    value = str(predicate["value"])
    evaluation = _evaluate_assignment(_instance_for_predicate(predicate), assignment)
    counterexample = (
        soft_label == "entailed"
        and assignment.get(variable) == value
        and evaluation["reference_agreement"] is False
    )
    label = "overbroad" if counterexample else soft_label
    return label, {
        **soft_result,
        "counterexample_candidate_id": candidate["candidate_id"],
        "counterexample_has_soft_value": assignment.get(variable) == value,
        "counterexample_hard_constraints_pass": evaluation["hard_constraints_pass"],
        "counterexample_reference_agreement": evaluation["reference_agreement"],
    }


PREDICATES: dict[str, PredicateFn] = {
    "assignment_equals": predicate_assignment_equals,
    "soft_preference_declared": predicate_soft_preference_declared,
    "candidate_soft_suboptimal": predicate_candidate_soft_suboptimal,
    "repair_to_reference": predicate_repair_to_reference,
    "candidate_acceptance_authorized": predicate_candidate_acceptance_authorized,
    "soft_preference_sufficient": predicate_soft_preference_sufficient,
}
PREDICATE_PATH_BY_TYPE = {
    "assignment_equals": EXECUTABLE_PREDICATE_PATHS[0],
    "soft_preference_declared": EXECUTABLE_PREDICATE_PATHS[1],
    "candidate_soft_suboptimal": EXECUTABLE_PREDICATE_PATHS[2],
    "repair_to_reference": EXECUTABLE_PREDICATE_PATHS[3],
    "candidate_acceptance_authorized": EXECUTABLE_PREDICATE_PATHS[4],
    "soft_preference_sufficient": EXECUTABLE_PREDICATE_PATHS[5],
}


def build_artifact(
    *,
    fixture: Mapping[str, Any] | None = None,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5501 result from exact helper predicates."""

    fixture_payload = build_fixture() if fixture is None else dict(fixture)
    report = evaluate_fixture(fixture_payload)
    ready = bool(report["helper_contract_fixture_ready"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifact": SOURCE_ARTIFACT_RELATIVE_PATH.as_posix(),
        "helper_contract_fixture_path": FIXTURE_RELATIVE_PATH.as_posix(),
        "executable_predicate_paths": list(EXECUTABLE_PREDICATE_PATHS),
        "test_paths": list(TEST_PATHS),
        "num_helper_contracts": report["num_helper_contracts"],
        "unsupported_contract_count": report["unsupported_contract_count"],
        "local_claim_label_accuracy": report["local_claim_label_accuracy"],
        "rolled_up_verdict_accuracy": report["rolled_up_verdict_accuracy"],
        "useful_repair_rate": report["useful_repair_rate"],
        "false_accept_rate": report["false_accept_rate"],
        "helper_contract_fixture_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict(ready, report["readiness_blockers"]),
        "fixture": fixture_payload,
        "contract_reports": report["contract_reports"],
        "repair_attempt_count": report["repair_attempt_count"],
        "useful_repair_count": report["useful_repair_count"],
        "false_accept_count": report["false_accept_count"],
        "readiness_blockers": report["readiness_blockers"],
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": [dict(item) for item in tests_run],
        "research_conductor_modified": False,
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
    """Write the helper fixture JSON and terminal Exp5501 result JSON."""

    fixture_payload = build_fixture()
    fixture_output = (
        Path(fixture_path) if fixture_path is not None else repo_root / FIXTURE_RELATIVE_PATH
    )
    fixture_output.parent.mkdir(parents=True, exist_ok=True)
    fixture_output.write_text(
        json.dumps(fixture_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    artifact = build_artifact(fixture=fixture_payload, tests_run=tests_run)
    result_output = (
        Path(result_path) if result_path is not None else repo_root / RESULT_RELATIVE_PATH
    )
    result_output.parent.mkdir(parents=True, exist_ok=True)
    result_output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise if the result can no longer support the Exp5501 claim."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        artifact.get("helper_contract_fixture_path") == FIXTURE_RELATIVE_PATH.as_posix(),
        "helper_contract_fixture_path",
    )
    _require(
        artifact.get("executable_predicate_paths") == list(EXECUTABLE_PREDICATE_PATHS),
        "executable_predicate_paths",
    )
    _require(artifact.get("test_paths") == list(TEST_PATHS), "test_paths")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("research_conductor_modified") is False, "research_conductor_modified")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    report = evaluate_fixture(artifact["fixture"])
    for field in (
        "num_helper_contracts",
        "unsupported_contract_count",
        "local_claim_label_accuracy",
        "rolled_up_verdict_accuracy",
        "useful_repair_rate",
        "false_accept_rate",
        "helper_contract_fixture_ready",
    ):
        _require(artifact.get(field) == report[field], field)
    _require(
        artifact.get("readiness_blockers") == report["readiness_blockers"], "readiness_blockers"
    )
    _require(artifact.get("contract_reports") == report["contract_reports"], "contract_reports")
    _require(
        artifact.get("repair_attempt_count") == report["repair_attempt_count"],
        "repair_attempt_count",
    )
    _require(
        artifact.get("useful_repair_count") == report["useful_repair_count"], "useful_repair_count"
    )
    _require(
        artifact.get("false_accept_count") == report["false_accept_count"], "false_accept_count"
    )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")


def readiness_blockers_for_report(
    *,
    num_contracts: int,
    unsupported_contract_count: int,
    local_claim_label_accuracy: float,
    rolled_up_verdict_accuracy: float,
    useful_repair_rate: float,
    false_accept_rate: float,
    repair_attempt_count: int,
) -> list[str]:
    """Return precise blockers for the helper-contract-ready gate."""

    checks = (
        (num_contracts > 0, "helper_contracts_missing"),
        (unsupported_contract_count > 0, "unsupported_negative_control_missing"),
        (local_claim_label_accuracy == 1.0, "local_claim_label_accuracy_below_1"),
        (rolled_up_verdict_accuracy == 1.0, "rolled_up_verdict_accuracy_below_1"),
        (repair_attempt_count > 0, "repair_attempts_missing"),
        (useful_repair_rate == 1.0, "useful_repair_rate_below_1"),
        (false_accept_rate == 0.0, "false_accept_rate_nonzero"),
    )
    return [name for passed, name in checks if not passed]


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return a terminal verdict that does not overstate helper prose."""

    if ready:
        return "complete: helper_contract_hierarchical_claim_fixture_ready_exact_predicates_authoritative"
    return "blocked: helper_contract_fixture_not_ready_" + "_".join(blockers)


def _contract(
    *,
    contract_id: str,
    contract_kind: str,
    statement: str,
    expected_verdict: str,
    claim_specs: Sequence[Mapping[str, Any]],
    repair_attempt: bool = False,
) -> JsonDict:
    claim_spans = []
    evidence_map = {}
    for spec in claim_specs:
        span_text = str(spec["span_text"])
        char_start = statement.index(span_text)
        claim = {
            "claim_id": spec["claim_id"],
            "span_text": span_text,
            "char_start": char_start,
            "char_end": char_start + len(span_text),
            "typed_claim": spec["typed_claim"],
            "expected_label": spec["expected_label"],
            "predicate": dict(spec["predicate"]),
            "evidence_refs": [dict(row) for row in spec["evidence_refs"]],
        }
        claim_spans.append(claim)
        evidence_map[str(spec["claim_id"])] = [dict(row) for row in spec["evidence_refs"]]
    return {
        "contract_id": contract_id,
        "contract_kind": contract_kind,
        "statement": statement,
        "expected_verdict": expected_verdict,
        "repair_attempt": repair_attempt,
        "claim_spans": claim_spans,
        "evidence_map": evidence_map,
    }


def _claim_spec(
    *,
    claim_id: str,
    span_text: str,
    typed_claim: str,
    expected_label: str,
    predicate: Mapping[str, Any],
    evidence_refs: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    return {
        "claim_id": claim_id,
        "span_text": span_text,
        "typed_claim": typed_claim,
        "expected_label": expected_label,
        "predicate": dict(predicate),
        "evidence_refs": list(evidence_refs)
        if evidence_refs is not None
        else [_evidence_ref(predicate)],
    }


def _evidence_ref(predicate: Mapping[str, Any]) -> JsonDict:
    return {
        "source_artifact": SOURCE_ARTIFACT_RELATIVE_PATH.as_posix(),
        "instance_id": predicate.get("instance_id"),
        "candidate_id": predicate.get("candidate_id"),
        "preference_id": predicate.get("preference_id"),
        "variable": predicate.get("variable"),
    }


def _validate_contract(contract: Mapping[str, Any]) -> None:
    _require(bool(contract.get("contract_id")), "contract_id")
    _require(
        contract.get("contract_kind")
        in {
            "acceptance_rule",
            "repair_rule",
            "negative_control_baseless",
            "negative_control_contradicted",
            "negative_control_overbroad",
        },
        "contract_kind",
    )
    statement = str(contract.get("statement", ""))
    _require(bool(statement), "statement")
    _require(contract.get("expected_verdict") in HELPER_VERDICTS, "expected_verdict")
    claim_spans = contract.get("claim_spans")
    evidence_map = contract.get("evidence_map")
    _require(isinstance(claim_spans, list) and bool(claim_spans), "claim_spans")
    _require(isinstance(evidence_map, dict) and bool(evidence_map), "evidence_map")
    for claim in claim_spans:
        _validate_claim(statement, claim, evidence_map)


def _validate_claim(
    statement: str,
    claim: Mapping[str, Any],
    evidence_map: Mapping[str, Any],
) -> None:
    claim_id = str(claim.get("claim_id", ""))
    _require(bool(claim_id), "claim_id")
    _require(claim_id in evidence_map, "claim_evidence_map")
    start = int(claim.get("char_start", -1))
    end = int(claim.get("char_end", -1))
    span_text = str(claim.get("span_text", ""))
    _require(0 <= start < end <= len(statement), "claim_span_offsets")
    _require(statement[start:end] == span_text, "claim_span_text")
    _require(claim.get("expected_label") in LOCAL_LABELS, "expected_label")
    predicate = _mapping(claim.get("predicate"))
    predicate_type = predicate.get("type")
    _require(bool(predicate_type), "predicate_type")
    if claim.get("expected_label") == "unsupported":
        _require(predicate_type == "unsupported", "unsupported_predicate")
    else:
        _require(predicate_type in PREDICATES, "executable_predicate")


def _source_fixture() -> JsonDict:
    fixture = fixture_mod.build_fixture()
    fixture_mod.validate_fixture(fixture)
    return fixture


def _instance_for_predicate(predicate: Mapping[str, Any]) -> JsonDict:
    instance_id = str(predicate["instance_id"])
    for instance in _source_fixture()["instances"]:
        if instance["instance_id"] == instance_id:
            return instance
    raise ValueError(f"missing_instance:{instance_id}")


def _candidate_for_predicate(predicate: Mapping[str, Any]) -> JsonDict:
    return _candidate_by_id(_instance_for_predicate(predicate), str(predicate["candidate_id"]))


def _candidate_by_id(instance: Mapping[str, Any], candidate_id: str) -> JsonDict:
    for candidate in instance["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise ValueError(f"missing_candidate:{candidate_id}")


def _evaluate_assignment(instance: Mapping[str, Any], assignment: Mapping[str, Any]) -> JsonDict:
    normalized = {str(key): str(value) for key, value in assignment.items()}
    reference = fixture_mod.solve_reference(instance)
    hard_ok = fixture_mod.hard_constraints_pass(instance, normalized)
    score = fixture_mod.soft_score(instance, normalized)
    reference_agreement = (
        reference["status"] == "optimal"
        and hard_ok
        and score == reference["objective_score"]
        and normalized == reference["assignment"]
        and fixture_mod.assignment_hash(normalized) == reference["assignment_hash"]
    )
    return {
        "reference": reference,
        "hard_constraints_pass": hard_ok,
        "soft_score": score,
        "reference_agreement": reference_agreement,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), "mapping")
    return value


def _rate(numerator: int | float, denominator: int) -> float:
    """Return a rounded rate with a zero denominator treated as no evidence."""

    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
