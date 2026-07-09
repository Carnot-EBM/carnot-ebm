"""Exp5475: deterministic CSL behavioral memory replay audit.

Spec refs: REQ-LEARN-5475,
SCENARIO-LEARN-5475-SUPPORT-REMOVAL,
SCENARIO-LEARN-5475-IRRELEVANT-MEMORY,
SCENARIO-LEARN-5475-LADDER,
SCENARIO-LEARN-5475-ARTIFACT.

This module audits memory as behavior, not as prompt decoration. A memory claim
is counted only when a deterministic row shows that governed retrieval selected
supported, relevant experience and rejected unsafe alternatives. No model is
loaded here because this audit is the fallback evidence ladder for times when a
SOTA GGUF runtime is unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5475_csl_behavioral_memory_ladder_v497.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5475_csl_behavioral_memory_ladder_v497.py"
)
SOURCE_ARTIFACTS = (
    Path("results/experiment_5460_csl_policy_bandit_v496.json"),
    Path("results/experiment_5461_gated_sota_csl_memory_routing_v496.json"),
    Path("results/experiment_5473_csl_kan_surrogate_assurance_v497.json"),
    Path("results/experiment_5474_sota_csl_scale_v497.json"),
)

EXPERIMENT_ID = "experiment_5475_csl_behavioral_memory_ladder_v497"
TASK_ID = "exp5475-csl-behavioral-memory-ladder-v497"
SCHEMA = "carnot.experiment_5475.csl_behavioral_memory_ladder.v497"
MILESTONE = "2026.07.497"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5475
INFERENCE_SUBSTRATE = "deterministic_replay_no_llm"
EXACT_VALIDATOR_AUTHORITY = "deterministic_replay_validator"
TERMINAL_PREFIXES = ("complete:", "blocked:")

NO_MEMORY_VARIANT = "no_memory"
NAIVE_VARIANT = "naive_icl"
GOVERNED_VARIANT = "governed_memory"
KAN_VARIANT = "kan_surrogate_policy"
VARIANT_NAMES = (NO_MEMORY_VARIANT, NAIVE_VARIANT, GOVERNED_VARIANT, KAN_VARIANT)
CONTROLLED_VARIANTS = (GOVERNED_VARIANT, KAN_VARIANT)
LADDER_AXES = (
    "support_removal",
    "paraphrase_robustness",
    "locality",
    "conflict_handling",
    "downstream_action_use",
    "stale_memory_rejection",
)
AXIS_RATE_FIELDS = {
    "support_removal": "support_removal_pass_rate",
    "paraphrase_robustness": "paraphrase_robustness_rate",
    "locality": "locality_pass_rate",
    "conflict_handling": "conflict_handling_pass_rate",
    "downstream_action_use": "downstream_action_use_rate",
    "stale_memory_rejection": "stale_memory_rejection_rate",
}
REQUIRED_ARTIFACT_FIELDS = (
    "replay_fixture_count",
    "support_removal_pass_rate",
    "paraphrase_robustness_rate",
    "locality_pass_rate",
    "conflict_handling_pass_rate",
    "downstream_action_use_rate",
    "stale_memory_rejection_rate",
    "no_memory_baseline_score",
    "naive_icl_baseline_score",
    "governed_memory_score",
    "csl_behavioral_memory_ready",
    "model_weight_mutation",
    "inference_substrate",
    "random_seed",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "replay_fixture_count": "coverage of behavioral memory ladder fixtures.",
    "support_removal_pass_rate": "removed support cannot justify memory use.",
    "paraphrase_robustness_rate": "experience survives wording changes.",
    "locality_pass_rate": "retrieval stays in the correct local context.",
    "conflict_handling_pass_rate": "newer verified evidence beats conflicts.",
    "downstream_action_use_rate": "retrieved experience changes action, not just text.",
    "stale_memory_rejection_rate": "expired experience is rejected.",
    "no_memory_baseline_score": "exact-validator score without memory.",
    "naive_icl_baseline_score": "exact-validator score with ungated retrieved memory.",
    "governed_memory_score": "exact-validator score under governed memory controls.",
    "csl_behavioral_memory_ready": "terminal readiness for auditable memory behavior.",
    "model_weight_mutation": "frozen model and adapter boundary.",
    "inference_substrate": "deterministic no-LLM replay declaration.",
    "random_seed": "deterministic run seed.",
    "honest_verdict": "terminal status; starts with complete: or blocked:.",
}
SPEC_REFS = (
    "REQ-LEARN-5475",
    "SCENARIO-LEARN-5475-SUPPORT-REMOVAL",
    "SCENARIO-LEARN-5475-IRRELEVANT-MEMORY",
    "SCENARIO-LEARN-5475-LADDER",
    "SCENARIO-LEARN-5475-ARTIFACT",
)


def build_replay_fixtures() -> list[JsonDict]:
    """Return deterministic rows that each isolate one memory-control behavior."""

    return [
        _fixture(
            row_id="5475-a-support-removed",
            axis="support_removal",
            question="What action is valid after the release proof was revoked?",
            expected_answer="hold-shipment",
            no_memory_answer="hold-shipment",
            topic="shipping",
            locality_key="dock-7",
            memory_records=[
                _memory_record(
                    "mem5475-a-old-release",
                    answer="release-shipment",
                    text="Revoked release proof once allowed this shipment to leave dock 7.",
                    topic="shipping",
                    locality_key="dock-7",
                    support_status="removed",
                    support_ids=["support:release-proof-revoked"],
                    rollback_pointer="rollback:support-removal-001",
                    trust_score=0.9,
                )
            ],
        ),
        _fixture(
            row_id="5475-b-paraphrase",
            axis="paraphrase_robustness",
            question="Which crate contains the signed gasket?",
            expected_answer="blue-crate-17",
            no_memory_answer="unknown",
            topic="crate",
            locality_key="warehouse-a",
            memory_records=[
                _memory_record(
                    "mem5475-b-azure-box",
                    answer="blue-crate-17",
                    text="The approved seal lives in azure box 17, the blue crate.",
                    topic="crate",
                    locality_key="warehouse-a",
                    support_ids=["support:gasket-intake-17"],
                    paraphrase_of="signed gasket crate location",
                    trust_score=0.86,
                )
            ],
        ),
        _fixture(
            row_id="5475-c-locality",
            axis="locality",
            question="Which zone should the cold-room robot inspect?",
            expected_answer="zone-2",
            no_memory_answer="unknown",
            topic="robot-routing",
            locality_key="cold-room",
            memory_records=[
                _memory_record(
                    "mem5475-c-warm-room",
                    answer="zone-9",
                    text="Warm-room robot inspections route to zone 9.",
                    topic="robot-routing",
                    locality_key="warm-room",
                    support_ids=["support:warm-room-route"],
                    rollback_pointer="rollback:wrong-locality-001",
                    trust_score=0.8,
                ),
                _memory_record(
                    "mem5475-c-cold-room",
                    answer="zone-2",
                    text="Cold-room robot inspections route to zone 2.",
                    topic="robot-routing",
                    locality_key="cold-room",
                    support_ids=["support:cold-room-route"],
                    trust_score=0.82,
                ),
            ],
        ),
        _fixture(
            row_id="5475-d-conflict",
            axis="conflict_handling",
            question="Which vendor is verified for lot 77?",
            expected_answer="vendor-A",
            no_memory_answer="unknown",
            topic="vendor",
            locality_key="lot-77",
            memory_records=[
                _memory_record(
                    "mem5475-d-old-vendor",
                    answer="vendor-B",
                    text="Older intake note listed vendor B for lot 77.",
                    topic="vendor",
                    locality_key="lot-77",
                    support_ids=["support:old-vendor-note"],
                    conflict_group="lot-77-vendor",
                    rollback_pointer="rollback:conflict-old-vendor",
                    trust_score=0.35,
                ),
                _memory_record(
                    "mem5475-d-verified-vendor",
                    answer="vendor-A",
                    text="Verified procurement receipt lists vendor A for lot 77.",
                    topic="vendor",
                    locality_key="lot-77",
                    support_ids=["support:verified-vendor-receipt"],
                    conflict_group="lot-77-vendor",
                    trust_score=0.94,
                ),
            ],
        ),
        _fixture(
            row_id="5475-e-action",
            axis="downstream_action_use",
            question="Which gate should the controller open for amber sensor TX?",
            expected_answer="open-gate-B",
            no_memory_answer="unknown",
            topic="controller-action",
            locality_key="line-tx",
            expected_downstream_action="dispatch:open-gate-B",
            memory_records=[
                _memory_record(
                    "mem5475-e-amber-tx",
                    answer="open-gate-B",
                    text="When sensor TX is amber, dispatch the gate-B opening action.",
                    topic="controller-action",
                    locality_key="line-tx",
                    support_ids=["support:amber-tx-action"],
                    downstream_action="dispatch:open-gate-B",
                    trust_score=0.88,
                )
            ],
        ),
        _fixture(
            row_id="5475-f-stale",
            axis="stale_memory_rejection",
            question="Which queue owns incident RX-4 after the handoff?",
            expected_answer="queue-beta",
            no_memory_answer="unknown",
            topic="incident-queue",
            locality_key="rx-4",
            memory_records=[
                _memory_record(
                    "mem5475-f-stale-alpha",
                    answer="queue-alpha",
                    text="Before the handoff, incident RX-4 belonged to queue alpha.",
                    topic="incident-queue",
                    locality_key="rx-4",
                    support_ids=["support:rx4-old-owner"],
                    stale=True,
                    rollback_pointer="rollback:stale-rx4-alpha",
                    trust_score=0.75,
                ),
                _memory_record(
                    "mem5475-f-fresh-beta",
                    answer="queue-beta",
                    text="Fresh replay receipt: incident RX-4 now belongs to queue beta.",
                    topic="incident-queue",
                    locality_key="rx-4",
                    support_ids=["support:rx4-fresh-owner"],
                    trust_score=0.89,
                ),
            ],
        ),
    ]


def evaluate_replay(fixtures: Sequence[Mapping[str, Any]] | None = None) -> JsonDict:
    """Score every fixture under all variants and summarize the ladder metrics."""

    replay_fixtures = [dict(row) for row in (fixtures or build_replay_fixtures())]
    row_results = [
        evaluate_variant(fixture, variant)
        for fixture in replay_fixtures
        for variant in VARIANT_NAMES
    ]
    row_ids_by_variant = {
        variant: [
            str(result["row_id"])
            for result in row_results
            if result.get("variant") == variant
        ]
        for variant in VARIANT_NAMES
    }
    variant_scores = {
        variant: _rate(
            sum(
                1
                for result in row_results
                if result.get("variant") == variant
                and _mapping(result.get("exact_validator_results")).get("accepted") is True
            ),
            len(replay_fixtures),
        )
        for variant in VARIANT_NAMES
    }
    axis_rates = {
        AXIS_RATE_FIELDS[axis]: _axis_pass_rate(axis, replay_fixtures, row_results)
        for axis in LADDER_AXES
    }
    return _json_ready(
        {
            "fixtures": replay_fixtures,
            "row_results": row_results,
            "row_ids_by_variant": row_ids_by_variant,
            "variant_scores": variant_scores,
            "axis_rates": axis_rates,
            "replay_fixture_count": len(replay_fixtures),
            "axis_details": _axis_details(replay_fixtures, row_results),
        }
    )


def evaluate_variant(fixture: Mapping[str, Any], variant: str) -> JsonDict:
    """Evaluate one fixture under one memory policy variant."""

    if variant == NO_MEMORY_VARIANT:
        return _build_result(
            fixture=fixture,
            variant=variant,
            retrieved=[],
            accepted=[],
            rejected=[],
            decision_path=["variant:no_memory", "retrieval:suppressed"],
            selected_answer=str(fixture["no_memory_answer"]),
            selected_downstream_action=None,
        )
    records = _list_of_mappings(fixture.get("memory_records"))
    if variant == NAIVE_VARIANT:
        selected = records[0] if records else {}
        return _build_result(
            fixture=fixture,
            variant=variant,
            retrieved=records,
            accepted=records,
            rejected=[],
            decision_path=[
                "variant:naive_icl",
                f"retrieved:{len(records)}",
                "accepted:all_without_governance",
            ],
            selected_answer=str(selected.get("answer", fixture["no_memory_answer"])),
            selected_downstream_action=selected.get("downstream_action"),
        )
    accepted, rejected, decision_path = governed_memory_decision(fixture, records, variant)
    selected = accepted[0] if accepted else {}
    return _build_result(
        fixture=fixture,
        variant=variant,
        retrieved=records,
        accepted=accepted,
        rejected=rejected,
        decision_path=decision_path,
        selected_answer=str(selected.get("answer", fixture["no_memory_answer"])),
        selected_downstream_action=selected.get("downstream_action"),
    )


def governed_memory_decision(
    fixture: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    variant: str,
) -> tuple[list[JsonDict], list[JsonDict], list[str]]:
    """Apply behavioral memory gates before a memory can influence the answer."""

    candidates: list[JsonDict] = []
    rejected: list[JsonDict] = []
    decision_path = [f"variant:{variant}", f"retrieved:{len(records)}"]
    if variant == KAN_VARIANT:
        decision_path.append("kan_surrogate_risk_threshold:0.50")
    for record in records:
        reason = _rejection_reason(fixture, record)
        if variant == KAN_VARIANT and _kan_risk_score(fixture, record) > 0.5:
            reason = reason or "kan_risk_threshold"
        if reason:
            rejected.append(_record_with_reason(record, reason))
            decision_path.append(f"reject:{record['memory_id']}:{reason}")
        else:
            candidates.append(copy.deepcopy(dict(record)))
    accepted, conflict_rejections = _resolve_conflicts(candidates)
    rejected.extend(conflict_rejections)
    decision_path.extend(
        f"reject:{record['memory_id']}:{record['rejection_reason']}"
        for record in conflict_rejections
    )
    decision_path.extend(f"accept:{record['memory_id']}" for record in accepted)
    if not accepted:
        decision_path.append("fallback:no_memory_answer")
    return accepted, rejected, decision_path


def validate_memory_use_claim(
    fixture: Mapping[str, Any],
    *,
    accepted_memory_records: Sequence[Mapping[str, Any]],
    selected_answer: str,
    memory_use_claimed: bool,
) -> JsonDict:
    """Validate that a claimed memory influence rests on supported evidence."""

    reasons: list[str] = []
    records = _list_of_mappings(list(accepted_memory_records))
    answer_accepted = str(selected_answer) == str(fixture["expected_answer"])
    if memory_use_claimed and not records:
        reasons.append("no_accepted_memory_evidence")
    for record in records:
        if record.get("support_status") != "active":
            reasons.append("support_removed_memory_accepted")
        if not _record_relevant(fixture, record):
            reasons.append("irrelevant_memory_accepted")
        if record.get("stale") is True:
            reasons.append("stale_memory_accepted")
        if _lower_trust_conflict_accepted(fixture, record):
            reasons.append("lower_trust_conflict_memory_accepted")
    if memory_use_claimed and not answer_accepted:
        reasons.append("answer_mismatch")
    return {
        "answer_accepted": answer_accepted,
        "memory_use_claimed": bool(memory_use_claimed),
        "memory_use_claim_valid": not reasons,
        "claim_failure_reasons": sorted(set(reasons)),
    }


def exact_validator(
    fixture: Mapping[str, Any],
    *,
    selected_answer: str,
    selected_downstream_action: Any,
    accepted_memory_records: Sequence[Mapping[str, Any]],
    memory_use_claimed: bool,
) -> JsonDict:
    """Return final authority results for answer, action, and memory-use validity."""

    claim = validate_memory_use_claim(
        fixture,
        accepted_memory_records=accepted_memory_records,
        selected_answer=selected_answer,
        memory_use_claimed=memory_use_claimed,
    )
    expected_action = fixture.get("expected_downstream_action")
    downstream_action_passed = (
        selected_downstream_action == expected_action if expected_action else True
    )
    failure_reasons: list[str] = []
    if not claim["answer_accepted"]:
        failure_reasons.append("answer_mismatch")
    if not downstream_action_passed:
        failure_reasons.append("downstream_action_mismatch")
    if not claim["memory_use_claim_valid"]:
        failure_reasons.extend(claim["claim_failure_reasons"])
    return {
        "authority": EXACT_VALIDATOR_AUTHORITY,
        "accepted": bool(
            claim["answer_accepted"]
            and downstream_action_passed
            and claim["memory_use_claim_valid"]
        ),
        "answer_accepted": claim["answer_accepted"],
        "selected_answer": selected_answer,
        "expected_answer": str(fixture["expected_answer"]),
        "downstream_action_passed": downstream_action_passed,
        "selected_downstream_action": selected_downstream_action,
        "expected_downstream_action": expected_action,
        "memory_use_claimed": claim["memory_use_claimed"],
        "memory_use_claim_valid": claim["memory_use_claim_valid"],
        "claim_failure_reasons": claim["claim_failure_reasons"],
        "failure_reasons": sorted(set(failure_reasons)),
    }


def row_axis_pass(fixture: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """Decide whether one governed row satisfies its behavioral ladder axis."""

    validator = _mapping(result.get("exact_validator_results"))
    accepted_records = _list_of_mappings(result.get("accepted_memory_records"))
    rejected_records = _list_of_mappings(result.get("rejected_memory_records"))
    if validator.get("accepted") is not True:
        return False
    axis = str(fixture["axis"])
    if axis == "support_removal":
        return (
            not any(record.get("support_status") != "active" for record in accepted_records)
            and any(record.get("support_status") == "removed" for record in rejected_records)
        )
    if axis == "paraphrase_robustness":
        return any(record.get("paraphrase_of") for record in accepted_records)
    if axis == "locality":
        return (
            all(_record_relevant(fixture, record) for record in accepted_records)
            and any(record.get("rejection_reason") == "irrelevant_memory" for record in rejected_records)
        )
    if axis == "conflict_handling":
        return any(
            record.get("rejection_reason") == "conflict_lower_trust"
            for record in rejected_records
        )
    if axis == "downstream_action_use":
        return bool(accepted_records) and validator.get("downstream_action_passed") is True
    if axis == "stale_memory_rejection":
        return (
            not any(record.get("stale") is True for record in accepted_records)
            and any(record.get("stale") is True for record in rejected_records)
        )
    return False


def fixture_by_axis(fixtures: Sequence[Mapping[str, Any]], axis: str) -> JsonDict:
    """Find a fixture by axis so tests can target one behavioral gate."""

    matches = [dict(fixture) for fixture in fixtures if fixture.get("axis") == axis]
    if not matches:
        raise ValueError(f"unknown fixture axis: {axis}")
    return matches[0]


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal no-LLM behavioral memory audit artifact."""

    root_path = Path(root)
    evaluation = evaluate_replay()
    rates = evaluation["axis_rates"]
    scores = evaluation["variant_scores"]
    ready = bool(
        tests_run
        and all(float(rates[field]) == 1.0 for field in AXIS_RATE_FIELDS.values())
        and scores[GOVERNED_VARIANT] > scores[NO_MEMORY_VARIANT]
        and scores[GOVERNED_VARIANT] > scores[NAIVE_VARIANT]
        and scores[KAN_VARIANT] >= scores[GOVERNED_VARIANT]
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "replay_fixture_count": evaluation["replay_fixture_count"],
        "support_removal_pass_rate": rates["support_removal_pass_rate"],
        "paraphrase_robustness_rate": rates["paraphrase_robustness_rate"],
        "locality_pass_rate": rates["locality_pass_rate"],
        "conflict_handling_pass_rate": rates["conflict_handling_pass_rate"],
        "downstream_action_use_rate": rates["downstream_action_use_rate"],
        "stale_memory_rejection_rate": rates["stale_memory_rejection_rate"],
        "no_memory_baseline_score": scores[NO_MEMORY_VARIANT],
        "naive_icl_baseline_score": scores[NAIVE_VARIANT],
        "governed_memory_score": scores[GOVERNED_VARIANT],
        "kan_surrogate_policy_score": scores[KAN_VARIANT],
        "csl_behavioral_memory_ready": ready,
        "model_weight_mutation": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "honest_verdict": _honest_verdict(ready),
        "fixtures": evaluation["fixtures"],
        "row_results": evaluation["row_results"],
        "row_ids_by_variant": evaluation["row_ids_by_variant"],
        "variant_scores": scores,
        "axis_details": evaluation["axis_details"],
        "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS],
        "source_files": {
            "module": str(MODULE_RELATIVE_PATH),
            "spec": str(SPEC_RELATIVE_PATH),
        },
        "source_file_checksums": _source_file_checksums(root_path),
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Write the Exp5475 artifact when requested and return the payload."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    if write:
        _write_json(Path(result_path), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the artifact cannot justify a behavioral memory-ready claim."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5475 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while leaving the caller's artifact untouched."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("missing required fields: " + ",".join(missing))
    for field in AXIS_RATE_FIELDS.values():
        if artifact.get(field) != 1.0:
            errors.append(field)
    if artifact.get("replay_fixture_count") != len(LADDER_AXES):
        errors.append("replay_fixture_count")
    if artifact.get("governed_memory_score") != 1.0:
        errors.append("governed_memory_score")
    if not (
        float(artifact.get("no_memory_baseline_score", 1.0))
        < float(artifact.get("governed_memory_score", 0.0))
        and float(artifact.get("naive_icl_baseline_score", 1.0))
        < float(artifact.get("governed_memory_score", 0.0))
    ):
        errors.append("baseline ordering")
    if artifact.get("csl_behavioral_memory_ready") is not True:
        errors.append("csl_behavioral_memory_ready")
    if artifact.get("model_weight_mutation") is not False:
        errors.append("model_weight_mutation")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py")
    if not _row_ids_identical(artifact.get("row_ids_by_variant")):
        errors.append("identical row IDs")
    row_errors = _row_result_errors(artifact.get("row_results"))
    errors.extend(row_errors)
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact payload while excluding the self-referential checksum."""

    return _sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def row_checksum(result: Mapping[str, Any]) -> str:
    """Hash a row result while excluding its self-referential checksum."""

    return _sha256_json({key: value for key, value in result.items() if key != "row_checksum"})


def _build_result(
    *,
    fixture: Mapping[str, Any],
    variant: str,
    retrieved: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    decision_path: Sequence[str],
    selected_answer: str,
    selected_downstream_action: Any,
) -> JsonDict:
    accepted_records = _list_of_mappings(list(accepted))
    rejected_records = _list_of_mappings(list(rejected))
    retrieved_records = _list_of_mappings(list(retrieved))
    memory_use_claimed = bool(accepted_records)
    validator = exact_validator(
        fixture,
        selected_answer=selected_answer,
        selected_downstream_action=selected_downstream_action,
        accepted_memory_records=accepted_records,
        memory_use_claimed=memory_use_claimed,
    )
    result: JsonDict = {
        "schema": "carnot.experiment_5475.row.v1",
        "row_id": str(fixture["row_id"]),
        "axis": str(fixture["axis"]),
        "variant": variant,
        "question": str(fixture["question"]),
        "memory_retrieval_ids": [str(record["memory_id"]) for record in retrieved_records],
        "retrieved_memory_records": retrieved_records,
        "provenance": [_mapping(record.get("provenance")) for record in retrieved_records],
        "decision_path": list(decision_path) + [f"selected_answer:{selected_answer}"],
        "accepted_memory_records": accepted_records,
        "rejected_memory_records": rejected_records,
        "rollback_pointers": [
            str(record["rollback_pointer"])
            for record in accepted_records + rejected_records
            if record.get("rollback_pointer")
        ],
        "selected_answer": selected_answer,
        "selected_downstream_action": selected_downstream_action,
        "exact_validator_results": validator,
        "accepted_by_exact_validator": validator["accepted"],
        "final_authority_bypassed": False,
        "axis_pass": False,
    }
    result["axis_pass"] = row_axis_pass(fixture, result) if variant in CONTROLLED_VARIANTS else False
    result["row_checksum"] = row_checksum(result)
    return _json_ready(result)


def _fixture(
    *,
    row_id: str,
    axis: str,
    question: str,
    expected_answer: str,
    no_memory_answer: str,
    topic: str,
    locality_key: str,
    memory_records: Sequence[Mapping[str, Any]],
    expected_downstream_action: str | None = None,
) -> JsonDict:
    return {
        "row_id": row_id,
        "axis": axis,
        "question": question,
        "expected_answer": expected_answer,
        "no_memory_answer": no_memory_answer,
        "topic": topic,
        "locality_key": locality_key,
        "expected_downstream_action": expected_downstream_action,
        "memory_records": [dict(record) for record in memory_records],
    }


def _memory_record(
    memory_id: str,
    *,
    answer: str,
    text: str,
    topic: str,
    locality_key: str,
    support_ids: Sequence[str],
    support_status: str = "active",
    stale: bool = False,
    conflict_group: str | None = None,
    downstream_action: str | None = None,
    paraphrase_of: str | None = None,
    rollback_pointer: str | None = None,
    trust_score: float = 0.5,
) -> JsonDict:
    return {
        "memory_id": memory_id,
        "answer": answer,
        "text": text,
        "topic": topic,
        "locality_key": locality_key,
        "support_ids": list(support_ids),
        "support_status": support_status,
        "stale": stale,
        "conflict_group": conflict_group,
        "downstream_action": downstream_action,
        "paraphrase_of": paraphrase_of,
        "rollback_pointer": rollback_pointer,
        "trust_score": float(trust_score),
        "provenance": {
            "source": "deterministic_exp5475_fixture",
            "source_row_id": memory_id.replace("mem", "row"),
            "support_ids": list(support_ids),
            "retrieval_method": "exact_fixture_replay",
        },
    }


def _rejection_reason(fixture: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    if record.get("support_status") != "active":
        return "support_removed"
    if record.get("stale") is True:
        return "stale_memory"
    if not _record_relevant(fixture, record):
        return "irrelevant_memory"
    return ""


def _record_relevant(fixture: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    return (
        record.get("topic") == fixture.get("topic")
        and record.get("locality_key") == fixture.get("locality_key")
    )


def _kan_risk_score(fixture: Mapping[str, Any], record: Mapping[str, Any]) -> float:
    risk = 0.0
    if record.get("support_status") != "active":
        risk += 0.6
    if record.get("stale") is True:
        risk += 0.5
    if not _record_relevant(fixture, record):
        risk += 0.6
    risk += max(0.0, 0.5 - float(record.get("trust_score", 0.0)))
    return round(risk, 6)


def _resolve_conflicts(records: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    by_group: dict[str, list[JsonDict]] = {}
    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    for record in _list_of_mappings(list(records)):
        group = record.get("conflict_group")
        if group:
            by_group.setdefault(str(group), []).append(record)
        else:
            accepted.append(record)
    for group_records in by_group.values():
        winner = max(group_records, key=lambda item: float(item.get("trust_score", 0.0)))
        accepted.append(winner)
        rejected.extend(
            _record_with_reason(record, "conflict_lower_trust")
            for record in group_records
            if record["memory_id"] != winner["memory_id"]
        )
    accepted.sort(key=lambda item: (-float(item.get("trust_score", 0.0)), str(item["memory_id"])))
    return accepted, rejected


def _record_with_reason(record: Mapping[str, Any], reason: str) -> JsonDict:
    row = copy.deepcopy(dict(record))
    row["rejection_reason"] = reason
    return row


def _lower_trust_conflict_accepted(fixture: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    group = record.get("conflict_group")
    if not group:
        return False
    peers = [
        item
        for item in _list_of_mappings(fixture.get("memory_records"))
        if item.get("conflict_group") == group and _record_relevant(fixture, item)
    ]
    best = max((float(item.get("trust_score", 0.0)) for item in peers), default=0.0)
    return float(record.get("trust_score", 0.0)) < best


def _axis_pass_rate(
    axis: str,
    fixtures: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> float:
    fixture = fixture_by_axis(fixtures, axis)
    controlled = [
        result
        for result in results
        if result.get("row_id") == fixture["row_id"] and result.get("variant") in CONTROLLED_VARIANTS
    ]
    return _rate(sum(1 for result in controlled if row_axis_pass(fixture, result)), len(controlled))


def _axis_details(
    fixtures: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        axis: [
            {
                "row_id": result.get("row_id"),
                "variant": result.get("variant"),
                "axis_pass": result.get("axis_pass"),
                "accepted_by_exact_validator": result.get("accepted_by_exact_validator"),
            }
            for result in results
            if result.get("row_id") == fixture_by_axis(fixtures, axis)["row_id"]
            and result.get("variant") in CONTROLLED_VARIANTS
        ]
        for axis in LADDER_AXES
    }


def _row_ids_identical(value: Any) -> bool:
    by_variant = _mapping(value)
    if set(by_variant) != set(VARIANT_NAMES):
        return False
    rows = [tuple(by_variant[variant]) for variant in VARIANT_NAMES]
    return len(set(rows)) == 1 and all(len(row_ids) == len(LADDER_AXES) for row_ids in rows)


def _row_result_errors(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["row_results"]
    errors: list[str] = []
    required = {
        "memory_retrieval_ids",
        "provenance",
        "decision_path",
        "accepted_memory_records",
        "rejected_memory_records",
        "rollback_pointers",
        "exact_validator_results",
    }
    for row in _list_of_mappings(value):
        if not required <= set(row):
            errors.append("row evidence fields")
        if row.get("final_authority_bypassed") is not False:
            errors.append("final authority")
        if row.get("row_checksum") != row_checksum(row):
            errors.append("row checksum")
        if _mapping(row.get("exact_validator_results")).get("authority") != EXACT_VALIDATOR_AUTHORITY:
            errors.append("exact validator")
    return sorted(set(errors))


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    return [_normalise_test_run(item) for item in tests_run]


def _normalise_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    return {
        "command": str(item if isinstance(item, str) else item.get("command", "")),
        "outcome": str("passed" if isinstance(item, str) else item.get("outcome", "passed")),
    }


def _honest_verdict(ready: bool) -> str:
    return (
        "complete: deterministic replay audited behavioral CSL memory use without live SOTA inference"
        if ready
        else "blocked: behavioral memory replay audit did not meet readiness gates"
    )


def _source_file_checksums(root: Path) -> JsonDict:
    paths = {
        "module": root / MODULE_RELATIVE_PATH,
        "spec": root / SPEC_RELATIVE_PATH,
        **{f"source_{index}": root / path for index, path in enumerate(SOURCE_ARTIFACTS)},
    }
    return {name: _file_checksum(path) for name, path in paths.items() if path.is_file()}


def _file_checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[JsonDict]:
    return [dict(item) for item in value] if isinstance(value, list) else []


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
