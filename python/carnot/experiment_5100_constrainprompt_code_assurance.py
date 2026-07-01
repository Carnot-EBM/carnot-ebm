"""Exp 5100: exact code assurance for prompt-defined verdict constraints.

Spec refs: REQ-VERIFY-5100, SCENARIO-VERIFY-5100.

This module turns a small set of plain-English prompt constraints into a fixed
logical tree of Python checks over verifier verdict JSON. The important
boundary is that acceptance comes only from local JSON parsing and explicit
field checks. A language model may help write future proposals, but this
prototype does not invoke one and never treats model judgment as verifier
authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]
CheckFn = Callable[[Any], tuple[bool, str | None]]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5100
EXPERIMENT_NAME = "experiment_5100_constrainprompt_code_assurance"
SCHEMA = "carnot.experiment_5100_constrainprompt_code_assurance.v468"
RESULT_RELATIVE_PATH = "results/experiment_5100_constrainprompt_code_assurance_v468.json"
SPEC_REFS = ["REQ-VERIFY-5100", "SCENARIO-VERIFY-5100"]
RUN_DATE = "20260701"
RANDOM_SEED = 20260701

SCHEMA_NAME = "verifier_verdict_schema_v1"
SCHEMA_PATH = (
    "python/carnot/experiment_5100_constrainprompt_code_assurance.py::"
    "VERIFIER_VERDICT_SCHEMA"
)
PARSER_BACKEND = "python_json_parser"
EXACT_CHECKER_BACKEND = "python_json_logical_tree"
INFERENCE_SUBSTRATE = "deterministic_python_json_logical_tree"

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "verdict",
    "confidence",
    "evidence_label",
    "evidence_refs",
    "claim_id",
    "checker_backend",
    "duration_s",
    "rationale",
)
ALLOWED_VERDICTS = frozenset({"accept", "reject", "abstain"})
ALLOWED_CONFIDENCE = frozenset({"low", "medium", "high"})
ACCEPT_EVIDENCE = frozenset({"schema_valid", "solver_verified"})
REJECT_EVIDENCE = frozenset(
    {"schema_missing_field", "solver_counterexample", "arithmetic_mismatch", "citation_gap"}
)
ABSTAIN_EVIDENCE = frozenset({"unsupported_prompt", "ambiguous_instruction"})
EVIDENCE_URI_RE = re.compile(r"^evidence://[a-z0-9_./:-]+$")
CLAIM_ID_RE = re.compile(r"^claim-[0-9]{4}$")

VERIFIER_VERDICT_SCHEMA: JsonDict = {
    "schema_name": SCHEMA_NAME,
    "canonicalization": "json.dumps(sort_keys=True,separators=(',',':'))",
    "fields": {
        "schema": {"type": "string", "const": SCHEMA_NAME},
        "verdict": {"type": "string", "enum": sorted(ALLOWED_VERDICTS)},
        "confidence": {"type": "string", "enum": sorted(ALLOWED_CONFIDENCE)},
        "evidence_label": {
            "type": "string",
            "enum": sorted(ACCEPT_EVIDENCE | REJECT_EVIDENCE | ABSTAIN_EVIDENCE),
        },
        "evidence_refs": {"type": "array", "items": "evidence_uri"},
        "claim_id": {"type": "string", "pattern": CLAIM_ID_RE.pattern},
        "checker_backend": {"type": "string", "const": EXACT_CHECKER_BACKEND},
        "duration_s": {"type": "number", "minimum": 0.0, "maximum": 30.0},
        "rationale": {"type": "string", "min_length": 12, "max_length": 240},
    },
}

MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "preferred_quant": "Q4_K_M",
    },
)
MANDATED_MODEL_IDS = tuple(row["hf_id"] for row in MODEL_SPECS)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "Terminal prefix states whether exact code checks accepted every prompt constraint or only a partial subset."
    },
    "duration_s": {
        "principle": "Wall-clock duration for deterministic JSON parsing, logical-tree checks, fixture evaluation, and artifact assembly."
    },
    "inference_substrate": {
        "principle": "Declares the actual deterministic Python JSON checker; live_llm_inference is forbidden when llm_invoked=false."
    },
    "preconditions_checked": {
        "principle": "Records schema path, prompt constraints, parser/checker backend, and LLM invocation state before checks run."
    },
    "model_specs": {
        "principle": "Carries the three mandated GGUF IDs as required methodology declarations without implying they were invoked."
    },
    "schema_name": {
        "principle": "Names the finite verifier verdict schema under exact code assurance."
    },
    "constraints_total": {
        "principle": "Counts selected prompt-level constraints before compilation."
    },
    "executable_constraints_total": {
        "principle": "Counts constraints that compiled to allow-listed executable checks."
    },
    "positive_tests_passed": {
        "principle": "True only when every known-good fixture is accepted by executable checks."
    },
    "negative_tests_passed": {
        "principle": "True only when every known-bad fixture is rejected with named failing checks."
    },
    "adversarial_tests_passed": {
        "principle": "True only when no-op, schema-spoof, and LLM-judge substitution fixtures are rejected."
    },
    "rejected_constraints": {
        "principle": "Lists prompt constraints not accepted into executable authority, rather than silently trusting them."
    },
    "llm_invoked": {
        "principle": "False for this prototype; any future true value must be accompanied by mandated GGUF proposal provenance and passing tests."
    },
    "exact_checker_backend": {
        "principle": "Names the executable Python logical tree that owns acceptance."
    },
    "flagged_adversarial": {
        "principle": "False only when fixture gates pass and the artifact contains no live-LLM or checker-authority contradiction."
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class PromptConstraint:
    """A prompt-level rule and the allow-listed check that can enforce it."""

    constraint_id: str
    prompt: str
    check_name: str


@dataclass(frozen=True)
class CompiledConstraintSet:
    """Executable logical tree plus any constraints refused by the compiler."""

    executable_constraints: tuple[PromptConstraint, ...]
    rejected_constraints: list[JsonDict]
    evaluation_tree: JsonDict


def build_prompt_constraints() -> tuple[PromptConstraint, ...]:
    """Select nine prompt-level constraints for the finite verdict schema."""

    return (
        PromptConstraint(
            "required_fields_and_schema",
            "Verdict JSON must be an object with all required fields and exact schema name.",
            "required_fields_and_schema",
        ),
        PromptConstraint(
            "verdict_enum",
            "The verdict field must be exactly accept, reject, or abstain.",
            "verdict_enum",
        ),
        PromptConstraint(
            "confidence_enum_and_mapping",
            "Confidence must be low, medium, or high; decisive verdicts cannot be low.",
            "confidence_enum_and_mapping",
        ),
        PromptConstraint(
            "verdict_evidence_consistency",
            "Evidence labels must match accept, reject, or abstain verdict semantics.",
            "verdict_evidence_consistency",
        ),
        PromptConstraint(
            "evidence_refs_for_decisive_verdicts",
            "Accept and reject verdicts must cite at least one evidence:// reference.",
            "evidence_refs_for_decisive_verdicts",
        ),
        PromptConstraint(
            "claim_id_format",
            "Claim IDs must use the stable claim-0000 decimal format.",
            "claim_id_format",
        ),
        PromptConstraint(
            "checker_backend_exact",
            "The checker_backend field must name the exact Python logical-tree checker.",
            "checker_backend_exact",
        ),
        PromptConstraint(
            "duration_bounds",
            "Verifier duration must be a measured finite number from 0 to 30 seconds.",
            "duration_bounds",
        ),
        PromptConstraint(
            "rationale_bounded",
            "Rationale must be present, bounded, and non-trivial.",
            "rationale_bounded",
        ),
    )


def compile_constraints(constraints: Sequence[PromptConstraint]) -> CompiledConstraintSet:
    """Compile allow-listed prompt constraints into a deterministic tree."""

    executable: list[PromptConstraint] = []
    rejected: list[JsonDict] = []
    for constraint in constraints:
        if constraint.check_name in CHECKS:
            executable.append(constraint)
        else:
            rejected.append(
                {
                    "constraint_id": constraint.constraint_id,
                    "reason": "no_allowlisted_executable_check",
                    "prompt": constraint.prompt,
                }
            )
    tree = {
        "backend": EXACT_CHECKER_BACKEND,
        "root": {"op": "all", "children": [row.constraint_id for row in executable]},
        "nodes": [
            {
                "node_id": row.constraint_id,
                "check_name": row.check_name,
                "authority": "python_exact_check",
                "prompt": row.prompt,
            }
            for row in executable
        ],
    }
    return CompiledConstraintSet(
        executable_constraints=tuple(executable),
        rejected_constraints=rejected,
        evaluation_tree=tree,
    )


def evaluate_candidate(
    candidate: str | Mapping[str, Any],
    compiled: CompiledConstraintSet,
) -> JsonDict:
    """Evaluate one candidate with local JSON parsing and exact checks."""

    payload: Any = candidate
    if isinstance(candidate, str):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return {
                "accepted": False,
                "failing_constraints": ["json_parse_error"],
                "rejection_reasons": ["json_parse_error"],
                "constraint_results": [
                    {
                        "constraint_id": "json_parse_error",
                        "accepted": False,
                        "reason": "json_parse_error",
                    }
                ],
                "llm_judge_used": False,
            }

    constraint_results = []
    failing_constraints = []
    rejection_reasons = []
    for constraint in compiled.executable_constraints:
        accepted, reason = CHECKS[constraint.check_name](payload)
        result = {
            "constraint_id": constraint.constraint_id,
            "accepted": accepted,
            "reason": reason,
        }
        constraint_results.append(result)
        if not accepted:
            failing_constraints.append(constraint.constraint_id)
            if reason is not None:
                rejection_reasons.append(reason)
    return {
        "accepted": not failing_constraints,
        "failing_constraints": failing_constraints,
        "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
        "constraint_results": constraint_results,
        "llm_judge_used": False,
    }


def build_fixture_sets() -> dict[str, list[JsonDict]]:
    """Return positive, negative, and adversarial/no-op verdict fixtures."""

    accept = _base_candidate(
        verdict="accept",
        confidence="high",
        evidence_label="solver_verified",
        evidence_refs=["evidence://solver/z3/transcript-0001"],
        claim_id="claim-1001",
        rationale="Z3 solver transcript verifies the claim under the finite schema.",
    )
    reject = _base_candidate(
        verdict="reject",
        confidence="medium",
        evidence_label="schema_missing_field",
        evidence_refs=["evidence://runtime/json/missing-field-0002"],
        claim_id="claim-1002",
        rationale="Runtime JSON validation found a required verifier verdict field missing.",
    )
    return {
        "positive": [
            {"fixture_id": "accept_solver_verified", "candidate": accept},
            {"fixture_id": "reject_schema_missing", "candidate": reject},
        ],
        "negative": [
            {"fixture_id": "missing_schema_field", "candidate": _without(accept, "schema")},
            {"fixture_id": "unknown_verdict", "candidate": _with(accept, verdict="maybe")},
            {
                "fixture_id": "low_confidence_accept",
                "candidate": _with(accept, confidence="low"),
            },
            {
                "fixture_id": "accept_with_reject_evidence",
                "candidate": _with(accept, evidence_label="citation_gap"),
            },
            {"fixture_id": "empty_decisive_evidence_refs", "candidate": _with(accept, evidence_refs=[])},
            {"fixture_id": "bad_claim_id", "candidate": _with(accept, claim_id="claim-x")},
            {
                "fixture_id": "wrong_checker_backend",
                "candidate": _with(accept, checker_backend="llm_judge"),
            },
            {"fixture_id": "negative_duration", "candidate": _with(accept, duration_s=-0.5)},
            {"fixture_id": "short_rationale", "candidate": _with(accept, rationale="ok")},
        ],
        "adversarial": [
            {
                "fixture_id": "noop_success_empty_evidence",
                "candidate": _with(accept, evidence_refs=[], rationale="ok"),
            },
            {
                "fixture_id": "llm_judge_substitution",
                "candidate": _with(accept, checker_backend="llm_judge"),
            },
            {
                "fixture_id": "schema_spoof_with_good_shape",
                "candidate": _with(accept, schema=f"{SCHEMA_NAME}_v2"),
            },
            {"fixture_id": "non_object_noop_json", "candidate": "[]"},
        ],
    }


def build_preconditions() -> JsonDict:
    """Record preconditions before any constraint acceptance decision."""

    return {
        "schema_path": SCHEMA_PATH,
        "schema_name": SCHEMA_NAME,
        "schema_fields": list(REQUIRED_SCHEMA_FIELDS),
        "prompt_constraints": [asdict(row) for row in build_prompt_constraints()],
        "parser_backend": PARSER_BACKEND,
        "checker_backend": EXACT_CHECKER_BACKEND,
        "llm_invoked": False,
        "llm_proposal_path": "disabled_for_exact_checker_prototype",
    }


def build_model_specs(*, llm_invoked: bool) -> list[JsonDict]:
    """Return mandated model declarations without implying unused inference."""

    status = "required_if_invoked" if llm_invoked else "not_invoked_reference_only"
    return [dict(row, invocation_status=status) for row in MODEL_SPECS]


def run(
    *,
    root: Path | str = REPO_ROOT,
    validation_commands: Sequence[str] = (),
) -> JsonDict:
    """Build the deterministic Exp 5100 artifact in memory."""

    del root
    started = time.perf_counter()
    constraints = build_prompt_constraints()
    compiled = compile_constraints(constraints)
    fixture_results = _evaluate_fixture_sets(compiled, build_fixture_sets())
    positive_passed = all(
        row["accepted"] is True for row in fixture_results if row["group"] == "positive"
    )
    negative_passed = all(
        row["accepted"] is False and row["failing_constraints"]
        for row in fixture_results
        if row["group"] == "negative"
    )
    adversarial_passed = all(
        row["accepted"] is False and row["failing_constraints"]
        for row in fixture_results
        if row["group"] == "adversarial"
    )
    all_executable = len(compiled.executable_constraints) == len(constraints)
    flagged = not (positive_passed and negative_passed and adversarial_passed)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "result_path": RESULT_RELATIVE_PATH,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": (
            "success_constrainprompt_code_assurance_exact_checks_passed"
            if all_executable and not flagged
            else "complete_constrainprompt_assurance_partial_constraints_only"
        ),
        "duration_s": round(time.perf_counter() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": build_preconditions(),
        "model_specs": build_model_specs(llm_invoked=False),
        "schema_name": SCHEMA_NAME,
        "constraints_total": len(constraints),
        "executable_constraints_total": len(compiled.executable_constraints),
        "positive_tests_passed": positive_passed,
        "negative_tests_passed": negative_passed,
        "adversarial_tests_passed": adversarial_passed,
        "rejected_constraints": compiled.rejected_constraints,
        "llm_invoked": False,
        "exact_checker_backend": EXACT_CHECKER_BACKEND,
        "flagged_adversarial": flagged,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_verdict_schema": VERIFIER_VERDICT_SCHEMA,
        "prompt_constraints": [asdict(row) for row in constraints],
        "evaluation_tree": compiled.evaluation_tree,
        "fixture_summary": _fixture_summary(fixture_results),
        "fixture_results": fixture_results,
        "validation_commands": list(validation_commands),
    }
    artifact["reproducibility_checksum"] = _checksum_for_artifact(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str | None = None,
    validation_commands: Sequence[str] = (),
) -> JsonDict:
    """Build, validate, and write the Exp 5100 JSON artifact."""

    active_root = Path(root)
    artifact = run(root=active_root, validation_commands=validation_commands)
    destination = Path(output_path) if output_path is not None else active_root / RESULT_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate that an Exp 5100 terminal artifact is internally coherent."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        field not in principles for field in REQUIRED_ARTIFACT_FIELDS
    ):
        raise ValueError("field_principles missing required principle annotations")
    if not str(artifact["honest_verdict"]).startswith(
        "success_constrainprompt_code_assurance_exact_checks_passed"
    ):
        raise ValueError("honest_verdict must use the success terminal prefix")
    if not isinstance(artifact["duration_s"], (int, float)) or artifact["duration_s"] < 0:
        raise ValueError("duration_s must be non-negative")
    if artifact.get("llm_invoked") is False and artifact["inference_substrate"] == "live_llm_inference":
        raise ValueError("live_llm_inference cannot be claimed when llm_invoked=false")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must name the deterministic checker")
    if artifact["schema_name"] != SCHEMA_NAME:
        raise ValueError("schema_name does not match the finite verdict schema")
    if not 5 <= int(artifact["constraints_total"]) <= 10:
        raise ValueError("constraints_total must be between five and ten")
    if artifact["executable_constraints_total"] != artifact["constraints_total"]:
        raise ValueError("executable_constraints_total must match constraints_total")
    if artifact["positive_tests_passed"] is not True:
        raise ValueError("positive_tests_passed must be true")
    if artifact["negative_tests_passed"] is not True:
        raise ValueError("negative_tests_passed must be true")
    if artifact["adversarial_tests_passed"] is not True:
        raise ValueError("adversarial_tests_passed must be true")
    if artifact["llm_invoked"] is not False:
        raise ValueError("llm_invoked must be false for this deterministic prototype")
    if artifact["exact_checker_backend"] != EXACT_CHECKER_BACKEND:
        raise ValueError("exact_checker_backend must be python_json_logical_tree")
    if artifact["flagged_adversarial"] is not False:
        raise ValueError("flagged_adversarial must remain false for the success artifact")
    preconditions = artifact.get("preconditions_checked")
    if not _preconditions_valid(preconditions):
        raise ValueError("preconditions_checked must record schema, constraints, backend, and LLM state")
    if {row.get("hf_id") for row in artifact["model_specs"]} != set(MANDATED_MODEL_IDS):
        raise ValueError("model_specs must include all mandated GGUF IDs")
    summary = artifact.get("fixture_summary")
    if not _fixture_summary_valid(summary):
        raise ValueError("fixture_summary must record all fixture groups and pass flags")
    if artifact["rejected_constraints"] != []:
        raise ValueError("rejected_constraints must be empty for the success verdict")


def main() -> int:
    """CLI entrypoint used by the conductor and focused tests."""

    root = Path(os.environ.get("CARNOT_EXP5100_ROOT", str(REPO_ROOT)))
    write_artifact(root=root)
    return 0


def _base_candidate(
    *,
    verdict: str,
    confidence: str,
    evidence_label: str,
    evidence_refs: list[str],
    claim_id: str,
    rationale: str,
) -> JsonDict:
    return {
        "schema": SCHEMA_NAME,
        "verdict": verdict,
        "confidence": confidence,
        "evidence_label": evidence_label,
        "evidence_refs": list(evidence_refs),
        "claim_id": claim_id,
        "checker_backend": EXACT_CHECKER_BACKEND,
        "duration_s": 0.125,
        "rationale": rationale,
    }


def _with(candidate: Mapping[str, Any], **updates: Any) -> JsonDict:
    payload = dict(candidate)
    payload.update(updates)
    return payload


def _without(candidate: Mapping[str, Any], field: str) -> JsonDict:
    payload = dict(candidate)
    payload.pop(field, None)
    return payload


def _check_required_fields_and_schema(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "candidate_not_json_object"
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in payload]
    if missing:
        return False, "missing_required_field"
    if payload.get("schema") != SCHEMA_NAME:
        return False, "schema_name_mismatch"
    return True, None


def _check_verdict_enum(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "candidate_not_json_object"
    if payload.get("verdict") not in ALLOWED_VERDICTS:
        return False, "verdict_not_allowed"
    return True, None


def _check_confidence_enum_and_mapping(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "candidate_not_json_object"
    confidence = payload.get("confidence")
    verdict = payload.get("verdict")
    if confidence not in ALLOWED_CONFIDENCE:
        return False, "confidence_not_allowed"
    if verdict in {"accept", "reject"} and confidence == "low":
        return False, "decisive_verdict_confidence_too_low"
    if verdict == "abstain" and confidence != "low":
        return False, "abstain_confidence_not_low"
    return True, None


def _check_verdict_evidence_consistency(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "candidate_not_json_object"
    verdict = payload.get("verdict")
    evidence_label = payload.get("evidence_label")
    allowed_by_verdict = {
        "accept": ACCEPT_EVIDENCE,
        "reject": REJECT_EVIDENCE,
        "abstain": ABSTAIN_EVIDENCE,
    }.get(verdict, frozenset())
    if evidence_label not in allowed_by_verdict:
        return False, "evidence_label_inconsistent_with_verdict"
    return True, None


def _check_evidence_refs_for_decisive_verdicts(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "candidate_not_json_object"
    evidence_refs = payload.get("evidence_refs")
    if not isinstance(evidence_refs, list):
        return False, "evidence_refs_not_list"
    if not all(isinstance(ref, str) and EVIDENCE_URI_RE.match(ref) for ref in evidence_refs):
        return False, "evidence_ref_invalid"
    if payload.get("verdict") in {"accept", "reject"} and not evidence_refs:
        return False, "decisive_verdict_missing_evidence_ref"
    return True, None


def _check_claim_id_format(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "candidate_not_json_object"
    claim_id = payload.get("claim_id")
    if not isinstance(claim_id, str) or not CLAIM_ID_RE.match(claim_id):
        return False, "claim_id_format_invalid"
    return True, None


def _check_checker_backend_exact(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "candidate_not_json_object"
    if payload.get("checker_backend") != EXACT_CHECKER_BACKEND:
        return False, "checker_backend_not_exact"
    return True, None


def _check_duration_bounds(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "candidate_not_json_object"
    duration = payload.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return False, "duration_not_number"
    if duration < 0.0 or duration > 30.0:
        return False, "duration_out_of_bounds"
    return True, None


def _check_rationale_bounded(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "candidate_not_json_object"
    rationale = payload.get("rationale")
    if not isinstance(rationale, str):
        return False, "rationale_not_string"
    if len(rationale.strip()) < 12 or len(rationale) > 240:
        return False, "rationale_length_out_of_bounds"
    return True, None


CHECKS: dict[str, CheckFn] = {
    "required_fields_and_schema": _check_required_fields_and_schema,
    "verdict_enum": _check_verdict_enum,
    "confidence_enum_and_mapping": _check_confidence_enum_and_mapping,
    "verdict_evidence_consistency": _check_verdict_evidence_consistency,
    "evidence_refs_for_decisive_verdicts": _check_evidence_refs_for_decisive_verdicts,
    "claim_id_format": _check_claim_id_format,
    "checker_backend_exact": _check_checker_backend_exact,
    "duration_bounds": _check_duration_bounds,
    "rationale_bounded": _check_rationale_bounded,
}


def _evaluate_fixture_sets(
    compiled: CompiledConstraintSet,
    fixture_sets: Mapping[str, Sequence[JsonDict]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for group, fixtures in fixture_sets.items():
        for fixture in fixtures:
            feedback = evaluate_candidate(fixture["candidate"], compiled)
            rows.append(
                {
                    "group": group,
                    "fixture_id": fixture["fixture_id"],
                    "accepted": feedback["accepted"],
                    "failing_constraints": feedback["failing_constraints"],
                    "rejection_reasons": feedback["rejection_reasons"],
                    "expected_accepted": group == "positive",
                }
            )
    return rows


def _fixture_summary(fixture_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    groups = sorted({str(row["group"]) for row in fixture_results})
    return {
        "groups": groups,
        "positive_count": sum(row["group"] == "positive" for row in fixture_results),
        "negative_count": sum(row["group"] == "negative" for row in fixture_results),
        "adversarial_count": sum(row["group"] == "adversarial" for row in fixture_results),
        "positive_tests_passed": all(
            row["accepted"] is True for row in fixture_results if row["group"] == "positive"
        ),
        "negative_tests_passed": all(
            row["accepted"] is False for row in fixture_results if row["group"] == "negative"
        ),
        "adversarial_tests_passed": all(
            row["accepted"] is False for row in fixture_results if row["group"] == "adversarial"
        ),
    }


def _fixture_summary_valid(summary: Any) -> bool:
    return isinstance(summary, Mapping) and summary.get("groups") == [
        "adversarial",
        "negative",
        "positive",
    ] and summary.get("positive_tests_passed") is True and summary.get(
        "negative_tests_passed"
    ) is True and summary.get("adversarial_tests_passed") is True


def _preconditions_valid(preconditions: Any) -> bool:
    return (
        isinstance(preconditions, Mapping)
        and preconditions.get("schema_path") == SCHEMA_PATH
        and preconditions.get("schema_name") == SCHEMA_NAME
        and bool(preconditions.get("prompt_constraints"))
        and preconditions.get("parser_backend") == PARSER_BACKEND
        and preconditions.get("checker_backend") == EXACT_CHECKER_BACKEND
        and preconditions.get("llm_invoked") is False
    )


def _checksum_for_artifact(artifact: Mapping[str, Any]) -> str:
    stable = {
        "schema_name": artifact["schema_name"],
        "prompt_constraints": artifact["prompt_constraints"],
        "evaluation_tree": artifact["evaluation_tree"],
        "fixture_results": artifact["fixture_results"],
        "model_ids": [row["hf_id"] for row in artifact["model_specs"]],
        "exact_checker_backend": artifact["exact_checker_backend"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
