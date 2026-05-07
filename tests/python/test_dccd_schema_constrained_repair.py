"""Tests for DCCD schema-constrained repair executor v2.

Spec: REQ-VERIFY-1428, SCENARIO-VERIFY-1428
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline.certificate_repair_executor import (
    ALLOWED_REPAIR_OUTPUT_SCHEMA,
    CertificateRepairRequest,
)
from carnot.pipeline.dccd_schema_constrained_repair import (
    DCCDRepairConfig,
    DCCDRepairOutputSchemaError,
    DraftConditionedSchemaRepairExecutor,
    build_dccd_repair_prompt,
    build_dccd_retry_prompt,
    classify_dccd_rejection,
    parse_dccd_repair_model_output,
)


def _request(**overrides: Any) -> CertificateRepairRequest:
    values: dict[str, Any] = {
        "case_id": "case_1428",
        "original_prompt": "Question: What is 2 + 2? Reasoning step: 2 + 2 = 5.",
        "current_certificate": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: add bound.",
        "repair_hint": "Repair the localized FoVer reasoning step before accepting.",
        "validator_error": "semantic_result=REPAIR_HINT; repair_required=True",
        "allowed_output_schema": ALLOWED_REPAIR_OUTPUT_SCHEMA,
    }
    values.update(overrides)
    return CertificateRepairRequest(**values)


def _dccd_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "draft_certificate": {
            "certificate_text": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: add bound.",
            "state": "REPAIR_HINT",
        },
        "repair_action": {
            "action_type": "STEP_REWRITE",
            "target": "localized FoVer reasoning step",
            "rationale": "2 + 2 must be 4 before the certificate can be SAT.",
        },
        "final_certificate": {
            "certificate_text": "<CARNOT_CERT_STATE:SAT>\nSAT",
            "state": "SAT",
        },
        "validator_metadata": {
            "expected_semantic_result": "SAT",
            "repair_hint_case_id": "case_1428",
        },
    }
    payload.update(overrides)
    return payload


def test_req1428_prompt_is_draft_conditioned_and_schema_constrained() -> None:
    """REQ-VERIFY-1428: the prompt separates draft, action, final, and metadata."""

    request = _request(original_prompt="x" * 96)
    prompt = build_dccd_repair_prompt(request, DCCDRepairConfig(max_field_chars=40))

    assert "REQ-VERIFY-1428" in prompt
    assert "draft_certificate" in prompt
    assert "repair_action" in prompt
    assert "final_certificate" in prompt
    assert "validator_metadata" in prompt
    assert "x" * 60 not in prompt
    assert "[truncated]" in prompt


def test_req1464_retry_prompt_flag_adds_exact_validation_error_context() -> None:
    """REQ-VERIFY-1464: baseline retry omits the exact error; context retry includes it."""

    request = _request()
    error = "invalid JSON repair output: missing final_certificate.state"
    baseline = build_dccd_retry_prompt(
        request,
        failed_output='{"final_certificate": {"certificate_text": "SAT"}}',
        validation_error_message=error,
        include_validation_error_context=False,
        config=DCCDRepairConfig(max_field_chars=90),
    )
    context = build_dccd_retry_prompt(
        request,
        failed_output='{"final_certificate": {"certificate_text": "SAT"}}',
        validation_error_message=error,
        include_validation_error_context=True,
        config=DCCDRepairConfig(max_field_chars=90),
    )

    assert "REQ-VERIFY-1464" in baseline
    assert "REQ-VERIFY-1464" in context
    assert "failed_model_output" in baseline
    assert "failed_model_output" in context
    assert error not in baseline
    assert error in context
    assert "validation_error_message" not in baseline
    assert "validation_error_message" in context


def test_req1428_parser_accepts_only_bounded_dccd_schema() -> None:
    """REQ-VERIFY-1428: schema validation is stricter than v1 flat JSON."""

    candidate = parse_dccd_repair_model_output(
        "prefix " + json.dumps(_dccd_payload()) + " suffix"
    )

    assert candidate.draft_state == "REPAIR_HINT"
    assert candidate.repair_action_type == "STEP_REWRITE"
    assert candidate.final_state == "SAT"
    assert candidate.final_certificate == "<CARNOT_CERT_STATE:SAT>\nSAT"
    assert candidate.validator_metadata["expected_semantic_result"] == "SAT"

    with pytest.raises(DCCDRepairOutputSchemaError, match="invalid JSON"):
        parse_dccd_repair_model_output("not json")

    with pytest.raises(DCCDRepairOutputSchemaError, match="JSON object"):
        parse_dccd_repair_model_output(json.dumps(["not", "an", "object"]))

    with pytest.raises(DCCDRepairOutputSchemaError, match="required section"):
        parse_dccd_repair_model_output(json.dumps({"final_certificate": {}}))

    with pytest.raises(DCCDRepairOutputSchemaError, match="unexpected"):
        parse_dccd_repair_model_output(
            json.dumps(_dccd_payload(unbounded_analysis="not allowed"))
        )

    with pytest.raises(DCCDRepairOutputSchemaError, match="draft_certificate"):
        parse_dccd_repair_model_output(
            json.dumps(_dccd_payload(draft_certificate="not an object"))
        )

    with pytest.raises(DCCDRepairOutputSchemaError, match="unexpected repair_action"):
        parse_dccd_repair_model_output(
            json.dumps(
                _dccd_payload(
                    repair_action={
                        "action_type": "STEP_REWRITE",
                        "target": "step",
                        "rationale": "fix",
                        "extra": "not allowed",
                    }
                )
            )
        )

    with pytest.raises(DCCDRepairOutputSchemaError, match="final_certificate.certificate_text"):
        parse_dccd_repair_model_output(
            json.dumps(
                _dccd_payload(final_certificate={"certificate_text": "", "state": "SAT"})
            )
        )

    with pytest.raises(DCCDRepairOutputSchemaError, match="exceeds"):
        parse_dccd_repair_model_output(
            json.dumps(
                _dccd_payload(
                    draft_certificate={
                        "certificate_text": "x" * 12,
                        "state": "REPAIR_HINT",
                    }
                )
            ),
            DCCDRepairConfig(max_output_chars=10),
        )

    with pytest.raises(DCCDRepairOutputSchemaError, match="final_certificate"):
        parse_dccd_repair_model_output(
            json.dumps(
                _dccd_payload(
                    draft_certificate={
                        "certificate_text": "draft",
                        "state": "REPAIR_HINT",
                    },
                    final_certificate={
                        "certificate_text": "<CARNOT_CERT_STATE:SAT>\n" + "x" * 12,
                        "state": "SAT",
                    }
                )
            ),
            DCCDRepairConfig(max_output_chars=10),
        )


def test_scenario1428_schema_invalid_candidate_skips_validator_handoff() -> None:
    """SCENARIO-VERIFY-1428: semantic validation never runs before schema validity."""

    validator_calls: list[str] = []

    def validator(_request: CertificateRepairRequest, _candidate: Any) -> dict[str, Any]:
        validator_calls.append("called")
        return {"constraint_passed": True, "semantic_result": "SAT"}

    executor = DraftConditionedSchemaRepairExecutor(
        generator=lambda _prompt: "not json",
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=validator,
    )

    result = executor.attempt(_request())

    assert validator_calls == []
    assert result.accepted is False
    assert result.schema_valid is False
    assert result.semantic_accepted is False
    assert result.fallback_reason == "schema_validation_failed"
    assert result.rejection_reason == "schema_validation_failed"


def test_req1428_timeout_paths_record_timeout_rejection() -> None:
    """REQ-VERIFY-1428: generator and validator timeouts produce one timeout reason."""

    def timeout_generator(_prompt: str) -> str:
        raise TimeoutError("bounded generation timeout")

    generator_timeout_executor = DraftConditionedSchemaRepairExecutor(
        generator=timeout_generator,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=lambda _request, _candidate: {},
    )

    generator_timeout = generator_timeout_executor.attempt(_request())
    assert generator_timeout.accepted is False
    assert generator_timeout.rejection_reason == "timeout"
    assert generator_timeout.schema_valid is False

    def validator_timeout(_request: CertificateRepairRequest, _candidate: Any) -> dict[str, Any]:
        raise TimeoutError("bounded validator timeout")

    validator_timeout_executor = DraftConditionedSchemaRepairExecutor(
        generator=lambda _prompt: json.dumps(_dccd_payload()),
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=validator_timeout,
    )

    validator_timeout_result = validator_timeout_executor.attempt(_request())
    assert validator_timeout_result.accepted is False
    assert validator_timeout_result.rejection_reason == "timeout"
    assert validator_timeout_result.schema_valid is True


def test_scenario1428_schema_valid_candidate_is_handed_to_validator() -> None:
    """SCENARIO-VERIFY-1428: the validator receives the schema-valid final certificate."""

    seen: list[tuple[str, str]] = []

    def validator(request: CertificateRepairRequest, candidate: Any) -> dict[str, Any]:
        seen.append((request.case_id, candidate.final_certificate))
        return {
            "constraint_passed": True,
            "semantic_result": "SAT",
            "repair_required": False,
            "false_acceptance": False,
        }

    executor = DraftConditionedSchemaRepairExecutor(
        generator=lambda _prompt: json.dumps(_dccd_payload()),
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=validator,
    )

    result = executor.attempt(_request())

    assert seen == [("case_1428", "<CARNOT_CERT_STATE:SAT>\nSAT")]
    assert result.accepted is True
    assert result.schema_valid is True
    assert result.semantic_accepted is True
    assert result.corrected_certificate == "<CARNOT_CERT_STATE:SAT>\nSAT"
    assert result.rejection_reason is None


def test_req1428_rejection_classifier_records_semantic_reason() -> None:
    """REQ-VERIFY-1428: semantic failures keep one auditable rejection reason."""

    assert classify_dccd_rejection("timeout", {}) == "timeout"
    assert classify_dccd_rejection("schema_validation_failed", {}) == "schema_validation_failed"
    assert (
        classify_dccd_rejection("generation_or_validation_failed", {})
        == "generation_or_validation_failed"
    )
    assert (
        classify_dccd_rejection(
            "semantic_validation_failed",
            {"fallback_reason": "no_validator_injected", "semantic_result": "REPAIR_HINT"},
        )
        == "validator_mismatch_no_validator_injected"
    )
    assert (
        classify_dccd_rejection(
            "semantic_validation_failed",
            {"constraint_passed": False, "semantic_result": "SAT"},
        )
        == "constraint_failed"
    )
    assert (
        classify_dccd_rejection(
            "semantic_validation_failed",
            {"constraint_passed": True, "semantic_result": "REPAIR_HINT"},
        )
        == "semantic_result_not_sat"
    )
    assert (
        classify_dccd_rejection(
            "semantic_validation_failed",
            {
                "constraint_passed": True,
                "semantic_result": "SAT",
                "repair_required": True,
            },
        )
        == "repair_required_still_true"
    )
    assert (
        classify_dccd_rejection(
            "semantic_validation_failed",
            {
                "constraint_passed": True,
                "semantic_result": "SAT",
                "repair_required": False,
                "false_acceptance": True,
            },
        )
        == "false_acceptance_guard"
    )
    assert (
        classify_dccd_rejection(
            "semantic_validation_failed",
            {
                "constraint_passed": True,
                "semantic_result": "SAT",
                "repair_required": False,
                "false_acceptance": False,
            },
        )
        == "semantic_validation_failed"
    )


def test_req1428_core_executor_has_no_closed_weight_sdk_imports() -> None:
    """REQ-VERIFY-1428: core v2 repair code depends on protocols, not vendor SDKs."""

    source = Path("python/carnot/pipeline/dccd_schema_constrained_repair.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint({"openai", "anthropic", "cohere"})
