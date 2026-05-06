"""Tests for the bounded certificate repair executor.

Spec: REQ-VERIFY-1414, SCENARIO-VERIFY-1414
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline.certificate_repair_executor import (
    ALLOWED_REPAIR_OUTPUT_SCHEMA,
    BoundedLocalLLMCertificateRepairExecutor,
    CertificateRepairConfig,
    CertificateRepairPipelineHook,
    CertificateRepairRequest,
    RepairOutputSchemaError,
    build_repair_prompt,
    parse_repair_model_output,
)


def _request(**overrides: Any) -> CertificateRepairRequest:
    values: dict[str, Any] = {
        "case_id": "case_1414",
        "original_prompt": "Question: What is 2 + 2? Reasoning step: 2 + 2 = 5.",
        "current_certificate": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: add bound.",
        "repair_hint": "Repair the localized FoVer reasoning step before accepting.",
        "validator_error": "semantic_result=REPAIR_HINT; repair_required=True",
        "allowed_output_schema": ALLOWED_REPAIR_OUTPUT_SCHEMA,
    }
    values.update(overrides)
    return CertificateRepairRequest(**values)


def test_req1414_prompt_is_bounded_and_contains_schema() -> None:
    """REQ-VERIFY-1414: prompt includes the repair contract within field bounds."""

    request = _request(original_prompt="x" * 120)
    prompt = build_repair_prompt(request, CertificateRepairConfig(max_field_chars=32))

    assert "REQ-VERIFY-1414" in prompt
    assert "REPAIR_HINT" in prompt
    assert '"corrected_certificate"' in prompt
    assert "x" * 40 not in prompt
    assert "[truncated]" in prompt


def test_req1414_parse_rejects_output_outside_allowed_schema() -> None:
    """REQ-VERIFY-1414: model output must match the allowed JSON schema."""

    candidate = parse_repair_model_output(
        "Here is the JSON: "
        + json.dumps(
            {
                "corrected_certificate": "<CARNOT_CERT_STATE:SAT>\nSAT",
                "corrected_reasoning_step": "2 + 2 = 4.",
                "metadata": {"edit": "arithmetic_fix"},
            }
        )
        + " done."
    )
    assert candidate.corrected_certificate == "<CARNOT_CERT_STATE:SAT>\nSAT"
    assert candidate.metadata == {"edit": "arithmetic_fix"}

    with pytest.raises(RepairOutputSchemaError, match="invalid JSON"):
        parse_repair_model_output("not json")

    with pytest.raises(RepairOutputSchemaError, match="JSON object"):
        parse_repair_model_output(json.dumps(["not", "an", "object"]))

    with pytest.raises(RepairOutputSchemaError, match="corrected_certificate"):
        parse_repair_model_output(json.dumps({"metadata": {"edit": "missing_cert"}}))

    with pytest.raises(RepairOutputSchemaError, match="unexpected"):
        parse_repair_model_output(
            json.dumps(
                {
                    "corrected_certificate": "<CARNOT_CERT_STATE:SAT>\nSAT",
                    "vendor_payload": "not allowed",
                }
            )
        )

    with pytest.raises(RepairOutputSchemaError, match="exceeds"):
        parse_repair_model_output(
            json.dumps({"corrected_certificate": "too long"}),
            CertificateRepairConfig(max_output_chars=3),
        )

    with pytest.raises(RepairOutputSchemaError, match="corrected_reasoning_step"):
        parse_repair_model_output(
            json.dumps(
                {
                    "corrected_certificate": "<CARNOT_CERT_STATE:SAT>\nSAT",
                    "corrected_reasoning_step": ["bad"],
                }
            )
        )

    with pytest.raises(RepairOutputSchemaError, match="metadata"):
        parse_repair_model_output(
            json.dumps(
                {
                    "corrected_certificate": "<CARNOT_CERT_STATE:SAT>\nSAT",
                    "metadata": ["bad"],
                }
            )
        )


def test_req1414_executor_accepts_only_semantically_validated_repairs() -> None:
    """REQ-VERIFY-1414: validation failure preserves the original repair hint."""

    def generator(_prompt: str) -> str:
        return json.dumps(
            {
                "corrected_certificate": "<CARNOT_CERT_STATE:SAT>\nSAT",
                "corrected_reasoning_step": "2 + 2 = 4.",
                "metadata": {"edit": "arithmetic_fix"},
            }
        )

    def validator(_request: CertificateRepairRequest, _candidate: Any) -> dict[str, Any]:
        return {
            "constraint_passed": False,
            "semantic_result": "REPAIR_HINT",
            "repair_required": True,
            "false_acceptance": False,
        }

    executor = BoundedLocalLLMCertificateRepairExecutor(
        generator=generator,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=validator,
    )

    result = executor.attempt(_request())

    assert result.accepted is False
    assert result.corrected_certificate is None
    assert result.fallback_reason == "validation_failed"
    assert result.preserved_repair_hint == "Repair the localized FoVer reasoning step before accepting."


def test_req1414_executor_fallbacks_for_schema_error_and_timeout() -> None:
    """REQ-VERIFY-1414: invalid model output and timeouts produce fallback results."""

    def validator(_request: CertificateRepairRequest, _candidate: Any) -> dict[str, Any]:
        return {}

    schema_executor = BoundedLocalLLMCertificateRepairExecutor(
        generator=lambda _prompt: "not json",
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=validator,
    )
    schema_result = schema_executor.attempt(_request())
    assert schema_result.accepted is False
    assert schema_result.fallback_reason == "schema_validation_failed"

    def timeout_generator(_prompt: str) -> str:
        raise TimeoutError("bounded timeout")

    timeout_executor = BoundedLocalLLMCertificateRepairExecutor(
        generator=timeout_generator,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=validator,
    )
    timeout_result = timeout_executor.attempt(_request())
    assert timeout_result.accepted is False
    assert timeout_result.fallback_reason == "timeout"


def test_scenario1414_executor_success_and_pipeline_hook_opt_in() -> None:
    """SCENARIO-VERIFY-1414: the pipeline hook is disabled until explicitly enabled."""

    calls: list[str] = []

    def generator(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps({"corrected_certificate": "<CARNOT_CERT_STATE:SAT>\nSAT"})

    def validator(_request: CertificateRepairRequest, _candidate: Any) -> dict[str, Any]:
        return {
            "constraint_passed": True,
            "semantic_result": "SAT",
            "repair_required": False,
            "false_acceptance": False,
            "full_pipeline_pass": True,
        }

    executor = BoundedLocalLLMCertificateRepairExecutor(
        generator=generator,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=validator,
    )

    disabled_hook = CertificateRepairPipelineHook(executor=executor, enabled=False)
    assert disabled_hook.attempt(_request()) is None
    assert calls == []

    enabled_hook = CertificateRepairPipelineHook(executor=executor, enabled=True)
    result = enabled_hook.attempt(_request())

    assert result is not None
    assert result.accepted is True
    assert result.corrected_certificate == "<CARNOT_CERT_STATE:SAT>\nSAT"
    assert result.local_model_used == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert len(calls) == 1


def test_req1414_core_executor_has_no_closed_weight_sdk_imports() -> None:
    """REQ-VERIFY-1414: core repair code does not import closed-weight SDKs."""

    source = Path("python/carnot/pipeline/certificate_repair_executor.py").read_text(
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
