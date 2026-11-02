"""Bounded local LLM executor for certificate `REPAIR_HINT` rows.

Spec: REQ-VERIFY-1414, SCENARIO-VERIFY-1414
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


ALLOWED_REPAIR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["corrected_certificate"],
    "properties": {
        "corrected_certificate": {
            "type": "string",
            "description": "Complete tag-first Carnot certificate after the local repair.",
        },
        "corrected_reasoning_step": {
            "type": "string",
            "description": "Optional corrected local reasoning step used to justify the certificate.",
        },
        "metadata": {
            "type": "object",
            "description": "JSON-compatible audit metadata about the bounded repair.",
        },
    },
}

GeneratorFn = Callable[[str], str]
ValidatorFn = Callable[["CertificateRepairRequest", "CertificateRepairCandidate"], Mapping[str, Any]]


class RepairOutputSchemaError(ValueError):
    """Raised when a local model response does not match the allowed schema."""


@dataclass(frozen=True)
class CertificateRepairConfig:
    """Runtime limits for one bounded repair attempt."""

    max_field_chars: int = 1800
    max_output_chars: int = 4000


@dataclass(frozen=True)
class CertificateRepairRequest:
    """Input contract for a single certificate repair attempt."""

    case_id: str
    original_prompt: str
    current_certificate: str
    repair_hint: str
    validator_error: str
    allowed_output_schema: Mapping[str, Any] = field(
        default_factory=lambda: dict(ALLOWED_REPAIR_OUTPUT_SCHEMA)
    )


@dataclass(frozen=True)
class CertificateRepairCandidate:
    """Schema-validated candidate emitted by the local repair model."""

    corrected_certificate: str
    corrected_reasoning_step: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CertificateRepairResult:
    """Accepted or fallback result from one bounded repair attempt."""

    case_id: str
    attempted: bool
    accepted: bool
    corrected_certificate: str | None
    corrected_reasoning_step: str | None
    local_model_used: str | None
    validation_result: dict[str, Any]
    fallback_reason: str | None
    preserved_repair_hint: str
    runtime_s: float


@dataclass
class CertificateRepairPipelineHook:
    """Disabled-by-default opt-in hook for pipeline repair execution."""

    executor: "BoundedLocalLLMCertificateRepairExecutor"
    enabled: bool = False

    def attempt(self, request: CertificateRepairRequest) -> CertificateRepairResult | None:
        """Run the executor only when the caller explicitly enables the hook."""

        if not self.enabled:
            return None
        return self.executor.attempt(request)


class BoundedLocalLLMCertificateRepairExecutor:
    """Execute one local open-weight LLM repair and validate before accepting."""

    def __init__(
        self,
        *,
        generator: GeneratorFn,
        model_spec: Mapping[str, Any],
        validator: ValidatorFn,
        config: CertificateRepairConfig | None = None,
    ) -> None:
        self._generator = generator
        self._model_spec = dict(model_spec)
        self._validator = validator
        self._config = config or CertificateRepairConfig()

    @property
    def model_id(self) -> str | None:
        """Return the auditable local model identifier used by this executor."""

        value = self._model_spec.get("hf_id") or self._model_spec.get("name")
        return str(value) if value else None

    def attempt(self, request: CertificateRepairRequest) -> CertificateRepairResult:
        """Try one bounded repair and accept only after semantic validation."""

        started = time.perf_counter()
        try:
            prompt = build_repair_prompt(request, self._config)
            raw_output = self._generator(prompt)
            candidate = parse_repair_model_output(raw_output, self._config)
            validation = dict(self._validator(request, candidate))
        except RepairOutputSchemaError as exc:
            return self._fallback(request, "schema_validation_failed", started, {"error": str(exc)})
        except TimeoutError as exc:
            return self._fallback(request, "timeout", started, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - live model failures are environment-specific.
            return self._fallback(
                request,
                f"generation_or_validation_failed:{type(exc).__name__}",
                started,
                {"error": str(exc)},
            )

        if not validation_accepts_repair(validation):
            return self._fallback(request, "validation_failed", started, validation)

        return CertificateRepairResult(
            case_id=request.case_id,
            attempted=True,
            accepted=True,
            corrected_certificate=candidate.corrected_certificate,
            corrected_reasoning_step=candidate.corrected_reasoning_step,
            local_model_used=self.model_id,
            validation_result=validation,
            fallback_reason=None,
            preserved_repair_hint=request.repair_hint,
            runtime_s=round(time.perf_counter() - started, 6),
        )

    def _fallback(
        self,
        request: CertificateRepairRequest,
        reason: str,
        started: float,
        validation: Mapping[str, Any],
    ) -> CertificateRepairResult:
        return CertificateRepairResult(
            case_id=request.case_id,
            attempted=True,
            accepted=False,
            corrected_certificate=None,
            corrected_reasoning_step=None,
            local_model_used=self.model_id,
            validation_result=dict(validation),
            fallback_reason=reason,
            preserved_repair_hint=request.repair_hint,
            runtime_s=round(time.perf_counter() - started, 6),
        )


def build_repair_prompt(
    request: CertificateRepairRequest,
    config: CertificateRepairConfig | None = None,
) -> str:
    """Build the bounded JSON-only prompt sent to the local GGUF model."""

    cfg = config or CertificateRepairConfig()
    fields = {
        "case_id": request.case_id,
        "original_prompt": _bounded(request.original_prompt, cfg.max_field_chars),
        "current_certificate": _bounded(request.current_certificate, cfg.max_field_chars),
        "repair_hint": _bounded(request.repair_hint, cfg.max_field_chars),
        "validator_error": _bounded(request.validator_error, cfg.max_field_chars),
    }
    schema = json.dumps(dict(request.allowed_output_schema), sort_keys=True, indent=2)
    payload = json.dumps(fields, sort_keys=True, indent=2)
    return (
        "REQ-VERIFY-1414 bounded Carnot certificate repair for REPAIR_HINT rows.\n"
        "Use only the local context below. Do not call external services. "
        "Return JSON only, with no markdown fences or explanatory prose.\n\n"
        "Input contract:\n"
        f"{payload}\n\n"
        "Allowed output schema:\n"
        f"{schema}\n"
    )


def parse_repair_model_output(
    text: str,
    config: CertificateRepairConfig | None = None,
) -> CertificateRepairCandidate:
    """Parse and validate the JSON candidate emitted by a local repair model."""

    cfg = config or CertificateRepairConfig()
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise RepairOutputSchemaError(f"invalid JSON repair output: {exc}") from exc
    if not isinstance(payload, dict):
        raise RepairOutputSchemaError("repair output must be a JSON object")

    allowed = set(ALLOWED_REPAIR_OUTPUT_SCHEMA["properties"])
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise RepairOutputSchemaError(f"unexpected repair output field(s): {unexpected}")

    certificate = payload.get("corrected_certificate")
    if not isinstance(certificate, str) or not certificate.strip():
        raise RepairOutputSchemaError("corrected_certificate must be a non-empty string")
    if len(certificate) > cfg.max_output_chars:
        raise RepairOutputSchemaError("corrected_certificate exceeds max_output_chars")

    reasoning = payload.get("corrected_reasoning_step")
    if reasoning is not None and not isinstance(reasoning, str):
        raise RepairOutputSchemaError("corrected_reasoning_step must be a string when present")

    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise RepairOutputSchemaError("metadata must be a JSON object when present")

    return CertificateRepairCandidate(
        corrected_certificate=certificate,
        corrected_reasoning_step=reasoning,
        metadata=dict(metadata),
    )


def validation_accepts_repair(validation: Mapping[str, Any]) -> bool:
    """Return True only when the existing semantic/scheduler contract passes."""

    return (
        validation.get("constraint_passed") is True
        and validation.get("semantic_result") == "SAT"
        and validation.get("repair_required") is not True
        and validation.get("false_acceptance") is not True
    )


def _bounded(value: object, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return f"{text[: max(0, max_chars - 12)]}[truncated]"


def _extract_json_object(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return stripped
    return stripped[start : end + 1]
