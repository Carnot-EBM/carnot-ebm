"""Draft-conditioned, schema-constrained repair executor v2.

The v1 certificate repair executor accepted a flat JSON object and then handed
the candidate to semantic validation. Exp 1427 showed that this left too much
room for empty prose, malformed JSON, and missing validator handoff evidence.
This module makes the repair shape explicit: the local model must echo the
draft certificate, describe the repair action, emit the final certificate, and
carry validator metadata in separate schema sections before semantic validation
is allowed to run.

Spec: REQ-VERIFY-1428, SCENARIO-VERIFY-1428
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from carnot.pipeline.certificate_repair_executor import (
    CertificateRepairRequest,
    validation_accepts_repair,
)


DCCD_REPAIR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "draft_certificate",
        "repair_action",
        "final_certificate",
        "validator_metadata",
    ],
    "properties": {
        "draft_certificate": {
            "type": "object",
            "additionalProperties": False,
            "required": ["certificate_text", "state"],
            "properties": {
                "certificate_text": {"type": "string"},
                "state": {"type": "string"},
            },
        },
        "repair_action": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action_type", "target", "rationale"],
            "properties": {
                "action_type": {"type": "string"},
                "target": {"type": "string"},
                "rationale": {"type": "string"},
            },
        },
        "final_certificate": {
            "type": "object",
            "additionalProperties": False,
            "required": ["certificate_text", "state"],
            "properties": {
                "certificate_text": {"type": "string"},
                "state": {"type": "string"},
            },
        },
        "validator_metadata": {
            "type": "object",
            "description": "JSON-compatible metadata for validator replay.",
        },
    },
}
"""Bounded DCCD repair schema used before semantic validation.

The schema is represented as a plain dict so prompts can embed it directly and
tests can inspect it without needing the optional ``jsonschema`` package.
"""

DCCD_REJECTION_REASONS: tuple[str, ...] = (
    "schema_validation_failed",
    "timeout",
    "generation_or_validation_failed",
    "validator_mismatch_no_validator_injected",
    "constraint_failed",
    "semantic_result_not_sat",
    "repair_required_still_true",
    "false_acceptance_guard",
    "semantic_validation_failed",
)
"""Stable rejection reasons emitted by repair executor v2."""

GeneratorFn = Callable[[str], str]
ValidatorFn = Callable[[CertificateRepairRequest, "DCCDRepairCandidate"], Mapping[str, Any]]


class DCCDRepairOutputSchemaError(ValueError):
    """Raised when the model output fails the DCCD repair schema."""


@dataclass(frozen=True)
class DCCDRepairConfig:
    """Runtime bounds for one DCCD repair attempt."""

    max_field_chars: int = 1800
    max_output_chars: int = 4000


@dataclass(frozen=True)
class DCCDRepairCandidate:
    """Schema-valid repair candidate ready for validator handoff."""

    draft_certificate: str
    draft_state: str
    repair_action_type: str
    repair_target: str
    repair_rationale: str
    final_certificate: str
    final_state: str
    validator_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DCCDRepairResult:
    """Accepted or rejected result from one schema-first repair attempt."""

    case_id: str
    attempted: bool
    accepted: bool
    schema_valid: bool
    semantic_accepted: bool
    corrected_certificate: str | None
    local_model_used: str | None
    validation_result: dict[str, Any]
    fallback_reason: str | None
    rejection_reason: str | None
    preserved_repair_hint: str
    runtime_s: float
    candidate: DCCDRepairCandidate | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit row for experiment artifacts."""

        payload = asdict(self)
        if self.candidate is not None:
            payload["candidate"] = asdict(self.candidate)
        return payload


class DraftConditionedSchemaRepairExecutor:
    """Run one local repair candidate through schema validation, then semantics."""

    def __init__(
        self,
        *,
        generator: GeneratorFn,
        model_spec: Mapping[str, Any],
        validator: ValidatorFn,
        config: DCCDRepairConfig | None = None,
    ) -> None:
        self._generator = generator
        self._model_spec = dict(model_spec)
        self._validator = validator
        self._config = config or DCCDRepairConfig()

    @property
    def model_id(self) -> str | None:
        """Return the auditable local model identifier selected for this attempt."""

        value = self._model_spec.get("hf_id") or self._model_spec.get("name")
        return str(value) if value else None

    def attempt(self, request: CertificateRepairRequest) -> DCCDRepairResult:
        """Try one repair and reject before validator handoff when schema fails."""

        started = time.perf_counter()
        try:
            prompt = build_dccd_repair_prompt(request, self._config)
            raw_output = self._generator(prompt)
            candidate = parse_dccd_repair_model_output(raw_output, self._config)
        except DCCDRepairOutputSchemaError as exc:
            return self._fallback(
                request,
                "schema_validation_failed",
                started,
                {"error": str(exc)},
                schema_valid=False,
                candidate=None,
            )
        except TimeoutError as exc:
            return self._fallback(
                request,
                "timeout",
                started,
                {"error": str(exc)},
                schema_valid=False,
                candidate=None,
            )
        except Exception as exc:  # pragma: no cover - live model failures vary by host.
            return self._fallback(
                request,
                "generation_or_validation_failed",
                started,
                {"error": f"{type(exc).__name__}: {exc}"},
                schema_valid=False,
                candidate=None,
            )

        try:
            validation = dict(self._validator(request, candidate))
        except TimeoutError as exc:
            return self._fallback(
                request,
                "timeout",
                started,
                {"error": str(exc)},
                schema_valid=True,
                candidate=candidate,
            )
        except Exception as exc:  # pragma: no cover - validator failures vary by integration.
            return self._fallback(
                request,
                "generation_or_validation_failed",
                started,
                {"error": f"{type(exc).__name__}: {exc}"},
                schema_valid=True,
                candidate=candidate,
            )

        if not validation_accepts_repair(validation):
            return self._fallback(
                request,
                "semantic_validation_failed",
                started,
                validation,
                schema_valid=True,
                candidate=candidate,
            )

        return DCCDRepairResult(
            case_id=request.case_id,
            attempted=True,
            accepted=True,
            schema_valid=True,
            semantic_accepted=True,
            corrected_certificate=candidate.final_certificate,
            local_model_used=self.model_id,
            validation_result=validation,
            fallback_reason=None,
            rejection_reason=None,
            preserved_repair_hint=request.repair_hint,
            runtime_s=round(time.perf_counter() - started, 6),
            candidate=candidate,
        )

    def _fallback(
        self,
        request: CertificateRepairRequest,
        fallback_reason: str,
        started: float,
        validation: Mapping[str, Any],
        *,
        schema_valid: bool,
        candidate: DCCDRepairCandidate | None,
    ) -> DCCDRepairResult:
        validation_dict = dict(validation)
        return DCCDRepairResult(
            case_id=request.case_id,
            attempted=True,
            accepted=False,
            schema_valid=schema_valid,
            semantic_accepted=False,
            corrected_certificate=None,
            local_model_used=self.model_id,
            validation_result=validation_dict,
            fallback_reason=fallback_reason,
            rejection_reason=classify_dccd_rejection(fallback_reason, validation_dict),
            preserved_repair_hint=request.repair_hint,
            runtime_s=round(time.perf_counter() - started, 6),
            candidate=candidate,
        )


def build_dccd_repair_prompt(
    request: CertificateRepairRequest,
    config: DCCDRepairConfig | None = None,
) -> str:
    """Build the bounded draft-conditioned JSON-only repair prompt."""

    cfg = config or DCCDRepairConfig()
    fields = {
        "case_id": request.case_id,
        "original_prompt": _bounded(request.original_prompt, cfg.max_field_chars),
        "draft_certificate": _bounded(request.current_certificate, cfg.max_field_chars),
        "repair_hint": _bounded(request.repair_hint, cfg.max_field_chars),
        "validator_feedback": _bounded(request.validator_error, cfg.max_field_chars),
    }
    payload = json.dumps(fields, sort_keys=True, indent=2)
    schema = json.dumps(DCCD_REPAIR_OUTPUT_SCHEMA, sort_keys=True, indent=2)
    return (
        "REQ-VERIFY-1428 draft-conditioned constrained Carnot repair.\n"
        "Return JSON only. No markdown fences, prose, chain-of-thought, or "
        "fields outside the allowed schema.\n\n"
        "Input contract:\n"
        f"{payload}\n\n"
        "Allowed DCCD repair output schema:\n"
        f"{schema}\n"
    )


def build_dccd_retry_prompt(
    request: CertificateRepairRequest,
    *,
    failed_output: str,
    validation_error_message: str,
    include_validation_error_context: bool,
    config: DCCDRepairConfig | None = None,
) -> str:
    """Build the Exp 1464 retry prompt without changing the first-attempt path.

    Exp 1464 compares two retry prompts after the same failed candidate.  The
    baseline retry sees the failed output only.  The context retry sees that
    same failed output plus the validator's exact complaint, so any measured
    delta is attributable to the new retry context rather than a different
    source case or output schema.
    """

    cfg = config or DCCDRepairConfig()
    fields = {
        "case_id": request.case_id,
        "original_prompt": _bounded(request.original_prompt, cfg.max_field_chars),
        "draft_certificate": _bounded(request.current_certificate, cfg.max_field_chars),
        "repair_hint": _bounded(request.repair_hint, cfg.max_field_chars),
        "failed_model_output": _bounded(failed_output, cfg.max_field_chars),
    }
    if include_validation_error_context:
        fields["validation_error_message"] = _bounded(
            validation_error_message,
            cfg.max_field_chars,
        )

    payload = json.dumps(fields, sort_keys=True, indent=2)
    schema = json.dumps(DCCD_REPAIR_OUTPUT_SCHEMA, sort_keys=True, indent=2)
    return (
        "REQ-VERIFY-1464 DCCD repair retry. Retry contract: repair the failed "
        "candidate while preserving the bounded DCCD output schema.\n"
        "Return JSON only. No markdown fences, prose, chain-of-thought, or "
        "fields outside the allowed schema.\n\n"
        "Retry input contract:\n"
        f"{payload}\n\n"
        "Allowed DCCD repair output schema:\n"
        f"{schema}\n"
    )


def parse_dccd_repair_model_output(
    text: str,
    config: DCCDRepairConfig | None = None,
) -> DCCDRepairCandidate:
    """Parse and validate one DCCD repair object before semantic validation."""

    cfg = config or DCCDRepairConfig()
    try:
        payload = json.loads(_extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise DCCDRepairOutputSchemaError(f"invalid JSON repair output: {exc}") from exc
    if not isinstance(payload, dict):
        raise DCCDRepairOutputSchemaError("repair output must be a JSON object")

    required = set(DCCD_REPAIR_OUTPUT_SCHEMA["required"])
    missing = sorted(required - set(payload))
    if missing:
        raise DCCDRepairOutputSchemaError(f"missing required section(s): {missing}")

    allowed = set(DCCD_REPAIR_OUTPUT_SCHEMA["properties"])
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise DCCDRepairOutputSchemaError(f"unexpected repair output field(s): {unexpected}")

    draft = _section(payload, "draft_certificate")
    action = _section(payload, "repair_action")
    final = _section(payload, "final_certificate")
    metadata = _section(payload, "validator_metadata")

    draft_text = _required_string(draft, "draft_certificate.certificate_text")
    draft_state = _required_string(draft, "draft_certificate.state")
    action_type = _required_string(action, "repair_action.action_type")
    target = _required_string(action, "repair_action.target")
    rationale = _required_string(action, "repair_action.rationale")
    final_text = _required_string(final, "final_certificate.certificate_text")
    final_state = _required_string(final, "final_certificate.state")

    if len(draft_text) > cfg.max_output_chars:
        raise DCCDRepairOutputSchemaError("draft_certificate.certificate_text exceeds max_output_chars")
    if len(final_text) > cfg.max_output_chars:
        raise DCCDRepairOutputSchemaError("final_certificate.certificate_text exceeds max_output_chars")

    return DCCDRepairCandidate(
        draft_certificate=draft_text,
        draft_state=draft_state,
        repair_action_type=action_type,
        repair_target=target,
        repair_rationale=rationale,
        final_certificate=final_text,
        final_state=final_state,
        validator_metadata=dict(metadata),
    )


def classify_dccd_rejection(
    fallback_reason: str | None,
    validation_result: Mapping[str, Any],
) -> str:
    """Classify one rejected v2 candidate into a stable audit reason."""

    fallback = str(fallback_reason or "")
    validation = dict(validation_result)
    if fallback == "schema_validation_failed":
        return "schema_validation_failed"
    if fallback == "timeout":
        return "timeout"
    if fallback == "generation_or_validation_failed":
        return "generation_or_validation_failed"
    if validation.get("fallback_reason") == "no_validator_injected":
        return "validator_mismatch_no_validator_injected"
    if validation.get("constraint_passed") is False:
        return "constraint_failed"
    if validation.get("semantic_result") != "SAT":
        return "semantic_result_not_sat"
    if validation.get("repair_required") is True:
        return "repair_required_still_true"
    if validation.get("false_acceptance") is True:
        return "false_acceptance_guard"
    return "semantic_validation_failed"


def _section(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise DCCDRepairOutputSchemaError(f"{name} must be a JSON object")

    if name in {"draft_certificate", "final_certificate"}:
        allowed = {"certificate_text", "state"}
    elif name == "repair_action":
        allowed = {"action_type", "target", "rationale"}
    else:
        return dict(value)

    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise DCCDRepairOutputSchemaError(f"unexpected {name} field(s): {unexpected}")
    return dict(value)


def _required_string(section: Mapping[str, Any], dotted_name: str) -> str:
    key = dotted_name.rsplit(".", 1)[-1]
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DCCDRepairOutputSchemaError(f"{dotted_name} must be a non-empty string")
    return value


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
