#!/usr/bin/env python3
"""Exp 1642 llguidance adapter for Carnot structured verdict records.

The adapter keeps Carnot's verifier output deterministic while giving local
GGUF/llama.cpp callers enough schema and grammar metadata to constrain the
generation step when optional `llguidance` bindings are present.  Hosts without
llguidance still get the same post-decode JSON validation and fail-closed
abstaining verdict records.

Spec: REQ-VERIFY-1642, SCENARIO-VERIFY-1642.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.pipeline.verdict_record import VerdictRecord, calibrated_confidence_from_energy
from carnot.verifiers.dccd_adapter import extract_json_object, validate_json_schema

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = "experiment_1642_llguidance.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE

EXPERIMENT_ID = 1642
RUN_DATE = "20260509"
ADAPTER_SCHEMA_VERSION = "carnot.llguidance_structured_verdict_adapter.v1"
SPEC_TRACES = ["REQ-VERIFY-1642", "SCENARIO-VERIFY-1642"]
REQUIRED_VERDICT_FIELDS = (
    "verdict",
    "energy",
    "calibrated_confidence",
    "producing_tier",
    "tier_reached",
    "rationale",
    "budget_ms_consumed",
    "repairs_applied",
    "extras",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "experiment_id",
    "adapter_module",
    "adapter_success",
    "llguidance_backend_available",
    "fallback_backend_available",
    "llama_cpp_adapter_ready",
    "structured_verdict_roundtrip",
    "invalid_output_abstains",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class LlGuidanceBackend:
    """JSON-safe status for the optional llguidance grammar compiler."""

    backend_name: str
    llguidance_backend_available: bool
    fallback_backend_available: bool
    grammar: str | None
    llguidance_version: str | None
    grammar_validation_message: str | None
    grammar_error: str | None

    def to_dict(self) -> JsonDict:
        """Return diagnostics that artifacts and verdict extras can serialize."""

        return {
            "backend_name": self.backend_name,
            "llguidance_backend_available": self.llguidance_backend_available,
            "fallback_backend_available": self.fallback_backend_available,
            "grammar_compiled": self.grammar is not None and self.grammar_error is None,
            "llguidance_version": self.llguidance_version,
            "grammar_validation_message": self.grammar_validation_message,
            "grammar_error": self.grammar_error,
        }


class LlGuidanceStructuredVerdictAdapter:
    """Bridge `VerdictRecord` JSON to llguidance/llama.cpp constrained decoding."""

    def __init__(
        self,
        *,
        schema: Mapping[str, Any] | None = None,
        llguidance_module: object | None = None,
        probe_llguidance: bool = True,
    ) -> None:
        self.schema = _json_clone(verdict_record_schema() if schema is None else schema)
        self._backend = _compile_llguidance_backend(
            self.schema,
            llguidance_module=llguidance_module,
            probe_llguidance=probe_llguidance,
        )

    @property
    def grammar(self) -> str | None:
        """Return the compiled grammar string, or None when fallback mode is active."""

        return self._backend.grammar

    def backend_diagnostics(self) -> JsonDict:
        """Expose backend status without forcing callers to import llguidance."""

        return self._backend.to_dict()

    def build_prompt(self, task_text: str) -> str:
        """Create the prompt paired with the schema for local structured generation."""

        return (
            "Return exactly one Carnot VerdictRecord JSON object. "
            "Do not include prose before or after the JSON.\n"
            f"Task:\n{task_text}\n"
            f"JSON schema:\n{_canonical_json(self.schema)}\n"
        )

    def build_llama_cpp_metadata(self, task_text: str) -> JsonDict:
        """Return prompt and kwargs a llama.cpp caller can use for generation."""

        llama_cpp_kwargs: JsonDict = {
            "temperature": 0.0,
            "max_tokens": 256,
        }
        if self.grammar is not None:
            llama_cpp_kwargs["grammar"] = self.grammar
        else:
            llama_cpp_kwargs["response_format"] = {"type": "json_object"}

        return {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "prompt": self.build_prompt(task_text),
            "json_schema": _json_clone(self.schema),
            "grammar": self.grammar,
            "backend": self.backend_diagnostics(),
            "llama_cpp_kwargs": llama_cpp_kwargs,
        }

    def record_to_json(self, record: VerdictRecord) -> str:
        """Serialize a Carnot verdict only after validating the public JSON shape."""

        payload = record.to_dict()
        schema_errors = validate_json_schema(self.schema, payload)
        if schema_errors:
            raise ValueError(f"VerdictRecord payload invalid: {schema_errors[0]}")
        return _canonical_json(payload)

    def parse_generated_verdict(self, raw_output: str) -> VerdictRecord:
        """Parse generated text into a `VerdictRecord`, abstaining on invalid JSON."""

        parsed = extract_json_object(raw_output)
        schema_errors = (
            ["$ is not a JSON object"]
            if parsed is None
            else validate_json_schema(self.schema, parsed)
        )
        if schema_errors:
            return self._abstain_record(
                schema_errors=schema_errors,
                parsed_payload={} if parsed is None else parsed,
            )

        return VerdictRecord(
            verdict=parsed["verdict"],
            energy=float(parsed["energy"]),
            calibrated_confidence=float(parsed["calibrated_confidence"]),
            producing_tier=int(parsed["producing_tier"]),
            tier_reached=int(parsed["tier_reached"]),
            rationale=str(parsed["rationale"]),
            budget_ms_consumed=float(parsed["budget_ms_consumed"]),
            repairs_applied=[str(item) for item in parsed["repairs_applied"]],
            extras=dict(parsed["extras"]),
        )

    def _abstain_record(
        self,
        *,
        schema_errors: Sequence[str],
        parsed_payload: Mapping[str, Any],
    ) -> VerdictRecord:
        energy = 1.0 + (0.1 * len(schema_errors))
        return VerdictRecord(
            verdict="abstain",
            energy=energy,
            calibrated_confidence=calibrated_confidence_from_energy(energy),
            producing_tier=0,
            tier_reached=0,
            rationale="structured_output_invalid",
            budget_ms_consumed=0.0,
            extras={
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "schema_errors": list(schema_errors),
                "parsed_payload": _json_clone(parsed_payload),
                "backend": self.backend_diagnostics(),
            },
        )


def verdict_record_schema() -> JsonDict:
    """Return the bounded JSON schema for Carnot's structured verdict record."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_VERDICT_FIELDS),
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail", "abstain"]},
            "energy": {"type": "number"},
            "calibrated_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "producing_tier": {"type": "integer", "minimum": 0},
            "tier_reached": {"type": "integer", "minimum": 0},
            "rationale": {"type": "string"},
            "budget_ms_consumed": {"type": "number", "minimum": 0.0},
            "repairs_applied": {"type": "array", "items": {"type": "string"}},
            "extras": {"type": "object"},
        },
    }


def compiler_uses_arbitrary_code_execution() -> bool:
    """Return whether adapter compilation evaluates model-generated code."""

    return False


def build_artifact(
    *,
    tests_run: Sequence[str] | None = None,
    llguidance_module: object | None = None,
    probe_llguidance: bool = True,
) -> JsonDict:
    """Build the deterministic Exp 1642 artifact without writing it."""

    adapter = LlGuidanceStructuredVerdictAdapter(
        llguidance_module=llguidance_module,
        probe_llguidance=probe_llguidance,
    )
    fixture = _fixture_verdict_record()
    encoded = adapter.record_to_json(fixture)
    roundtrip = adapter.parse_generated_verdict(encoded)
    invalid = adapter.parse_generated_verdict(
        '{"verdict":"pass","energy":"low","calibrated_confidence":1.4}'
    )
    metadata = adapter.build_llama_cpp_metadata("Verify the candidate response.")
    diagnostics = adapter.backend_diagnostics()

    structured_verdict_roundtrip = (
        roundtrip.verdict == fixture.verdict
        and roundtrip.energy == fixture.energy
        and roundtrip.extras == fixture.extras
    )
    invalid_output_abstains = (
        invalid.verdict == "abstain" and bool(invalid.extras.get("schema_errors"))
    )
    llama_cpp_adapter_ready = (
        bool(metadata["prompt"])
        and isinstance(metadata["json_schema"], dict)
        and bool(metadata["llama_cpp_kwargs"])
    )
    adapter_success = (
        structured_verdict_roundtrip
        and invalid_output_abstains
        and llama_cpp_adapter_ready
        and diagnostics["fallback_backend_available"]
        and not compiler_uses_arbitrary_code_execution()
    )

    artifact: JsonDict = {
        "status": "complete" if adapter_success else "blocked",
        "run_date": RUN_DATE,
        "experiment_id": EXPERIMENT_ID,
        "spec_traces": SPEC_TRACES,
        "adapter_module": "scripts.experiment_1642_llguidance",
        "adapter_class": "LlGuidanceStructuredVerdictAdapter",
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "adapter_success": adapter_success,
        "llguidance_backend_available": diagnostics["llguidance_backend_available"],
        "fallback_backend_available": diagnostics["fallback_backend_available"],
        "grammar_compiled": diagnostics["grammar_compiled"],
        "llama_cpp_adapter_ready": llama_cpp_adapter_ready,
        "structured_verdict_roundtrip": structured_verdict_roundtrip,
        "invalid_output_abstains": invalid_output_abstains,
        "arbitrary_code_execution_path_introduced": compiler_uses_arbitrary_code_execution(),
        "backend_diagnostics": diagnostics,
        "llama_cpp_metadata": metadata,
        "adapter_rows": {
            "roundtrip": roundtrip.to_dict(),
            "invalid": invalid.to_dict(),
        },
        "tests_run": list(tests_run or []),
        "honest_verdict": (
            "complete: llguidance structured-verdict adapter exposes llama.cpp "
            "metadata and preserves deterministic abstaining fallback validation"
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert the Exp 1642 artifact is internally consistent."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    assert not missing, f"missing required fields: {missing}"
    if artifact["status"] == "complete":
        assert artifact["adapter_success"] is True, "adapter_success required for complete"
    assert artifact["structured_verdict_roundtrip"] is True, "structured_verdict_roundtrip"
    assert artifact["invalid_output_abstains"] is True, "invalid_output_abstains"
    assert artifact["llama_cpp_adapter_ready"] is True, "llama_cpp_adapter_ready"
    assert artifact["fallback_backend_available"] is True, "fallback_backend_available"
    assert (
        artifact["arbitrary_code_execution_path_introduced"] is False
    ), "arbitrary code execution is forbidden"


def run_experiment(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    tests_run: Sequence[str] | None = None,
    llguidance_module: object | None = None,
    probe_llguidance: bool = True,
) -> JsonDict:
    """Write the Exp 1642 terminal artifact and return the JSON payload."""

    artifact = build_artifact(
        tests_run=tests_run,
        llguidance_module=llguidance_module,
        probe_llguidance=probe_llguidance,
    )
    return _write_json(output_path, artifact)


def _compile_llguidance_backend(
    schema: Mapping[str, Any],
    *,
    llguidance_module: object | None,
    probe_llguidance: bool,
) -> LlGuidanceBackend:
    module = llguidance_module
    if module is None:
        if not probe_llguidance:
            return _fallback_backend("llguidance probing disabled")
        try:
            module = importlib.import_module("llguidance")
        except ImportError:
            return _fallback_backend("llguidance not installed")

    matcher = getattr(module, "LLMatcher", None)
    grammar_from_schema = getattr(matcher, "grammar_from_json_schema", None)
    if not callable(grammar_from_schema):
        return _fallback_backend("llguidance LLMatcher.grammar_from_json_schema unavailable")

    try:
        grammar = grammar_from_schema(
            _json_clone(schema),
            overrides={"whitespace_flexible": False},
        )
        validator = getattr(matcher, "validate_grammar", None)
        validation_message = str(validator(grammar)) if callable(validator) else ""
        if validation_message and not validation_message.startswith("WARNING"):
            return _fallback_backend(validation_message, grammar=str(grammar))
        version_fn = getattr(module, "get_version", None)
        version = str(version_fn()) if callable(version_fn) else None
    except Exception as exc:
        return _fallback_backend(f"{type(exc).__name__}: {exc}")

    return LlGuidanceBackend(
        backend_name="llguidance",
        llguidance_backend_available=True,
        fallback_backend_available=True,
        grammar=str(grammar),
        llguidance_version=version,
        grammar_validation_message=validation_message or None,
        grammar_error=None,
    )


def _fallback_backend(error: str, *, grammar: str | None = None) -> LlGuidanceBackend:
    return LlGuidanceBackend(
        backend_name="post_decode_fallback",
        llguidance_backend_available=False,
        fallback_backend_available=True,
        grammar=grammar,
        llguidance_version=None,
        grammar_validation_message=None,
        grammar_error=error,
    )


def _fixture_verdict_record() -> VerdictRecord:
    return VerdictRecord(
        verdict="pass",
        energy=0.0,
        calibrated_confidence=0.93,
        producing_tier=3,
        tier_reached=3,
        rationale="constraints_satisfied",
        budget_ms_consumed=2.5,
        repairs_applied=["none"],
        extras={"case_id": "case-1642", "source": "fixture"},
    )


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
